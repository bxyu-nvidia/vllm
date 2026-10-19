# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Join vLLM and Mooncake JSON logs for one full prefix hash.

Enable VLLM_MAMBA_CACHE_TRACE=1 on every vLLM process and
MOONCAKE_CACHE_TRACE=1 on the Mooncake master/storage processes. Set the same
VLLM_MAMBA_CACHE_TRACE_RUN_ID on all processes. Preserve logs from startup.
The logs include token IDs. No tensor downloads or additional storage RPCs are
performed. Debug logging is synchronous and is not intended for benchmarking.

Example:
    .venv/bin/python tools/mamba_cache_trace.py --hash FULL_HEX_HASH \
        --run-id experiment-1 prefill.log decode.log mooncake.log

GPU counts are per residency and Mamba group. Store counts are per object key
and include rank-local transfers; they must not be summed as request resumes.
Missing events/history and disk reclamation without storage-side evidence
remain unknown. Timestamps are host observation times; compare monotonic times
only inside one process. Consumers are observed scheduler completions, not
per-token causal attribution.

"""

import argparse
import json
import sys
from collections.abc import Iterator
from pathlib import Path

MARKER = "MAMBA_CACHE_TRACE "


def read_events(paths: list[Path], run_id: str | None) -> Iterator[dict[str, object]]:
    seen = set()
    for path in paths:
        with path.open(errors="replace") as stream:
            for line_number, line in enumerate(stream, 1):
                if MARKER not in line:
                    continue
                try:
                    event = json.loads(line.split(MARKER, 1)[1])
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Truncated/invalid trace at {path}:{line_number}"
                    ) from error
                if run_id is not None and event.get("run_id") != run_id:
                    continue
                identity = (event["instance_id"], event.get("sequence"))
                if identity[1] is not None:
                    if identity in seen:
                        continue
                    seen.add(identity)
                yield event


def report(
    paths: list[Path], prefix_hash: str, run_id: str | None = None
) -> dict[str, object]:
    """Reconstruct observed residencies, store operations, and request tokens."""
    gpu: dict[str, dict] = {}
    objects: dict[str, dict] = {}
    request_ids: set[tuple[str | None, str]] = set()
    for event in read_events(paths, run_id):
        key = event.get("object_key", "")
        if event.get("prefix_hash") != prefix_hash and not key.endswith(
            "@" + prefix_hash
        ):
            continue
        rid = event.get("request_id") or event.get("producer_request_id")
        if rid:
            request_ids.add((event.get("engine_id"), rid))
        residency = event.get("residency_id")
        if residency:
            state = gpu.setdefault(residency, {"history": [], "consumers": {}})
            state["history"].append(event)
            if "registered_ns" in event and event["sequence"] >= state.get(
                "sequence", -1
            ):
                state.update({k: v for k, v in event.items() if k != "consumers"})
            consumers = event.get("consumers", [])
            if "consumer" in event:
                consumers = [*consumers, event["consumer"]]
            if event["event"] == "checkpoint_compute":
                consumers = [event]
            for consumer in consumers:
                consumer_id = consumer["request_id"]
                request_ids.add((event.get("engine_id"), consumer_id))
                consumer_identity = (consumer_id, consumer["resumed_ns"])
                previous = state["consumers"].get(consumer_identity)
                if (
                    previous is None
                    or consumer["computed_end"] >= previous["computed_end"]
                ):
                    state["consumers"][consumer_identity] = consumer
        if key:
            object_identity = (event.get("tenant_id", "default"), key)
            state = objects.setdefault(
                object_identity,
                {
                    "object_key": key,
                    "tenant_id": object_identity[0],
                    "observed_lookup_hits": 0,
                    "observed_successful_loads": 0,
                    "transfers": [],
                    "replicas": {},
                    "storage_reclamations": [],
                },
            )
            if event["event"] == "mooncake_lookup_hit":
                state["observed_lookup_hits"] += 1
            elif event["event"] in ("mooncake_load", "mooncake_put"):
                state["transfers"].append(event)
                if event["event"] == "mooncake_load" and event["success"]:
                    state["observed_successful_loads"] += 1
            elif "replica_id" in event:
                replica_key = f"{event['instance_id']}:{event['replica_id']}"
                previous = state["replicas"].get(replica_key)
                if (
                    previous is None
                    or event["timestamp_ns"] >= previous["timestamp_ns"]
                ):
                    state["replicas"][replica_key] = event
            elif event["event"] == "mooncake_storage_reclaimed":
                state["storage_reclamations"].append(event)

    requests: dict[tuple[str, str], dict] = {}
    for event in read_events(paths, run_id):
        rid = event.get("request_id")
        if (event.get("engine_id"), rid) not in request_ids:
            continue
        if event["event"] != "request_tokens":
            continue
        identity = (event["instance_id"], rid)
        request = requests.setdefault(
            identity,
            {
                "instance_id": event["instance_id"],
                "engine_id": event.get("engine_id"),
                "request_id": rid,
                "prompt_length": event["prompt_length"],
                "tokens": {},
            },
        )
        request["tokens"].update(enumerate(event["token_ids"], event["start"]))
    for request in requests.values():
        tokens = request.pop("tokens")
        end = max(tokens, default=-1) + 1
        ordered = [tokens.get(i) for i in range(end)]
        prompt_length = request["prompt_length"]
        request["prompt_token_ids"] = ordered[:prompt_length]
        request["output_token_ids"] = ordered[prompt_length:]
        request["has_missing_tokens"] = None in ordered or end < prompt_length
    for state in gpu.values():
        state["consumers"] = list(state["consumers"].values())
        state["history"].sort(key=lambda event: event["sequence"])
    for state in objects.values():
        state["replicas"] = list(state["replicas"].values())
    return {
        "prefix_hash": prefix_hash,
        "run_id": run_id,
        "gpu_residencies": list(gpu.values()),
        "mooncake_objects": list(objects.values()),
        "requests": list(requests.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hash", required=True, help="Full prefix hash in hex")
    parser.add_argument("--run-id")
    parser.add_argument("logs", nargs="+", type=Path)
    args = parser.parse_args()
    try:
        result = report(args.logs, args.hash.lower(), args.run_id)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    json.dump(result, sys.stdout, indent=2)
    print()


if __name__ == "__main__":
    main()
