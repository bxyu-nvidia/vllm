# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Filter removal summaries for a full prefix hash.

Enable VLLM_MAMBA_CACHE_TRACE=1 and MOONCAKE_CACHE_TRACE=1, with Mooncake
-minloglevel=0. Each producer emits only hash_removed records. GPU counts cover
one residency/group; Mooncake counts cover one object key across its replicas.
The store sees replica-list requests, not completed transfers or computation.

Example:
    .venv/bin/python tools/mamba_cache_trace.py --hash FULL_HEX_HASH server.log

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
    """Return matching removal summaries without merging distinct residencies."""
    return {
        "prefix_hash": prefix_hash,
        "run_id": run_id,
        "removals": [
            event
            for event in read_events(paths, run_id)
            if event.get("event") == "hash_removed"
            and (
                event.get("prefix_hash") == prefix_hash
                or str(event.get("object_key", "")).endswith("@" + prefix_hash)
            )
        ],
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
