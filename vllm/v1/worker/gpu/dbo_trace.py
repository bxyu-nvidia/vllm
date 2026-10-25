# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in per-forward diagnostics, reusing DBO timing events without GPU waits.

GPU offsets use each forward's first compute event, NOT a cross-rank clock.
Host monotonic timestamps can be compared only on the same host. NVTX ranges
identify the CUDA submissions in Nsight, which supplies actual kernel starts.
"""

import json
import os
import socket
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, TextIO, TypedDict

import torch

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.forward_context import ForwardContext
    from vllm.v1.worker.gpu.input_batch import InputBatch
    from vllm.v1.worker.ubatch_utils import UBatchSlices

logger = init_logger(__name__)


class MicrobatchWorkload(TypedDict):
    microbatch: int
    query_tokens: int
    padding_tokens: int
    query_lens: list[int]
    prefix_lens_cpu: list[int]
    causal_attention_pairs_cpu: int
    dp_padded_tokens: list[int]


def microbatch_workloads(
    batch: "InputBatch",
    slices: "UBatchSlices",
    contexts: list["ForwardContext"],
) -> list[MicrobatchWorkload]:
    """Snapshot CPU lengths before async runner buffers are reused.

    Prefix lengths are CPU estimates (potential upper bounds with speculation).
    Causal pairs are a dense-attention work proxy, not FLOPs or predicted time.
    Token slices can cut requests; account for the prefix of the cut request.
    Padding-only microbatches have no logical queries or attention work.
    """
    result = []
    for index, (s, context) in enumerate(zip(slices, contexts)):
        queries, prefixes = [], []
        for req in range(
            s.request_slice.start, min(s.request_slice.stop, batch.num_reqs)
        ):
            req_start = int(batch.query_start_loc_np[req])
            start = max(s.token_slice.start, req_start)
            end = min(
                s.token_slice.stop,
                int(batch.query_start_loc_np[req + 1]),
                batch.num_tokens,
            )
            if end > start:
                queries.append(end - start)
                prefixes.append(
                    int(batch.num_computed_tokens_np[req]) + start - req_start
                )
        metadata = context.dp_metadata
        result.append(
            MicrobatchWorkload(
                microbatch=index,
                query_tokens=sum(queries),
                padding_tokens=max(1, s.num_tokens) - sum(queries),
                query_lens=queries,
                prefix_lens_cpu=prefixes,
                causal_attention_pairs_cpu=sum(
                    q * c + q * (q + 1) // 2 for q, c in zip(queries, prefixes)
                ),
                dp_padded_tokens=(
                    metadata.num_tokens_across_dp_cpu.tolist()
                    if metadata is not None
                    else []
                ),
            )
        )
    return result


class DBOTraceWriter:
    """One JSONL file per worker; write after the existing output-copy wait."""

    def __init__(
        self, directory: str, *, dp_rank: int, device_index: int | None
    ) -> None:
        self.hostname = socket.gethostname()
        self.pid = os.getpid()
        self.dp_rank = dp_rank
        self.device_index = device_index
        self._step = 0
        self._file: TextIO | None = None
        path = Path(directory) / (
            f"dbo-{self.hostname}-dp{dp_rank}-pid{self.pid}-{time.time_ns()}.jsonl"
        )
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._file = path.open("w", buffering=1)
        except OSError:
            logger.warning(
                "Cannot open DBO trace %s; disabling diagnostics", path, exc_info=True
            )
        else:
            logger.info("Writing per-forward DBO diagnostics to %s", path)

    def new_trace(
        self,
        batch: "InputBatch",
        slices: "UBatchSlices",
        contexts: list["ForwardContext"],
    ) -> "DBOTrace | None":
        if self._file is None:
            return None
        trace = DBOTrace(
            self, self._step, microbatch_workloads(batch, slices, contexts)
        )
        self._step += 1
        return trace

    def write(self, record: dict[str, object]) -> None:
        if self._file is None:
            return
        try:
            self._file.write(json.dumps(record, separators=(",", ":")) + "\n")
        except OSError:
            logger.warning(
                "Cannot write DBO trace; disabling diagnostics", exc_info=True
            )
            self.close()

    def close(self) -> None:
        if self._file is not None:
            file, self._file = self._file, None
            try:
                file.close()
            except OSError:
                logger.warning("Cannot close DBO trace", exc_info=True)


@dataclass
class _Span:
    microbatch: int
    phase: str
    ordinal: int
    start: torch.Event
    host_start_ns: int
    end: torch.Event | None = None
    host_end_ns: int = 0


class DBOTrace:
    """One eligible forward, indexed by a worker-local sequence and NVTX IDs."""

    def __init__(
        self, writer: DBOTraceWriter, step: int, workloads: list[MicrobatchWorkload]
    ) -> None:
        self.writer, self.step, self.workloads = writer, step, workloads
        self._active: dict[int, _Span] = {}
        self._spans: list[_Span] = []
        self._counts: dict[tuple[int, str], int] = {}
        self.created_unix_ns = time.time_ns()

    def start(self, ubatch: int, phase: str, event: torch.Event) -> None:
        ordinal = self._counts.get((ubatch, phase), 0)
        self._counts[ubatch, phase] = ordinal + 1
        span = _Span(ubatch, phase, ordinal, event, time.monotonic_ns())
        self._active[ubatch] = span
        self._spans.append(span)
        torch.cuda.nvtx.range_push(
            f"dbo:step={self.step}:dp={self.writer.dp_rank}:ubatch={ubatch}"
            f":{phase}={ordinal}"
        )

    def stop(self, ubatch: int, event: torch.Event) -> None:
        span = self._active.pop(ubatch)
        span.end, span.host_end_ns = event, time.monotonic_ns()
        torch.cuda.nvtx.range_pop()

    def resolve(self, origin: torch.Event) -> None:
        """Serialize completed stream windows; no waits or new CUDA events."""
        assert not self._active
        windows = []
        totals = [{"compute_seconds": 0.0, "comm_seconds": 0.0} for _ in self.workloads]
        for span in self._spans:
            assert span.end is not None
            start = origin.elapsed_time(span.start) / 1000
            end = origin.elapsed_time(span.end) / 1000
            totals[span.microbatch][f"{span.phase}_seconds"] += end - start
            windows.append(
                {
                    "microbatch": span.microbatch,
                    "phase": span.phase,
                    "ordinal": span.ordinal,
                    "gpu_start_offset_seconds": start,
                    "gpu_end_offset_seconds": end,
                    "host_start_monotonic_ns": span.host_start_ns,
                    "host_end_monotonic_ns": span.host_end_ns,
                }
            )
        self.writer.write(
            {
                "schema_version": 1,
                "hostname": self.writer.hostname,
                "pid": self.writer.pid,
                "dp_rank": self.writer.dp_rank,
                "device_index": self.writer.device_index,
                "step": self.step,
                "created_unix_ns": self.created_unix_ns,
                "microbatches": [
                    dict(work, **total) for work, total in zip(self.workloads, totals)
                ],
                "windows": windows,
            }
        )
