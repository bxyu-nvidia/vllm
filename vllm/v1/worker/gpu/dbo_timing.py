# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA-event timing of eager AG/RS DBO stream windows."""

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.dbo_trace import DBOTrace

from vllm.v1.metrics.dbo import DBOTimingStats


def _union(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    result: list[tuple[float, float]] = []
    for start, end in sorted(intervals):
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(end, result[-1][1]))
        else:
            result.append((start, end))
    return result


def summarize_dbo_timing(
    compute: list[tuple[float, float]], comm: list[tuple[float, float]]
) -> DBOTimingStats:
    """Summarize timestamp intervals in seconds without double counting."""
    compute, comm = _union(compute), _union(comm)
    overlap = 0.0
    i = j = 0
    while i < len(compute) and j < len(comm):
        c_start, c_end = compute[i]
        m_start, m_end = comm[j]
        overlap += max(0.0, min(c_end, m_end) - max(c_start, m_start))
        if c_end <= m_end:
            i += 1
        else:
            j += 1
    intervals = compute + comm
    return DBOTimingStats(
        compute_seconds=sum(end - start for start, end in compute),
        comm_seconds=sum(end - start for start, end in comm),
        overlap_seconds=overlap,
        wall_seconds=(
            max(end for _, end in intervals) - min(start for start, _ in intervals)
            if intervals
            else 0.0
        ),
    )


class DBOTimingEvents:
    """One forward's events, resolved after the existing output-copy wait.

    Only one microbatch thread submits work at a time. Record windows after
    stream waits and close them before waits/yields so parked time is excluded.
    This recorder is attached only to the eager AG/RS handoff protocol.
    """

    def __init__(self, *, trace: "DBOTrace | None" = None) -> None:
        self.trace = trace
        self._active: dict[int, tuple[str, torch.Stream, torch.Event]] = {}
        self._intervals: dict[str, list[tuple[torch.Event, torch.Event]]] = {
            "compute": [],
            "comm": [],
        }
        self._origin: torch.Event | None = None

    def start(self, ubatch: int, phase: str, stream: torch.Stream) -> None:
        assert ubatch not in self._active
        event = torch.Event(enable_timing=True)
        event.record(stream)
        if self._origin is None:
            self._origin = event
        self._active[ubatch] = (phase, stream, event)
        if self.trace is not None:
            self.trace.start(ubatch, phase, event)

    def stop(self, ubatch: int) -> None:
        phase, stream, start = self._active.pop(ubatch)
        end = torch.Event(enable_timing=True)
        end.record(stream)
        self._intervals[phase].append((start, end))
        if self.trace is not None:
            self.trace.stop(ubatch, end)

    def resolve(self) -> DBOTimingStats:
        """Read completed events; never synchronize or wait for GPU work here."""
        assert not self._active
        origin = self._origin
        assert origin is not None
        assert all(
            end.query() for spans in self._intervals.values() for _, end in spans
        )
        intervals = {
            phase: [
                (origin.elapsed_time(start) / 1000, origin.elapsed_time(end) / 1000)
                for start, end in spans
            ]
            for phase, spans in self._intervals.items()
        }
        if self.trace is not None:
            self.trace.resolve(origin)
            self.trace = None
        return summarize_dbo_timing(intervals["compute"], intervals["comm"])
