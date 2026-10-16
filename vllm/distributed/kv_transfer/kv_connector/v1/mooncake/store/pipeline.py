# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Timing and live backlog for the Mooncake lookup and load pipelines."""

import threading
import time
from collections.abc import Callable
from typing import Literal, TypedDict

PIPELINE_KEY = "_pipeline"
PIPELINE_BUCKETS = (
    0.001,
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
    30.0,
    60.0,
    120.0,
)


class PipelineSnapshot(TypedDict):
    stage: str
    rank: int
    queued: int
    active: int
    oldest_pending_seconds: float
    queue_wait_seconds: list[float]
    execution_seconds: list[float]


class PipelineTracker:
    """Track jobs until completion, including cancelled queued jobs.

    Lookup rank -1 denotes scheduler-side, engine-wide work. Load ranks are
    worker ranks, not distinct logical requests across tensor-parallel workers.
    """

    def __init__(
        self,
        stage: Literal["lookup", "load"],
        rank: int,
        *,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.stage = stage
        self.rank = rank
        self._clock = clock
        self._lock = threading.Lock()
        self._next_ticket = 0
        self._pending: dict[int, tuple[float, float | None]] = {}
        self._queue_wait: list[float] = []
        self._execution: list[float] = []

    def enqueue(self) -> int:
        with self._lock:
            ticket = self._next_ticket
            self._next_ticket += 1
            self._pending[ticket] = (self._clock(), None)
            return ticket

    def start(self, ticket: int) -> None:
        with self._lock:
            enqueued, _ = self._pending[ticket]
            now = self._clock()
            self._pending[ticket] = (enqueued, now)
            self._queue_wait.append(now - enqueued)

    def finish(self, ticket: int) -> None:
        with self._lock:
            pending = self._pending.pop(ticket, None)
            if pending is not None and pending[1] is not None:
                self._execution.append(self._clock() - pending[1])

    def snapshot(self) -> PipelineSnapshot:
        """Read live gauges and drain timing samples exactly once."""
        with self._lock:
            now = self._clock()
            active = sum(started is not None for _, started in self._pending.values())
            oldest = max(
                (now - enqueued for enqueued, _ in self._pending.values()),
                default=0.0,
            )
            result = PipelineSnapshot(
                stage=self.stage,
                rank=self.rank,
                queued=len(self._pending) - active,
                active=active,
                oldest_pending_seconds=oldest,
                queue_wait_seconds=self._queue_wait,
                execution_seconds=self._execution,
            )
            self._queue_wait = []
            self._execution = []
            return result
