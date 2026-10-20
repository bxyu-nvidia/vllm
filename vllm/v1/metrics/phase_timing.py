# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in CPU/CUDA phase timings, aggregated into periodic JSON log records.

VLLM_PREFILL_TIMING=1 instruments prefill producers and standalone engines;
decode-only consumers are excluded. Standalone records may include decode.
VLLM_PREFILL_TIMING_INTERVAL_SECONDS defaults to 10. Worker sampling is controlled
by VLLM_PREFILL_TIMING_SAMPLE_EVERY (default 1); scheduler CPU ranges are unsampled.

CPU durations include blocking/submission, not just CPU execution. CUDA durations
are elapsed intervals on the entry stream, including waits and launch gaps, not
kernel-only compute time. Other streams are included only through existing
dependencies. Nested ranges overlap and must not be added. Pending CUDA results
are assigned to the window in which they become readable, without synchronizing.
Percentiles use at most the most recent 4096 observations per phase/window.
"""

import functools
import json
import math
import os
import socket
import time
from collections import Counter, deque
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Concatenate, ParamSpec, TypeVar

import torch

import vllm.envs as envs
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)
P = ParamSpec("P")
R = TypeVar("R")
T = TypeVar("T")


@dataclass
class _Samples:
    count: int = 0
    total: float = 0.0
    maximum: float = 0.0
    recent: deque[float] = field(default_factory=lambda: deque(maxlen=4096))

    def add(self, value: float) -> None:
        self.count += 1
        self.total += value
        self.maximum = max(self.maximum, value)
        self.recent.append(value)

    def summary(self) -> dict[str, float | int]:
        values = sorted(self.recent)
        return {
            "count": self.count,
            "total": self.total,
            "mean": self.total / self.count,
            "max": self.maximum,
            "percentile_samples": len(values),
            **{
                f"p{p}": values[math.ceil(p / 100 * len(values)) - 1]
                for p in (50, 90, 99)
            },
        }


class PhaseTimer:
    """Bounded, non-blocking timing collector owned by one scheduler/worker.

    This collector is used only on the owning execution thread, outside compiled
    model code. It never inserts cross-stream waits or calls synchronize().
    """

    def __init__(
        self,
        *,
        labels: dict[str, str | int],
        interval_seconds: float = 10.0,
        sample_every: int = 1,
        cuda: bool = False,
        max_pending: int = 256,
        active: bool = True,
    ) -> None:
        if not math.isfinite(interval_seconds) or interval_seconds <= 0:
            raise ValueError("VLLM_PREFILL_TIMING_INTERVAL_SECONDS must be positive")
        if sample_every < 1 or max_pending < 1:
            raise ValueError(
                "Timing sample interval and pending limit must be positive"
            )
        self.labels = labels
        self.interval_seconds = interval_seconds
        self.sample_every = sample_every
        self.cuda = cuda
        self.active = active
        self._max_pending = max_pending
        self._allocated = 0
        self._free: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        self._pending: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []
        self._cpu: dict[str, _Samples] = {}
        self._gpu: dict[str, _Samples] = {}
        self._batch: dict[str, _Samples] = {}
        self._counts: Counter[str] = Counter()
        self._window_start: float | None = None
        self._last_step_start: float | None = None
        self._step = 0
        self._depth = 0

    def begin_step(self, *, dummy: bool = False) -> None:
        """Select a real worker step, excluding warmup and graph capture."""
        self.active = False
        if dummy:
            return
        if self.cuda and torch.cuda.is_current_stream_capturing():
            return
        now = time.monotonic()
        if self._window_start is None:
            self._window_start = now
        self.flush()
        self._counts["real_steps"] += 1
        if self._last_step_start is not None:
            self._add(
                self._cpu, "execute_start_gap", (now - self._last_step_start) * 1000
            )
        self._last_step_start = now
        self.active = self._step % self.sample_every == 0
        self._step += 1
        if self.active:
            self._counts["sampled_steps"] += 1

    def observe_batch(self, **values: int) -> None:
        """Record workload sizes for the selected worker steps."""
        if self.active:
            for name, value in values.items():
                self._add(self._batch, name, value)

    @staticmethod
    def _add(samples: dict[str, _Samples], name: str, value: float) -> None:
        entry = samples.get(name)
        if entry is None:
            entry = samples[name] = _Samples()
        entry.add(value)

    @contextmanager
    def measure(self, name: str, *, cuda: bool = False) -> Iterator[None]:
        """Measure a host scope and optionally its entry-stream CUDA interval."""
        if not self.active:
            yield
            return
        if self._window_start is None:
            self._window_start = time.monotonic()
        pair = None
        stream = None
        if cuda and self.cuda:
            if torch.cuda.is_current_stream_capturing():
                self._counts["cuda_ranges_skipped_capture"] += 1
            else:
                if self._free:
                    pair = self._free.pop()
                elif self._allocated < self._max_pending:
                    pair = (
                        torch.cuda.Event(enable_timing=True),
                        torch.cuda.Event(enable_timing=True),
                    )
                    self._allocated += 1
                else:
                    self._counts["cuda_ranges_dropped_capacity"] += 1
                if pair is not None:
                    stream = torch.cuda.current_stream()
                    pair[0].record(stream)
        self._depth += 1
        start = time.perf_counter_ns()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter_ns() - start) / 1e6
            self._add(self._cpu, name, elapsed_ms)
            if pair is not None:
                pair[1].record(stream)
                self._pending.append((name, *pair))
            self._depth -= 1
            if self._depth == 0:
                self.flush()

    def flush(self, *, force: bool = False) -> dict[str, object] | None:
        """Log ready observations; leave incomplete CUDA events queued."""
        pending = []
        for name, start, end in self._pending:
            if end.query():
                self._add(self._gpu, name, start.elapsed_time(end))
                self._free.append((start, end))
            else:
                pending.append((name, start, end))
        self._pending = pending
        now = time.monotonic()
        if self._window_start is None:
            return None
        elapsed = now - self._window_start
        if not force and elapsed < self.interval_seconds:
            return None
        if not (self._cpu or self._gpu or self._batch or self._counts):
            return None
        record: dict[str, object] = {
            **self.labels,
            "timestamp": time.time(),
            "window_seconds": elapsed,
            "sample_every": self.sample_every,
            "cpu_ms": {k: v.summary() for k, v in self._cpu.items()},
            "cuda_elapsed_ms": {k: v.summary() for k, v in self._gpu.items()},
            "batch": {k: v.summary() for k, v in self._batch.items()},
            "counts": dict(self._counts),
            "pending_cuda_ranges": len(self._pending),
        }
        # Preserve bounded samples so analysis can pool token-size percentiles.
        if (tokens := self._batch.get("scheduled_tokens")) is not None:
            record["scheduled_token_samples"] = list(tokens.recent)
        logger.info("VLLM_PREFILL_TIMING %s", json.dumps(record, separators=(",", ":")))
        self._cpu.clear()
        self._gpu.clear()
        self._batch.clear()
        self._counts.clear()
        self._window_start = now
        return record


def make_phase_timer(
    config: "VllmConfig", *, component: str, device: torch.device | None = None
) -> PhaseTimer | None:
    """Enable timing on prefill/standalone engines, never decode consumers."""
    if not envs.VLLM_PREFILL_TIMING:
        return None
    transfer = config.kv_transfer_config
    role = transfer.kv_role if transfer is not None else "standalone"
    if role == "kv_consumer":
        return None
    parallel = config.parallel_config
    return PhaseTimer(
        labels={
            "component": component,
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "dp_rank": parallel.data_parallel_index,
            "rank": parallel.rank,
            "scope": "prefill_engine" if role == "kv_producer" else "mixed_engine",
        },
        interval_seconds=envs.VLLM_PREFILL_TIMING_INTERVAL_SECONDS,
        sample_every=(
            envs.VLLM_PREFILL_TIMING_SAMPLE_EVERY if device is not None else 1
        ),
        cuda=device is not None and device.type == "cuda",
        active=device is None,
    )


def measure_phase(
    owner: object, name: str, *, cuda: bool = False
) -> AbstractContextManager[None]:
    """Return a no-op when profiling is disabled, including bare test runners."""
    timer = getattr(owner, "phase_timing", None)
    if isinstance(timer, PhaseTimer):
        return timer.measure(name, cuda=cuda)
    return nullcontext()


def timed_phase(
    name: str, *, cuda: bool = False, step: bool = False
) -> Callable[[Callable[Concatenate[T, P], R]], Callable[Concatenate[T, P], R]]:
    """Instrument a scheduler/runner method outside compiled model code."""

    def decorate(fn: Callable[Concatenate[T, P], R]) -> Callable[Concatenate[T, P], R]:
        @functools.wraps(fn)
        def wrapped(owner: T, /, *args: P.args, **kwargs: P.kwargs) -> R:
            timer = getattr(owner, "phase_timing", None)
            if not isinstance(timer, PhaseTimer):
                return fn(owner, *args, **kwargs)
            if step:
                # MRV2 execute_model's third positional argument is dummy_run.
                dummy = kwargs.get("dummy_run", args[2] if len(args) > 2 else False)
                timer.begin_step(dummy=bool(dummy))
            with timer.measure(name, cuda=cuda):
                return fn(owner, *args, **kwargs)

        return wrapped

    return decorate
