# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only tests of aggregation, sampling and nonblocking CUDA event handling."""

import json
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock

import pytest
import torch

from vllm.v1.metrics import phase_timing as pt


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    clock = SimpleNamespace(seconds=0.0)
    monkeypatch.setattr(pt.time, "monotonic", lambda: clock.seconds)
    monkeypatch.setattr(pt.time, "perf_counter_ns", lambda: int(clock.seconds * 1e9))
    monkeypatch.setattr(pt.logger, "info", Mock())
    return clock


def report(timer: pt.PhaseTimer) -> dict[str, Any]:
    result = timer.flush(force=True)
    assert result is not None
    return result


def test_cpu_summary_window_and_json(clock: SimpleNamespace) -> None:
    timer = pt.PhaseTimer(labels={"component": "scheduler"}, interval_seconds=10)
    for duration in (0.001, 0.002, 0.007):
        with timer.measure("schedule"):
            clock.seconds += duration
    assert timer.flush() is None
    clock.seconds = 10
    result = timer.flush()
    assert result is not None
    stats = cast(dict[str, Any], result["cpu_ms"])["schedule"]
    assert stats == pytest.approx(
        dict(
            count=3,
            total=10,
            mean=10 / 3,
            max=7,
            percentile_samples=3,
            p50=2,
            p90=7,
            p99=7,
        )
    )
    assert result["window_seconds"] == 10
    args = cast(Mock, pt.logger.info).call_args.args
    assert args[0] == "VLLM_PREFILL_TIMING %s"
    assert json.loads(args[1]) == result
    assert timer.flush(force=True) is None
    with timer.measure("schedule"):
        clock.seconds += 0.004
    assert report(timer)["cpu_ms"]["schedule"]["count"] == 1


@pytest.fixture
def cuda_events(monkeypatch: pytest.MonkeyPatch) -> list[Mock]:
    events: list[Mock] = []

    def make_event(*, enable_timing: bool) -> Mock:
        assert enable_timing
        event = Mock(spec=["record", "query", "elapsed_time"])
        event.query.return_value = False
        event.elapsed_time.return_value = 3.5
        events.append(event)
        return event

    monkeypatch.setattr(torch.cuda, "Event", make_event)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: "entry_stream")
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: pytest.fail("GPU sync"))
    monkeypatch.setattr(
        torch.accelerator, "synchronize", lambda: pytest.fail("GPU sync")
    )
    return events


def test_pending_events_do_not_block_and_pool_is_bounded(
    clock: SimpleNamespace, cuda_events: list[Mock]
) -> None:
    timer = pt.PhaseTimer(labels={}, cuda=True, max_pending=1)
    with timer.measure("forward", cuda=True):
        clock.seconds += 0.002
    first = report(timer)
    assert first["pending_cuda_ranges"] == 1
    assert first["cuda_elapsed_ms"] == {}
    assert first["cpu_ms"]["forward"]["total"] == 2
    cuda_events[0].elapsed_time.assert_not_called()

    with timer.measure("forward", cuda=True):
        pass
    dropped = report(timer)
    assert len(cuda_events) == 2
    assert dropped["counts"]["cuda_ranges_dropped_capacity"] == 1
    assert dropped["cpu_ms"]["forward"]["count"] == 1

    cuda_events[1].query.return_value = True
    ready = report(timer)
    assert ready["pending_cuda_ranges"] == 0
    assert ready["cuda_elapsed_ms"]["forward"]["mean"] == 3.5
    cuda_events[0].elapsed_time.assert_called_once_with(cuda_events[1])
    with timer.measure("forward", cuda=True):
        pass
    assert len(cuda_events) == 2  # Completed pair is reused, not reallocated.
    for event in cuda_events:
        assert event.record.call_count == 2
        event.record.assert_called_with("entry_stream")


def test_sampling_excludes_dummy_steps_and_keeps_sample_phase(
    clock: SimpleNamespace, cuda_events: list[Mock]
) -> None:
    class Runner:
        phase_timing = pt.PhaseTimer(labels={}, cuda=True, sample_every=2, active=False)

        @pt.timed_phase("execute", cuda=True, step=True)
        def execute(self, output=None, tensors=None, dummy_run=False) -> int:
            self.phase_timing.observe_batch(tokens=8)
            return 42

        @pt.timed_phase("sample", cuda=True)
        def sample(self) -> None:
            pass

    runner = Runner()
    assert runner.execute(None, None, True) == 42  # Positional dummy_run.
    runner.sample()
    assert not cuda_events
    assert runner.phase_timing.flush(force=True) is None
    for _ in range(3):
        runner.execute()
        runner.sample()
        clock.seconds += 0.001
    result = report(runner.phase_timing)
    assert result["counts"]["real_steps"] == 3
    assert result["counts"]["sampled_steps"] == 2
    assert result["cpu_ms"]["execute"]["count"] == 2
    assert result["cpu_ms"]["sample"]["count"] == 2
    assert result["cpu_ms"]["execute_start_gap"]["count"] == 2
    assert result["batch"]["tokens"]["total"] == 16


def test_capture_does_not_insert_events(
    clock: SimpleNamespace, cuda_events: list[Mock], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    timer = pt.PhaseTimer(labels={}, cuda=True)
    with timer.measure("forward", cuda=True):
        pass
    assert not cuda_events
    assert report(timer)["counts"]["cuda_ranges_skipped_capture"] == 1
    timer.begin_step()
    with timer.measure("execute", cuda=True):
        pass
    assert timer.flush(force=True) is None


def test_nested_ranges_and_exceptions_preserve_behavior(clock: SimpleNamespace) -> None:
    timer = pt.PhaseTimer(labels={})
    with pytest.raises(ValueError, match="original error"), timer.measure("outer"):
        clock.seconds += 0.001
        with timer.measure("inner"):
            clock.seconds += 0.002
            raise ValueError("original error")
    stats = report(timer)["cpu_ms"]
    assert stats["outer"]["total"] == 3
    assert stats["inner"]["total"] == 2


def test_disabled_timer_does_not_touch_cuda_or_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VLLM_PREFILL_TIMING", "0")
    monkeypatch.setattr(torch.cuda, "Event", lambda **_: pytest.fail("CUDA event"))
    assert pt.make_phase_timer(cast(Any, None), component="worker") is None

    @pt.timed_phase("execute", cuda=True, step=True)
    def method(owner: object, value: int) -> int:
        return value + 1

    assert method(object(), 7) == 8


@pytest.mark.parametrize("role", ["kv_producer", "kv_consumer", None])
def test_factory_roles_and_scheduler_sampling(
    monkeypatch: pytest.MonkeyPatch, role: str | None
) -> None:
    monkeypatch.setenv("VLLM_PREFILL_TIMING", "1")
    monkeypatch.setenv("VLLM_PREFILL_TIMING_SAMPLE_EVERY", "4")
    config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(kv_role=role) if role else None,
        parallel_config=SimpleNamespace(data_parallel_index=2, rank=3),
    )
    timer = pt.make_phase_timer(
        cast(Any, config), component="worker", device=torch.device("cuda")
    )
    scheduler = pt.make_phase_timer(cast(Any, config), component="scheduler")
    if role == "kv_consumer":
        assert timer is None and scheduler is None
        return
    assert timer is not None and scheduler is not None
    assert timer.sample_every == 4 and not timer.active
    assert scheduler.sample_every == 1 and scheduler.active
    assert timer.labels["dp_rank"] == 2
    assert timer.labels["scope"] == ("prefill_engine" if role else "mixed_engine")


def test_percentiles_are_bounded_but_totals_cover_entire_window(
    clock: SimpleNamespace,
) -> None:
    timer = pt.PhaseTimer(labels={})
    with timer.measure("execute"):
        for value in range(5000):
            timer.observe_batch(tokens=value)
    stats = report(timer)["batch"]["tokens"]
    assert stats["count"] == 5000
    assert stats["total"] == sum(range(5000))
    assert stats["percentile_samples"] == 4096
    assert stats["p50"] == 2951


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_real_cuda_graph_replay_timing() -> None:
    # Exercise real events around a replay without inserting events into capture.
    tensor = torch.zeros(1024, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        tensor.add_(1)
    timer = pt.PhaseTimer(labels={}, cuda=True)
    with timer.measure("forward", cuda=True):
        graph.replay()
    # Only the test synchronizes, to make the result deterministic at assertion.
    torch.accelerator.synchronize()
    result = report(timer)
    assert result["pending_cuda_ranges"] == 0
    assert result["cuda_elapsed_ms"]["forward"]["count"] == 1
    assert result["cuda_elapsed_ms"]["forward"]["total"] > 0
    assert torch.equal(tensor, torch.ones_like(tensor))
