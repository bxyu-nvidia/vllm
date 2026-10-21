# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from pathlib import Path
from queue import Queue
from unittest.mock import Mock

import prometheus_client
import pytest
from transformers import GPT2Config

from vllm.config import ModelConfig, VllmConfig
from vllm.v1.metrics.batching import UNDERFILL_REASONS, PrefillBatchingStats
from vllm.v1.metrics.loggers import PrometheusStatLogger
from vllm.v1.metrics.prometheus import unregister_vllm_metrics
from vllm.v1.metrics.stats import SchedulerStats


def test_batching_records_work_without_histograms_and_only_underfill_reasons():
    stats = PrefillBatchingStats()
    stats.observe_batch([], total_tokens=2, requests=2, token_budget=128, reasons=[])
    stats.observe_batch(
        [64, 64],
        total_tokens=128,
        requests=2,
        token_budget=128,
        reasons=["mamba_alignment"],
    )
    stats.observe_batch(
        [16],
        total_tokens=17,
        requests=2,
        token_budget=128,
        reasons=[
            "mamba_alignment",
            "mamba_alignment",
            "request_tail",
            "mamba_alignment_budget_too_small",
            "mamba_alignment_budget_too_small",
        ],
    )
    assert stats.batches == 2
    assert stats.batch_tokens == 145
    assert stats.batch_requests == 4  # Includes one decode row in the mixed batch.
    assert stats.request_tokens == 144
    assert stats.request_forward_observations == 3
    assert stats.unused_token_slots == 111
    assert stats.underfilled_batches == 1
    assert stats.underfilled_reasons == {
        "mamba_alignment": 1,
        "request_tail": 1,
        "mamba_alignment_budget_too_small": 1,
    }


def test_python_logger_exports_prefill_counter_deltas(tmp_path: Path) -> None:
    GPT2Config(architectures=["GPT2LMHeadModel"]).save_pretrained(tmp_path)
    config = VllmConfig(model_config=ModelConfig(model=str(tmp_path)))
    try:
        logger = PrometheusStatLogger(config)
        stats = PrefillBatchingStats(completed_requests=1, request_forwards=3)
        stats.observe_batch(
            [32, 64],
            total_tokens=96,
            requests=2,
            token_budget=128,
            reasons=[
                reason
                for reason in UNDERFILL_REASONS
                if reason.startswith("mamba_alignment")
            ],
        )
        stats.kv_lookup_completed = 2
        stats.kv_lookup_duration_ns = 6_000_000
        stats.observe_delays(
            {"pending": "kv_lookup_pending"}, scheduled_tokens={}, has_prefill=False
        )
        logger.record(SchedulerStats(prefill_batching_stats=stats), None)
        logger.record(SchedulerStats(), None)
        samples = {
            sample.name: sample.value
            for family in prometheus_client.REGISTRY.collect()
            for sample in family.samples
            if sample.name.startswith("vllm:prefill_")
            and not sample.labels.get("reason")
        }
        assert samples["vllm:prefill_batch_tokens_total"] == 96
        assert samples["vllm:prefill_batches_total"] == 1
        assert samples["vllm:prefill_request_tokens_total"] == 96
        assert samples["vllm:prefill_request_forward_observations_total"] == 2
        assert samples["vllm:prefill_kv_lookup_completed_total"] == 2
        assert samples["vllm:prefill_kv_lookup_duration_ns_total"] == 6_000_000
        assert samples["vllm:prefill_delay_steps_total"] == 1
        for field in ("delayed_requests", "blocked_requests"):
            assert (
                prometheus_client.REGISTRY.get_sample_value(
                    f"vllm:prefill_{field}_by_reason_total",
                    {
                        "model_name": str(tmp_path),
                        "engine": "0",
                        "reason": "kv_lookup_pending",
                    },
                )
                == 1
            )
        assert samples["vllm:prefill_completed_requests_total"] == 1
        assert samples["vllm:prefill_request_forwards_total"] == 3
        reason_samples = {
            sample.labels["reason"]: sample.value
            for family in prometheus_client.REGISTRY.collect()
            for sample in family.samples
            if sample.name == "vllm:prefill_underfilled_batches_by_reason_total"
        }
        for reason in UNDERFILL_REASONS:
            if reason.startswith("mamba_alignment"):
                assert reason_samples[reason] == 1
        assert not any(key.endswith("_bucket") for key in samples)
    finally:
        unregister_vllm_metrics()


@pytest.mark.parametrize("lockstep", [False, True])
def test_dp_load_stats_roundtrip_with_prefill_batching_field(lockstep: bool) -> None:
    """Load-balancing messages must not put request counts in batching stats."""
    from vllm.v1.engine import EngineCoreOutputs
    from vllm.v1.engine.core import DPEngineCoreProc, EngineCoreProc
    from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

    core_type = DPEngineCoreProc if lockstep else EngineCoreProc
    core = object.__new__(core_type)
    core.publish_dp_lb_stats = True
    core.last_counts = (0, 0)
    core.scheduler = Mock()
    core.scheduler.get_request_counts.return_value = (3, 7)
    core.scheduler.get_kv_cache_usage.return_value = 0.25
    core.output_queue = Queue()
    core.step_counter = 11
    core.current_wave = 2

    core._maybe_publish_request_counts()
    destination, output = core.output_queue.get_nowait()
    decoded = MsgpackDecoder(EngineCoreOutputs).decode(MsgpackEncoder().encode(output))
    assert destination == -1
    stats = decoded.scheduler_stats
    assert stats is not None
    assert (stats.num_running_reqs, stats.num_waiting_reqs) == (3, 7)
    assert stats.kv_cache_usage == 0.25
    assert stats.prefill_batching_stats is None
    if lockstep:
        assert (stats.step_counter, stats.current_wave) == (11, 2)

    core._maybe_publish_request_counts()
    assert core.output_queue.empty()


def test_delay_counts_include_idle_steps_and_distinguish_clipped_from_blocked():
    stats = PrefillBatchingStats()
    stats.observe_delays(
        {"a": "mamba_alignment", "b": "kv_lookup_pending", "c": "kv_lookup_pending"},
        scheduled_tokens={"a": 64},
        has_prefill=True,
    )
    stats.observe_delays(
        {"b": "kv_lookup_pending", "c": "kv_load_pending"},
        scheduled_tokens={},
        has_prefill=False,
    )
    stats.observe_delays({}, scheduled_tokens={"decode": 1}, has_prefill=False)
    assert stats.delay_steps == 2
    assert stats.delayed_requests == {
        "mamba_alignment": 1,
        "kv_lookup_pending": 3,
        "kv_load_pending": 1,
    }
    assert stats.blocked_requests == {"kv_lookup_pending": 3, "kv_load_pending": 1}
