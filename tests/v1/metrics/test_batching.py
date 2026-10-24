# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from pathlib import Path
from queue import Queue
from unittest.mock import Mock

import msgspec
import prometheus_client
import pytest
from transformers import GPT2Config

from vllm.config import ModelConfig, ObservabilityConfig, VllmConfig
from vllm.engine.arg_utils import EngineArgs
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.v1.metrics.batching import (
    OUTCOME_COUNTER_FIELDS,
    UNDERFILL_REASONS,
    PrefillBatchingStats,
)
from vllm.v1.metrics.dbo import DBOStats
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
        stats.observe_outcomes(
            {"scheduled": 1, "mamba_alignment": 1, "not_examined": 2},
            {"mamba_alignment": 2},
            underfilled=True,
        )
        stats.readiness_steps = 3
        stats.ready_requests = 4
        stats.ready_unscheduled_requests = 1
        stats.unexamined_requests = 2
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
        assert samples["vllm:prefill_readiness_steps_total"] == 3
        assert samples["vllm:prefill_ready_requests_total"] == 4
        assert samples["vllm:prefill_ready_unscheduled_requests_total"] == 1
        assert samples["vllm:prefill_unexamined_requests_total"] == 2
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
        assert samples["vllm:prefill_outcome_steps_total"] == 1
        assert samples["vllm:prefill_underfilled_outcome_steps_total"] == 1
        for field, (label, values) in OUTCOME_COUNTER_FIELDS.items():
            for value in values:
                assert prometheus_client.REGISTRY.get_sample_value(
                    f"vllm:prefill_{field}_total",
                    {"model_name": str(tmp_path), "engine": "0", label: value},
                ) == getattr(stats, field).get(value, 0)
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
    assert stats.dbo_stats is None
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


def test_outcome_averages_have_separate_idle_full_and_underfilled_denominators() -> (
    None
):
    stats = PrefillBatchingStats()
    stats.observe_outcomes({}, {}, underfilled=False)
    stats.observe_outcomes({"kv_lookup_pending": 2}, {}, underfilled=False)
    stats.observe_outcomes({"scheduled": 4}, {}, underfilled=False)
    stats.observe_outcomes(
        {"scheduled": 1, "mamba_alignment": 1, "not_examined": 2},
        {"mamba_alignment": 2},
        underfilled=True,
    )
    stats.observe_outcomes({"scheduled": 2}, {}, underfilled=True)
    assert stats.outcome_steps == 4
    assert stats.underfilled_outcome_steps == 2
    assert stats.request_outcomes == {
        "scheduled": 7,
        "kv_lookup_pending": 2,
        "mamba_alignment": 1,
        "not_examined": 2,
    }
    assert stats.underfilled_request_outcomes == {
        "scheduled": 3,
        "mamba_alignment": 1,
        "not_examined": 2,
    }
    assert stats.underfilled_unexamined_by_stop_reason == {"mamba_alignment": 2}


def test_dbo_flag_is_opt_in():
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    assert not EngineArgs.from_cli_args(parser.parse_args([])).enable_dbo_metrics
    assert EngineArgs.from_cli_args(
        parser.parse_args(["--enable-dbo-metrics"])
    ).enable_dbo_metrics


@pytest.mark.parametrize("enabled", [False, True])
def test_dbo_metrics_export_per_engine_without_counting_empty_updates(
    tmp_path: Path, enabled: bool
):
    GPT2Config(architectures=["GPT2LMHeadModel"]).save_pretrained(tmp_path)
    config = VllmConfig(
        model_config=ModelConfig(model=str(tmp_path)),
        observability_config=ObservabilityConfig(enable_dbo_metrics=enabled),
    )
    try:
        logger = PrometheusStatLogger(config, engine_indexes=[0, 1])
        stats = SchedulerStats(
            dbo_stats=DBOStats(
                outcome="executed",
                microbatch_tokens=[9, 9],
                microbatch_requests=[1, 1],
                microbatch_padding_tokens=[0, 6],
            )
        )
        # Exercise Python's nested dataclass MessagePack transport.
        decoded = msgspec.msgpack.decode(
            msgspec.msgpack.encode(stats), type=SchedulerStats
        )
        assert decoded.dbo_stats == stats.dbo_stats
        for _ in range(2):
            logger.record(decoded, None, engine_idx=0)
        logger.record(SchedulerStats(), None, engine_idx=0)
        logger.record(
            SchedulerStats(dbo_stats=DBOStats(outcome="disabled")), None, engine_idx=1
        )
        samples = [
            sample
            for family in prometheus_client.REGISTRY.collect()
            for sample in family.samples
            if sample.name.startswith("vllm:dbo_")
            and not sample.name.endswith("_created")
        ]
        if not enabled:
            assert not samples
            return

        def value(name, engine, **labels):
            return [
                sample.value
                for sample in samples
                if sample.name == "vllm:dbo_" + name
                and sample.labels
                == {"model_name": str(tmp_path), "engine": str(engine), **labels}
            ]

        assert value("steps_total", 0, outcome="executed") == [2]
        assert value("steps_total", 1, outcome="disabled") == [1]
        assert value("microbatch_tokens_sum", 0, microbatch="0") == [18]
        assert value("microbatch_tokens_count", 0, microbatch="1") == [2]
        assert value("microbatch_requests_sum", 0, microbatch="1") == [2]
        assert value("microbatch_padding_tokens_sum", 0, microbatch="1") == [12]
        assert not [
            s for s in samples if s.labels["engine"] == "1" and "microbatch" in s.name
        ]
    finally:
        unregister_vllm_metrics()
