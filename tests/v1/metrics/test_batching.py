# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from pathlib import Path

import prometheus_client
from transformers import GPT2Config

from vllm.config import ModelConfig, VllmConfig
from vllm.v1.metrics.batching import PrefillBatchingStats
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
        reasons=["mamba_alignment", "mamba_alignment", "request_tail"],
    )
    assert stats.batches == 2
    assert stats.batch_tokens == 145
    assert stats.batch_requests == 4  # Includes one decode row in the mixed batch.
    assert stats.request_tokens == 144
    assert stats.request_forward_observations == 3
    assert stats.unused_token_slots == 111
    assert stats.underfilled_batches == 1
    assert stats.underfilled_reasons == {"mamba_alignment": 1, "request_tail": 1}


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
            reasons=["request_tail"],
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
        assert samples["vllm:prefill_completed_requests_total"] == 1
        assert samples["vllm:prefill_request_forwards_total"] == 3
        assert not any(key.endswith("_bucket") for key in samples)
    finally:
        unregister_vllm_metrics()
