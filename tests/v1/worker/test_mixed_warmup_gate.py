# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the max_num_reqs gate on the V2 mixed prefill+decode warmup."""

from types import SimpleNamespace

import pytest

from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.worker.gpu import warmup
from vllm.v1.worker.gpu.warmup import run_mixed_prefill_decode_warmup


def _fail(*args, **kwargs):
    raise AssertionError("worker callback must not run when warmup is skipped")


@pytest.mark.parametrize("max_num_reqs", [1, 0])
def test_mixed_warmup_skipped_for_single_seq(max_num_reqs):
    """A mixed prefill+decode step needs >=2 requests; with max_num_reqs < 2
    the warmup must be skipped without touching the worker callbacks."""
    runner = SimpleNamespace(is_pooling_model=False, max_num_reqs=max_num_reqs)

    assert (
        run_mixed_prefill_decode_warmup(
            runner,
            worker_execute_model=_fail,
            worker_sample_tokens=_fail,
            num_tokens=128,
        )
        is False
    )


@pytest.mark.parametrize("fail_warmup", [False, True])
def test_kernel_warmup_restores_uncalibrated_adaptive_manager(monkeypatch, fail_warmup):
    """Startup must warm fixed drafts before calibration and retain its manager."""
    manager = SimpleNamespace(cost_tables=None)
    rejection_sampler = SimpleNamespace(enable_adaptive_verification=True)
    runner = SimpleNamespace(
        adaptive_verification=manager,
        rejection_sampler=rejection_sampler,
    )

    def run_steps(model_runner, execute, sample):
        assert model_runner.adaptive_verification is None
        assert not model_runner.rejection_sampler.enable_adaptive_verification
        if fail_warmup:
            raise RuntimeError("warmup failed")

    monkeypatch.setattr(warmup, "_warmup_kernels", run_steps)
    if fail_warmup:
        with pytest.raises(RuntimeError, match="warmup failed"):
            warmup.warmup_kernels(runner, _fail, _fail)
    else:
        warmup.warmup_kernels(runner, _fail, _fail)
    assert runner.adaptive_verification is manager
    assert manager.cost_tables is None
    assert rejection_sampler.enable_adaptive_verification


@pytest.mark.parametrize("last_pp_rank", [False, True])
@pytest.mark.parametrize(
    "schedule,async_scheduling,expect_drafts",
    [
        ([(1, 1024, 0)], True, False),
        ([(1, 4, 0), (5, 8, 3)], True, False),
        ([(1, 2, 0), (3, 4, 3)], True, True),
        ([(1, 4, 3)], True, True),
        ([(1, 4, 0)], False, True),
        (None, True, True),
    ],
)
def test_kernel_warmup_does_not_consume_disabled_mtp_drafts(
    monkeypatch, last_pp_rank, schedule, async_scheduling, expect_drafts
):
    from tests.v1.worker.test_gpu_warmup_blocks import (
        _attention_group,
        _make_runner,
    )

    runner = _make_runner([_attention_group()], 5, num_spec_steps=5)
    runner.is_last_pp_rank = last_pp_rank
    runner.scheduler_config.async_scheduling = async_scheduling
    runner.vllm_config.speculative_config = SimpleNamespace(
        num_speculative_tokens_per_batch_size=schedule
    )
    # No GPU kernels run: inspect the actual hand-built SchedulerOutputs.
    monkeypatch.setattr(warmup.torch.accelerator, "synchronize", lambda: None)
    outputs: list[SchedulerOutput] = []
    warmup.warmup_kernels(runner, outputs.append, lambda _: None)
    decode_outputs = [o for o in outputs if o.scheduled_cached_reqs.req_ids]
    assert decode_outputs
    assert set(outputs[0].num_scheduled_tokens.values()) == {7}
    assert any(o.scheduled_spec_decode_tokens for o in decode_outputs) == expect_drafts
    if not expect_drafts:
        for output in decode_outputs:
            assert set(output.num_scheduled_tokens.values()) == {1}
    assert runner.num_speculative_steps == 5
    assert runner.decode_query_len == 6
