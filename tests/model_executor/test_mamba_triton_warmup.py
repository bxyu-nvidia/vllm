# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm.model_executor.warmup.mamba_triton_warmup import _warm_batch_memcpy_kernel
from vllm.platforms import current_platform


@pytest.mark.skipif(not current_platform.is_cuda_alike(), reason="CUDA is required")
def test_mamba_batch_memcpy_kernel_compiles_on_gpu() -> None:
    _warm_batch_memcpy_kernel(torch.device("cuda"))
    torch.accelerator.synchronize("cuda")


@pytest.mark.parametrize(
    "tokens,sequences,expected",
    [
        (1, 1, [1]),
        (256, 5, [1, 2, 4, 8]),
        (64, 64, [1, 2, 4, 8, 16, 32, 64]),
        (16960, 256, [1, 2, 4, 8, 16, 32, 64, 128]),
    ],
)
def test_ssd_warmup_covers_serving_chunk_buckets(
    monkeypatch, tokens, sequences, expected
):
    from vllm.model_executor.layers.mamba import mamba_mixer2 as mod
    from vllm.model_executor.layers.mamba.ops.ssd_chunk_scan import (
        chunk_scan_chunk_bucket,
    )

    monkeypatch.setattr(
        mod,
        "get_current_vllm_config",
        lambda: SimpleNamespace(
            scheduler_config=SimpleNamespace(
                max_num_batched_tokens=tokens, max_num_seqs=sequences
            )
        ),
    )
    monkeypatch.setattr(torch.accelerator, "empty_cache", lambda: None)
    layer = SimpleNamespace(
        _ssd_kernels_warmed_up=False,
        model_config=SimpleNamespace(get_mamba_chunk_size=lambda: 128),
        _warmup_ssd_kernel_shape=Mock(),
        prefix="test",
    )
    projected = torch.empty(1)
    mod.MambaMixer2._warmup_ssd_kernels(layer, projected)
    calls = layer._warmup_ssd_kernel_shape.call_args_list
    assert [call.args[1:] for call in calls] == [
        (chunks, initial) for chunks in expected for initial in (False, True)
    ]
    assert all(call.args[0] is projected for call in calls)
    assert layer._ssd_kernels_warmed_up
    # Single-chunk warmup can never populate the large-prefill cache entry.
    assert chunk_scan_chunk_bucket(130) == chunk_scan_chunk_bucket(140) == 128
    assert chunk_scan_chunk_bucket(1) != chunk_scan_chunk_bucket(130)
    mod.MambaMixer2._warmup_ssd_kernels(layer, projected)
    assert layer._warmup_ssd_kernel_shape.call_count == len(calls)


def test_ssd_failed_warmup_is_not_marked_complete(monkeypatch):
    from vllm.model_executor.layers.mamba import mamba_mixer2 as mod

    monkeypatch.setattr(
        mod,
        "get_current_vllm_config",
        lambda: SimpleNamespace(
            scheduler_config=SimpleNamespace(max_num_batched_tokens=1, max_num_seqs=1)
        ),
    )
    monkeypatch.setattr(torch.accelerator, "empty_cache", lambda: None)
    layer = SimpleNamespace(
        _ssd_kernels_warmed_up=False,
        model_config=SimpleNamespace(get_mamba_chunk_size=lambda: 128),
        _warmup_ssd_kernel_shape=Mock(
            side_effect=[RuntimeError("allocation failed"), None]
        ),
        prefix="test",
    )
    mod.MambaMixer2._warmup_ssd_kernels(layer, torch.empty(1))
    assert not layer._ssd_kernels_warmed_up
    layer._warmup_ssd_kernel_shape.side_effect = None
    mod.MambaMixer2._warmup_ssd_kernels(layer, torch.empty(1))
    assert layer._ssd_kernels_warmed_up


@pytest.mark.parametrize("initial", [False, True])
def test_ssd_warmup_uses_multichunk_packed_inputs_and_state_dtype(monkeypatch, initial):
    from vllm.model_executor.layers.mamba import mamba_mixer2 as mod

    combined = Mock()
    monkeypatch.setattr(mod, "mamba_chunk_scan_combined_varlen", combined)
    layer = SimpleNamespace(
        model_config=SimpleNamespace(get_mamba_chunk_size=lambda: 16),
        num_heads=4,
        n_groups=2,
        tp_size=2,
        head_dim=8,
        ssm_state_size=4,
        get_state_dtype=lambda: (torch.float32, torch.float32),
        A=torch.zeros(2),
        D=torch.ones(2),
        dt_bias=torch.zeros(2),
    )
    mod.MambaMixer2._warmup_ssd_kernel_shape(
        layer, torch.empty(1, dtype=torch.bfloat16), 4, initial
    )
    args = combined.call_args.kwargs
    assert args["x"].shape == (64, 2, 8)
    assert args["x"].stride() == (24, 8, 1)
    assert args["B"].stride() == args["C"].stride() == (24, 4, 1)
    assert args["x"].dtype == args["B"].dtype == args["out"].dtype == torch.bfloat16
    assert args["state_dtype"] == torch.float32
    assert args["cu_chunk_seqlens"].tolist() == [0, 16, 32, 48, 64]
    assert args["cu_seqlens"].tolist() == [0, 64]
    assert args["last_chunk_indices"].tolist() == [3]
    assert args["seq_idx"].tolist() == [0, 0, 0, 0]
    if initial:
        assert args["initial_states"].shape == (1, 2, 8, 4)
        assert args["initial_states"].dtype == torch.float32
    else:
        assert args["initial_states"] is None


@pytest.mark.parametrize("chunks,expected", [(1, 1), (3, 4), (65, 128), (140, 128)])
def test_chunk_scan_dispatch_passes_workload_to_autotuner(
    monkeypatch, chunks, expected
):
    from vllm.model_executor.layers.mamba.ops import ssd_chunk_scan as mod

    launched = {}

    class Kernel:
        def __getitem__(self, grid):
            def launch(**kwargs):
                launched.update(kwargs)
                launched["grid"] = grid({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 8})

            return launch

    monkeypatch.setattr(mod, "_chunk_scan_fwd_kernel", Kernel())
    monkeypatch.setattr(
        mod.triton,
        "next_power_of_2",
        lambda n: 1 << (n - 1).bit_length(),
        raising=False,
    )
    length = chunks * 16
    x = torch.empty(length, 2, 8)
    mod._chunk_scan_fwd(
        cb=torch.empty(chunks, 1, 16, 16),
        x=x,
        dt=torch.empty(2, chunks, 16),
        dA_cumsum=torch.empty(2, chunks, 16),
        C=torch.empty(length, 1, 4),
        states=torch.empty(chunks, 2, 8, 4),
        cu_chunk_seqlens=torch.arange(0, length + 1, 16, dtype=torch.int32),
        out=torch.empty_like(x),
        seq_idx=torch.zeros(chunks, dtype=torch.int32),
    )
    assert launched["nchunks_bucket"] == expected
    assert launched["nheads"] == 2
    # Bucketing changes tuning only: the launch still processes every real chunk.
    assert launched["grid"] == (1, chunks, 2)
