# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import NvFp4MoeBackend
from vllm.model_executor.layers.quantization.utils import flashinfer_fp4_moe
from vllm.model_executor.layers.quantization.utils.flashinfer_fp4_moe import (
    prepare_nvfp4_moe_layer_for_fi_or_cutlass,
)
from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
    align_fp4_moe_hidden_dim_for_fi,
    align_fp4_moe_weights_for_fi,
)


def test_shared_nvfp4_input_scales_have_writable_storage(monkeypatch):
    monkeypatch.setattr(flashinfer_fp4_moe, "swizzle_blockscale", lambda x: x)

    num_experts = 3
    layer = SimpleNamespace(
        activation=SimpleNamespace(is_gated=False),
        moe_config=SimpleNamespace(
            moe_parallel_config=SimpleNamespace(enable_eplb=False)
        ),
    )
    w13 = torch.zeros((num_experts, 2, 1), dtype=torch.uint8)
    w2 = torch.zeros((num_experts, 2, 1), dtype=torch.uint8)
    w13_scale = torch.zeros((num_experts, 2, 1), dtype=torch.float8_e4m3fn)
    w2_scale = torch.zeros((num_experts, 2, 1), dtype=torch.float8_e4m3fn)
    weight_scale = torch.ones(num_experts)

    outputs = prepare_nvfp4_moe_layer_for_fi_or_cutlass(
        backend=NvFp4MoeBackend.FLASHINFER_CUTLASS,
        layer=layer,
        w13=w13,
        w13_scale=w13_scale,
        w13_scale_2=weight_scale,
        a13_scale=torch.tensor([1.0, 2.0, 3.0]),
        w2=w2,
        w2_scale=w2_scale,
        w2_scale_2=weight_scale,
        a2_scale=torch.tensor([4.0, 5.0, 6.0]),
        is_act_and_mul=False,
    )
    a13_scale, a2_scale = outputs[3], outputs[7]

    torch.testing.assert_close(a13_scale, torch.full((num_experts,), 3.0))
    torch.testing.assert_close(a2_scale, torch.full((num_experts,), 6.0))
    distinct_values = torch.arange(num_experts, dtype=torch.float32)
    a13_scale.copy_(distinct_values)
    a2_scale.copy_(distinct_values)
    torch.testing.assert_close(a13_scale, distinct_values)
    torch.testing.assert_close(a2_scale, distinct_values)


def test_align_trtllm_fp4_moe_hidden_dim_noop():
    w13 = torch.arange(2 * 8 * 256, dtype=torch.uint8).reshape(2, 8, 256)
    w13_scale = torch.arange(2 * 8 * 32, dtype=torch.uint8).reshape(2, 8, 32)
    w2 = torch.arange(2 * 512 * 4, dtype=torch.uint8).reshape(2, 512, 4)
    w2_scale = torch.arange(2 * 512 * 1, dtype=torch.uint8).reshape(2, 512, 1)

    out_w13, out_w13_scale, out_w2, out_w2_scale, padded_hidden = (
        align_fp4_moe_hidden_dim_for_fi(w13, w13_scale, w2, w2_scale)
    )

    assert padded_hidden == 512
    assert out_w13 is w13
    assert out_w13_scale is w13_scale
    assert out_w2 is w2
    assert out_w2_scale is w2_scale


def test_align_trtllm_fp4_moe_hidden_dim_pads_to_256_multiple():
    hidden_dim = 2688
    padded_hidden_dim = 2816

    w13 = torch.arange(2 * 12 * (hidden_dim // 2), dtype=torch.uint8).reshape(
        2, 12, hidden_dim // 2
    )
    w13_scale = torch.arange(2 * 12 * (hidden_dim // 16), dtype=torch.uint8).reshape(
        2, 12, hidden_dim // 16
    )

    w2 = torch.arange(2 * hidden_dim * 6, dtype=torch.uint8).reshape(2, hidden_dim, 6)
    w2_scale = torch.arange(2 * hidden_dim * 2, dtype=torch.uint8).reshape(
        2, hidden_dim, 2
    )

    out_w13, out_w13_scale, out_w2, out_w2_scale, out_hidden_dim = (
        align_fp4_moe_hidden_dim_for_fi(w13, w13_scale, w2, w2_scale)
    )

    assert out_hidden_dim == padded_hidden_dim
    assert out_w13.shape == (2, 12, padded_hidden_dim // 2)
    assert out_w13_scale.shape == (2, 12, padded_hidden_dim // 16)
    assert out_w2.shape == (2, padded_hidden_dim, 6)
    assert out_w2_scale.shape == (2, padded_hidden_dim, 2)

    torch.testing.assert_close(out_w13[:, :, : hidden_dim // 2], w13)
    torch.testing.assert_close(out_w13_scale[:, :, : hidden_dim // 16], w13_scale)
    torch.testing.assert_close(out_w2[:, :hidden_dim, :], w2)
    torch.testing.assert_close(out_w2_scale[:, :hidden_dim, :], w2_scale)

    assert torch.count_nonzero(out_w13[:, :, hidden_dim // 2 :]) == 0
    assert torch.count_nonzero(out_w13_scale[:, :, hidden_dim // 16 :]) == 0
    assert torch.count_nonzero(out_w2[:, hidden_dim:, :]) == 0
    assert torch.count_nonzero(out_w2_scale[:, hidden_dim:, :]) == 0


def test_align_trtllm_fp4_moe_intermediate_pads_gate_and_up_separately():
    # Qwen3.8-Flash-Next has intermediate_size=640. At TP4 each rank owns
    # 160 values, which TRTLLM pads to the next 64-value boundary (192).
    intermediate = 160
    padded_intermediate = 192
    hidden_dim = 32

    gate = torch.ones((1, intermediate, hidden_dim // 2), dtype=torch.uint8)
    up = torch.full_like(gate, 2)
    w13 = torch.cat((gate, up), dim=1)
    gate_scale = torch.full((1, intermediate, hidden_dim // 16), 3, dtype=torch.uint8)
    up_scale = torch.full_like(gate_scale, 4)
    w13_scale = torch.cat((gate_scale, up_scale), dim=1)
    w2 = torch.full((1, hidden_dim, intermediate // 2), 5, dtype=torch.uint8)
    w2_scale = torch.full((1, hidden_dim, intermediate // 16), 6, dtype=torch.uint8)

    out_w13, out_w13_scale, out_w2, out_w2_scale, out_intermediate = (
        align_fp4_moe_weights_for_fi(
            w13,
            w13_scale,
            w2,
            w2_scale,
            is_act_and_mul=True,
            min_alignment=64,
        )
    )

    assert out_intermediate == padded_intermediate
    torch.testing.assert_close(out_w13[:, :intermediate], gate)
    torch.testing.assert_close(
        out_w13[:, padded_intermediate : padded_intermediate + intermediate], up
    )
    torch.testing.assert_close(out_w13_scale[:, :intermediate], gate_scale)
    torch.testing.assert_close(
        out_w13_scale[:, padded_intermediate : padded_intermediate + intermediate],
        up_scale,
    )

    assert torch.count_nonzero(out_w13[:, intermediate:padded_intermediate]) == 0
    assert torch.count_nonzero(out_w13[:, padded_intermediate + intermediate :]) == 0
    assert torch.count_nonzero(out_w13_scale[:, intermediate:padded_intermediate]) == 0
    assert (
        torch.count_nonzero(out_w13_scale[:, padded_intermediate + intermediate :]) == 0
    )
    torch.testing.assert_close(out_w2[:, :, : intermediate // 2], w2)
    torch.testing.assert_close(out_w2_scale[:, :, : intermediate // 16], w2_scale)
    assert torch.count_nonzero(out_w2[:, :, intermediate // 2 :]) == 0
    assert torch.count_nonzero(out_w2_scale[:, :, intermediate // 16 :]) == 0


@pytest.mark.parametrize("dwdp", [False, True])
def test_nvfp4_reload_refreshes_scales_without_moving_captured_storage(dwdp):
    from vllm.model_executor.layers.fused_moe.experts.prims_nvfp4_moe import (
        PrimsNvFp4Experts,
    )
    from vllm.model_executor.layers.fused_moe.experts.trtllm_nvfp4_moe import (
        TrtLlmNvFp4ExpertsBase,
    )
    from vllm.model_executor.layers.quantization.online.nvfp4 import (
        Nvfp4OnlineMoEMethod,
    )

    cls = PrimsNvFp4Experts if dwdp else TrtLlmNvFp4ExpertsBase
    experts = cls.__new__(cls)
    experts.quant_config = SimpleNamespace(
        g1_alphas=torch.tensor([2.0, 4.0]),
        g2_alphas=torch.ones(2),
        a2_gscale=torch.tensor([3.0, 5.0]),
        _w1=SimpleNamespace(scale=None),
        _w2=SimpleNamespace(scale=None),
    )
    method = object.__new__(Nvfp4OnlineMoEMethod)
    method.moe_quant_config = experts.quant_config
    method.moe_kernel = SimpleNamespace(fused_experts=experts)
    experts.moe_config = SimpleNamespace(is_act_and_mul=True)
    experts.is_situ = False
    experts._gemm1_clamp_limit_unfolded = torch.tensor([8.0, 16.0])
    experts._gemm1_beta_unfolded = torch.tensor([4.0, 12.0])
    experts.gemm1_clamp_limit = torch.empty(2)
    experts.gemm1_beta = torch.empty(2)
    experts.gemm1_alpha = torch.ones(2)
    layer = torch.nn.Module()
    for name in ("g1_scale_c", "gemm1_clamp_limit", "gemm1_beta", "gemm1_alpha"):
        layer.register_parameter(
            name, torch.nn.Parameter(torch.zeros(2), requires_grad=False)
        )
    layer.dwdp_manager = SimpleNamespace(merge_free=True) if dwdp else None
    for name in ("w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale"):
        layer.register_parameter(
            name, torch.nn.Parameter(torch.ones(2, 2), requires_grad=False)
        )
    pointers = {name: param.data_ptr() for name, param in layer.named_parameters()}
    for factor in (1.0, 2.0):
        layer.w13_weight_scale_2 = torch.tensor([2.0, 4.0]) * factor
        layer.w2_weight_scale_2 = torch.tensor([7.0, 9.0]) * factor
        method.finalize_weight_reload(layer)
        assert experts.quant_config._w1.scale is layer.w13_weight_scale
        assert experts.quant_config._w2.scale is layer.w2_weight_scale
        torch.testing.assert_close(
            experts.quant_config.g2_alphas, layer.w2_weight_scale_2
        )
        torch.testing.assert_close(layer.g1_scale_c, torch.tensor([6.0, 20.0]) * factor)
        torch.testing.assert_close(
            layer.gemm1_clamp_limit, torch.tensor([4.0, 4.0]) / factor
        )
        torch.testing.assert_close(layer.gemm1_beta, torch.tensor([2.0, 3.0]) / factor)
        assert experts.g1_scale_c is layer.g1_scale_c
        assert experts.gemm1_clamp_limit is layer.gemm1_clamp_limit
        assert {
            name: param.data_ptr() for name, param in layer.named_parameters()
        } == pointers
        if dwdp:
            assert isinstance(experts, PrimsNvFp4Experts)
            assert experts._resident_weights is not None
            assert experts._dwdp_scales_ready
            assert (
                experts._resident_weights[0].data_ptr() == layer.w13_weight.data_ptr()
            )
