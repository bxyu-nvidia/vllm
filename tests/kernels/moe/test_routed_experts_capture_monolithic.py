# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Routed-expert capture on the monolithic (fused router + experts) MoE path.

A monolithic kernel that ``supports_routing_replay_capture`` must write, into
the leading rows of ``routing_replay_out``, exactly the experts it routed each
token to. The router logits plant an unambiguous top-k per token, so the test
checks the reported experts against that answer and not just their range.
Capture must preserve the uncaptured numerical output for the same inputs.
The NVFP4 graph case also checks replay with changing routes and per-token scales.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch

from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
    RoutingMethodType,
    fp8_w8a8_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.experts.trtllm_bf16_moe import (
    TrtLlmBf16ExpertsMonolithic,
)
from vllm.model_executor.layers.fused_moe.experts.trtllm_fp8_moe import (
    TrtLlmFp8ExpertsMonolithic,
)
from vllm.model_executor.layers.fused_moe.experts.trtllm_nvfp4_moe import (
    TrtLlmNvFp4ExpertsMonolithic,
)
from vllm.model_executor.layers.fused_moe.modular_kernel import (
    FusedMoEExpertsMonolithic,
)
from vllm.platforms import current_platform

try:
    from vllm.utils.flashinfer import has_flashinfer_trtllm_fused_moe
except ImportError:
    pytest.skip("flashinfer not available", allow_module_level=True)

if not has_flashinfer_trtllm_fused_moe() or not current_platform.is_cuda():
    pytest.skip(
        "Requires FlashInfer TRT-LLM fused MoE on CUDA",
        allow_module_level=True,
    )

if not current_platform.is_device_capability_family(100):
    pytest.skip(
        "TRT-LLM fused MoE kernels require SM100+",
        allow_module_level=True,
    )

NUM_EXPERTS = 32
NUM_GROUPS = 4  # DeepSeekV3 grouped routing
TOPK_GROUPS = 2
HIDDEN = 1024
INTERMEDIATE = 1024
DEVICE = torch.device("cuda:0")

# Fixed weights and activations: capture must not change the numerical output.
Kernel = tuple[
    FusedMoEExpertsMonolithic,
    Callable[[torch.Tensor, torch.Tensor | None], torch.Tensor],
]


def _planted_routing(num_tokens: int, top_k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Router logits with one unambiguous answer under every routing method.

    Each token gives +4 to ``top_k`` experts drawn from at most two expert
    groups and -4 to the rest, so softmax top-k, sigmoid top-k and DeepSeekV3
    grouped top-k all pick the same experts, with margins no rounding can flip.
    Returns the logits and the planted experts of each token, sorted.
    """
    group_size = NUM_EXPERTS // NUM_GROUPS
    planted = []
    for _ in range(num_tokens):
        groups = torch.randperm(NUM_GROUPS)[:TOPK_GROUPS]
        candidates = (groups[:, None] * group_size + torch.arange(group_size)).flatten()
        planted.append(candidates[torch.randperm(len(candidates))[:top_k]])
    experts = torch.stack(planted)
    logits = torch.full((num_tokens, NUM_EXPERTS), -4.0).scatter_(1, experts, 4.0)
    return logits.to(DEVICE), experts.sort(dim=-1).values.to(DEVICE)


def _routing_kwargs(routing_method: RoutingMethodType) -> dict:
    if routing_method != RoutingMethodType.DeepSeekV3:
        return {}
    return dict(
        num_expert_group=NUM_GROUPS,
        topk_group=TOPK_GROUPS,
        e_score_correction_bias=torch.zeros(
            NUM_EXPERTS, device=DEVICE, dtype=torch.bfloat16
        ),
        routed_scaling_factor=1.0,
    )


def _moe_config(
    top_k: int,
    routing_method: RoutingMethodType,
    activation: MoEActivation = MoEActivation.SILU,
) -> FusedMoEConfig:
    return FusedMoEConfig(
        num_experts=NUM_EXPERTS,
        experts_per_token=top_k,
        hidden_dim=HIDDEN,
        intermediate_size=INTERMEDIATE,
        num_local_experts=NUM_EXPERTS,
        num_logical_experts=NUM_EXPERTS,
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
        in_dtype=torch.bfloat16,
        activation=activation,
        device=DEVICE,
        routing_method=routing_method,
        max_num_tokens=16,
    )


def _bf16_kernel(top_k: int, routing_method: RoutingMethodType) -> Kernel:
    from flashinfer import shuffle_matrix_a
    from flashinfer.fused_moe import convert_to_block_layout

    def block_major_k(w: torch.Tensor) -> torch.Tensor:
        """(E, M, K) -> the BlockMajorK layout ``trtllm_bf16_moe`` expects."""
        return torch.stack(
            [
                convert_to_block_layout(shuffle_matrix_a(e.view(torch.uint8), 64), 128)
                for e in w
            ]
        ).view(torch.bfloat16)

    experts = TrtLlmBf16ExpertsMonolithic(
        moe_config=_moe_config(top_k, routing_method),
        quant_config=FusedMoEQuantConfig.make(
            quant_dtype=None,
            per_act_token_quant=False,
            per_out_ch_quant=False,
            block_shape=None,
        ),
    )
    w13 = block_major_k(
        torch.randn(NUM_EXPERTS, 2 * INTERMEDIATE, HIDDEN, device=DEVICE).bfloat16()
        * 0.1
    )
    w2 = block_major_k(
        torch.randn(NUM_EXPERTS, HIDDEN, INTERMEDIATE, device=DEVICE).bfloat16() * 0.1
    )

    inputs = (torch.randn(16, HIDDEN, device=DEVICE) * 0.1).bfloat16()

    def forward(
        router_logits: torch.Tensor, routing_replay_out: torch.Tensor | None
    ) -> torch.Tensor:
        hidden_states = inputs[: len(router_logits)]
        result = experts.apply(
            hidden_states=hidden_states,
            w1=w13,
            w2=w2,
            router_logits=router_logits,
            activation=MoEActivation.SILU,
            global_num_experts=NUM_EXPERTS,
            expert_map=None,
            a1q_scale=None,
            apply_router_weight_on_input=False,
            routing_replay_out=routing_replay_out,
            **_routing_kwargs(routing_method),
        )
        assert isinstance(result, torch.Tensor)
        return result

    return experts, forward


def _fp8_block_kernel(top_k: int, routing_method: RoutingMethodType) -> Kernel:
    from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
        _shuffle_deepseek_fp8_moe_weights,
    )

    block = 128
    w13, w2 = _shuffle_deepseek_fp8_moe_weights(
        torch.randn(NUM_EXPERTS, 2 * INTERMEDIATE, HIDDEN, device=DEVICE).to(
            torch.float8_e4m3fn
        ),
        torch.randn(NUM_EXPERTS, HIDDEN, INTERMEDIATE, device=DEVICE).to(
            torch.float8_e4m3fn
        ),
    )
    experts = TrtLlmFp8ExpertsMonolithic(
        moe_config=_moe_config(top_k, routing_method),
        quant_config=fp8_w8a8_moe_quant_config(
            w1_scale=torch.ones(
                NUM_EXPERTS, 2 * INTERMEDIATE // block, HIDDEN // block, device=DEVICE
            ),
            w2_scale=torch.ones(
                NUM_EXPERTS, HIDDEN // block, INTERMEDIATE // block, device=DEVICE
            ),
            block_shape=[block, block],
            per_act_token_quant=False,
        ),
    )

    inputs = (torch.randn(16, HIDDEN, device=DEVICE) * 0.1).to(torch.float8_e4m3fn)

    def forward(
        router_logits: torch.Tensor, routing_replay_out: torch.Tensor | None
    ) -> torch.Tensor:
        num_tokens = len(router_logits)
        hidden_states = inputs[:num_tokens]
        result = experts.apply(
            hidden_states=hidden_states,
            w1=w13,
            w2=w2,
            router_logits=router_logits,
            activation=MoEActivation.SILU,
            global_num_experts=NUM_EXPERTS,
            expert_map=None,
            # Per-block activation scales, (num_tokens, hidden / 128).
            a1q_scale=torch.ones(num_tokens, HIDDEN // block, device=DEVICE),
            apply_router_weight_on_input=False,
            routing_replay_out=routing_replay_out,
            **_routing_kwargs(routing_method),
        )
        assert isinstance(result, torch.Tensor)
        return result

    return experts, forward


def _nvfp4_kernel(
    top_k: int,
    routing_method: RoutingMethodType,
    *,
    activation: MoEActivation = MoEActivation.SILU,
    per_token: bool = False,
) -> Kernel:
    from flashinfer import fp4_quantize

    block = 16
    global_scale = torch.tensor(1.0, device=DEVICE)

    def quantize(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Packed fp4 values and their fp8 per-16 block scales, (E, M, K/16)."""
        q, scale = fp4_quantize(
            w, global_scale, block, sf_use_ue8m0=False, is_sf_swizzled_layout=False
        )
        return q, scale.view(torch.float8_e4m3fn).reshape(*w.shape[:2], -1)

    w13, w13_scale = quantize(
        torch.randn(
            NUM_EXPERTS,
            INTERMEDIATE * (2 if activation == MoEActivation.SILU else 1),
            HIDDEN,
            device=DEVICE,
        ).bfloat16()
    )
    w2, w2_scale = quantize(
        torch.randn(NUM_EXPERTS, HIDDEN, INTERMEDIATE, device=DEVICE).bfloat16()
    )
    ones = torch.ones(NUM_EXPERTS, device=DEVICE)
    experts = TrtLlmNvFp4ExpertsMonolithic(
        moe_config=_moe_config(top_k, routing_method, activation),
        per_token_activation=per_token,
        quant_config=FusedMoEQuantConfig.make(
            quant_dtype="nvfp4",
            per_act_token_quant=False,
            per_out_ch_quant=False,
            block_shape=None,
            w1_scale=w13_scale,
            w2_scale=w2_scale,
            g1_alphas=ones,
            g2_alphas=ones,
            a1_gscale=global_scale,
            a2_gscale=ones,
        ),
    )

    inputs = (torch.randn(16, HIDDEN, device=DEVICE) * 0.1).bfloat16()

    def forward(
        router_logits: torch.Tensor, routing_replay_out: torch.Tensor | None
    ) -> torch.Tensor:
        hidden_states = inputs[: len(router_logits)]
        # apply() reinterprets the packed uint8 activation scales itself.
        if per_token:
            hidden_q, hidden_scale = hidden_states, None
        else:
            hidden_q, hidden_scale = fp4_quantize(
                hidden_states,
                global_scale,
                block,
                sf_use_ue8m0=False,
                is_sf_swizzled_layout=False,
            )
        result = experts.apply(
            hidden_states=hidden_q,
            w1=w13,
            w2=w2,
            router_logits=router_logits,
            activation=activation,
            global_num_experts=NUM_EXPERTS,
            expert_map=None,
            a1q_scale=hidden_scale,
            apply_router_weight_on_input=False,
            routing_replay_out=routing_replay_out,
            **_routing_kwargs(routing_method),
        )
        assert isinstance(result, torch.Tensor)
        return result

    return experts, forward


KERNELS = {"bf16": _bf16_kernel, "fp8_block": _fp8_block_kernel, "nvfp4": _nvfp4_kernel}


@pytest.mark.parametrize("kernel", list(KERNELS))
@pytest.mark.parametrize(
    "routing_method", [RoutingMethodType.Renormalize, RoutingMethodType.DeepSeekV3]
)
@pytest.mark.parametrize("top_k", [2, 4])
@pytest.mark.parametrize("num_tokens", [2, 7, 16])
def test_routing_replay_reports_the_routed_experts(
    kernel: str, routing_method: RoutingMethodType, top_k: int, num_tokens: int
) -> None:
    torch.manual_seed(0)
    experts, forward = KERNELS[kernel](top_k, routing_method)
    assert experts.supports_routing_replay_capture()
    router_logits, planted = _planted_routing(num_tokens, top_k)
    # One spare row: the kernel must write the batch's rows and nothing else.
    routing_replay_out = torch.full(
        (num_tokens + 1, top_k), -1, dtype=torch.int16, device=DEVICE
    )

    expected = forward(router_logits, None).clone()
    actual = forward(router_logits, routing_replay_out)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    reported = routing_replay_out[:num_tokens].long().sort(dim=-1).values
    assert torch.equal(reported, planted), (reported, planted)
    assert (routing_replay_out[num_tokens:] == -1).all()


@pytest.mark.parametrize("per_token", [False, True])
def test_nvfp4_capture_graph_replay_updates_routes_and_preserves_output(
    per_token: bool,
) -> None:
    """Nemotron's non-gated experts with changing routes across decode replays."""
    torch.manual_seed(0)
    top_k, num_tokens = 4, 7
    _, forward = _nvfp4_kernel(
        top_k,
        RoutingMethodType.DeepSeekV3,
        activation=MoEActivation.RELU2_NO_MUL,
        per_token=per_token,
    )
    logits, planted = _planted_routing(num_tokens, top_k)
    replay_out = torch.full(
        (num_tokens + 1, top_k), -1, dtype=torch.int16, device=DEVICE
    )
    # Warm both variants before capture so compilation is outside the graph.
    forward(logits, None)
    forward(logits, replay_out)
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output = forward(logits, replay_out)
    original_logits = logits.clone()
    for shift in (0, NUM_EXPERTS // NUM_GROUPS, NUM_EXPERTS // 2):
        # Rotate whole expert groups, preserving valid grouped routing.
        logits.copy_(original_logits.roll(shift, dims=-1))
        expected = forward(logits, None).clone()
        replay_out.fill_(-1)
        graph.replay()
        torch.testing.assert_close(graph_output, expected, rtol=0, atol=0)
        torch.testing.assert_close(
            replay_out[:num_tokens].long().sort(dim=-1).values,
            ((planted + shift) % NUM_EXPERTS).sort(dim=-1).values,
        )
        assert (replay_out[num_tokens:] == -1).all()
