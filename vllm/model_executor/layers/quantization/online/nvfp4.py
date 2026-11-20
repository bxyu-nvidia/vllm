# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterator
from contextlib import contextmanager

import torch
from torch.nn import Module

from vllm._custom_ops import scaled_fp4_quant
from vllm.model_executor.layers.fused_moe import RoutedExperts, SharedExperts
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.moe_output import UnfinalizedMoEOutput
from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import (
    NvFp4MoeBackend,
    convert_to_nvfp4_moe_kernel_format,
    make_nvfp4_moe_kernel,
    make_nvfp4_moe_quant_config,
    select_nvfp4_moe_backend,
)
from vllm.model_executor.layers.quantization.online.moe_base import (
    OnlineMoEMethodBase,
)
from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
    trtllm_nvfp4_hidden_alignment,
)
from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
    FLOAT4_E2M1_MAX,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    amax_for_moe_weight_quant,
    kNvfp4DynamicToken,
    kNvfp4Static,
    weight_amax,
)
from vllm.model_executor.utils import replace_parameter
from vllm.platforms import current_platform

FLOAT8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max


def _quantize_moe_weight_to_nvfp4(
    weight: torch.Tensor,
    moe_tp_size: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize stacked MoE expert weights ``(E, N, K)`` to NVFP4.

    One FP32 global scale per expert plus per-block (group-16) FP8 scales,
    matching the ModelOpt NVFP4 checkpoint layout. Returns packed FP4 weights
    ``(E, N, K // 2)``, block scales ``(E, N, K // 16)``, and the per-expert
    global scale ``(E,)`` stored as ``amax / (fp4_max * fp8_max)``.
    """
    assert weight.dim() == 3, f"expected 3D expert weights, got {weight.shape}"
    k = weight.shape[-1]
    assert k % 16 == 0, f"last dim must be a multiple of 16, got {k}"

    amax = weight_amax(weight.flatten(1), dim=-1).to(torch.float32)
    amax = amax_for_moe_weight_quant(amax, moe_tp_size).clamp_min(1e-8)
    global_scale = (FLOAT4_E2M1_MAX * FLOAT8_E4M3_MAX) / amax
    weight_scale_2 = (1.0 / global_scale).to(torch.float32)

    # Keep the original BF16/FP16 values as the quantizer input. Folding each
    # expert's FP32 global scale into the weight would add a BF16/FP16 rounding
    # before the group-16 scale and E2M1 values are selected.
    weight = weight.contiguous()
    quantized_experts = [
        scaled_fp4_quant(
            expert_weight,
            expert_scale,
            is_sf_swizzled_layout=False,
        )
        for expert_weight, expert_scale in zip(
            weight,
            global_scale,
            strict=True,
        )
    ]
    qweight = torch.stack([quantized for quantized, _ in quantized_experts])
    block_scale = torch.stack([block_scale for _, block_scale in quantized_experts])
    return (
        qweight,
        block_scale,
        weight_scale_2,
    )


class Nvfp4OnlineMoEMethod(OnlineMoEMethodBase):
    """Online NVFP4 MoE quantization with per-token activation scales.

    Quantizes fp16/bf16 expert weights to NVFP4 at load time; the FlashInfer
    Prims and TRTLLM kernels compute per-token activation scales at runtime.
    Blackwell (SM100) only.
    """

    activation_quant_key = kNvfp4DynamicToken

    def __init__(
        self,
        *,
        moe: FusedMoEConfig,
    ):
        if not current_platform.is_device_capability_family(100):
            raise ValueError(
                "nvfp4_per_token online quantization requires a Blackwell (SM100) GPU."
            )
        super().__init__(moe)
        self.nvfp4_backend, self.experts_cls = select_nvfp4_moe_backend(
            config=self.moe,
            weight_key=kNvfp4Static,
            activation_key=self.activation_quant_key,
        )

    def create_weights(
        self,
        layer: RoutedExperts,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        from vllm.model_executor.offloader import get_offloader
        from vllm.model_executor.offloader.dwdp import DWDPWeightOffloader, expert_range

        offloader = get_offloader()
        if isinstance(offloader, DWDPWeightOffloader):
            if (
                self.nvfp4_backend != NvFp4MoeBackend.FLASHINFER_PRIMS
                or self.moe.has_bias
                or not offloader.config.merge_free
            ):
                raise ValueError(
                    "Online NVFP4 DWDP requires merge-free flashinfer_prims "
                    "experts without bias"
                )
            layer.dwdp_manager = offloader.manager
            layer.dwdp_expert_range = expert_range(
                num_experts,
                offloader.config.rank,
                offloader.config.world_size,
            )
            start, end = layer.dwdp_expert_range
            num_experts = end - start
            self.supports_pre_processed_weights = False
        super().create_weights(
            layer,
            num_experts,
            hidden_size,
            intermediate_size_per_partition,
            params_dtype,
            **extra_weight_attrs,
        )

    @contextmanager
    def _dwdp_weights(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        if layer.dwdp_manager is None:
            yield layer.w13_weight, layer.w2_weight
            return
        with layer.dwdp_manager.layer(layer.layer_name, x.shape[0]) as weights:
            w13, w2, s13, s2 = weights[:4]
            quant = self.moe_quant_config
            assert quant is not None
            old_scales = quant.w1_scale, quant.w2_scale
            quant._w1.scale, quant._w2.scale = s13, s2
            try:
                yield w13, w2
            finally:
                quant._w1.scale, quant._w2.scale = old_scales

    def apply(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: SharedExperts | None,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor | UnfinalizedMoEOutput:
        assert not self.is_monolithic
        assert self.moe_kernel is not None
        with self._dwdp_weights(layer, x) as (w13, w2):
            return self.moe_kernel.apply(
                x,
                w13,
                w2,
                topk_weights,
                topk_ids,
                activation=layer.activation,
                global_num_experts=layer.global_num_experts,
                expert_map=layer.expert_map,
                apply_router_weight_on_input=layer.apply_router_weight_on_input,
                shared_experts=shared_experts,
                shared_experts_input=shared_experts_input,
            )

    @property
    def per_token_activation(self) -> bool:
        return self.activation_quant_key == kNvfp4DynamicToken

    def process_weights_after_loading(self, layer: Module) -> None:
        if (
            layer.dwdp_manager is not None
            and layer.dwdp_manager.initialized
            and not layer.dwdp_manager.refitting
        ):
            raise RuntimeError("Cannot reload immutable DWDP expert weights")
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return

        self._quantize_weights(layer)
        self._setup_kernel(layer)

        layer._already_called_process_weights_after_loading = True

    def _quantize_weights(self, layer: Module) -> None:
        moe_tp_size = self.moe.tp_size
        w13, w13_scale, w13_scale_2 = _quantize_moe_weight_to_nvfp4(
            layer.w13_weight, moe_tp_size
        )
        w2, w2_scale, w2_scale_2 = _quantize_moe_weight_to_nvfp4(
            layer.w2_weight, moe_tp_size
        )

        replace_parameter(layer, "w13_weight", w13)
        replace_parameter(layer, "w13_weight_scale", w13_scale)
        replace_parameter(layer, "w13_weight_scale_2", w13_scale_2)
        replace_parameter(layer, "w2_weight", w2)
        replace_parameter(layer, "w2_weight_scale", w2_scale)
        replace_parameter(layer, "w2_weight_scale_2", w2_scale_2)

        # Neutral (1.0) activation global scales: the kernel derives per-token
        # scales at runtime, so the output scalars reduce to the weight scales.
        ones = torch.ones(layer.num_experts, device=w13.device, dtype=torch.float32)
        replace_parameter(layer, "w13_input_scale", ones)
        replace_parameter(layer, "w2_input_scale", ones.clone())

    def _setup_kernel(self, layer: RoutedExperts) -> None:
        (
            w13,
            w13_scale,
            w13_scale_2,
            a13_scale,
            w2,
            w2_scale,
            w2_scale_2,
            a2_scale,
        ) = convert_to_nvfp4_moe_kernel_format(
            nvfp4_backend=self.nvfp4_backend,
            layer=layer,
            w13=layer.w13_weight,
            w13_scale=layer.w13_weight_scale,
            w13_scale_2=layer.w13_weight_scale_2,
            a13_scale=layer.w13_input_scale,
            w2=layer.w2_weight,
            w2_scale=layer.w2_weight_scale,
            w2_scale_2=layer.w2_weight_scale_2,
            a2_scale=layer.w2_input_scale,
            is_act_and_mul=self.moe.is_act_and_mul,
            trtllm_hidden_alignment=trtllm_nvfp4_hidden_alignment(
                per_token_activation=self.per_token_activation,
                is_act_and_mul=self.moe.is_act_and_mul,
            ),
        )

        replace_parameter(layer, "w13_weight", w13)
        replace_parameter(layer, "w13_weight_scale", w13_scale)
        replace_parameter(layer, "w13_weight_scale_2", w13_scale_2)
        replace_parameter(layer, "w13_input_scale", a13_scale)
        replace_parameter(layer, "w2_weight", w2)
        replace_parameter(layer, "w2_weight_scale", w2_scale)
        replace_parameter(layer, "w2_weight_scale_2", w2_scale_2)
        replace_parameter(layer, "w2_input_scale", a2_scale)

        if layer.dwdp_manager is not None:
            metadata_names = (
                "w13_weight_scale_2",
                "w2_weight_scale_2",
                "w13_input_scale",
                "w2_input_scale",
            )
            projections = (
                w13.detach(),
                w2.detach(),
                w13_scale.detach(),
                w2_scale.detach(),
                *(getattr(layer, name).detach().view(-1, 1) for name in metadata_names),
            )
            replicated = layer.dwdp_manager.register_layer(
                layer.layer_name,
                layer.global_num_experts,
                projections,
                replicated_indices=range(4, 8),
            )
            for index, name in enumerate(metadata_names, 4):
                # Profiling starts only after manager initialization. Unit input
                # scales also make the precomputed kernel transforms well-defined.
                if not layer.dwdp_manager.refitting:
                    replicated[index].fill_(1.0)
                replace_parameter(layer, name, replicated[index].view(-1))
            if layer.dwdp_manager.refitting:
                # Keep the kernel's resident views on the IPC-exported storage.
                resident = layer.dwdp_manager.local_weights(layer.layer_name)
                for name, tensor in zip(
                    ("w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale"),
                    resident[:4],
                ):
                    replace_parameter(layer, name, tensor)
            w13_scale_2 = layer.w13_weight_scale_2
            w2_scale_2 = layer.w2_weight_scale_2

        if self.moe_kernel is None:
            self.moe_quant_config = self.get_fused_moe_quant_config(layer)
            assert self.experts_cls is not None
            self.moe_kernel = make_nvfp4_moe_kernel(
                moe_quant_config=self.moe_quant_config,
                moe_config=self.moe,
                experts_cls=self.experts_cls,
                backend=self.nvfp4_backend,
                routing_tables=layer._expert_routing_tables(),
                per_token_activation=self.per_token_activation,
            )
        else:
            # Reload creates new scale tensors; derived kernel scales must use
            # their new values before layerwise reload restores captured storage.
            assert self.moe_quant_config is not None
            assert self.moe_quant_config.g1_alphas is not None
            assert self.moe_quant_config.g2_alphas is not None
            self.moe_quant_config.g1_alphas.copy_(w13_scale_2)
            self.moe_quant_config.g2_alphas.copy_(w2_scale_2)

        self.moe_kernel.fused_experts.process_weights_after_loading(layer)

    def finalize_weight_reload(self, layer: RoutedExperts) -> None:
        """Refresh derived tensors after storage restoration and DWDP replication."""
        from vllm.model_executor.layers.fused_moe.experts.trtllm_nvfp4_moe import (
            TrtLlmNvFp4ExpertsBase,
        )

        assert self.moe_quant_config is not None and self.moe_kernel is not None
        experts = self.moe_kernel.fused_experts
        if not isinstance(experts, TrtLlmNvFp4ExpertsBase):
            raise NotImplementedError(
                "Online NVFP4 refit requires FlashInfer Prims or TRTLLM experts"
            )
        quant = self.moe_quant_config
        quant._w1.scale = layer.w13_weight_scale
        quant._w2.scale = layer.w2_weight_scale
        assert quant.g1_alphas is not None and quant.g2_alphas is not None
        quant.g1_alphas.copy_(layer.w13_weight_scale_2)
        quant.g2_alphas.copy_(layer.w2_weight_scale_2)
        experts.refresh_after_weight_reload(layer)

    def get_fused_moe_quant_config(self, layer: torch.nn.Module) -> FusedMoEQuantConfig:
        return make_nvfp4_moe_quant_config(
            backend=self.nvfp4_backend,
            w13_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            w13_scale_2=layer.w13_weight_scale_2,
            w2_scale_2=layer.w2_weight_scale_2,
            a13_scale=layer.w13_input_scale,
            a2_scale=layer.w2_input_scale,
            swiglu_limit=getattr(layer, "swiglu_limit", None),
            layer=layer,
        )
