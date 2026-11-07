# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.model_executor.layers.fused_moe.config import FusedMoEParallelConfig
from vllm.model_executor.layers.fused_moe.experts.trtllm_nvfp4_moe import (
    TrtLlmNvFp4ExpertsModular,
)


class PrimsNvFp4Experts(TrtLlmNvFp4ExpertsModular):
    """Prims NVFP4 experts with optional merge-free DWDP storage."""

    @staticmethod
    def _supports_parallel_config(config: FusedMoEParallelConfig) -> bool:
        return (
            config.tp_size == config.dp_size == config.ep_size == 1
            and not config.enable_eplb
            and not config.use_batched_activation_format
        )

    def process_weights_after_loading(self, layer) -> None:
        super().process_weights_after_loading(layer)
        self._dwdp_scales_ready = False
        self._resident_weights = None
        if layer.dwdp_manager is not None and layer.dwdp_manager.merge_free:
            self._resident_weights = (
                layer.w13_weight.detach(),
                layer.w13_weight_scale.detach(),
                layer.w2_weight.detach(),
                layer.w2_weight_scale.detach(),
            )
            self._resident_expert_offset = layer.dwdp_expert_range[0]

    def _routed_moe(self, **kwargs):
        from flashinfer.fused_moe import prims_ts_fp4_block_scale_routed_moe

        resident = self._resident_weights
        if resident is not None and not self._dwdp_scales_ready:
            self.g1_scale_c.copy_(self._compute_g1_scale_c())
            if self.gemm1_clamp_limit is not None and not self.is_situ:
                self.gemm1_clamp_limit.copy_(
                    self._gemm1_clamp_limit_unfolded / self.quant_config.g1_alphas
                )
            if self.gemm1_beta is not None and not self.is_situ:
                self.gemm1_beta.copy_(
                    self._gemm1_beta_unfolded / self.quant_config.g1_alphas
                )
            self._dwdp_scales_ready = True
        if resident is not None:
            kwargs.update(
                resident_gemm1_weights=resident[0],
                resident_gemm1_weights_scale=resident[1],
                resident_gemm2_weights=resident[2],
                resident_gemm2_weights_scale=resident[3],
                resident_expert_offset=self._resident_expert_offset,
            )
        return prims_ts_fp4_block_scale_routed_moe(**kwargs)
