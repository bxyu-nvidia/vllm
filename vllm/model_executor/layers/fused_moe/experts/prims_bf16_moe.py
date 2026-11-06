# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.model_executor.layers.fused_moe.config import FusedMoEParallelConfig
from vllm.model_executor.layers.fused_moe.experts.trtllm_bf16_moe import (
    TrtLlmBf16ExpertsModular,
    view_as_block_major_k,
)
from vllm.model_executor.layers.fused_moe.utils import fi_moe_largest_bucket


class PrimsBf16Experts(TrtLlmBf16ExpertsModular):
    """Prims grouped GEMMs with vLLM routing and expert-major packed weights."""

    @staticmethod
    def _supports_parallel_config(config: FusedMoEParallelConfig) -> bool:
        return (
            config.tp_size == config.dp_size == config.ep_size == 1
            and not config.enable_eplb
            and not config.use_batched_activation_format
        )

    def process_weights_after_loading(self, layer) -> None:
        self._resident_weights = None
        if layer.dwdp_manager is not None and layer.dwdp_manager.merge_free:
            assert layer.dwdp_expert_range is not None
            # Parameters carry bound weight-loader callbacks back to the layer.
            self._resident_weights = (
                view_as_block_major_k(layer.w13_weight.detach()),
                view_as_block_major_k(layer.w2_weight.detach()),
            )
            self._resident_expert_offset = layer.dwdp_expert_range[0]

    def _routed_moe(self, **kwargs):
        from flashinfer.fused_moe import prims_ts_bf16_routed_moe

        kwargs["tune_max_num_tokens"] = fi_moe_largest_bucket(self.moe_config)
        resident = getattr(self, "_resident_weights", None)
        if resident is not None:
            kwargs.update(
                resident_gemm1_weights=resident[0],
                resident_gemm2_weights=resident[1],
                resident_expert_offset=self._resident_expert_offset,
            )
        return prims_ts_bf16_routed_moe(**kwargs)
