# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration for model weight offloading."""

import warnings
from typing import TYPE_CHECKING, Literal

from pydantic import Field, model_validator

from vllm.config.compilation import CUDAGraphMode
from vllm.config.utils import config

if TYPE_CHECKING:
    from vllm.config.speculative import SpeculativeConfig


OffloadBackend = Literal["auto", "uva", "prefetch", "dwdp"]


@config
class DWDPConfig:
    """Peer expert storage for independent, same-host prefill workers."""

    rank: int = Field(default=0, ge=0)
    """Weight-sharing rank, independent of vLLM's DP/EP ranks."""
    world_size: int = Field(default=4, ge=2)
    """Number of independent workers sharing weights."""
    host: str = "127.0.0.1"
    """Address of rank zero's startup rendezvous server."""
    port: int = Field(default=29590, ge=1, le=65535)
    """Dedicated startup port; do not reuse the engine distributed port."""
    timeout_seconds: int = Field(default=600, ge=1)
    """Startup and coordinated shutdown timeout."""
    slice_bytes: int = Field(default=0, ge=0)
    """Maximum peer DMA request size; zero copies each shard in one request."""
    prefetch: bool = True
    """Overlap the next MoE layer's copies; disable for a synchronous baseline."""
    collect_metrics: bool = False
    """Record local CUDA timing events for prefetch, exposed waits and MoE compute."""
    collect_expert_usage: bool = False
    """Log distinct local/remote experts per batch and layer; synchronizes GPU."""
    merge_free: bool = False
    """Read resident experts directly with Prims instead of copying them to staging."""
    allow_cuda_graph: bool = False
    """Allow CUDA graph capture; compilation without graphs needs no opt-in."""

    def verify_speculative_config(
        self,
        speculative_config: "SpeculativeConfig | None",
        *,
        enforce_eager: bool,
        cudagraph_mode: CUDAGraphMode | None,
    ) -> None:
        """Validate target-only DWDP with a replicated MTP drafter."""
        if speculative_config is None:
            return
        if speculative_config.method != "mtp":
            raise ValueError("DWDP currently supports only MTP speculative decoding")
        if not enforce_eager and cudagraph_mode != CUDAGraphMode.NONE:
            raise ValueError(
                "DWDP with MTP does not support CUDA graphs; use --enforce-eager "
                'or --compilation-config \'{"mode":3,"cudagraph_mode":"NONE"}\''
            )

    @model_validator(mode="after")
    def validate_rank(self) -> "DWDPConfig":
        if self.collect_expert_usage and self.allow_cuda_graph:
            raise ValueError("DWDP expert usage diagnostics do not support CUDA graphs")
        if self.rank >= self.world_size:
            raise ValueError("DWDP rank must be less than world_size")
        return self


@config
class UVAOffloadConfig:
    """Configuration for UVA (Unified Virtual Addressing) CPU offloading.

    Uses zero-copy access from CPU-pinned memory. Simple but requires
    fast CPU-GPU interconnect.
    """

    cpu_offload_gb: float = Field(default=0, ge=0)
    """The space in GiB to offload to CPU, per GPU. Default is 0, which means
    no offloading. Intuitively, this argument can be seen as a virtual way to
    increase the GPU memory size. For example, if you have one 24 GB GPU and
    set this to 10, virtually you can think of it as a 34 GB GPU. Then you can
    load a 13B model with BF16 weight, which requires at least 26GB GPU memory.
    Note that this requires fast CPU-GPU interconnect, as part of the model is
    loaded from CPU memory to GPU memory on the fly in each model forward pass.
    This uses UVA (Unified Virtual Addressing) for zero-copy access.
    """

    cpu_offload_params: set[str] = Field(default_factory=set)
    """The set of parameter name segments to target for CPU offloading.
    Unmatched parameters are not offloaded. If this set is empty, parameters
    are offloaded non-selectively until the memory limit defined by
    `cpu_offload_gb` is reached.
    Examples:
        - For parameter name "mlp.experts.w2_weight":
            - "experts" or "experts.w2_weight" will match.
            - "expert" or "w2" will NOT match (must be exact segments).
    This allows distinguishing parameters like "w2_weight" and "w2_weight_scale".
    """


@config
class PrefetchOffloadConfig:
    """Configuration for prefetch-based CPU offloading.

    Groups layers and uses async H2D prefetch to hide transfer latency.
    """

    offload_group_size: int = Field(default=0, ge=0)
    """Group every N layers together. Offload last `offload_num_in_group`
    layers of each group. Default is 0 (disabled).
    Example: group_size=8, num_in_group=2 offloads layers 6,7,14,15,22,23,...
    Unlike cpu_offload_gb, this uses explicit async prefetching to hide transfer
    latency.
    """

    offload_num_in_group: int = Field(default=1, ge=1)
    """Number of layers to offload per group.
    Must be <= offload_group_size. Default is 1."""

    offload_prefetch_step: int = Field(default=1, ge=0)
    """Number of layers to prefetch ahead.
    Higher values hide more latency but use more GPU memory. Default is 1."""

    offload_params: set[str] = Field(default_factory=set)
    """The set of parameter name segments to target for prefetch offloading.
    Unmatched parameters are not offloaded. If this set is empty, ALL
    parameters of each offloaded layer are offloaded.
    Uses segment matching: "w13_weight" matches "mlp.experts.w13_weight"
    but not "mlp.experts.w13_weight_scale".
    """


@config
class OffloadConfig:
    """Configuration for model weight offloading to reduce GPU memory usage."""

    offload_backend: OffloadBackend = "auto"
    """The backend for weight offloading. Options:
    - "auto": Selects based on which sub-config has non-default values
      (prefetch if offload_group_size > 0, uva if cpu_offload_gb > 0).
    - "uva": UVA (Unified Virtual Addressing) zero-copy offloading.
    - "prefetch": Async prefetch with group-based layer offloading.
    - "dwdp": Fetch routed expert weights from independent peer workers.
    """

    uva: UVAOffloadConfig = Field(default_factory=UVAOffloadConfig)
    """Parameters for UVA offloading backend."""

    prefetch: PrefetchOffloadConfig = Field(default_factory=PrefetchOffloadConfig)
    """Parameters for prefetch offloading backend."""

    dwdp: DWDPConfig | None = None
    """Enable same-host peer expert storage with a separate startup group."""

    @model_validator(mode="after")
    def validate_offload_config(self) -> "OffloadConfig":
        """Validate offload configuration constraints."""
        if self.dwdp is not None:
            if (
                self.offload_backend not in ("auto", "dwdp")
                or self.uva.cpu_offload_gb > 0
                or self.prefetch.offload_group_size > 0
            ):
                raise ValueError("DWDP cannot be combined with CPU weight offloading")
        elif self.offload_backend == "dwdp":
            raise ValueError("The dwdp offload backend requires dwdp configuration")
        if self.offload_backend == "prefetch" or self.prefetch.offload_group_size > 0:
            if self.prefetch.offload_num_in_group > self.prefetch.offload_group_size:
                raise ValueError(
                    f"offload_num_in_group ({self.prefetch.offload_num_in_group})"
                    f" must be <= offload_group_size"
                    f" ({self.prefetch.offload_group_size})"
                )
            if self.prefetch.offload_prefetch_step < 1:
                raise ValueError(
                    f"offload_prefetch_step"
                    f" ({self.prefetch.offload_prefetch_step})"
                    f" must be >= 1 when prefetch offloading is enabled"
                    f" (offload_group_size > 0)"
                )

        # Warn if both backends have non-default values
        uva_active = self.uva.cpu_offload_gb > 0
        prefetch_active = self.prefetch.offload_group_size > 0
        if self.offload_backend == "uva" and prefetch_active:
            warnings.warn(
                "Prefetch offload fields are set but offload_backend='uva'. "
                "Prefetch settings will be ignored.",
                stacklevel=2,
            )
        elif self.offload_backend == "prefetch" and uva_active:
            warnings.warn(
                "UVA offload fields are set but offload_backend='prefetch'. "
                "UVA settings will be ignored.",
                stacklevel=2,
            )
        elif self.offload_backend == "auto" and uva_active and prefetch_active:
            warnings.warn(
                "Both UVA and prefetch offload fields are set with "
                "offload_backend='auto'. Prefetch backend will be selected. "
                "Set offload_backend explicitly to suppress this warning.",
                stacklevel=2,
            )
        return self

    def compute_hash(self) -> str:
        """Provide a hash that uniquely identifies all the offload configs.

        All fields are included because PrefetchOffloader patches module
        forwards and inserts custom ops (wait_prefetch, start_prefetch)
        into the computation graph. Changing any offload setting can
        alter which layers are hooked and how prefetch indices are
        computed, so the compilation cache must distinguish them.
        """
        from vllm.config.utils import get_hash_factors, hash_factors

        factors = get_hash_factors(self, ignored_factors=set())
        hash_str = hash_factors(factors)
        return hash_str
