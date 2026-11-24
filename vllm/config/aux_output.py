# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration for execution auxiliary outputs."""

from pydantic import Field

from vllm.config.utils import config


@config
class AuxOutputConfig:
    """Configuration for auxiliary-output delivery."""

    enable_return_routed_experts: bool = False
    """Capture and return routed-experts auxiliary outputs."""

    use_capture_handoff: bool = False
    """Use the engine capture adapter to join prefill and decode expert routes.

    Required for NIXL pull P/D with routed experts. Every request must use the
    configured capture adapter; ordinary serving cannot return complete routes.
    Routes are retained only until delivery to the adapter, which owns
    prior-turn lineage. No local or Mooncake route cache is allocated.
    Training reuses prior-turn routes from TQ. Local and Mooncake prefix
    reuse stops where recorded routes end; inference computes the remaining
    suffix, including the previous turn's last sampled token. Decode can
    also reuse the prefix whose routes were delivered by the prefill adapter.
    This changes output delivery only, not the compiled model graph.
    """

    max_bytes: int | None = Field(default=None, gt=0)
    """LRU capacity, or ``None`` to derive it from the KV cache capacity.

    Unused with ``use_capture_handoff``, which has no reusable route cache.
    """

    @property
    def enabled(self) -> bool:
        """Whether any execution auxiliary output is enabled."""
        return self.enable_return_routed_experts

    def compute_hash(self) -> str:
        """Hash AuxOutput settings that alter the model forward graph."""
        from vllm.config.utils import hash_factors

        return hash_factors(
            {"enable_return_routed_experts": self.enable_return_routed_experts}
        )
