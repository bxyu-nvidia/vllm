# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV cache metrics tracking."""

import random
import time
from collections import OrderedDict, deque
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_utils import BlockHashWithGroupId, KVCacheBlock

from vllm.v1.metrics.stats import KVCacheEvictionEvent


class BlockMetricsState:
    """Tracks lifecycle metrics for a single KV cache block."""

    def __init__(self):
        now_ns = time.monotonic_ns()
        self.birth_time_ns = now_ns
        self.last_access_ns = now_ns
        # Bounded to prevent unbounded growth if a block is accessed many times.
        self.access_history: deque[int] = deque(maxlen=4)

    def record_access(self) -> None:
        now_ns = time.monotonic_ns()
        self.last_access_ns = now_ns
        self.access_history.append(now_ns)

    def get_lifetime_seconds(self) -> float:
        now_ns = time.monotonic_ns()
        return (now_ns - self.birth_time_ns) / 1e9

    def get_idle_time_seconds(self) -> float:
        now_ns = time.monotonic_ns()
        return (now_ns - self.last_access_ns) / 1e9

    def get_reuse_gaps_seconds(self) -> list[float]:
        if len(self.access_history) < 2:
            return []
        history = list(self.access_history)
        return [(history[i] - history[i - 1]) / 1e9 for i in range(1, len(history))]


class KVCacheMetricsCollector:
    """Collects KV cache residency metrics with sampling."""

    def __init__(
        self, sample_rate: float = 0.01, *, eviction_history_limit: int = 65536
    ):
        assert 0 < sample_rate <= 1.0, (
            f"sample_rate must be in (0, 1.0], got {sample_rate}"
        )
        assert eviction_history_limit > 0
        self.sample_rate = sample_rate
        self.eviction_history_limit = eviction_history_limit
        self._evicted_hashes: OrderedDict[BlockHashWithGroupId, int] = OrderedDict()
        self._turnover: dict[str, dict[str, int]] = {}

        self.block_metrics: dict[int, BlockMetricsState] = {}

        self._eviction_events: list[KVCacheEvictionEvent] = []

    def _group_counts(self, group_id: int) -> dict[str, int]:
        group = str(group_id)
        if group not in self._turnover:
            self._turnover[group] = dict.fromkeys(
                (
                    "evicted",
                    "lookup_hit",
                    "lookup_miss",
                    "miss_after_eviction",
                    "history_dropped",
                ),
                0,
            )
        return self._turnover[group]

    def on_prefix_evicted(self, key: "BlockHashWithGroupId", group_id: int) -> None:
        """Remember loss of the last resident copy due to block recycling."""
        self._group_counts(group_id)["evicted"] += 1
        self._evicted_hashes[key] = group_id
        self._evicted_hashes.move_to_end(key)
        if len(self._evicted_hashes) > self.eviction_history_limit:
            _, dropped_group = self._evicted_hashes.popitem(last=False)
            self._group_counts(dropped_group)["history_dropped"] += 1

    def on_prefix_inserted(self, key: "BlockHashWithGroupId") -> None:
        self._evicted_hashes.pop(key, None)

    def on_prefix_lookup(
        self, key: "BlockHashWithGroupId", group_id: int, *, hit: bool
    ) -> None:
        """Count probes, including repeated probes; these are not token counts."""
        counts = self._group_counts(group_id)
        counts["lookup_hit" if hit else "lookup_miss"] += 1
        if not hit and key in self._evicted_hashes:
            counts["miss_after_eviction"] += 1

    def drain_turnover(self) -> dict[str, dict[str, int]]:
        """Drain unsampled counters without forgetting the eviction history."""
        counts, self._turnover = self._turnover, {}
        return counts

    def should_sample_block(self) -> bool:
        return random.random() < self.sample_rate

    def on_block_allocated(self, block: "KVCacheBlock") -> None:
        if self.should_sample_block():
            self.block_metrics[block.block_id] = BlockMetricsState()

    def on_block_accessed(self, block: "KVCacheBlock") -> None:
        metrics = self.block_metrics.get(block.block_id)
        if metrics:
            metrics.record_access()

    def on_block_evicted(self, block: "KVCacheBlock") -> None:
        metrics = self.block_metrics.pop(block.block_id, None)
        if not metrics:
            return

        lifetime = metrics.get_lifetime_seconds()
        idle_time = metrics.get_idle_time_seconds()
        reuse_gaps = tuple(metrics.get_reuse_gaps_seconds())

        self._eviction_events.append(
            KVCacheEvictionEvent(
                lifetime_seconds=lifetime,
                idle_seconds=idle_time,
                reuse_gaps_seconds=reuse_gaps,
            )
        )

    def reset(self) -> None:
        """Clear all state on cache reset."""
        self._evicted_hashes.clear()
        self._turnover.clear()
        self.block_metrics.clear()
        self._eviction_events.clear()

    def drain_events(self) -> list[KVCacheEvictionEvent]:
        events = self._eviction_events
        self._eviction_events = []
        return events
