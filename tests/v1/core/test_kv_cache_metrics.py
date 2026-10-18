# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import patch

import pytest

from vllm.v1.core.kv_cache_metrics import (
    BlockMetricsState,
    KVCacheMetricsCollector,
)
from vllm.v1.core.kv_cache_utils import KVCacheBlock


class TestBlockMetricsState:
    def test_init(self):
        with patch("time.monotonic_ns", return_value=1000000000):
            state = BlockMetricsState()
            assert state.birth_time_ns == 1000000000
            assert state.last_access_ns == 1000000000
            assert len(state.access_history) == 0

    def test_access_tracking(self):
        with patch("time.monotonic_ns", return_value=1000000000):
            state = BlockMetricsState()

        with patch("time.monotonic_ns", return_value=2000000000):
            state.record_access()

        assert state.last_access_ns == 2000000000
        assert list(state.access_history) == [2000000000]

    def test_ring_buffer_wraps_at_4(self):
        with patch("time.monotonic_ns", return_value=1000000000):
            state = BlockMetricsState()

        for i in range(5):
            t = 1000000000 + (i + 1) * 1000000000
            with patch("time.monotonic_ns", return_value=t):
                state.record_access()

        assert len(state.access_history) == 4
        assert list(state.access_history) == [
            3000000000,
            4000000000,
            5000000000,
            6000000000,
        ]

    def test_lifetime(self):
        with patch("time.monotonic_ns", return_value=1000000000):
            state = BlockMetricsState()
        with patch("time.monotonic_ns", return_value=6500000000):
            assert abs(state.get_lifetime_seconds() - 5.5) < 0.001

    def test_idle_time(self):
        with patch("time.monotonic_ns", return_value=1000000000):
            state = BlockMetricsState()
        state.last_access_ns = 2000000000
        with patch("time.monotonic_ns", return_value=5200000000):
            assert abs(state.get_idle_time_seconds() - 3.2) < 0.001

    def test_reuse_gaps(self):
        with patch("time.monotonic_ns", return_value=1000000000):
            state = BlockMetricsState()

        base = 1000000000
        for offset in [0, 1.5, 3.0, 5.5]:
            state.access_history.append(base + int(offset * 1e9))

        gaps = state.get_reuse_gaps_seconds()
        assert len(gaps) == 3
        assert gaps[0] == 1.5 and gaps[1] == 1.5 and gaps[2] == 2.5

    def test_ring_wrap_only_gives_3_gaps(self):
        # 5 accesses in size-4 buffer = 3 gaps
        with patch("time.monotonic_ns", return_value=1000000000):
            state = BlockMetricsState()

        for i in range(5):
            state.access_history.append(1000000000 + i * 1000000000)

        assert len(state.get_reuse_gaps_seconds()) == 3


class TestKVCacheMetricsCollector:
    def test_sample_rate_validation(self):
        with pytest.raises(AssertionError):
            KVCacheMetricsCollector(sample_rate=-0.1)
        with pytest.raises(AssertionError):
            KVCacheMetricsCollector(sample_rate=1.5)
        with pytest.raises(AssertionError):
            KVCacheMetricsCollector(sample_rate=0.0)

    def test_sampling(self):
        c = KVCacheMetricsCollector(sample_rate=1.0)
        assert sum(1 for _ in range(100) if c.should_sample_block()) == 100

        c = KVCacheMetricsCollector(sample_rate=0.5)
        samples = sum(1 for _ in range(1000) if c.should_sample_block())
        assert 400 < samples < 600

    def test_alloc(self):
        c = KVCacheMetricsCollector(sample_rate=1.0)

        blocks = [KVCacheBlock(block_id=i) for i in range(5)]
        with patch("time.monotonic_ns", return_value=1000000000):
            for block in blocks:
                c.on_block_allocated(block)

        assert len(c.block_metrics) == 5

    def test_access(self):
        c = KVCacheMetricsCollector(sample_rate=1.0)
        block = KVCacheBlock(block_id=0)

        with patch("time.monotonic_ns", return_value=1000000000):
            c.on_block_allocated(block)

        for i in range(3):
            t = 1000000000 + (i + 1) * 1000000000
            with patch("time.monotonic_ns", return_value=t):
                c.on_block_accessed(block)

        assert len(c.block_metrics[0].access_history) == 3

    def test_evict_no_accesses(self):
        # lifetime should equal idle if never accessed
        c = KVCacheMetricsCollector(sample_rate=1.0)

        block = KVCacheBlock(block_id=0)
        with patch("time.monotonic_ns", return_value=1000000000):
            c.on_block_allocated(block)

        with patch("time.monotonic_ns", return_value=6000000000):
            c.on_block_evicted(block)

        events = c.drain_events()
        assert len(events) == 1
        assert abs(events[0].lifetime_seconds - 5.0) < 0.001
        assert abs(events[0].idle_seconds - 5.0) < 0.001

    def test_evict(self):
        c = KVCacheMetricsCollector(sample_rate=1.0)

        block = KVCacheBlock(block_id=0)
        with patch("time.monotonic_ns", return_value=1000000000):
            c.on_block_allocated(block)

        with patch("time.monotonic_ns", return_value=2000000000):
            c.on_block_accessed(block)
        with patch("time.monotonic_ns", return_value=3000000000):
            c.on_block_accessed(block)

        with patch("time.monotonic_ns", return_value=4000000000):
            c.on_block_evicted(block)

        events = c.drain_events()
        assert len(events) == 1
        sample = events[0]
        assert abs(sample.lifetime_seconds - 3.0) < 0.001
        assert abs(sample.idle_seconds - 1.0) < 0.001
        assert sample.reuse_gaps_seconds == (1.0,)
        assert 0 not in c.block_metrics

    def test_reset(self):
        c = KVCacheMetricsCollector(sample_rate=1.0)

        with patch("time.monotonic_ns", return_value=1000000000):
            for i in range(5):
                c.on_block_allocated(KVCacheBlock(block_id=i))

        assert len(c.block_metrics) == 5
        c.reset()
        assert len(c.block_metrics) == 0

        with patch("time.monotonic_ns", return_value=2000000000):
            c.on_block_allocated(KVCacheBlock(block_id=10))
        assert 10 in c.block_metrics

    def test_huge_time_jump(self):
        c = KVCacheMetricsCollector(sample_rate=1.0)

        block = KVCacheBlock(block_id=0)
        with patch("time.monotonic_ns", return_value=1000000000):
            c.on_block_allocated(block)

        with patch("time.monotonic_ns", return_value=9999999999999999):
            c.on_block_evicted(block)

        events = c.drain_events()
        assert len(events) == 1
        assert events[0].lifetime_seconds > 0


def test_kv_cache_metrics_collector_smoke() -> None:
    """Simple smoke test for KVCacheMetricsCollector on CPU."""
    collector = KVCacheMetricsCollector(sample_rate=1.0)
    block = KVCacheBlock(block_id=123)

    # Allocate at t = 1.0s.
    with patch("time.monotonic_ns", return_value=1_000_000_000):
        collector.on_block_allocated(block)

    # Access at t = 2.0s and t = 3.0s.
    with patch("time.monotonic_ns", return_value=2_000_000_000):
        collector.on_block_accessed(block)
    with patch("time.monotonic_ns", return_value=3_000_000_000):
        collector.on_block_accessed(block)

    # Evict at t = 4.0s.
    with patch("time.monotonic_ns", return_value=4_000_000_000):
        collector.on_block_evicted(block)

    events = collector.drain_events()
    assert len(events) == 1

    event = events[0]
    # Lifetime: 1.0s → 4.0s.
    assert abs(event.lifetime_seconds - 3.0) < 1e-6
    # Idle: last access at 3.0s, evicted at 4.0s.
    assert abs(event.idle_seconds - 1.0) < 1e-6
    # One reuse gap between the two accesses.
    assert event.reuse_gaps_seconds == (1.0,)


def test_inventory_counts_physical_blocks_once_and_preserves_idle_cache():
    """Two request refs and two hash aliases must not double-count GPU bytes."""
    from types import SimpleNamespace

    import torch

    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.core.cache_inventory import collect_cache_inventory
    from vllm.v1.core.kv_cache_utils import BlockHash, make_block_hash_with_group_id
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        KVCacheGroupSpec,
        MambaSpec,
    )

    pool = BlockPool(8, True, 4)
    attention, mamba, pinned = pool.get_new_blocks(3)
    attention.ref_cnt += 1
    h = make_block_hash_with_group_id(BlockHash(b"shared"), 0)
    attention.set_block_hash(h)
    pool.cached_block_hashes_by_block[attention.block_id] = {
        h,
        make_block_hash_with_group_id(BlockHash(b"alias"), 0),
    }
    mamba.set_block_hash(make_block_hash_with_group_id(BlockHash(b"shared"), 1))
    groups = [
        KVCacheGroupSpec(
            ["attn"],
            FullAttentionSpec(
                block_size=4, num_kv_heads=1, head_size=2, dtype=torch.float32
            ),
        ),
        KVCacheGroupSpec(
            ["mamba"],
            MambaSpec(block_size=4, shapes=((2, 2),), dtypes=(torch.float32,)),
        ),
    ]
    manager = SimpleNamespace(
        block_pool=pool,
        kv_cache_config=SimpleNamespace(kv_cache_groups=groups),
        coordinator=SimpleNamespace(
            single_type_managers=[
                SimpleNamespace(req_to_blocks={"a": [attention], "b": [attention]}),
                SimpleNamespace(req_to_blocks={"a": [mamba]}),
            ]
        ),
    )
    first = collect_cache_inventory(manager)
    assert first["attention.in_use_blocks"] == 1
    assert first["attention.in_use_bytes"] == groups[0].kv_cache_spec.page_size_bytes
    assert first["attention.cached_blocks"] == 1
    assert first["attention.distinct_prefixes"] == 2
    assert first["mamba.in_use_blocks"] == 1
    assert first["pool.active_requests"] == 2
    assert first["pool.unattributed_in_use_blocks"] == 1
    assert first["pool.unused_blocks"] == 4
    assert pinned.ref_cnt == 1  # Sampling must not touch references.

    manager.coordinator.single_type_managers[1].req_to_blocks.clear()
    pool.free_blocks([mamba])
    second = collect_cache_inventory(manager)
    assert second["mamba.in_use_blocks"] == 0
    assert second["mamba.idle_cached_blocks"] == 1
    assert second["mamba.idle_cached_bytes"] == 16
    assert second["mamba.distinct_prefixes"] == 1
    assert second["pool.reclaimable_blocks"] == 5
