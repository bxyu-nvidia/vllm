# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Removal summaries must distinguish probes, loads and completed computation."""

import json

import pytest

from tests.v1.core.prefix_cache.test_partial_prefix_cache_primitives import make_request
from vllm.utils.hashing import sha256
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.cache_trace import CacheTrace
from vllm.v1.core.kv_cache_utils import init_none_hash

pytestmark = pytest.mark.cpu_test


@pytest.fixture
def traced_pool(monkeypatch):
    monkeypatch.setenv("VLLM_MAMBA_CACHE_TRACE", "1")
    init_none_hash(sha256)
    records: list[dict[str, object]] = []
    pool = BlockPool(4, True, 2)
    pool.trace = CacheTrace(sink=records.append)
    pool.trace.groups = {0: ["model.layers.0.mixer"]}
    return pool, pool.trace, records


def register(pool, request_id="producer"):
    request = make_request(request_id, [1, 2, 3, 4, 5, 6], 2, sha256)
    request.num_computed_tokens = 4
    block = pool.get_new_blocks(1)[0]
    key = pool.cache_partial_block(request, block, 4, 0, 6)
    return request, block, key


def test_only_removal_emits_with_distinct_usage_counts_and_computed_hashes(traced_pool):
    pool, trace, records = traced_pool
    producer, block, key = register(pool)
    for _ in range(3):
        assert pool.get_cached_block(producer.block_hashes[1], [0]) == [block]
    expected: set[str] = set()
    for request_id, tail in [("a", [7, 8]), ("b", [9, 10])]:
        consumer = make_request(request_id, [1, 2, 3, 4, *tail, 11, 12], 2, sha256)
        trace.adopt(request_id, [block], 4)
        consumer.num_computed_tokens = 6
        consumer.num_in_flight_tokens = 2
        trace.progress(consumer)
        entry = trace.entries[block.block_id, key]
        assert entry.checkpoint_resume_count == len(expected)
        assert entry.computed_hashes == expected
        consumer.num_in_flight_tokens = 0
        trace.progress(consumer)
        trace.progress(consumer)
        expected.add(consumer.block_hashes[2].hex())
        trace.finish(consumer)
    trace.release("producer", [block], retired=True)
    pool.free_blocks([block])
    assert records == []
    pool.evict_blocks({block.block_id})
    trace.remove(block.block_id, [key], "duplicate_notification")
    pool.get_new_blocks(3)
    assert len(records) == 1
    record = records[0]
    assert record["event"] == "hash_removed"
    assert record["lookup_hit_count"] == 3
    assert record["checkpoint_resume_count"] == 2
    assert record["successful_load_count"] == 0
    assert set(record["computed_hashes"]) == expected
    assert record["prefix_hash"] == producer.block_hashes[1].hex()
    assert not trace.entries


def test_retirement_and_removal_use_first_retirement_and_monotonic_time(
    traced_pool, monkeypatch
):
    from vllm.v1.core import cache_trace

    pool, trace, records = traced_pool
    clock = [100]
    monkeypatch.setattr(cache_trace.time, "monotonic_ns", lambda: clock[0])
    _, block, _ = register(pool)
    clock[0] = 130
    trace.release("producer", [block], retired=True)
    clock[0] = 150
    trace.release("producer", [block], retired=True)
    clock[0] = 200
    pool.evict_blocks({block.block_id})
    assert records[0]["lifetime_ns"] == 100
    assert records[0]["time_to_retirement_ns"] == 30


def test_moves_preserve_counts_and_duplicate_residencies_are_independent(traced_pool):
    pool, trace, records = traced_pool
    request, source, key = register(pool)
    duplicate = pool.get_new_blocks(1)[0]
    pool.cache_partial_block(request, duplicate, 4, 0, 6)
    trace.lookup(source.block_id, key)
    destination = pool.get_new_blocks(1)[0]
    pool.move_block_hashes(source, destination)
    assert records == []
    pool.evict_blocks({destination.block_id})
    pool.evict_blocks({duplicate.block_id})
    assert [r["lookup_hit_count"] for r in records] == [1, 0]
    assert [r["block_id"] for r in records] == [
        destination.block_id,
        duplicate.block_id,
    ]


def test_external_load_is_not_a_resume_and_detach_cancels_pending_use(traced_pool):
    pool, trace, records = traced_pool
    trace.expect_external("consumer", 4)
    request, block, _ = register(pool, "consumer")
    trace.loaded(request, [block])
    trace.loaded(request, [block])
    trace.detach(request.request_id)
    request.num_computed_tokens = 6
    trace.progress(request)
    assert records == []
    pool.evict_blocks({block.block_id})
    assert records[0]["successful_load_count"] == 1
    assert records[0]["checkpoint_resume_count"] == 0
    assert records[0]["computed_hashes"] == []


def test_unfinished_checkpoint_has_unknown_lifetime_and_reset_emits_once_per_alias(
    traced_pool,
):
    pool, trace, records = traced_pool
    request, block, _ = register(pool)
    request.num_computed_tokens = 0
    pool.cache_partial_block(request, block, 2, 0, 6)
    pool.free_blocks([block])
    assert pool.reset_prefix_cache()
    assert len(records) == 2
    assert {r["reason"] for r in records} == {"cache_reset"}
    unfinished = next(r for r in records if r["prefix_length"] == 2)
    assert unfinished["created_ns"] is None
    assert unfinished["lifetime_ns"] is None
    assert unfinished["time_to_retirement_ns"] is None


def test_report_filters_and_deduplicates_removal_summaries(traced_pool, tmp_path):
    from tools.mamba_cache_trace import report

    pool, trace, records = traced_pool
    _, block, key = register(pool)
    h = trace.entries[block.block_id, key].prefix_hash
    pool.evict_blocks({block.block_id})
    path = tmp_path / "trace.log"
    path.write_text("INFO MAMBA_CACHE_TRACE " + json.dumps(records[0]) + "\n")
    assert report([path, path], h)["removals"] == records
    assert report([path], "absent")["removals"] == []


def test_report_rejects_truncated_records(tmp_path):
    from tools.mamba_cache_trace import report

    path = tmp_path / "trace.log"
    path.write_text('INFO MAMBA_CACHE_TRACE {"event":')
    with pytest.raises(ValueError, match="Truncated/invalid"):
        report([path], "abcd")


def test_tracing_is_disabled_by_default(monkeypatch):
    monkeypatch.delenv("VLLM_MAMBA_CACHE_TRACE", raising=False)
    assert BlockPool(2, True, 2).trace is None
