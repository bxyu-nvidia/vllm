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
    pool.free_blocks([block])
    assert records == []
    pool.evict_blocks({block.block_id})
    trace.remove(block.block_id, [key], "duplicate_notification")
    pool.get_new_blocks(3)
    assert len(records) == 1
    record = records[0]
    assert record["event"] == "hash_removed"
    assert not {
        "engine_id",
        "instance_id",
        "created_ns",
        "retired_ns",
        "removed_ns",
        "lifetime_ns",
        "time_to_retirement_ns",
        "computed_hashes",
        "model_namespace",
        "model_revision",
        "dp_rank",
        "tp_size",
        "pp_size",
        "worker_ranks",
        "rank_scope",
        "medium",
        "schema_version",
        "run_id",
        "reason",
    }.intersection(record)
    assert record["lookup_hit_count"] == 3
    assert record["checkpoint_resume_count"] == 2
    assert record["successful_load_count"] == 0
    assert record["computed_hash_count_direct"] == len(expected)
    assert record["computed_hash_count_descendants"] == len(expected)
    assert record["prefix_hash"] == producer.block_hashes[1].hex()
    assert not trace.entries


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
    assert records[0]["computed_hash_count_direct"] == 0
    assert records[0]["computed_hash_count_descendants"] == 0


def test_reset_emits_once_per_alias(
    traced_pool,
):
    pool, trace, records = traced_pool
    request, block, _ = register(pool)
    request.num_computed_tokens = 0
    pool.cache_partial_block(request, block, 2, 0, 6)
    pool.free_blocks([block])
    assert pool.reset_prefix_cache()
    assert len(records) == 2
    assert {record["prefix_length"] for record in records} == {2, 4}
    assert all(record["computed_hash_count_descendants"] == 0 for record in records)


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


def test_direct_children_deduplicate_branches_and_wait_for_completed_boundaries(
    traced_pool,
):
    pool, trace, records = traced_pool
    _, block, _ = register(pool)
    for request_id, tail in [
        ("a", [7, 8, 9, 10]),
        ("b", [7, 8, 11, 12]),
        ("c", [13, 14, 15, 16]),
    ]:
        consumer = make_request(request_id, [1, 2, 3, 4, *tail], 2, sha256)
        trace.adopt(request_id, [block], 4)
        consumer.num_computed_tokens = 5
        trace.progress(consumer)  # Resumed, but no complete child hash yet.
        entry = next(iter(trace.by_block[block.block_id].values()))
        if request_id == "a":
            assert not entry.direct_hashes
            assert not entry.computed_hashes
        consumer.num_computed_tokens = 8
        consumer.num_in_flight_tokens = 2
        trace.progress(consumer)
        if request_id == "a":
            assert entry.direct_hashes == {consumer.block_hashes[2].hex()}
            assert entry.computed_hashes == entry.direct_hashes
        consumer.num_in_flight_tokens = 0
        trace.progress(consumer)
        trace.progress(consumer)  # Reobserving progress must not add counts.
        trace.finish(consumer)
    pool.evict_blocks({block.block_id})
    assert records[0]["checkpoint_resume_count"] == 3
    assert records[0]["computed_hash_count_direct"] == 2
    assert records[0]["computed_hash_count_descendants"] == 5


def test_report_keeps_distinct_records_with_same_process_local_sequence(tmp_path):
    from tools.mamba_cache_trace import report

    path = tmp_path / "trace.log"
    records = [
        {
            "event": "hash_removed",
            "prefix_hash": "abcd",
            "sequence": 1,
            "block_id": block_id,
        }
        for block_id in [1, 2]
    ]
    path.write_text(
        "".join("MAMBA_CACHE_TRACE " + json.dumps(record) + "\n" for record in records)
    )
    assert report([path, path], "abcd")["removals"] == records
