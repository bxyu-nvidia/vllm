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
    monkeypatch.delenv("VLLM_MAMBA_EVICT_ON_DIRECT_DESCENDANT", raising=False)
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


@pytest.mark.parametrize("emit_trace", [False, True])
def test_post_use_eviction_waits_for_direct_completion_and_preserves_readers(
    monkeypatch, emit_trace
):
    from unittest.mock import Mock

    from vllm.v1.core import cache_trace

    monkeypatch.setenv("VLLM_MAMBA_EVICT_ON_DIRECT_DESCENDANT", "1")
    monkeypatch.setenv("VLLM_MAMBA_CACHE_TRACE", "1" if emit_trace else "0")
    log = Mock()
    monkeypatch.setattr(cache_trace.logger, "info", log)
    init_none_hash(sha256)
    pool = BlockPool(8, True, 2)
    trace = pool.trace
    assert trace is not None
    trace.groups = {0: ["mamba"]}
    producer, block, _ = register(pool)
    consumer = make_request("consumer", [1, 2, 3, 4, 7, 8, 9, 10], 2, sha256)
    pool.touch([block, block])  # Producer plus two active readers.
    trace.adopt("consumer", [block], 4)
    trace.adopt("other_reader", [block], 4)
    free_before = pool.get_num_free_blocks()

    consumer.num_computed_tokens = 8
    consumer.num_in_flight_tokens = 4
    trace.progress(consumer)
    assert pool.get_cached_block(producer.block_hashes[1], [0]) == [block]
    consumer.num_in_flight_tokens = 3  # Resumed, incomplete direct hash.
    trace.progress(consumer)
    assert pool.get_cached_block(producer.block_hashes[1], [0]) == [block]
    consumer.num_in_flight_tokens = 2  # First direct hash is now complete.
    trace.progress(consumer)
    assert pool.get_cached_block(producer.block_hashes[1], [0]) is None
    assert block.ref_cnt == 3
    assert pool.get_num_free_blocks() == free_before
    assert not trace.entries
    assert log.call_count == int(emit_trace)
    if emit_trace:
        record = json.loads(log.call_args.args[1])
        assert record["computed_hash_count_direct"] == 1
        assert record["computed_hash_count_descendants"] == 1
        assert record["checkpoint_resume_count"] == 1
    trace.progress(consumer)
    assert log.call_count == int(emit_trace)

    pool.free_blocks([block, block])
    assert block.ref_cnt == 1
    assert pool.get_num_free_blocks() == free_before
    pool.free_blocks([block])
    assert pool.get_num_free_blocks() == free_before + 1
    assert pool.get_new_blocks(1) == [block]


def test_post_use_eviction_follows_copied_state_and_keeps_reused_source(monkeypatch):
    monkeypatch.setenv("VLLM_MAMBA_EVICT_ON_DIRECT_DESCENDANT", "1")
    monkeypatch.delenv("VLLM_MAMBA_CACHE_TRACE", raising=False)
    init_none_hash(sha256)
    pool = BlockPool(8, True, 2)
    trace = pool.trace
    trace.groups = {0: ["mamba"]}
    producer, source, _ = register(pool)
    consumer = make_request("consumer", [1, 2, 3, 4, 7, 8], 2, sha256)
    trace.adopt("consumer", [source], 4)
    destination = pool.get_new_blocks(1)[0]
    pool.move_block_hashes(source, destination)
    pool.free_blocks([destination])  # Cached CoW copy, with no remaining reader.
    replacement = make_request("replacement", [11, 12, 13, 14], 2, sha256)
    pool.cache_partial_block(replacement, source, 4, 0, 6)
    consumer.num_computed_tokens = 6
    trace.progress(consumer)
    assert pool.get_cached_block(producer.block_hashes[1], [0]) is None
    assert pool.get_cached_block(replacement.block_hashes[1], [0]) == [source]
    assert source.ref_cnt == 1
    assert pool.get_new_blocks(1) == [destination]


def test_post_use_eviction_ignores_failed_loads_and_non_mamba_groups(monkeypatch):
    monkeypatch.setenv("VLLM_MAMBA_EVICT_ON_DIRECT_DESCENDANT", "1")
    monkeypatch.delenv("VLLM_MAMBA_CACHE_TRACE", raising=False)
    init_none_hash(sha256)
    pool = BlockPool(8, True, 2)
    trace = pool.trace
    trace.groups = {0: ["mamba"]}
    request, block, _ = register(pool, "consumer")
    attention = pool.get_new_blocks(1)[0]
    pool.cache_partial_block(request, attention, 4, 1, 6)
    trace.expect_external("consumer", 4)
    trace.loaded(request, [block, attention])
    assert pool.get_cached_block(request.block_hashes[1], [0]) == [block]
    trace.detach("consumer")  # Failed/preempted work cannot trigger eviction.
    request.num_computed_tokens = 6
    trace.progress(request)
    assert pool.get_cached_block(request.block_hashes[1], [0]) == [block]
    trace.adopt("consumer", [block, attention], 4)
    trace.progress(request)
    assert pool.get_cached_block(request.block_hashes[1], [0]) is None
    assert pool.get_cached_block(request.block_hashes[1], [1]) == [attention]


@pytest.mark.parametrize("external_second_use", [False, True])
def test_two_direct_uses_protect_across_residencies_until_explicit_reset(
    monkeypatch, external_second_use
):
    monkeypatch.setenv("VLLM_MAMBA_EVICT_ON_DIRECT_DESCENDANT", "1")
    monkeypatch.delenv("VLLM_MAMBA_CACHE_TRACE", raising=False)
    init_none_hash(sha256)
    pool = BlockPool(8, True, 2)
    trace = pool.trace
    trace.groups = {0: ["mamba"]}

    for use_number in range(1, 4):
        producer, block, _ = register(pool)
        consumer = make_request(f"use-{use_number}", [1, 2, 3, 4, 7, 8], 2, sha256)
        if use_number == 2 and external_second_use:
            trace.expect_external(consumer.request_id, 4)
            trace.loaded(consumer, [block])
        else:
            trace.adopt(consumer.request_id, [block], 4)
        consumer.num_computed_tokens = 5
        trace.progress(consumer)
        assert pool.get_cached_block(producer.block_hashes[1], [0]) == [block]
        consumer.num_computed_tokens = 6
        trace.progress(consumer)
        trace.progress(consumer)  # One completed use, not two observations.
        trace.finish(consumer)
        if use_number == 1:
            assert pool.get_cached_block(producer.block_hashes[1], [0]) is None
        else:
            assert pool.get_cached_block(producer.block_hashes[1], [0]) == [block]
            assert not pool.reset_prefix_cache()  # Active references block reset.
            # Protection does not pin the hash against normal/explicit eviction.
            pool.evict_blocks({block.block_id})
            assert pool.get_cached_block(producer.block_hashes[1], [0]) is None
        pool.free_blocks([block])

    assert pool.reset_prefix_cache()
    producer, block, _ = register(pool)
    consumer = make_request("after-reset", [1, 2, 3, 4, 7, 8], 2, sha256)
    trace.adopt(consumer.request_id, [block], 4)
    consumer.num_computed_tokens = 6
    trace.progress(consumer)
    assert pool.get_cached_block(producer.block_hashes[1], [0]) is None


def test_reader_finishing_after_hash_removal_still_contributes_to_protection(
    monkeypatch,
):
    monkeypatch.setenv("VLLM_MAMBA_EVICT_ON_DIRECT_DESCENDANT", "1")
    monkeypatch.delenv("VLLM_MAMBA_CACHE_TRACE", raising=False)
    init_none_hash(sha256)
    pool = BlockPool(8, True, 2)
    trace = pool.trace
    trace.groups = {0: ["mamba"]}
    producer, block, _ = register(pool)
    pool.touch([block, block])
    readers = [make_request(name, [1, 2, 3, 4, 7, 8], 2, sha256) for name in ["a", "b"]]
    for reader in readers:
        trace.adopt(reader.request_id, [block], 4)
        reader.num_computed_tokens = 6
    trace.progress(readers[0])
    assert pool.get_cached_block(producer.block_hashes[1], [0]) is None
    assert block.ref_cnt == 3
    trace.progress(readers[1])  # The entry is gone, but its second reader completed.
    for reader in readers:
        trace.finish(reader)
    pool.free_blocks([block, block, block])
    _, recached, _ = register(pool)
    # Exercise another alias without a third use of the protected prefix.
    # Otherwise that third use could hide a missed second-reader completion.
    pool.cache_partial_block(producer, recached, 2, 0, 6)
    consumer = make_request("c", [1, 2, 3, 4, 7, 8], 2, sha256)
    trace.adopt(consumer.request_id, [recached], 2)
    consumer.num_computed_tokens = 4
    trace.progress(consumer)
    assert pool.get_cached_block(producer.block_hashes[1], [0]) == [recached]


def test_protection_is_per_group_and_prevents_alias_eviction(monkeypatch):
    monkeypatch.setenv("VLLM_MAMBA_EVICT_ON_DIRECT_DESCENDANT", "1")
    monkeypatch.delenv("VLLM_MAMBA_CACHE_TRACE", raising=False)
    init_none_hash(sha256)
    pool = BlockPool(8, True, 2)
    trace = pool.trace
    trace.groups = {0: ["mamba0"], 1: ["mamba1"]}
    for name in ["a", "b"]:
        producer, block, _ = register(pool)
        consumer = make_request(name, [1, 2, 3, 4, 7, 8], 2, sha256)
        trace.adopt(name, [block], 4)
        consumer.num_computed_tokens = 6
        trace.progress(consumer)
        trace.finish(consumer)
        if name == "a":
            pool.free_blocks([block])
    assert pool.get_cached_block(producer.block_hashes[1], [0]) == [block]

    other_group = pool.get_new_blocks(1)[0]
    pool.cache_partial_block(producer, other_group, 4, 1, 6)
    consumer = make_request("other-group", [1, 2, 3, 4, 7, 8], 2, sha256)
    trace.adopt(consumer.request_id, [other_group], 4)
    consumer.num_computed_tokens = 6
    trace.progress(consumer)
    assert pool.get_cached_block(producer.block_hashes[1], [1]) is None
    assert pool.get_cached_block(producer.block_hashes[1], [0]) == [block]

    pool.cache_partial_block(producer, block, 2, 0, 6)
    consumer = make_request("alias", [1, 2, 3, 4, 7, 8], 2, sha256)
    trace.adopt(consumer.request_id, [block], 2)
    consumer.num_computed_tokens = 4
    trace.progress(consumer)
    assert pool.get_cached_block(producer.block_hashes[0], [0]) == [block]
    assert pool.get_cached_block(producer.block_hashes[1], [0]) == [block]


def test_direct_use_observer_reports_same_child_after_gpu_removal(monkeypatch):
    """Two admitted readers must protect the store even after GPU eviction."""
    monkeypatch.setenv("VLLM_MAMBA_EVICT_ON_DIRECT_DESCENDANT", "1")
    init_none_hash(sha256)
    pool = BlockPool(8, True, 2)
    trace = pool.trace
    trace.groups = {0: ["mamba"]}
    reports = []
    trace.on_direct_use = lambda group, prefix: reports.append((group, prefix))
    producer, block, _ = register(pool)
    readers = [make_request(name, [1, 2, 3, 4, 7, 8], 2, sha256) for name in ["a", "b"]]
    for reader in readers:
        trace.adopt(reader.request_id, [block], 4)
        reader.num_computed_tokens = 6
        reader.num_in_flight_tokens = 2
        trace.progress(reader)
    assert not reports
    for reader in readers:
        reader.num_in_flight_tokens = 0
        trace.progress(reader)
        trace.progress(reader)
    assert reports == [(0, producer.block_hashes[1].hex())] * 2
    assert pool.get_cached_block(producer.block_hashes[1], [0]) is None
