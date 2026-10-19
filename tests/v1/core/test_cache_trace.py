# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Checkpoint accounting must follow residency, not probes or block IDs.

Exercise the real pool with a recording sink: GPU tensors are irrelevant to
these metadata transitions. Scheduler completion is represented by the same
computed/in-flight counters used by the production observer.
"""

import copy
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
    records = []
    pool = BlockPool(4, True, 2)
    pool.trace = CacheTrace(sink=lambda record: records.append(copy.deepcopy(record)))
    pool.trace.groups = {0: ["model.layers.0.mixer"]}
    return pool, pool.trace, records


def register(pool, request_id="producer"):
    request = make_request(request_id, [1, 2, 3, 4, 5, 6], 2, sha256)
    request.num_computed_tokens = 4
    block = pool.get_new_blocks(1)[0]
    key = pool.cache_partial_block(request, block, 4, 0, 6)
    return request, block, key


def test_probe_adoption_and_completed_resume_are_distinct(traced_pool):
    pool, trace, records = traced_pool
    producer, block, key = register(pool)
    consumer = make_request("consumer", [1, 2, 3, 4, 7, 8], 2, sha256)
    for _ in range(3):
        assert pool.get_cached_block(producer.block_hashes[1], [0]) == [block]
    pool.touch([block])
    trace.adopt(consumer.request_id, [block], 4)
    consumer.num_computed_tokens = 6
    consumer.num_in_flight_tokens = 2
    trace.progress(consumer)
    entry = trace.entries[block.block_id, key]
    assert entry.lookup_hit_count == 3
    assert entry.checkpoint_resume_count == 0
    consumer.num_in_flight_tokens = 0
    consumer.append_output_token_ids([9])
    trace.progress(consumer)
    trace.progress(consumer)
    assert entry.checkpoint_resume_count == 1
    assert entry.successful_load_count == 0
    assert entry.consumers[0]["computed_start"] == 4
    assert entry.consumers[0]["computed_end"] == 6
    assert any(
        r["event"] == "request_tokens" and r["token_ids"] == [9] for r in records
    )


def test_retirement_pinning_removal_and_reallocation(traced_pool):
    pool, trace, records = traced_pool
    _, block, key = register(pool)
    # A transfer retains a reference after the producing request retires it.
    pool.touch([block])
    pool.free_blocks([block])
    trace.release("producer", [block], retired=True)
    entry = trace.entries[block.block_id, key]
    assert entry.retired_ns is not None
    assert entry.eviction_eligible_ns is None
    assert entry.hash_removed_ns is None
    pool.free_blocks([block])
    assert entry.eviction_eligible_ns is not None
    pool.evict_blocks({block.block_id})
    assert entry.hash_removed_ns is not None
    assert entry.storage_reclaimed_ns is None
    # Allocate all slots, including the retired one.
    pool.get_new_blocks(3)
    assert entry.storage_reclaimed_ns is not None
    assert trace.generations[block.block_id] == 2
    events = [r["event"] for r in records]
    assert events.index("checkpoint_retired") < events.index("eviction_eligible")
    assert events.index("hash_removed") < events.index("storage_reclaimed")


def test_copy_on_write_preserves_history_and_duplicate_hashes_are_separate(traced_pool):
    pool, trace, records = traced_pool
    request, source, key = register(pool)
    duplicate = pool.get_new_blocks(1)[0]
    pool.cache_partial_block(request, duplicate, 4, 0, 6)
    source_entry = trace.entries[source.block_id, key]
    duplicate_entry = trace.entries[duplicate.block_id, key]
    assert source_entry.residency_id != duplicate_entry.residency_id
    destination = pool.get_new_blocks(1)[0]
    pool.move_block_hashes(source, destination)
    assert trace.entries[destination.block_id, key] is source_entry
    assert source_entry.block_id == destination.block_id
    assert source_entry.hash_removed_ns is None
    assert not any(r["event"] == "hash_removed" for r in records)
    pool.evict_blocks({destination.block_id})
    assert pool.get_cached_block(request.block_hashes[1], [0]) == [duplicate]
    assert duplicate_entry.hash_removed_ns is None


def test_external_load_does_not_invent_a_producer_or_resume(traced_pool):
    pool, trace, records = traced_pool
    trace.expect_external("consumer", 4)
    request, block, key = register(pool, "consumer")
    entry = trace.entries[block.block_id, key]
    assert entry.producer_request_id is None
    assert entry.origin == "external"
    trace.loaded(request, [block])
    trace.loaded(request, [block])
    assert entry.successful_load_count == 1
    assert entry.checkpoint_resume_count == 0
    trace.detach(request.request_id)
    request.num_computed_tokens = 6
    trace.progress(request)
    assert entry.checkpoint_resume_count == 0


def test_reset_removes_aliases_but_does_not_claim_overwrite(traced_pool):
    pool, trace, records = traced_pool
    request, block, key = register(pool)
    pool.cache_partial_block(request, block, 2, 0, 6)
    assert len(trace.entries) == 2
    pool.free_blocks([block])
    assert pool.reset_prefix_cache()
    assert not trace.entries
    removed = [r for r in records if r["event"] == "hash_removed"]
    assert len(removed) == 2
    assert {r["prefix_length"] for r in removed} == {2, 4}
    assert all(r["reason"] == "cache_reset" for r in removed)
    assert all(r["storage_reclaimed_ns"] is None for r in removed)


def test_tracing_is_disabled_by_default(monkeypatch):
    monkeypatch.delenv("VLLM_MAMBA_CACHE_TRACE", raising=False)
    assert BlockPool(2, True, 2).trace is None


def test_in_place_overwrite_waits_for_completion(traced_pool):
    pool, trace, records = traced_pool
    request, block, key = register(pool)
    previous = trace.entries[block.block_id, key]
    pool.cache_partial_block(request, block, 6, 0, 6)
    assert previous.hash_removed_ns is not None
    assert previous.storage_reclaimed_ns is None
    request.num_computed_tokens = 6
    request.num_in_flight_tokens = 2
    trace.progress(request)
    assert previous.storage_reclaimed_ns is None
    request.num_in_flight_tokens = 0
    trace.progress(request)
    assert previous.storage_reclaimed_ns is not None
    assert any(
        r["event"] == "storage_reclaimed" and r["reason"] == "checkpoint_overwritten"
        for r in records
    )


def test_report_joins_tokens_and_deduplicates_repeated_logs(traced_pool, tmp_path):
    from tools.mamba_cache_trace import report

    pool, trace, records = traced_pool
    producer, block, key = register(pool)
    consumer = make_request("consumer", [1, 2, 3, 4, 7, 8], 2, sha256)
    trace.adopt(consumer.request_id, [block], 4)
    consumer.num_computed_tokens = 6
    consumer.append_output_token_ids([9])
    trace.progress(consumer)
    trace.release("producer", [block], retired=True)
    h = trace.entries[block.block_id, key].prefix_hash
    trace.emit("mooncake_lookup_hit", object_key=f"model@group:0@{h}")
    trace.emit("mooncake_load", object_key=f"model@group:0@{h}", success=True)
    path = tmp_path / "trace.log"
    path.write_text(
        "\n".join("INFO MAMBA_CACHE_TRACE " + json.dumps(r) for r in records)
    )
    result = report([path, path], h)
    assert len(result["gpu_residencies"]) == 1
    assert result["gpu_residencies"][0]["checkpoint_resume_count"] == 1
    assert result["mooncake_objects"][0]["observed_successful_loads"] == 1
    assert result["mooncake_objects"][0]["observed_lookup_hits"] == 1
    requests = {r["request_id"]: r for r in result["requests"]}
    assert requests["consumer"]["output_token_ids"] == [9]
    assert requests["producer"]["prompt_token_ids"] == [1, 2, 3, 4, 5, 6]


def test_report_rejects_truncated_records(tmp_path):
    from tools.mamba_cache_trace import report

    path = tmp_path / "trace.log"
    path.write_text('INFO MAMBA_CACHE_TRACE {"event":')
    with pytest.raises(ValueError, match="Truncated/invalid"):
        report([path], "abcd")
