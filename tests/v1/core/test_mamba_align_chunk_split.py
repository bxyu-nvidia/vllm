# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mamba "align" prefill chunk splitting (`_mamba_block_aligned_split`).

Invariant: slot `p` holds the state after exactly `(p + 1) * block_size` tokens.
State is written at chunk ends, so a chunk ending mid-block leaves its slot at
the wrong offset, and a later chunk crossing that boundary publishes it anyway.
Requests resuming from it then restore a truncated state (#43559).
"""

import sys
from types import SimpleNamespace

import pytest
import torch

from vllm.utils.math_utils import cdiv
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
    is_mamba_prefill_checkpoint_valid,
)
from vllm.v1.request import Request

from .utils import create_requests

pytestmark = pytest.mark.cpu_test

# Mirrors the deployment where the poisoning was observed (Qwen3.6-27B): mamba
# block 1600, MTP with 3 draft tokens, prompts shorter than 2 mamba blocks.
ATTN_BLOCK_SIZE = 16
MAMBA_BLOCK_SIZE = 1600
NUM_SPEC = 3
PROMPT_LEN = 2002
MAMBA_GROUP_ID = 1


@pytest.mark.parametrize(
    (
        "query_start",
        "query_end",
        "checkpoint_position",
        "hash_size",
        "block_size",
        "alignment",
        "expected",
    ),
    [
        (0, 100, 88, 8, 64, 16, False),
        (0, 100, 96, 8, 64, 16, True),
        (0, 100, 96, 8, 64, None, False),
        (8, 100, 96, 8, 64, 16, False),
        (15360, 16896, 16768, 128, 1536, 16, False),
        (15360, 16960, 16896, 128, 1536, 16, True),
        (15360, 16960, 16768, 128, 1536, 16, True),
        (16064, 18433, 18432, 128, 1536, 16, False),
    ],
)
def test_mamba_prefill_checkpoint_valid(
    query_start: int,
    query_end: int,
    checkpoint_position: int,
    hash_size: int,
    block_size: int,
    alignment: int | None,
    expected: bool,
) -> None:
    assert (
        is_mamba_prefill_checkpoint_valid(
            query_start=query_start,
            query_end=query_end,
            checkpoint_position=checkpoint_position,
            hash_block_size=hash_size,
            mamba_block_size=block_size,
            checkpoint_alignment=alignment,
        )
        is expected
    )


def _make_hybrid_kv_cache_manager(
    num_prefill_checkpoint_blocks: int = 0,
) -> KVCacheManager:
    config = KVCacheConfig(
        num_blocks=10000,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["full_layer"],
                FullAttentionSpec(
                    block_size=ATTN_BLOCK_SIZE,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            ),
            KVCacheGroupSpec(
                ["mamba_layer"],
                MambaSpec(
                    block_size=MAMBA_BLOCK_SIZE,
                    shapes=((1, 1),),
                    dtypes=(torch.float32,),
                    mamba_cache_mode="align",
                    num_speculative_blocks=NUM_SPEC,
                    num_prefill_checkpoint_blocks=num_prefill_checkpoint_blocks,
                    prefill_checkpoint_alignment=(
                        16 if num_prefill_checkpoint_blocks > 0 else None
                    ),
                ),
            ),
        ],
    )
    return KVCacheManager(
        config,
        max_model_len=262144,
        scheduler_block_size=MAMBA_BLOCK_SIZE,
        hash_block_size=ATTN_BLOCK_SIZE,
        enable_caching=True,
        use_eagle=True,
    )


def _split(
    request: Request,
    num_new_tokens: int,
    use_eagle: bool = True,
    use_eagle_block_drop: bool | None = None,
    partial_hit: bool = False,
    num_prefill_checkpoint_blocks: int = 0,
    max_num_scheduled_tokens: int = 16384,
) -> int:
    """Call the real `Scheduler._mamba_block_aligned_split` on a stub self."""
    if use_eagle_block_drop is None:
        use_eagle_block_drop = use_eagle
    stub = SimpleNamespace(
        block_size=MAMBA_BLOCK_SIZE,
        cache_config=SimpleNamespace(block_size=MAMBA_BLOCK_SIZE),
        use_eagle_block_drop=use_eagle_block_drop,
        max_num_scheduled_tokens=max_num_scheduled_tokens,
        scheduler_config=SimpleNamespace(long_prefill_token_threshold=0),
        # `prefix_match_unit` finer than the block size (#46384).
        mamba_partial_cache_hit=partial_hit,
        mamba_shared_prefix_checkpoint=False,
        hash_block_size=ATTN_BLOCK_SIZE,
        mamba_has_prefill_checkpoint_blocks=(num_prefill_checkpoint_blocks > 0),
        mamba_prefill_checkpoint_alignment=(
            16 if num_prefill_checkpoint_blocks > 0 else None
        ),
    )
    return Scheduler._mamba_block_aligned_split(stub, request, num_new_tokens)


@pytest.mark.parametrize(
    ("prompt_len", "num_new_tokens", "use_eagle", "expected"),
    [
        (2002, 2002, False, 2002),
        (3602, 2000, False, MAMBA_BLOCK_SIZE),
        (3602, 3602, True, 3602),
        (2002, 2002, True, 2002),
        (3602, 512, True, 0),
        (3602, 1700, True, MAMBA_BLOCK_SIZE),
    ],
)
def test_internal_checkpoint_split(
    prompt_len: int, num_new_tokens: int, use_eagle: bool, expected: int
) -> None:
    (request,) = create_requests(1, num_tokens=prompt_len, block_size=ATTN_BLOCK_SIZE)
    assert (
        _split(
            request,
            num_new_tokens,
            use_eagle=use_eagle,
            num_prefill_checkpoint_blocks=1,
        )
        == expected
    )


def test_partial_checkpoint_resume_only_splits_without_export() -> None:
    prompt_len = 3602
    resume_at = 1984
    (request,) = create_requests(1, num_tokens=prompt_len, block_size=ATTN_BLOCK_SIZE)
    request.num_computed_tokens = resume_at

    # Without an internal checkpoint export, the forward must stop at the
    # physical Mamba boundary so it does not publish a stale running state.
    assert (
        _split(
            request,
            prompt_len - resume_at,
            use_eagle=False,
            partial_hit=True,
        )
        == MAMBA_BLOCK_SIZE - resume_at % MAMBA_BLOCK_SIZE
    )

    # The fixed checkpoint column aliases the input state. A separate export
    # destination lets the forward finish without overwriting that input.
    assert (
        _split(
            request,
            prompt_len - resume_at,
            use_eagle=False,
            partial_hit=True,
            num_prefill_checkpoint_blocks=1,
        )
        == prompt_len - resume_at
    )


def test_disabling_eagle_block_drop_keeps_the_trailing_cache_boundary() -> None:
    (request,) = create_requests(1, num_tokens=3602, block_size=ATTN_BLOCK_SIZE)

    with_drop = _split(request, request.num_tokens, use_eagle_block_drop=True)
    without_drop = _split(request, request.num_tokens, use_eagle_block_drop=False)

    assert with_drop == MAMBA_BLOCK_SIZE
    assert without_drop == 2 * MAMBA_BLOCK_SIZE


def _run_chunked_prefill(
    manager: KVCacheManager, request: Request, budgets: list[int]
) -> dict[int, int]:
    """Prefill `request`, one step per entry in `budgets`.

    A zero-token split means the budget cannot fund an aligned chunk; the
    scheduler defers the request to a later step, so this does too.

    Returns physical mamba block id -> the token offset of the state it holds,
    mirroring the GDN kernel: the running slot ends up at the chunk end.
    """
    mamba_manager = manager.coordinator.single_type_managers[MAMBA_GROUP_ID]
    state_at: dict[int, int] = {}
    # `budgets` fragments the first steps; afterwards the request is alone and
    # gets as much as it can use, so the prefill always finishes.
    for step in range(len(budgets) + 64):
        computed = request.num_computed_tokens
        if computed >= request.num_tokens:
            break
        budget = budgets[step] if step < len(budgets) else request.num_tokens
        num_new = _split(request, min(request.num_tokens - computed, budget))
        if num_new == 0:
            continue
        assert (
            manager.allocate_slots(request, num_new, num_lookahead_tokens=NUM_SPEC)
            is not None
        )
        request.num_computed_tokens = computed + num_new
        blocks = mamba_manager.req_to_blocks[request.request_id]
        running = cdiv(request.num_computed_tokens, MAMBA_BLOCK_SIZE) - 1
        state_at[blocks[running].block_id] = request.num_computed_tokens
    return state_at


def _count_cached_boundary_states(
    manager: KVCacheManager, request: Request, state_at: dict[int, int]
) -> int:
    """Assert every hash-cached mamba slot holds the state its hash claims.

    Covers both full-block snapshots (`(p + 1) * block_size`) and the
    partial-tail entries align mode registers at an exact token count.

    Returns the number of cached slots checked.
    """
    mamba_manager = manager.coordinator.single_type_managers[MAMBA_GROUP_ID]
    checked = 0
    for pos, block in enumerate(mamba_manager.req_to_blocks[request.request_id]):
        if block.is_null or block.block_hash is None:
            continue
        claimed = block.block_hash_num_tokens
        assert state_at.get(block.block_id) == claimed, (
            f"mamba slot {pos} is hashed as state@{claimed} but holds "
            f"state@{state_at.get(block.block_id)}"
        )
        checked += 1
    return checked


def _prefill(prompt_len: int, budgets: list[int]) -> int:
    manager = _make_hybrid_kv_cache_manager()
    (request,) = create_requests(1, num_tokens=prompt_len, block_size=ATTN_BLOCK_SIZE)
    state_at = _run_chunked_prefill(manager, request, budgets)
    assert request.num_computed_tokens == prompt_len, "prefill did not complete"
    return _count_cached_boundary_states(manager, request, state_at)


def test_fragmented_first_chunk_does_not_poison_mamba_prefix_cache() -> None:
    """EAGLE zeroes `last_cache_position` for prompts under two blocks.

    Past that point any chunk end used to be accepted, so a short first chunk
    (concurrent prefills sharing the budget) left slot 0 at state@364 while the
    next chunk crossed 1600 and published slot 0 as state@1600.
    """
    _prefill(PROMPT_LEN, budgets=[364, PROMPT_LEN])


def test_fragmented_tail_chunk_does_not_poison_mamba_prefix_cache() -> None:
    """Same poisoning one block in, where a hit is still cacheable.

    `last_cache_position` is 1600, so the chunk ending there is cached. The next
    chunk used to be free to stop mid-block (slot 1 at state@2600) and the one
    after it crossed 3200, publishing slot 1 as state@3200.
    """
    assert _prefill(3602, budgets=[1600, 1000, 3602]) > 0


@pytest.mark.parametrize("first_chunk", [800, 900, 1599, 1601, 2000])
def test_intermediate_chunk_ends_stay_block_aligned(first_chunk: int) -> None:
    """Every non-final prefill chunk must end on a mamba block boundary."""
    _prefill(PROMPT_LEN, budgets=[first_chunk, PROMPT_LEN, PROMPT_LEN])


@pytest.mark.parametrize(
    ("block_size", "prompt_len", "budgets"),
    [
        # Kimi-K3-scale mamba blocks: TP8 shards the recurrent state 8 ways
        # (~1.5k block), DEP16 keeps it whole (~12k block). The first budget
        # walks the request up to `last_cache_position`; the second is the
        # sub-block fragment that lands in the unguarded tail region.
        (1536, 30000, [27648, 1024, 30000]),
        (12288, 30000, [12288, 4000, 30000]),
        (12288, 41000, [24576, 8000, 41000]),
    ],
)
def test_poisoning_is_block_size_independent(
    monkeypatch: pytest.MonkeyPatch,
    block_size: int,
    prompt_len: int,
    budgets: list[int],
) -> None:
    """The invariant is per-block, so large mamba blocks are not safer."""
    monkeypatch.setattr(sys.modules[__name__], "MAMBA_BLOCK_SIZE", block_size)
    assert _prefill(prompt_len, budgets=budgets) > 0


@pytest.mark.parametrize("partial_hit", [False, True])
@pytest.mark.parametrize("resume_at", [331, 1599, 1601, 2531, 3011])
@pytest.mark.parametrize(
    ("num_prefill_checkpoint_blocks", "use_eagle"),
    [(0, True), (1, False)],
)
def test_unaligned_resume_never_runs_past_its_block(
    partial_hit: bool,
    resume_at: int,
    num_prefill_checkpoint_blocks: int,
    use_eagle: bool,
) -> None:
    """A prefill resuming mid-block must re-align before crossing a boundary.

    Reachable with a finer `prefix_match_unit` (its partial-tail stop ends a
    chunk off-grid by design) and with unaligned external tokens from a KV
    connector.
    """
    prompt_len = 3602
    (request,) = create_requests(1, num_tokens=prompt_len, block_size=ATTN_BLOCK_SIZE)
    # Under eagle the partial-tail stop sits one hash unit below the prompt's
    # last hash boundary: eagle matches a unit past its candidate and drops it,
    # so nothing proves that last boundary.
    tail_stop = prompt_len // ATTN_BLOCK_SIZE * ATTN_BLOCK_SIZE
    if use_eagle:
        tail_stop -= ATTN_BLOCK_SIZE

    pos, ends = resume_at, []
    while pos < prompt_len:
        request.num_computed_tokens = pos
        num_new = _split(
            request,
            prompt_len - pos,
            use_eagle=use_eagle,
            partial_hit=partial_hit,
            num_prefill_checkpoint_blocks=num_prefill_checkpoint_blocks,
        )
        assert num_new > 0, f"no progress at {pos}"
        if pos % MAMBA_BLOCK_SIZE != 0:
            block_end = (pos // MAMBA_BLOCK_SIZE + 1) * MAMBA_BLOCK_SIZE
            assert pos + num_new <= block_end, (
                f"chunk [{pos}, {pos + num_new}) starts mid-block and runs past "
                f"{block_end}; the slot holding state@{pos} gets hashed as "
                f"state@{block_end}"
            )
        pos += num_new
        ends.append(pos)

    for end in ends[:-1]:
        aligned = end % MAMBA_BLOCK_SIZE == 0
        assert aligned or (partial_hit and end == tail_stop), (
            f"intermediate chunk end {end} is neither block-aligned nor the "
            f"partial-tail stop ({tail_stop})"
        )


@pytest.mark.parametrize("spec_blocks", [0, 3])
def test_short_prefill_reserves_separate_checkpoint_and_preserves_it(
    monkeypatch: pytest.MonkeyPatch,
    spec_blocks: int,
) -> None:
    """One forward exports a reusable prefix without changing the live table."""
    monkeypatch.setattr(sys.modules[__name__], "NUM_SPEC", spec_blocks)
    manager = _make_hybrid_kv_cache_manager(num_prefill_checkpoint_blocks=1)
    mamba = manager.coordinator.single_type_managers[MAMBA_GROUP_ID]
    (request,) = create_requests(1, num_tokens=1000, block_size=ATTN_BLOCK_SIZE)
    free_before = manager.block_pool.get_num_free_blocks()
    assert (
        _split(request, 1000, partial_hit=True, num_prefill_checkpoint_blocks=1) == 1000
    )
    assert manager.allocate_slots(request, 1000) is not None
    ids = manager.get_mamba_checkpoint_block_ids([request.request_id])
    checkpoint_id = ids[MAMBA_GROUP_ID][request.request_id]
    blocks = mamba.req_to_blocks[request.request_id]
    assert len(blocks) == 1 + spec_blocks
    assert checkpoint_id not in [block.block_id for block in blocks]
    assert (
        free_before - manager.block_pool.get_num_free_blocks()
        == cdiv(1000, ATTN_BLOCK_SIZE) + 2 + spec_blocks
    )
    checkpoint = manager.block_pool.blocks[checkpoint_id]
    assert checkpoint.block_hash_num_tokens == 976
    live_id = blocks[0].block_id

    # Simulate completed prefill followed by a decode step. The checkpoint
    # survives while the worker continues mutating the same live block.
    request.num_computed_tokens = 1000
    request.append_output_token_ids(42)
    manager.new_step_starts()
    assert manager.allocate_slots(request, 1) is not None
    assert manager.get_mamba_checkpoint_block_ids([request.request_id]) == {}
    assert mamba.req_to_blocks[request.request_id][0].block_id == live_id
    assert checkpoint.block_hash_num_tokens == 976
    assert checkpoint.ref_cnt == 1

    manager.free(request)
    assert checkpoint.ref_cnt == 0
    assert checkpoint.block_hash_num_tokens == 976
    assert manager.block_pool.get_num_free_blocks() == free_before

    # A sibling can resume from the exported state after the producer is freed.
    (sibling,) = create_requests(
        1, num_tokens=1000, block_size=ATTN_BLOCK_SIZE, req_ids=["sibling"]
    )
    computed, hit, _ = manager.get_computed_blocks(sibling)
    assert hit == 976
    assert computed.blocks[MAMBA_GROUP_ID][-1].block_id == checkpoint_id
    assert (
        manager.allocate_slots(
            sibling,
            1000 - hit,
            num_new_computed_tokens=hit,
            new_computed_blocks=computed,
        )
        is not None
    )
    assert checkpoint.block_hash_num_tokens == 976
    assert mamba.req_to_blocks[sibling.request_id][0].block_id != checkpoint_id
    manager.free(sibling)


def test_short_checkpoint_allocation_respects_pool_capacity() -> None:
    manager = _make_hybrid_kv_cache_manager(num_prefill_checkpoint_blocks=1)
    (request,) = create_requests(1, num_tokens=1000, block_size=ATTN_BLOCK_SIZE)
    # Attention, live Mamba, and speculation fit; the extra checkpoint does not.
    without_checkpoint = cdiv(1000, ATTN_BLOCK_SIZE) + 1 + NUM_SPEC
    held = manager.block_pool.get_new_blocks(
        manager.block_pool.get_num_free_blocks() - without_checkpoint
    )
    assert manager.allocate_slots(request, 1000) is None
    mamba = manager.coordinator.single_type_managers[MAMBA_GROUP_ID]
    assert not mamba._separate_checkpoint_blocks
    assert manager.block_pool.get_num_free_blocks() == without_checkpoint
    manager.block_pool.free_blocks(held)


@pytest.mark.parametrize("spec_blocks", [0, 3])
@pytest.mark.parametrize(
    "prompt_len,cached_tokens",
    [(97000, 85888), (99328, 88064), (97000, 94720), (94336, 85888)],
)
@pytest.mark.parametrize("retention_interval", [None, 0])
def test_cached_prefill_finishes_with_separate_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    spec_blocks: int,
    prompt_len: int,
    cached_tokens: int,
    retention_interval: int | None,
) -> None:
    """A cached resume exports its checkpoint and tail in one budgeted forward.

    Preserve the input hash, publish the exported checkpoint, and never publish
    an intermediate physical boundary whose state was not computed.
    """
    monkeypatch.setattr(sys.modules[__name__], "ATTN_BLOCK_SIZE", 128)
    monkeypatch.setattr(sys.modules[__name__], "MAMBA_BLOCK_SIZE", 8576)
    monkeypatch.setattr(sys.modules[__name__], "NUM_SPEC", spec_blocks)
    manager = _make_hybrid_kv_cache_manager(num_prefill_checkpoint_blocks=1)
    manager.coordinator.retention_interval = retention_interval
    mamba = manager.coordinator.single_type_managers[MAMBA_GROUP_ID]
    free_before = manager.block_pool.get_num_free_blocks()
    (producer,) = create_requests(
        1, num_tokens=cached_tokens + 128, block_size=128, req_ids=["producer"]
    )
    while producer.num_computed_tokens < producer.num_prompt_tokens:
        count = _split(
            producer,
            min(16384, producer.num_prompt_tokens - producer.num_computed_tokens),
            partial_hit=True,
            num_prefill_checkpoint_blocks=1,
        )
        assert count > 0
        assert manager.allocate_slots(producer, count) is not None
        producer.num_computed_tokens += count
        _, retained = manager.take_kv_cache_block_copies()
        manager.block_pool.free_blocks(retained)
        manager.new_step_starts()
    manager.free(producer)
    (request,) = create_requests(
        1, num_tokens=prompt_len, block_size=128, req_ids=["consumer"]
    )
    computed, hit, _ = manager.get_computed_blocks(request)
    assert hit == cached_tokens
    source = computed.blocks[MAMBA_GROUP_ID][-1]
    source_hash = source.block_hash
    request.num_computed_tokens = hit
    remaining = prompt_len - hit
    assert remaining <= 16384
    assert (
        _split(request, remaining, partial_hit=True, num_prefill_checkpoint_blocks=1)
        == remaining
    )
    # Adopt the real cache hit, just as schedule() does for a waiting request.
    request.num_computed_tokens = 0
    needed = (
        cdiv(prompt_len, 128)
        + spec_blocks
        + 3
        + int(cdiv(prompt_len, MAMBA_BLOCK_SIZE) > cdiv(hit, MAMBA_BLOCK_SIZE))
    )
    # Admission must account for the extra checkpoint before allocating anything.
    assert (
        manager.allocate_slots(
            request,
            remaining,
            num_new_computed_tokens=hit,
            new_computed_blocks=computed,
            reserved_blocks=free_before - needed + 1,
        )
        is None
    )
    assert request.request_id not in mamba._separate_checkpoint_blocks
    assert manager.block_pool.get_num_free_blocks() == free_before
    assert (
        manager.allocate_slots(
            request,
            remaining,
            num_new_computed_tokens=hit,
            new_computed_blocks=computed,
        )
        is not None
    )
    assert free_before - manager.block_pool.get_num_free_blocks() == needed
    checkpoint_id = manager.get_mamba_checkpoint_block_ids([request.request_id])[
        MAMBA_GROUP_ID
    ][request.request_id]
    checkpoint = manager.block_pool.blocks[checkpoint_id]
    checkpoint_position = prompt_len // 128 * 128 - 128
    blocks = mamba.req_to_blocks[request.request_id]
    live = blocks[cdiv(prompt_len, MAMBA_BLOCK_SIZE) - 1]
    assert checkpoint_id not in [block.block_id for block in blocks]
    assert len({checkpoint_id, source.block_id, live.block_id}) == 3
    assert checkpoint.block_hash_num_tokens == checkpoint_position
    assert source.block_hash == source_hash
    assert source.block_hash_num_tokens == hit
    # No state was materialized at any crossed interior physical boundary.
    for boundary in range(
        (hit // MAMBA_BLOCK_SIZE + 1) * MAMBA_BLOCK_SIZE,
        prompt_len,
        MAMBA_BLOCK_SIZE,
    ):
        assert (
            manager.block_pool.get_cached_block(
                request.block_hashes[boundary // 128 - 1], [MAMBA_GROUP_ID]
            )
            is None
        )
    _, retained = manager.take_kv_cache_block_copies()
    manager.block_pool.free_blocks(retained)
    request.num_computed_tokens = prompt_len
    request.append_output_token_ids(42)
    manager.new_step_starts()
    assert manager.allocate_slots(request, 1) is not None
    assert manager.get_mamba_checkpoint_block_ids([request.request_id]) == {}
    assert checkpoint.block_hash_num_tokens == checkpoint_position
    manager.free(request)
    assert checkpoint.ref_cnt == 0
    assert manager.block_pool.get_num_free_blocks() == free_before
    manager.new_step_starts()
    (sibling,) = create_requests(
        1, num_tokens=prompt_len, block_size=128, req_ids=["sibling"]
    )
    blocks, hit, _ = manager.get_computed_blocks(sibling)
    assert hit == checkpoint_position
    assert blocks.blocks[MAMBA_GROUP_ID][-1].block_id == checkpoint_id
