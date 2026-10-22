# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Internal prefill checkpoints for Mamba2 align-mode prefix caching."""

from types import SimpleNamespace

import pytest
import torch

from tests.v1.attention.utils import (
    BatchSpec,
    create_common_attn_metadata,
    create_vllm_config,
)
from vllm.v1.attention.backends.mamba2_attn import (
    Mamba2AttentionMetadataBuilder,
)
from vllm.v1.attention.backends.mamba_attn import BaseMambaAttentionMetadataBuilder
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID
from vllm.v1.kv_cache_interface import MambaSpec

compute_chunk_metadata = BaseMambaAttentionMetadataBuilder._compute_chunk_metadata


def test_no_checkpoint_offsets_preserves_legacy_chunking():
    """Passing no offsets must reproduce the pre-feature chunk layout."""
    num_computed = torch.tensor([0, 300])
    qsl = torch.tensor([0, 700, 1000])

    cu, seq_idx, last, ckpt = compute_chunk_metadata(256, 2, num_computed, qsl)

    # Row 0: 700 fresh tokens from sequence position 0 -> 256, 256, 188.
    # Row 1: 300 new tokens resuming at 300. 300 % 256 == 44, so it finishes
    # its partial chunk with 256 - 44 == 212 tokens, then 88.
    assert cu == [0, 256, 512, 700, 912, 1000]
    assert seq_idx == [0, 0, 0, 1, 1]
    assert last == [2, 4]
    assert ckpt == [-1, -1]


def test_chunking_realigns_to_the_grid_after_a_checkpoint():
    """After the forced break, chunks resume breaking on chunk_size.

    100 -> then 156 to reach the 256 grid, then full chunks. Without the
    realignment the second chunk would span 100..356 and cross a boundary.
    """
    num_computed = torch.tensor([0])
    qsl = torch.tensor([0, 700])

    cu, _, _, _ = compute_chunk_metadata(
        256, 1, num_computed, qsl, checkpoint_offsets_p=[100]
    )

    assert cu == [0, 100, 256, 512, 700]


def test_checkpoint_on_a_resumed_request_is_relative_to_the_query():
    """Offsets count from the query start, not the sequence start."""
    num_computed = torch.tensor([300])
    qsl = torch.tensor([0, 500])

    cu, _, _, ckpt = compute_chunk_metadata(
        256, 1, num_computed, qsl, checkpoint_offsets_p=[212]
    )

    # Row resumes at sequence position 300. 300 % 256 == 44, so the partial
    # chunk is 212 tokens and already ends exactly at the checkpoint (query
    # offset 212 == sequence position 512) — no extra split needed.
    assert cu == [0, 212, 468, 500]
    assert ckpt == [0]


MAMBA_BLOCK_SIZE = 256
CHUNK_SIZE = 256
DEVICE = torch.device("cpu")


def _create_mamba2_builder(
    mamba_cache_mode: str = "align",
    num_prefill_checkpoint_blocks: int = 1,
    prefill_checkpoint_alignment: int | None = 1,
) -> Mamba2AttentionMetadataBuilder:
    vllm_config = create_vllm_config(
        model_name="Qwen/Qwen3.5-0.8B", block_size=MAMBA_BLOCK_SIZE
    )
    vllm_config.cache_config.mamba_cache_mode = mamba_cache_mode
    # get_mamba_chunk_size() reads `mamba_chunk_size` then `chunk_size`.
    vllm_config.model_config.hf_text_config.mamba_chunk_size = CHUNK_SIZE
    spec = MambaSpec(
        block_size=MAMBA_BLOCK_SIZE,
        shapes=((16, 64),),
        dtypes=(torch.float16,),
        mamba_cache_mode=mamba_cache_mode,
        num_prefill_checkpoint_blocks=num_prefill_checkpoint_blocks,
        prefill_checkpoint_alignment=prefill_checkpoint_alignment,
    )
    builder = Mamba2AttentionMetadataBuilder(
        kv_cache_spec=spec,
        layer_names=["layer.0"],
        vllm_config=vllm_config,
        device=DEVICE,
    )
    vllm_config.cache_config.hash_block_size = MAMBA_BLOCK_SIZE
    vllm_config.cache_config.cache_hit_alignment_tokens = MAMBA_BLOCK_SIZE
    return builder


def _build(builder, seq_lens, query_lens):
    common = create_common_attn_metadata(
        BatchSpec(seq_lens=seq_lens, query_lens=query_lens),
        MAMBA_BLOCK_SIZE,
        DEVICE,
        arange_block_indices=True,
    )
    # create_common_attn_metadata leaves is_prefilling unset; the builder
    # asserts on it. Every row here is a full prefill.
    common = common.replace(is_prefilling=torch.ones(len(seq_lens), dtype=torch.bool))
    return builder.build(common_prefix_len=0, common_attn_metadata=common)


def test_builder_emits_one_checkpoint_entry_per_prefill_row():
    """Declined rows stay in place, masked, and a chunk ends on the checkpoint."""
    builder = _create_mamba2_builder()
    # Row 0 (seq_len 900) checkpoints at 768; row 1 (seq_len 100) is too short
    # for a checkpoint at all.
    meta = _build(builder, seq_lens=[900, 100], query_lens=[900, 100])

    assert meta.checkpoint_chunk_idx is not None
    assert meta.checkpoint_meta is not None
    assert meta.checkpoint_chunk_idx.numel() == 2
    # The exporter masks row 1 off by its zero offset and null block, but
    # still loads its chunk index first, so the placeholder must be in range.
    assert meta.checkpoint_meta.checkpoint_offsets.tolist() == [768, 0]
    assert meta.checkpoint_meta.state_indices[1].item() == NULL_BLOCK_ID
    assert meta.checkpoint_chunk_idx[1].item() == 0
    # The checkpoint's token offset is the end of its chunk. Asserting it here
    # also proves the chunk split actually landed on the checkpoint.
    ckpt_chunk = meta.checkpoint_chunk_idx[0]
    assert meta.cu_chunk_seqlen_p[ckpt_chunk + 1].item() == 768


def test_update_block_table_regathers_checkpoint_blocks():
    """Groups sharing a spec reuse one group's metadata via update_block_table.

    Checkpoint destinations are block-table entries, so each group needs its
    own: layers of different groups share physical tensors, and reusing the
    first group's indices writes every group's checkpoint into its blocks.
    """
    builder = _create_mamba2_builder()
    meta = _build(builder, seq_lens=[900, 100], query_lens=[900, 100])
    assert meta.checkpoint_meta is not None

    # A second group's table: same shape as the first, disjoint block ids.
    other_table = torch.arange(1000, 1016, dtype=torch.int32).reshape(2, 8)
    updated = builder.update_block_table(
        meta, other_table, torch.zeros(1000, dtype=torch.int64)
    )

    # Row 0 checkpoints in column cdiv(900, 256) - 2 == 2; row 1 declined.
    assert updated.checkpoint_meta is not None
    assert updated.checkpoint_meta.state_indices.tolist() == [
        other_table[0, 2].item(),
        NULL_BLOCK_ID,
    ]


def test_builder_emits_nothing_outside_align_mode():
    builder = _create_mamba2_builder(mamba_cache_mode="none")
    meta = _build(builder, seq_lens=[900], query_lens=[900])

    assert meta.checkpoint_chunk_idx is None


@pytest.mark.parametrize(
    "seq_len,query_start,drop_eagle,offset",
    [
        (1000, 0, False, 896),
        (97000, 85888, True, 10880),
        (99328, 88064, True, 11136),
        (97000, 94720, True, 2048),
    ],
)
def test_separate_checkpoint_metadata_uses_explicit_group_destination(
    seq_len,
    query_start,
    drop_eagle,
    offset,
):
    """Export beside the live table without overwriting a resumed input state."""
    from vllm.model_executor.layers.mamba.checkpoint import (
        MambaPrefillCheckpointBuilder,
    )

    config = SimpleNamespace(
        cache_config=SimpleNamespace(
            mamba_cache_mode="align",
            hash_block_size=128,
            cache_hit_alignment_tokens=128,
        ),
        speculative_config=SimpleNamespace(use_eagle_block_drop=lambda: drop_eagle),
    )
    spec = MambaSpec(
        block_size=8576,
        shapes=((1, 1),),
        dtypes=(torch.float32,),
        mamba_cache_mode="align",
        num_prefill_checkpoint_blocks=1,
        prefill_checkpoint_alignment=1,
    )
    builder = MambaPrefillCheckpointBuilder(config, spec)
    query_len = seq_len - query_start
    common = create_common_attn_metadata(
        BatchSpec(seq_lens=[seq_len, 100], query_lens=[query_len, 100]),
        8576,
        DEVICE,
        arange_block_indices=True,
    )
    common.mamba_checkpoint_block_ids = [71, -1]
    checkpoint = builder.build(common, [0, 1])
    assert checkpoint is not None
    assert checkpoint.offsets == [offset, 0]
    assert checkpoint.state_indices.tolist() == [71, NULL_BLOCK_ID]
    cu, _, _, chunk_idx = compute_chunk_metadata(
        256,
        2,
        torch.tensor([query_start, 0]),
        torch.tensor([0, query_len, query_len + 100]),
        checkpoint_offsets_p=checkpoint.offsets,
    )
    assert cu[chunk_idx[0] + 1] == offset
    # Another group must use its own dedicated destination.
    common.mamba_checkpoint_block_ids = [91, -1]
    other = builder.build(common, [0, 1])
    assert other.state_indices.tolist() == [91, NULL_BLOCK_ID]
    # Profiling without scheduler allocations must not export to a bogus slot.
    common.mamba_checkpoint_block_ids = None
    assert builder.build(common, [0, 1]) is None


def test_group_metadata_reuse_does_not_replace_separate_checkpoint_destinations():
    from vllm.v1.attention.backends.mamba2_attn import Mamba2AttentionBackend
    from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec
    from vllm.v1.worker.gpu.attn_utils import build_attn_metadata
    from vllm.v1.worker.utils import AttentionGroup

    builders = [_create_mamba2_builder(), _create_mamba2_builder()]
    for builder in builders:
        cache_config = builder.checkpoint_builder.vllm_config.cache_config
        cache_config.hash_block_size = 128
        cache_config.cache_hit_alignment_tokens = 128
    spec = builders[0].kv_cache_spec
    groups = [
        [
            AttentionGroup(
                backend=Mamba2AttentionBackend,
                layer_names=[f"layer{i}"],
                kv_cache_spec=spec,
                kv_cache_group_id=i,
                metadata_builders=[builder],
            )
        ]
        for i, builder in enumerate(builders)
    ]
    qsl = torch.tensor([0, 200], dtype=torch.int32)
    meta = build_attn_metadata(
        attn_groups=groups,
        num_reqs=1,
        num_tokens=200,
        query_start_loc_gpu=qsl,
        query_start_loc_cpu=qsl,
        max_query_len=200,
        seq_lens=torch.tensor([200]),
        max_seq_len=200,
        block_tables=[
            torch.tensor([[10]], dtype=torch.int32),
            torch.tensor([[20]], dtype=torch.int32),
        ],
        slot_mappings=torch.zeros((2, 200), dtype=torch.int64),
        kv_cache_config=KVCacheConfig(
            num_blocks=100,
            kv_cache_tensors=[],
            kv_cache_groups=[KVCacheGroupSpec([f"layer{i}"], spec) for i in range(2)],
        ),
        seq_lens_cpu_upper_bound=torch.tensor([200]),
        is_prefilling=torch.tensor([True]),
        mamba_checkpoint_block_ids={0: [71], 1: [91]},
    )
    assert meta["layer0"].checkpoint_meta.state_indices.tolist() == [71]
    assert meta["layer1"].checkpoint_meta.state_indices.tolist() == [91]
    assert meta["layer0"].checkpoint_meta.offsets == [128]
    assert meta["layer1"].checkpoint_meta.offsets == [128]
