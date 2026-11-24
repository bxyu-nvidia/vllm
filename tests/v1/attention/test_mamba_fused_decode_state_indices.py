# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The fused decode state-index kernel (VLLM_MAMBA_FUSED_DECODE_STATE_INDICES)
must fill the CUDA graph buffers exactly as mamba_get_block_table_tensor plus
the copies in _update_metadata_for_cudagraph_capture do."""

from types import SimpleNamespace

import pytest
import torch

from tests.v1.attention.utils import BatchSpec, create_common_attn_metadata
from vllm.config.compilation import CUDAGraphMode
from vllm.triton_utils import triton
from vllm.v1.attention.backends.mamba_attn import (
    BaseMambaAttentionMetadata,
    BaseMambaAttentionMetadataBuilder,
    _decode_state_indices_kernel,
)
from vllm.v1.attention.backends.utils import (
    NULL_BLOCK_ID,
    mamba_get_block_table_tensor,
)
from vllm.v1.kv_cache_interface import MambaSpec

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Triton kernel needs a GPU"
)

NUM_SPEC_TOKENS = 5
BLOCK_SIZE = 16


def _reference(block_table, seq_lens, spec, num_decodes, num_rows, out_width):
    """The unfused path: gather, then copy into the buffer and pad with NULL."""
    gathered = mamba_get_block_table_tensor(block_table, seq_lens, spec, "align")
    out = torch.full(
        (num_rows, out_width), -7, dtype=torch.int32, device=block_table.device
    )
    out[:num_decodes].copy_(gathered[:num_decodes, :out_width])
    out[num_decodes:] = NULL_BLOCK_ID
    return out


@pytest.mark.parametrize("num_speculative_blocks", [0, NUM_SPEC_TOKENS])
@pytest.mark.parametrize("num_decodes,num_rows", [(1, 1), (4, 4), (5, 8), (37, 40)])
def test_kernel_matches_unfused_path(num_speculative_blocks, num_decodes, num_rows):
    device = torch.device("cuda")
    torch.manual_seed(num_decodes * 10 + num_speculative_blocks)
    spec = MambaSpec(
        block_size=BLOCK_SIZE,
        shapes=((1,), (1,)),
        dtypes=(torch.float32,),
        mamba_cache_mode="align",
        num_speculative_blocks=num_speculative_blocks,
    )
    max_blocks = 64
    # Non-contiguous rows, to check the row stride is honoured.
    block_table = torch.randint(
        1, 10_000, (num_rows, max_blocks + 3), dtype=torch.int32, device=device
    )[:, :max_blocks]
    max_seq_len = (max_blocks - num_speculative_blocks - 1) * BLOCK_SIZE
    seq_lens = torch.randint(
        1, max_seq_len, (num_rows,), dtype=torch.int32, device=device
    )
    seq_lens[0] = 0  # Zero-length rows clamp to block 0.
    if num_rows > 1:
        seq_lens[1] = BLOCK_SIZE  # Last token at the end of a block.
    if num_rows > 2:
        seq_lens[2] = BLOCK_SIZE + 1  # First token of the next block.

    out_width = 1 + NUM_SPEC_TOKENS
    out = torch.full((64, out_width), -7, dtype=torch.int32, device=device)
    replay = torch.full((64,), -7, dtype=torch.int32, device=device)
    _decode_state_indices_kernel[(num_rows,)](
        block_table,
        block_table.stride(0),
        seq_lens,
        out,
        out.stride(0),
        replay,
        num_decodes,
        num_rows,
        BLOCK_SIZE,
        SRC_WIDTH=1 + num_speculative_blocks,
        OUT_WIDTH=out_width,
        OUT_WIDTH_POW2=triton.next_power_of_2(out_width),
        NULL_ID=NULL_BLOCK_ID,
        WRITE_REPLAY=True,
    )

    expected = _reference(block_table, seq_lens, spec, num_decodes, num_rows, out_width)
    torch.testing.assert_close(out[:num_rows], expected, rtol=0, atol=0)
    torch.testing.assert_close(replay[:num_rows], expected[:, 0], rtol=0, atol=0)
    # Rows past num_rows are left alone.
    assert (out[num_rows:] == -7).all()
    assert (replay[num_rows:] == -7).all()


class _Builder(BaseMambaAttentionMetadataBuilder[BaseMambaAttentionMetadata]):
    metadata_cls = BaseMambaAttentionMetadata


def _build(monkeypatch, fused: bool, num_speculative_blocks: int, seq_lens):
    monkeypatch.setenv("VLLM_MAMBA_FUSED_DECODE_STATE_INDICES", "1" if fused else "0")
    device = torch.device("cuda")
    config = SimpleNamespace(
        cache_config=SimpleNamespace(
            block_size=BLOCK_SIZE,
            mamba_cache_mode="align",
            use_replayssm=False,
            replayssm_buffer_len=16,
        ),
        compilation_config=SimpleNamespace(
            cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
            max_cudagraph_capture_size=None,
        ),
        speculative_config=SimpleNamespace(
            num_speculative_tokens=NUM_SPEC_TOKENS, parallel_drafting=False
        ),
        num_speculative_tokens=NUM_SPEC_TOKENS,
        use_flashinfer_replayssm=False,
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
        scheduler_config=SimpleNamespace(max_num_seqs=64, async_scheduling=True),
        model_config=SimpleNamespace(max_model_len=4096),
    )
    spec = MambaSpec(
        block_size=BLOCK_SIZE,
        shapes=((1,), (1,)),
        dtypes=(torch.float32,),
        mamba_cache_mode="align",
        num_speculative_blocks=num_speculative_blocks,
    )
    builder = _Builder(spec, ["layer0"], config, device)
    assert builder.fuse_decode_state_indices == fused
    query_lens = [1 + NUM_SPEC_TOKENS] * len(seq_lens)
    common = create_common_attn_metadata(
        BatchSpec(seq_lens=seq_lens, query_lens=query_lens),
        block_size=BLOCK_SIZE,
        device=device,
    )
    common = common.replace(is_prefilling=torch.zeros(len(seq_lens), dtype=torch.bool))
    num_accepted = torch.tensor(
        [1 + i % (1 + NUM_SPEC_TOKENS) for i in range(len(seq_lens))],
        dtype=torch.int32,
        device=device,
    )
    return builder.build(0, common, num_accepted_tokens=num_accepted)


@pytest.mark.parametrize("num_speculative_blocks", [0, NUM_SPEC_TOKENS])
def test_builder_metadata_matches_unfused(monkeypatch, num_speculative_blocks):
    seq_lens = [6, 17, 33, 100, 255]
    unfused = _build(monkeypatch, False, num_speculative_blocks, seq_lens)
    fused = _build(monkeypatch, True, num_speculative_blocks, seq_lens)

    assert fused.num_decodes == unfused.num_decodes == len(seq_lens)
    assert fused.num_prefills == unfused.num_prefills == 0
    torch.testing.assert_close(
        fused.state_indices_tensor_d, unfused.state_indices_tensor_d, rtol=0, atol=0
    )
    torch.testing.assert_close(
        fused.num_accepted_tokens, unfused.num_accepted_tokens, rtol=0, atol=0
    )
    assert fused.state_indices_tensor_p.numel() == 0
