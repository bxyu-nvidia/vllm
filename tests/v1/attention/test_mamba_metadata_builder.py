# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for BaseMambaAttentionMetadataBuilder batch classification."""

from types import SimpleNamespace

import pytest
import torch

from tests.v1.attention.utils import MockMambaBuilder
from vllm.config.compilation import CUDAGraphMode


def _make_vllm_config(
    max_model_len: int,
    max_num_seqs: int,
    num_speculative_tokens: int = 0,
    block_size: int | None = None,
):
    """Create a minimal mock VllmConfig with only the fields the builder
    accesses, avoiding any model download / HF config inspection."""
    speculative_config = (
        SimpleNamespace(
            num_speculative_tokens=num_speculative_tokens,
            parallel_drafting=False,
        )
        if num_speculative_tokens > 0
        else None
    )
    return SimpleNamespace(
        use_flashinfer_replayssm=False,
        cache_config=SimpleNamespace(
            block_size=block_size,
            mamba_cache_mode="none",
            use_replayssm=False,
            replayssm_buffer_len=16,
        ),
        compilation_config=SimpleNamespace(
            cudagraph_mode=CUDAGraphMode.FULL,
            max_cudagraph_capture_size=None,
        ),
        speculative_config=speculative_config,
        num_speculative_tokens=num_speculative_tokens,
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
        scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs),
        model_config=SimpleNamespace(max_model_len=max_model_len),
    )


def test_mamba_single_token_prompt_runs_as_prefill():
    seq_lens = [8, 9, 1]
    config = _make_vllm_config(256, len(seq_lens), block_size=16)
    metadata = MockMambaBuilder.build_mamba_metadata(
        config,
        seq_lens=seq_lens,
        query_lens=[1] * len(seq_lens),
        is_prefilling=[False, False, True],
    )

    assert metadata.num_decodes == 2
    assert metadata.num_prefills == 1
    assert metadata.has_initial_states_p.tolist() == [False]


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="requires CUDA"
            ),
        ),
    ],
)
@pytest.mark.parametrize("num_reqs", [0, 1, 65])
@pytest.mark.parametrize("num_speculative_blocks", [0, 2, 5])
def test_mamba_aligned_block_table_strides_and_padded_rows(
    device: str, num_reqs: int, num_speculative_blocks: int
):
    from vllm.v1.attention.backends.utils import mamba_get_block_table_tensor
    from vllm.v1.kv_cache_interface import MambaSpec

    lengths = torch.tensor([0, 1, 127, 128, 129, 255, 256, 257, 511, 512, 513])
    seq_storage = lengths.repeat(12).to(device)
    seq_lens = seq_storage[::2][:num_reqs]
    storage = torch.arange((num_reqs + 2) * 64, dtype=torch.int32, device=device)
    block_table = storage.view(num_reqs + 2, 64)[:, ::2]
    spec = MambaSpec(
        block_size=128,
        shapes=((16, 64),),
        dtypes=(torch.float16,),
        num_speculative_blocks=num_speculative_blocks,
    )
    # Compute a CPU oracle so this cannot accidentally compare two fused calls.
    first = ((seq_lens.cpu() - 1) // 128).clamp(min=0)
    columns = first[:, None] + torch.arange(1 + num_speculative_blocks)
    expected = block_table.cpu()[:num_reqs].gather(1, columns)
    actual = mamba_get_block_table_tensor(block_table, seq_lens, spec, "align")
    torch.testing.assert_close(actual.cpu(), expected)
    assert (
        mamba_get_block_table_tensor(block_table, seq_lens, spec, "none") is block_table
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_mamba_aligned_block_table_graph_reads_updated_lengths():
    from vllm.v1.attention.backends.utils import mamba_get_block_table_tensor
    from vllm.v1.kv_cache_interface import MambaSpec

    spec = MambaSpec(block_size=128, shapes=((16, 64),), dtypes=(torch.float16,))
    table = torch.arange(8, dtype=torch.int32, device="cuda").view(2, 4)
    lengths = torch.tensor([1, 129], dtype=torch.int32, device="cuda")
    # Compile outside capture.
    mamba_get_block_table_tensor(table, lengths, spec, "align")
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = mamba_get_block_table_tensor(table, lengths, spec, "align")
    lengths.copy_(torch.tensor([257, 0], dtype=torch.int32, device="cuda"))
    graph.replay()
    assert result.cpu().tolist() == [[2], [4]]
