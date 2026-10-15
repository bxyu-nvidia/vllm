# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright (c) 2024, Tri Dao, Albert Gu.
# Adapted from https://github.com/state-spaces/mamba/blob/v2.2.4/mamba_ssm/ops/triton/ssd_combined.py

# ruff: noqa: E501

import torch
from einops import rearrange
from packaging import version

from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

from .ssd_bmm import _bmm_chunk_fwd
from .ssd_chunk_scan import _chunk_scan_fwd
from .ssd_chunk_state import _chunk_cumsum_fwd, _chunk_state_fwd
from .ssd_state_passing import _state_passing_fwd

TRITON_22 = version.parse(triton.__version__) >= version.parse("2.2.0")


@triton.jit
def _export_mamba2_prefill_checkpoint_kernel(
    offsets,
    state_indices,
    query_start_loc,
    last_chunk_indices,
    states,
    x,
    conv_cache,
    ssm_cache,
    stride_states_chunk: tl.int64,
    stride_x_dim: tl.int64,
    stride_x_token: tl.int64,
    stride_conv_block: tl.int64,
    stride_conv_dim: tl.int64,
    stride_conv_token: tl.int64,
    stride_ssm_block: tl.int64,
    SSM_SIZE: tl.constexpr,
    CONV_DIM: tl.constexpr,
    CONV_WIDTH: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    NULL_STATE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offset = tl.load(offsets + row)
    if offset == 0:
        return
    state_idx = tl.load(state_indices + row).to(tl.int64)
    if state_idx == NULL_STATE:
        return
    first_chunk = tl.load(last_chunk_indices + row - 1, row > 0, other=-1) + 1
    chunk = first_chunk + offset // CHUNK_SIZE - 1
    idx = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    state = tl.load(
        states + chunk.to(tl.int64) * stride_states_chunk + idx,
        idx < SSM_SIZE,
        other=0,
    )
    tl.store(ssm_cache + state_idx * stride_ssm_block + idx, state, idx < SSM_SIZE)

    dim = idx // CONV_WIDTH
    token = tl.load(query_start_loc + row) + offset - CONV_WIDTH + idx % CONV_WIDTH
    conv = tl.load(
        x + dim * stride_x_dim + token.to(tl.int64) * stride_x_token,
        dim < CONV_DIM,
        other=0,
    )
    tl.store(
        conv_cache
        + state_idx * stride_conv_block
        + dim * stride_conv_dim
        + idx % CONV_WIDTH * stride_conv_token,
        conv,
        dim < CONV_DIM,
    )


def export_mamba2_prefill_checkpoint(
    offsets: torch.Tensor,
    state_indices: torch.Tensor,
    query_start_loc: torch.Tensor,
    last_chunk_indices: torch.Tensor,
    states: torch.Tensor,
    x: torch.Tensor,
    conv_cache: torch.Tensor,
    ssm_cache: torch.Tensor,
    chunk_size: int,
    conv_width: int,
) -> None:
    """Save a chunk-aligned SSD state and its convolution input window."""
    ssm_size = states.shape[1] * states.shape[2] * states.shape[3]
    block = 1024
    _export_mamba2_prefill_checkpoint_kernel[
        (offsets.numel(), triton.cdiv(max(ssm_size, x.shape[0] * conv_width), block))
    ](
        offsets,
        state_indices,
        query_start_loc,
        last_chunk_indices,
        states,
        x,
        conv_cache,
        ssm_cache,
        states.stride(0),
        x.stride(0),
        x.stride(1),
        conv_cache.stride(0),
        conv_cache.stride(1),
        conv_cache.stride(2),
        ssm_cache.stride(0),
        SSM_SIZE=ssm_size,
        CONV_DIM=x.shape[0],
        CONV_WIDTH=conv_width,
        CHUNK_SIZE=chunk_size,
        NULL_STATE=NULL_BLOCK_ID,
        BLOCK=block,
    )


def is_int_pow_2(n):
    return isinstance(n, int) and n > 0 and (n & (n - 1)) == 0


def _mamba_chunk_scan_combined_fwd(
    x,
    dt,
    A,
    B,
    C,
    chunk_size,
    out,
    D=None,
    z=None,
    dt_bias=None,
    initial_states=None,
    return_intermediate_states=False,
    seq_idx=None,
    cu_seqlens=None,
    cu_chunk_seqlens=None,
    last_chunk_indices=None,
    dt_softplus=False,
    dt_limit=(0.0, float("inf")),
    state_dtype=None,
):
    assert is_int_pow_2(chunk_size), "chunk_size must be integer power of 2"
    seqlen, nheads, headdim = x.shape
    _, ngroups, dstate = B.shape
    assert nheads % ngroups == 0
    assert B.shape == (seqlen, ngroups, dstate)
    assert dt.shape == (seqlen, nheads)
    assert A.shape == (nheads,)
    assert C.shape == B.shape
    if z is not None:
        assert z.shape == x.shape
    if D is not None:
        assert D.shape == (nheads, headdim) or D.shape == (nheads,)
    if seq_idx is not None:
        assert seq_idx.shape == (cu_chunk_seqlens.shape[0] - 1,)
    if B.stride(-1) != 1:
        B = B.contiguous()
    if C.stride(-1) != 1:
        C = C.contiguous()
    if (
        x.stride(-1) != 1 and x.stride(0) != 1
    ):  # Either M or K dimension should be contiguous
        x = x.contiguous()
    if (
        z is not None and z.stride(-1) != 1 and z.stride(0) != 1
    ):  # Either M or K dimension should be contiguous
        z = z.contiguous()
    if D is not None and D.stride(-1) != 1:
        D = D.contiguous()
    assert cu_seqlens is not None, "Assuming varlen input - must supply cu_seqlens"

    if initial_states is not None:
        assert initial_states.shape == (len(cu_seqlens) - 1, nheads, headdim, dstate)

    # This function executes 5 sub-functions for computing mamba
    # - a good resource is the blog https://goombalab.github.io/blog/2024/mamba2-part3-algorithm/
    #   which has a minimal implementation to understand the below operations
    # - as explained by the blog, mamba is a special case of causal attention
    # - the idea is to chunk the attention matrix and compute each
    #   submatrix separately using different optimizations.
    # - see the blog and paper for a visualization of the submatrices
    #   which we refer to in the comments below

    # 1. Compute chunked cumsum of A * dt
    # - here dt may go through a softplus activation
    dA_cumsum, dt = _chunk_cumsum_fwd(
        dt,
        A,
        chunk_size,
        cu_chunk_seqlens,
        dt_bias=dt_bias,
        dt_softplus=dt_softplus,
        dt_limit=dt_limit,
    )

    # 2. Compute the state for each intra-chunk
    # (right term of low-rank factorization of off-diagonal blocks; B terms)
    states = _chunk_state_fwd(
        B, x, dt, dA_cumsum, cu_chunk_seqlens, states_in_fp32=True
    )

    # 3. Compute the inter-chunk SSM recurrence; produces correct SSM states at chunk boundaries
    # (middle term of factorization of off-diag blocks; A terms)
    # - parallelized across sequences using last_chunk_indices to derive
    #   per-sequence chunk ranges. Each sequence's state passing runs independently.
    states = _state_passing_fwd(
        rearrange(states, "... p n -> ... (p n)"),
        dA_cumsum,  # (nheads, nchunks, chunk_size)
        last_chunk_indices,
        initial_states=rearrange(initial_states, "... p n -> ... (p n)")
        if initial_states is not None
        else None,  # (batch, nheads, headdim*dstate)
        out_dtype=state_dtype if state_dtype is not None else C.dtype,
    )
    states = rearrange(states, "... (p n) -> ... p n", n=dstate)

    # 4. Compute batched matrix multiply for C_j^T B_i terms
    CB = _bmm_chunk_fwd(C, B, chunk_size, cu_chunk_seqlens, output_dtype=torch.float32)

    # 5. Scan and compute the diagonal blocks, taking into
    #    account past causal states.
    # - if initial states are provided, then states information will be
    #   augmented with initial_states.
    # - to do this properly, we need to account for example changes in
    #   the continuous batch, therefore we introduce pseudo chunks, which is
    #   a chunk that is split up each time an example changes.
    # - in each (pseudo) chunk, we detect if the previous (pseudo) chunk had
    #   a seq_idx change, in which case we take states information from
    #   init_states.
    _chunk_scan_fwd(
        CB,
        x,
        dt,
        dA_cumsum,
        C,
        states,
        cu_chunk_seqlens,
        out,  # in-place update
        seq_idx,
        D=D,
        z=z,
        initial_states=initial_states,
    )

    if return_intermediate_states:
        return states
    else:
        return states[last_chunk_indices]


def mamba_chunk_scan_combined_varlen(
    x,
    dt,
    A,
    B,
    C,
    chunk_size,
    cu_seqlens,
    cu_chunk_seqlens,
    last_chunk_indices,
    seq_idx,
    out,
    D=None,
    z=None,
    dt_bias=None,
    initial_states=None,
    dt_softplus=False,
    dt_limit=(0.0, float("inf")),
    return_intermediate_states=False,
    state_dtype=None,
):
    """Argument:
        x: (seqlen, nheads, headdim)
        dt: (seqlen, nheads)
        A: (nheads)
        B: (seqlen, ngroups, dstate)
        C: (seqlen, ngroups, dstate)
        chunk_size: int
        cu_seqlens: (batch + 1,)
        cu_chunk_seqlens: (nchunks + 1,)
        last_chunk_indices: (batch,)
        seq_idx: (nchunks,)
        out: (seqlen, nheads, headdim) preallocated output tensor
        D: (nheads, headdim) or (nheads,)
        z: (seqlen, nheads, headdim)
        dt_bias: (nheads,)
        initial_states: (batch, nheads, headdim, dstate)
        dt_softplus: Whether to apply softplus to dt
        out: (seqlen, nheads, headdim) preallocated output tensor
        state_dtype: The data type of the ssm state
    Return:
        varlen_states: (batch, nheads, headdim, dstate)
    """
    assert cu_seqlens is not None, "cu_seqlens must be provided assuming varlen input"
    assert seq_idx is not None

    varlen_states = _mamba_chunk_scan_combined_fwd(
        x,
        dt,
        A,
        B,
        C,
        chunk_size,
        out,
        D=D,
        z=z,
        dt_bias=dt_bias,
        initial_states=initial_states,
        return_intermediate_states=return_intermediate_states,
        seq_idx=seq_idx,
        cu_seqlens=cu_seqlens,
        cu_chunk_seqlens=cu_chunk_seqlens,
        last_chunk_indices=last_chunk_indices,
        dt_softplus=dt_softplus,
        dt_limit=dt_limit,
        state_dtype=state_dtype,
    )

    return varlen_states


from vllm.platforms import current_platform  # noqa: E402

if current_platform.is_cpu():
    import vllm.model_executor.layers.mamba.ops.cpu.mamba_ssm as cpu_mamba_ssm

    _mamba_chunk_scan_combined_fwd = cpu_mamba_ssm._mamba_chunk_scan_combined_fwd_cpu  # type: ignore
