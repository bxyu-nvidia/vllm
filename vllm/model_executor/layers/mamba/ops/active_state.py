# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _copy_state_kernel(
    src,
    dst,
    src_ids,
    dst_ids,
    reset,
    has_initial,
    accepted,
    seed,
    src_stride: tl.int64,
    dst_stride: tl.int64,
    src_row_stride: tl.constexpr,
    dst_row_stride: tl.constexpr,
    SRC_COLS: tl.constexpr,
    SIZE: tl.constexpr,
    SRC_NULL: tl.constexpr,
    DST_NULL: tl.constexpr,
    HAS_RESET: tl.constexpr,
    HAS_INITIAL: tl.constexpr,
    HAS_ACCEPTED: tl.constexpr,
    STOCHASTIC: tl.constexpr,
    PHILOX_ROUNDS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row, col = tl.program_id(1), tl.program_id(2)
    if HAS_RESET:
        should_copy = tl.load(reset + row)
        if not should_copy:
            return
    src_col = col
    if HAS_ACCEPTED:
        src_col = tl.minimum(tl.maximum(tl.load(accepted + row) - 1, 0), SRC_COLS - 1)
    src_id = tl.load(src_ids + row * src_row_stride + src_col).to(tl.int64)
    dst_id = tl.load(dst_ids + row * dst_row_stride + col).to(tl.int64)
    if dst_id < 0 or dst_id == DST_NULL:
        return
    valid = (src_id >= 0) & (src_id != SRC_NULL)
    if HAS_INITIAL:
        valid = valid & tl.load(has_initial + row)
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(
        src + src_id * src_stride + offsets, mask=(offsets < SIZE) & valid, other=0
    ).to(tl.float32)
    if STOCHASTIC:
        truncated = value.to(tl.float16, fp_downcast_rounding="rtz")
        bits = truncated.to(tl.uint16, bitcast=True)
        away = (bits + 1).to(tl.uint16).to(tl.float16, bitcast=True)
        low, high = truncated.to(tl.float32), away.to(tl.float32)
        probability = (value - low) / (high - low)
        random_offsets = (dst_id * SIZE + offsets).to(tl.uint32)
        random = tl.rand(tl.load(seed), random_offsets, n_rounds=PHILOX_ROUNDS)
        rounded = tl.where(random < probability, away, truncated)
        rounded = tl.where(tl.abs(value) <= 65504, rounded, value.to(tl.float16))
        tl.store(dst + dst_id * dst_stride + offsets, rounded, mask=offsets < SIZE)
    else:
        tl.store(dst + dst_id * dst_stride + offsets, value, mask=offsets < SIZE)


def copy_mamba_state(
    src: torch.Tensor,
    dst: torch.Tensor,
    src_ids: torch.Tensor,
    dst_ids: torch.Tensor,
    *,
    reset: torch.Tensor | None = None,
    has_initial: torch.Tensor | None = None,
    accepted: torch.Tensor | None = None,
    src_null: int = -1,
    dst_null: int = 0,
    stochastic: bool = False,
    philox_rounds: int = 0,
) -> None:
    """Copy paged SSM states, optionally initializing or committing live slots."""
    if dst_ids.numel() == 0:
        return
    assert src.shape[1:] == dst.shape[1:]
    size = src[0].numel()
    assert src[0].is_contiguous() and dst[0].is_contiguous()
    src_ids = src_ids.unsqueeze(1) if src_ids.ndim == 1 else src_ids
    dst_ids = dst_ids.unsqueeze(1) if dst_ids.ndim == 1 else dst_ids
    assert src_ids.shape[0] == dst_ids.shape[0]
    assert accepted is not None or src_ids.shape[1] == dst_ids.shape[1]
    seed = torch.randint(0, 2**31, (1,), device=src.device) if stochastic else None
    _copy_state_kernel[(triton.cdiv(size, 1024), dst_ids.shape[0], dst_ids.shape[1])](
        src,
        dst,
        src_ids,
        dst_ids,
        reset,
        has_initial,
        accepted,
        seed,
        src.stride(0),
        dst.stride(0),
        src_ids.stride(0),
        dst_ids.stride(0),
        src_ids.shape[1],
        size,
        src_null,
        dst_null,
        reset is not None,
        has_initial is not None,
        accepted is not None,
        stochastic,
        philox_rounds or 10,
        1024,
    )
