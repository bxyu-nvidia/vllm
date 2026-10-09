# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Diagnose ReplaySSM rounding at checkpoint and prefix-resume boundaries.

Run on one CUDA GPU with FlashInfer 0.7.0.post1; no model weights are needed:
    .venv/bin/python benchmarks/replayssm/check_fp16_cache.py

The default verify length is six (one token plus five drafts). Use --tokens 1
for ordinary decode and --reference-only for the CPU arithmetic example. This
is a deliberately sensitive input, not a model accuracy evaluation or timing.
"""

import argparse
import json

import torch


def materialize(state, x_cache, b_cache, dt_cache, a, rand_seed, philox_rounds):
    from flashinfer.mamba.replayssm_materialize import replayssm_materialize

    def table(value):
        return torch.tensor([value], dtype=torch.int64, device=state.device)

    ptrs_and_strides = []
    for tensor in (state, x_cache, b_cache, dt_cache):
        ptrs_and_strides.extend((table(tensor.data_ptr()), table(tensor.stride(0))))
    replayssm_materialize(
        *ptrs_and_strides,
        table(a.data_ptr()),
        table(0),
        table(0),
        torch.tensor([[0]], dtype=torch.int32, device=state.device),
        torch.tensor([[1]], dtype=torch.int32, device=state.device),
        torch.tensor([0], dtype=torch.int32, device=state.device),
        torch.tensor([8], dtype=torch.int32, device=state.device),
        torch.tensor([0], dtype=torch.int32, device=state.device),
        state_dtype=state.dtype,
        input_dtype=x_cache.dtype,
        matrixA_dtype=a.dtype,
        dim=state.size(2),
        dstate=state.size(3),
        num_heads=state.size(1),
        heads_per_group=state.size(1),
        max_window=16,
        ring_buffer_len=x_cache.size(2),
        rand_seed=rand_seed,
        philox_rounds=philox_rounds,
    )


def run_case(state_dtype, philox_rounds, mode, algorithm, tokens):
    from flashinfer.mamba.checkpointing_ssu import (
        allocate_checkpointing_ssu_scratch,
        checkpointing_ssu,
    )

    device = torch.device("cuda")
    heads, dim, dstate = 8, 64, 128
    # Eight committed tokens plus up to eight new tokens force a checkpoint when
    # max_window=8. Both histories represent exactly the same mathematical state.
    window = 8 if mode == "checkpoint" else 16
    ring_len = window + tokens
    state = torch.zeros(2, heads, dim, dstate, dtype=state_dtype, device=device)
    state[0, :, :, 0] = 1
    x_cache = torch.zeros(2, heads, ring_len, dim, dtype=torch.bfloat16, device=device)
    b_cache = torch.zeros(2, 1, ring_len, dstate, dtype=torch.bfloat16, device=device)
    dt_cache = torch.zeros(2, heads, ring_len, device=device)
    x_cache[0, :, 0, :] = 1
    b_cache[0, 0, 0, 0] = 1
    dt_cache[0, :, 0] = 17 / 4096
    a = torch.zeros(heads, device=device)
    rand_seed = (
        torch.tensor([12345], dtype=torch.int64, device=device)
        if philox_rounds
        else None
    )
    slot = 0
    committed = torch.tensor([8, 0], dtype=torch.int32, device=device)
    if mode == "prefix-resume":
        materialize(state, x_cache, b_cache, dt_cache, a, rand_seed, philox_rounds)
        slot = 1

    # A=0 gives unit decay; the new token contributes zero. C reads only the
    # state coordinate whose exact value is 1 + 17/4096, for every output row.
    x = torch.zeros(1, tokens, heads, dim, dtype=torch.bfloat16, device=device)
    dt = torch.zeros(
        1, tokens, heads, 1, dtype=torch.bfloat16, device=device
    ).expand_as(x)
    b = torch.zeros(1, tokens, 1, dstate, dtype=torch.bfloat16, device=device)
    c = torch.zeros_like(b)
    c[..., 0] = 1
    out = torch.empty_like(x)
    cb_scaled, cum_adt, cb_old = allocate_checkpointing_ssu_scratch(
        1, heads, tokens, window, torch.bfloat16, device
    )
    checkpointing_ssu(
        state,
        x_cache,
        b_cache,
        dt_cache,
        torch.zeros(2, dtype=torch.int32, device=device),
        committed,
        x,
        dt,
        a.as_strided((heads, dim, dstate), (1, 0, 0)),
        b,
        c,
        out,
        state_batch_indices=torch.tensor([slot], dtype=torch.int32, device=device),
        rand_seed=rand_seed,
        philox_rounds=philox_rounds,
        cb_scaled=cb_scaled,
        cumAdt_vec=cum_adt,
        cb_old=cb_old,
        algorithm=algorithm,
        d_split=1,
    )
    expected = torch.full_like(out, 1 + 17 / 4096)
    torch.accelerator.synchronize()
    return {
        "dtype": str(state_dtype),
        "philox_rounds": philox_rounds,
        "algorithm": algorithm,
        "tokens": tokens,
        "mode": mode,
        "expected_bf16": expected.flatten()[0].item(),
        "observed_values": out.unique().float().tolist(),
        "matching_fraction": (out == expected).float().mean().item(),
        "max_abs_error": (out.float() - expected.float()).abs().max().item(),
        "finite": out.isfinite().all().item(),
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--reference-only", action="store_true")
    parser.add_argument("--tokens", type=int, choices=range(1, 9), default=6)
    parser.add_argument(
        "--algorithm", choices=["monolith", "two-kernel", "both"], default="both"
    )
    args = parser.parse_args()
    exact = torch.tensor(1 + 17 / 4096, dtype=torch.float32)
    print(
        json.dumps(
            {
                "exact_state": exact.item(),
                "direct_bf16": exact.bfloat16().item(),
                "fp16_intermediate": exact.half().item(),
                "via_fp16_bf16": exact.half().bfloat16().item(),
            }
        ),
        flush=True,
    )
    if args.reference_only:
        return
    if not torch.cuda.is_available():
        parser.error("CUDA is required; --reference-only runs just the CPU arithmetic")

    import flashinfer

    print(f"FlashInfer {flashinfer.__version__}; GPU {torch.cuda.get_device_name()}")
    algorithms = (
        ["monolith", "two-kernel"] if args.algorithm == "both" else [args.algorithm]
    )
    for algorithm in algorithms:
        for dtype, rounds in (
            (torch.float32, 0),
            (torch.float16, 0),
            (torch.float16, 5),
        ):
            for mode in ("live", "checkpoint", "prefix-resume"):
                result = run_case(dtype, rounds, mode, algorithm, args.tokens)
                print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
