# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Diagnose ReplaySSM rounding at checkpoint and prefix-resume boundaries.

Run on one CUDA GPU with FlashInfer 0.7.0.post1; no model weights are needed:
    .venv/bin/python benchmarks/replayssm/check_fp16_cache.py

The default verify length is six (one token plus five drafts). Use --tokens 1
for ordinary decode and --reference-only for the CPU arithmetic example. This
is a deliberately sensitive input, not a model accuracy evaluation or timing.
The main reference is the unrounded FP32 recurrence. Prefix resume also reports
agreement with the actual stored state: FP16 quantization alone is not a bad read.
Use --steps 128 for randomized repeated checkpoints and --prefix-every 17 to
include prefix materialization. These report errors, not model accuracy or timing.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch


def materialize(
    state,
    x_cache,
    b_cache,
    dt_cache,
    a,
    rand_seed,
    philox_rounds,
    *,
    count=8,
    start=0,
    window=16,
    src=0,
    dst=1,
):
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
        torch.tensor([[src]], dtype=torch.int32, device=state.device),
        torch.tensor([[dst]], dtype=torch.int32, device=state.device),
        torch.tensor([start], dtype=torch.int32, device=state.device),
        torch.tensor([count], dtype=torch.int32, device=state.device),
        torch.tensor([0], dtype=torch.int32, device=state.device),
        state_dtype=state.dtype,
        input_dtype=x_cache.dtype,
        matrixA_dtype=a.dtype,
        dim=state.size(2),
        dstate=state.size(3),
        num_heads=state.size(1),
        heads_per_group=state.size(1),
        max_window=window,
        ring_buffer_len=x_cache.size(2),
        rand_seed=rand_seed,
        philox_rounds=philox_rounds,
    )


def run_case(
    state_dtype, philox_rounds, mode, algorithm, tokens, *, checkpoint_window=8
):
    from flashinfer.mamba.checkpointing_ssu import (
        allocate_checkpointing_ssu_scratch,
        checkpointing_ssu,
    )

    device = torch.device("cuda")
    heads, dim, dstate = 8, 64, 128
    # Eight committed tokens plus up to eight new tokens force a checkpoint when
    # max_window=8. Both histories represent exactly the same mathematical state.
    window = checkpoint_window if mode == "checkpoint" else 16
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
    committed = torch.tensor(
        [window if mode == "checkpoint" else 8, 0], dtype=torch.int32, device=device
    )
    if mode == "cache-readout":
        # Both cache entries are exactly representable in FP16. Their small
        # difference disappears when the matmul operands are rounded to BF16.
        state[0, :, :, 0] = 1 + 1 / 1024
        state[0, :, :, 1] = 1
        committed.zero_()
    if mode == "prefix-resume":
        materialize(state, x_cache, b_cache, dt_cache, a, rand_seed, philox_rounds)
        slot = 1
    stored_prefix = (
        state[slot, :, :, 0].float().clone() if mode == "prefix-resume" else None
    )

    # A=0 gives unit decay; the new token contributes zero. C reads only the
    # state coordinate whose exact value is 1 + 17/4096, for every output row.
    x = torch.zeros(1, tokens, heads, dim, dtype=torch.bfloat16, device=device)
    dt = torch.zeros(
        1, tokens, heads, 1, dtype=torch.bfloat16, device=device
    ).expand_as(x)
    b = torch.zeros(1, tokens, 1, dstate, dtype=torch.bfloat16, device=device)
    c = torch.zeros_like(b)
    c[..., 0] = 1
    if mode == "cache-readout":
        c[..., 1] = -1
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
    expected = torch.full_like(
        out, 1 / 1024 if mode == "cache-readout" else 1 + 17 / 4096
    )
    torch.accelerator.synchronize()
    result = {
        "dtype": str(state_dtype),
        "philox_rounds": philox_rounds,
        "algorithm": algorithm,
        "tokens": tokens,
        "window": window,
        "mode": mode,
        "expected_bf16": expected.flatten()[0].item(),
        "observed_values": out.unique().float().tolist(),
        "matching_fraction": (out == expected).float().mean().item(),
        "max_abs_error": (out.float() - expected.float()).abs().max().item(),
        "finite": out.isfinite().all().item(),
    }
    if stored_prefix is not None:
        stored_readout = stored_prefix.to(out.dtype)[None, None].expand_as(out)
        result.update(
            stored_prefix_values=stored_prefix.unique().tolist(),
            stored_readout_matching_fraction=(out == stored_readout)
            .float()
            .mean()
            .item(),
        )
    return result


def error_metrics(actual, reference):
    actual, reference = actual.float(), reference.float()
    diff = actual - reference
    return {
        "finite": bool(torch.isfinite(actual).all() & torch.isfinite(reference).all()),
        "matching_fraction": (actual == reference).float().mean().item(),
        "max_abs_error": diff.abs().max().item(),
        "relative_rms": (
            diff.square().mean().sqrt()
            / reference.square().mean().sqrt().clamp_min(1e-20)
        ).item(),
    }


def cache_metrics(actual, unrounded, rounds):
    metrics = error_metrics(actual, unrounded.half())
    rounded = unrounded.half()
    lo = torch.where(
        rounded.float() > unrounded,
        torch.nextafter(rounded, torch.full_like(rounded, -torch.inf)),
        rounded,
    )
    hi = torch.where(
        rounded.float() < unrounded,
        torch.nextafter(rounded, torch.full_like(rounded, torch.inf)),
        rounded,
    )
    allowed = ((actual >= lo) & (actual <= hi)) if rounds else actual == rounded
    metrics["outside_rounding_bounds_fraction"] = (~allowed).float().mean().item()
    return metrics


def recurrence_readout(state, x, dt, a, b, c):
    """Independent FP32 recurrence and contraction; dt is already processed."""
    state = state.float().clone()
    heads_per_group = state.shape[0] // b.shape[1]
    outputs = []
    for t in range(x.shape[0]):
        bt = b[t].float().repeat_interleave(heads_per_group, dim=0)
        ct = c[t].float().repeat_interleave(heads_per_group, dim=0)
        state = (
            state * torch.exp(a * dt[t])[:, None, None]
            + x[t].float()[:, :, None] * (dt[t, :, None] * bt)[:, None, :]
        )
        outputs.append((state * ct[:, None, :]).sum(-1))
    return torch.stack(outputs)


def run_trajectory(algorithm, tokens, rounds, steps, window, prefix_every):
    """Separate dtype-path errors from accumulated cache quantization.

    The matched FP32 oracle starts each call with precisely the FP16 cache's
    stored values. The independent FP32 trajectory keeps its full precision.
    Host bookkeeping accepts 1..T tokens and discards the remaining drafts.
    """
    from flashinfer.mamba.checkpointing_ssu import (
        allocate_checkpointing_ssu_scratch,
        checkpointing_ssu,
    )

    torch.manual_seed(20261011)
    device = "cuda"
    heads, dim, dstate = 8, 64, 128
    ring_len = window + tokens
    low = (torch.randn(2, heads, dim, dstate, device=device) * 0.1).half()
    full = low.float()
    xc = torch.zeros(2, heads, ring_len, dim, dtype=torch.bfloat16, device=device)
    bc = torch.zeros(2, 1, ring_len, dstate, dtype=torch.bfloat16, device=device)
    dc = torch.zeros(2, heads, ring_len, device=device)
    a = -torch.logspace(-3, 1, heads, device=device)
    bias = torch.full((heads, 1), -4.0, dtype=torch.bfloat16, device=device)
    scratch = allocate_checkpointing_ssu_scratch(
        1, heads, tokens, window, torch.bfloat16, device
    )
    index = torch.tensor([1], dtype=torch.int32, device=device)
    start = committed = 0
    summary = {}

    def record(name, metrics, step):
        bucket = summary.setdefault(name, {"calls": 0, "finite": True})
        bucket["calls"] += 1
        bucket["finite"] &= metrics["finite"]
        for key, value in metrics.items():
            if key == "finite":
                continue
            reducer = min if key == "matching_fraction" else max
            bucket[key] = reducer(bucket.get(key, value), value)
        if not metrics["finite"] or metrics.get("outside_rounding_bounds_fraction", 0):
            bucket.setdefault("first_bad_step", step)

    for step in range(steps):
        x = torch.randn(1, tokens, heads, dim, device=device).bfloat16()
        dt = torch.randn(1, tokens, heads, 1, device=device).bfloat16().expand_as(x)
        b = torch.randn(1, tokens, 1, dstate, device=device).bfloat16()
        c = torch.randn_like(b)
        seed = torch.tensor([12345 + step], dtype=torch.int64, device=device)
        starts = torch.tensor([0, start], dtype=torch.int32, device=device)
        counts = torch.tensor([0, committed], dtype=torch.int32, device=device)
        checkpoint = committed + tokens > window
        history = (torch.arange(committed, device=device) + start) % ring_len
        ref_x = torch.cat((xc[1, :, history].transpose(0, 1), x[0]))
        ref_b = torch.cat((bc[1, :, history].transpose(0, 1), b[0]))
        ref_c = torch.cat((torch.zeros_like(ref_b[:committed]), c[0]))
        ref_dt = torch.cat(
            (
                dc[1, :, history].T,
                torch.nn.functional.softplus(
                    dt[0, :, :, 0].float() + bias[:, 0].float()
                ),
            )
        )
        exact = recurrence_readout(low[1], ref_x, ref_dt, a, ref_b, ref_c)
        matched = low.float()
        outputs = []
        for state in (matched, full, low):
            out = torch.empty_like(x)
            # Independent rings keep one invocation's writes out of the oracle.
            rings = (
                (xc, bc, dc) if state is low else (xc.clone(), bc.clone(), dc.clone())
            )
            checkpointing_ssu(
                state,
                *rings,
                starts,
                counts,
                x,
                dt,
                a.as_strided((heads, dim, dstate), (1, 0, 0)),
                b,
                c,
                out,
                dt_bias=bias.expand(heads, dim),
                dt_softplus=True,
                state_batch_indices=index,
                rand_seed=seed if state is low and rounds else None,
                philox_rounds=rounds if state is low else 0,
                cb_scaled=scratch[0],
                cumAdt_vec=scratch[1],
                cb_old=scratch[2],
                algorithm=algorithm,
                d_split=1,
            )
            outputs.append(out)
        phase = "checkpoint" if checkpoint else "live"
        record(
            f"{phase}_matched_input_output", error_metrics(outputs[2], outputs[0]), step
        )
        record(
            f"{phase}_full_fp32_trajectory", error_metrics(outputs[2], outputs[1]), step
        )
        record(
            f"{phase}_same_cache_fp32_recurrence",
            error_metrics(outputs[2][0], exact[-tokens:].bfloat16()),
            step,
        )
        record(
            f"{phase}_cache",
            cache_metrics(low[1], matched[1], rounds if checkpoint else 0),
            step,
        )
        accepted = step % tokens + 1
        if checkpoint:
            start = (start + committed) % ring_len
            committed = 0
        committed += accepted

        if prefix_every and (step + 1) % prefix_every == 0:
            matched = low.float()
            for state in (matched, full, low):
                materialize(
                    state,
                    xc,
                    bc,
                    dc,
                    a,
                    seed if state is low and rounds else None,
                    rounds if state is low else 0,
                    count=committed,
                    start=start,
                    window=window,
                    src=1,
                    dst=0,
                )
            record("prefix_cache", cache_metrics(low[0], matched[0], rounds), step)
            low[1].copy_(low[0])
            full[1].copy_(full[0])
            start = committed = 0
    return {
        "mode": "trajectory",
        "algorithm": algorithm,
        "philox_rounds": rounds,
        "steps": steps,
        "window": window,
        "tokens": tokens,
        "prefix_every": prefix_every,
        "metrics": summary,
    }


def loaded_replay_libraries():
    """Report actual loaded Linux DSOs, including any prebuilt provider path."""
    try:
        lines = Path("/proc/self/maps").read_text().splitlines()
    except OSError:
        return None
    return sorted(
        {
            line.split(maxsplit=5)[-1]
            for line in lines
            if ".so" in line
            and ("checkpointing_ssu_" in line or "replayssm_materialize_" in line)
        }
    )


def replay_source_info():
    """Identify JIT headers; markers alone do not establish binary freshness."""
    from flashinfer.jit import env as jit_env

    root = jit_env.FLASHINFER_INCLUDE_DIR / "flashinfer" / "mamba"
    markers = {
        "kernel_checkpointing_ssu.cuh": ("WRITE_READOUT", "smem.checkpoint_state"),
        "kernel_checkpointing_ssu_main.cuh": (
            "output_state_ptr<state_t, CHECKPOINT>",
            "smem.checkpoint_state",
        ),
        "kernel_checkpointing_ssu_common.cuh": (
            "struct CheckpointReadoutStorage",
            "return smem.checkpoint_state",
        ),
    }
    headers = []
    for name, required in markers.items():
        path = root / name
        try:
            data = path.read_bytes()
        except OSError as exc:
            headers.append({"path": str(path.resolve()), "error": str(exc)})
            continue
        headers.append(
            {
                "path": str(path.resolve()),
                "sha256": hashlib.sha256(data).hexdigest(),
                "patch_markers_present": all(
                    marker.encode() in data for marker in required
                ),
            }
        )
    return {
        "python_executable": sys.executable,
        "jit_workspace": str(jit_env.FLASHINFER_WORKSPACE_DIR),
        "headers": headers,
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--reference-only", action="store_true")
    parser.add_argument(
        "--verify-patch",
        action="store_true",
        help="Require patched headers and exact checkpoint output before trajectories",
    )
    parser.add_argument("--tokens", type=int, choices=range(1, 9), default=6)
    parser.add_argument(
        "--steps",
        type=int,
        default=0,
        help="Run a randomized trajectory instead of boundary cases",
    )
    parser.add_argument("--window", type=int, choices=[6, 8, 16], default=16)
    parser.add_argument(
        "--prefix-every",
        type=int,
        default=0,
        help="Materialize and resume every N trajectory steps; 0 disables",
    )
    parser.add_argument(
        "--algorithm", choices=["monolith", "two-kernel", "both"], default="both"
    )
    args = parser.parse_args()
    if args.steps < 0 or args.prefix_every < 0 or args.tokens > args.window:
        parser.error("steps/prefix-every must be nonnegative and tokens <= window")
    exact = torch.tensor(1 + 17 / 4096, dtype=torch.float32)
    print(
        json.dumps(
            {
                "exact_state": exact.item(),
                "direct_bf16": exact.bfloat16().item(),
                "fp16_intermediate": exact.half().item(),
                "via_fp16_bf16": exact.half().bfloat16().item(),
                "reference": "unrounded FP32 recurrence, before cache quantization",
                "cached_dot_fp32": 1 / 1024,
                "cached_dot_bf16_operands": (
                    torch.tensor([1 + 1 / 1024, 1]).bfloat16().float()
                    * torch.tensor([1, -1])
                )
                .sum()
                .item(),
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
    print(json.dumps({"flashinfer_path": flashinfer.__file__}), flush=True)
    source_info = replay_source_info()
    print(json.dumps({"replay_source_info": source_info}), flush=True)
    if args.verify_patch and not all(
        entry.get("patch_markers_present", False) for entry in source_info["headers"]
    ):
        parser.error(
            "The imported FlashInfer JIT headers are missing the readout patch"
        )
    algorithms = (
        ["monolith", "two-kernel"] if args.algorithm == "both" else [args.algorithm]
    )
    if args.verify_patch:
        for algorithm in algorithms:
            for rounds in (0, 5):
                result = run_case(
                    torch.float16,
                    rounds,
                    "checkpoint",
                    algorithm,
                    args.tokens,
                    checkpoint_window=args.window,
                )
                print(json.dumps({"patch_smoke_test": result}), flush=True)
                if not result["finite"] or result["matching_fraction"] != 1.0:
                    print(
                        json.dumps(
                            {"loaded_replay_libraries": loaded_replay_libraries()}
                        ),
                        flush=True,
                    )
                    parser.error(
                        "Patched checkpoint behavior was not observed; verify the "
                        "build before interpreting trajectory or model accuracy results"
                    )
    for algorithm in algorithms:
        if args.steps:
            for rounds in (0, 5):
                print(
                    json.dumps(
                        run_trajectory(
                            algorithm,
                            args.tokens,
                            rounds,
                            args.steps,
                            args.window,
                            args.prefix_every,
                        )
                    ),
                    flush=True,
                )
            continue
        for dtype, rounds in (
            (torch.float32, 0),
            (torch.float16, 0),
            (torch.float16, 5),
        ):
            for mode in ("live", "checkpoint", "prefix-resume", "cache-readout"):
                result = run_case(dtype, rounds, mode, algorithm, args.tokens)
                print(json.dumps(result), flush=True)
    print(
        json.dumps({"loaded_replay_libraries": loaded_replay_libraries()}), flush=True
    )


if __name__ == "__main__":
    main()
