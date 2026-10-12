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

Use --compare-standard --steps 128 to compare against ordinary FlashInfer SSU,
using vLLM's paged multi-token API. This runs FP32, FP16 RN, and FP16 SR controls
on identical synthetic inputs (one request, default TP4 shape: 32 heads/2 groups).
It does not exercise vLLM scheduling, convolution, model weights, or PD transport.
Both kernels are also compared to a shared sequential PyTorch recurrence that
never quantizes its state. --reference-dtype float64 checks oracle precision.
--per-step prints the error trajectory; --dump-dir saves each case's worst
ReplaySSM accepted-output error with all inputs needed to reproduce that call.

Add --precision-ablation to emulate all eight combinations of retaining/removing
BF16 rounding of state readout, scaled CB, and reconstruction operands. Each call
uses the real kernel's pre-call cache/history; these are local counterfactuals,
not independent trajectories or replacement serving kernels. Native-emulation
residuals against the real kernel must be checked before attributing its error.
Use --ablate-dump FILE.pt to run the same analysis on a saved call without CUDA.
"""

import argparse
import hashlib
import itertools
import json
import sys
import tempfile
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
        heads_per_group=state.size(1) // b_cache.size(1),
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
    return recurrence_scan(state, x, dt, a, b, c)[0]


def recurrence_scan(state, x, dt, a, b, c, d=None, *, dtype=torch.float32):
    """Sequential PyTorch oracle with no cache quantization or matrix multiplies.

    Inputs are token-major; dt is already processed. Return every output and
    post-token state so the caller can commit only the accepted prefix. This is
    a numerical reference, not a bitwise oracle for CUDA fast math/FMA ordering.
    """
    state = state.to(dtype).clone()
    x, dt, a, b, c = (t.to(dtype) for t in (x, dt, a, b, c))
    heads_per_group = state.shape[0] // b.shape[1]
    outputs, states = [], []
    for t in range(x.shape[0]):
        bt = b[t].repeat_interleave(heads_per_group, dim=0)
        ct = c[t].repeat_interleave(heads_per_group, dim=0)
        state = (
            state * torch.exp(a * dt[t])[:, None, None]
            + x[t, :, :, None] * (dt[t, :, None] * bt)[:, None, :]
        )
        out = (state * ct[:, None, :]).sum(-1)
        if d is not None:
            out = out + x[t] * d.to(dtype)
        outputs.append(out)
        states.append(state)
    return torch.stack(outputs), torch.stack(states)


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


class ComparisonStats:
    """Aggregate errors over elements, retaining the worst call and its step."""

    def __init__(self):
        self.calls = self.elements = self.matches = 0
        self.error_squared = self.reference_squared = 0.0
        self.max_abs_error = self.worst_relative_rms = 0.0
        self.worst_step = None
        self.first_nonfinite_step = None

    def update(self, actual, reference, step):
        assert actual.shape == reference.shape
        # Preserve an optional FP64 reference when computing differences.
        actual, reference = actual.double(), reference.double()
        self.calls += 1
        self.elements += actual.numel()
        self.matches += (actual == reference).sum().item()
        if not bool(actual.isfinite().all() & reference.isfinite().all()):
            if self.first_nonfinite_step is None:
                self.first_nonfinite_step = step
            return {"finite": False, "relative_rms": None}
        diff = actual - reference
        error_squared = diff.double().square().sum().item()
        reference_squared = reference.double().square().sum().item()
        relative_rms = (error_squared / max(reference_squared, 1e-40)) ** 0.5
        self.error_squared += error_squared
        self.reference_squared += reference_squared
        self.max_abs_error = max(self.max_abs_error, diff.abs().max().item())
        if self.worst_step is None or relative_rms > self.worst_relative_rms:
            self.worst_relative_rms, self.worst_step = relative_rms, step
        return {
            "finite": True,
            "relative_rms": relative_rms,
            "max_abs_error": diff.abs().max().item(),
        }

    def result(self):
        finite = self.first_nonfinite_step is None
        return {
            "calls": self.calls,
            "elements": self.elements,
            "finite": finite,
            "matching_fraction": self.matches / self.elements,
            "max_abs_error": self.max_abs_error if finite else None,
            "relative_rms": (
                (self.error_squared / max(self.reference_squared, 1e-40)) ** 0.5
                if finite
                else None
            ),
            "worst_call_relative_rms": self.worst_relative_rms if finite else None,
            "worst_step": self.worst_step,
            "first_nonfinite_step": self.first_nonfinite_step,
        }


def replay_precision_variants(state, history, x, dt, a, b, c, d, *, checkpoint):
    """Emulate ReplaySSM factorization with explicit optional BF16 conversions.

    All arithmetic uses FP32 elementwise operations/reductions, never matmul
    (so TF32 settings cannot silently affect the controls). CUDA fast math,
    MMA accumulation order, and fused operations are not reproduced bitwise.
    History is chronological, token-major (x, processed dt, B). Outputs remain
    FP32 here; the caller applies the common final BF16 output conversion.
    Reconstructed states precede cache quantization, including stochastic rounding.
    """
    state = state.float()
    x, dt, a, b, c, d = (v.float() for v in (x, dt, a, b, c, d))
    ox, odt, ob = (v.float() for v in history)
    hpg = state.shape[0] // b.shape[1]
    b, c, ob = (v.repeat_interleave(hpg, dim=1) for v in (b, c, ob))
    cum = (dt * a).cumsum(0)
    old_cum = (odt * a).cumsum(0)
    total = old_cum[-1] if len(ox) else torch.zeros_like(a)
    causal = torch.ones(len(x), len(x), device=x.device, dtype=torch.bool).tril()
    delta = cum[:, None] - cum[None, :]
    # Mask before exp too: unused future-token terms can otherwise overflow.
    decay = torch.exp(delta.masked_fill(~causal[:, :, None], -torch.inf))
    cb = (c[:, None] * b[None, :]).sum(-1) * decay * dt[None, :]
    cb_old = (
        (c[:, None] * ob[None, :]).sum(-1)
        * torch.exp(cum[:, None] + total - old_cum[None, :])
        * odt[None, :]
    )
    scaled_b = ob * (torch.exp(total - old_cum) * odt)[:, :, None]

    def rounded(value, enabled):
        return value.bfloat16().float() if enabled else value

    def reconstruct(round_operand):
        return state * torch.exp(total)[:, None, None] + (
            ox[:, :, :, None] * rounded(scaled_b, round_operand)[:, :, None, :]
        ).sum(0)

    reconstructed = {flag: reconstruct(flag) for flag in (False, True)}
    for readout, cb_round, reconstruction in itertools.product((True, False), repeat=3):
        removed = [
            name
            for name, enabled in (
                ("readout", readout),
                ("cb", cb_round),
                ("reconstruction", reconstruction),
            )
            if not enabled
        ]
        name = "without_" + "_".join(removed) if removed else "native"
        base = reconstructed[reconstruction] if checkpoint else state
        y = (c[:, :, None, :] * rounded(base, readout)[None]).sum(-1)
        beta = torch.exp(cum)
        if not checkpoint:
            beta = beta * torch.exp(total)
        y = y * beta[:, :, None]
        y = y + (rounded(cb, cb_round)[:, :, :, None] * x[None]).sum(1)
        if not checkpoint:
            y = y + (rounded(cb_old, cb_round)[:, :, :, None] * ox[None]).sum(1)
        yield name, y + d * x, reconstructed[reconstruction]


def replay_history(snapshot, start, count):
    """Extract only committed tokens, including a wrapped ring buffer."""
    xc, dc, bc = (snapshot[k] for k in ("x_cache", "dt_cache", "b_cache"))
    indices = (torch.arange(count, device=xc.device) + start) % xc.shape[2]
    return tuple(t[0].index_select(1, indices).transpose(0, 1) for t in (xc, dc, bc))


class PrecisionAblationStats:
    """Condition every ablation on the same actual pre-call cache and history."""

    def __init__(self):
        self.stats = {}

    def update(self, snapshot, actual, *, start, count, checkpoint, accepted, step):
        history = replay_history(snapshot, start, count)
        state = snapshot["replay_state"][0]
        x, b, c = (snapshot[k][0] for k in ("x", "b", "c"))
        a, d = (snapshot[k] for k in ("a", "d"))
        raw_dt = snapshot["dt"][0, :, :, 0]
        bias = snapshot["dt_bias"][:, 0]
        dt = torch.nn.functional.softplus(raw_dt.float() + bias.float())
        ref_state = state.double()
        if count:
            _, states = recurrence_scan(
                ref_state,
                history[0],
                history[1],
                a,
                history[2],
                torch.zeros_like(history[2]),
                dtype=torch.float64,
            )
            ref_state = states[-1]
        reference, _ = recurrence_scan(
            ref_state,
            x,
            torch.nn.functional.softplus(raw_dt.double() + bias.double()),
            a,
            b,
            c,
            d,
            dtype=torch.float64,
        )
        phase = "checkpoint" if checkpoint else "live" if count else "empty_history"
        stats = self.stats.setdefault(phase, {})

        def record(name, value, ref):
            stats.setdefault(name, ComparisonStats()).update(value, ref, step)

        record("kernel_vs_local_reference", actual[:accepted], reference[:accepted])
        record(
            "output_rounding_floor",
            reference[:accepted].bfloat16(),
            reference[:accepted],
        )
        for name, out, reconstructed in replay_precision_variants(
            state, history, x, dt, a, b, c, d, checkpoint=checkpoint
        ):
            out = out[:accepted].bfloat16()
            record(f"{name}/output_vs_local_reference", out, reference[:accepted])
            if name in ("native", "without_reconstruction") and count:
                record(
                    f"{name}/reconstruction_vs_local_reference",
                    reconstructed,
                    ref_state,
                )
            if name == "native":
                record("native_emulation_vs_kernel", out, actual[:accepted])

    def result(self):
        return {
            phase: {name: stat.result() for name, stat in metrics.items()}
            for phase, metrics in self.stats.items()
        }


def ablate_saved_call(path):
    """Analyze a saved pre-call snapshot on CPU without importing FlashInfer."""
    dump = torch.load(path, map_location="cpu", weights_only=True)
    if dump.get("format_version") != 1:
        raise ValueError("Expected a format_version=1 --compare-standard dump")
    stats = PrecisionAblationStats()
    stats.update(
        dump["tensors"],
        dump["tensors"]["replay_out"][0],
        start=dump["ring_start"],
        count=dump["committed"],
        checkpoint=dump["checkpoint"],
        accepted=dump["accepted"],
        step=dump["step"],
    )
    return {
        "mode": "precision-ablation-dump",
        "file": str(path),
        "case": dump["case"],
        "step": dump["step"],
        "scope": "Local FP32 arithmetic emulation; not a modified CUDA kernel",
        "metrics": stats.result(),
    }


def run_standard_comparison(args, state_dtype, rounds, algorithm, device="cuda"):
    """Compare real SSU kernels with shared inputs but independent live state.

    Ordinary SSU writes all T speculative states, then the next call reads the
    accepted one. ReplaySSM commits only those accepted inputs to its ring.
    Neither arm consumes rejected states. The same-state probe has no history;
    it isolates output arithmetic without inherited state divergence. SR uses
    equal seed values, but different kernels need not map Philox draws equally.
    """
    from flashinfer.mamba import selective_state_update
    from flashinfer.mamba.checkpointing_ssu import (
        allocate_checkpointing_ssu_scratch,
        checkpointing_ssu,
    )

    tokens, window = args.tokens, args.window
    heads, groups, dim, dstate = args.heads, args.groups, 64, 128
    ring_len = window + tokens
    rng = torch.Generator(device=device).manual_seed(args.seed)

    def randn(*shape):
        return torch.randn(*shape, generator=rng, device=device)

    # All controls start with identical values, including the FP32 control.
    initial = (randn(heads, dim, dstate) * 0.1).half().to(state_dtype)
    reference_dtype = getattr(torch, args.reference_dtype)
    reference_state = initial.to(reference_dtype).clone()
    standard = initial.expand(tokens, -1, -1, -1).clone()
    replay = initial.expand(2, -1, -1, -1).clone()
    xc = torch.zeros(2, heads, ring_len, dim, dtype=torch.bfloat16, device=device)
    bc = torch.zeros(2, groups, ring_len, dstate, dtype=torch.bfloat16, device=device)
    dc = torch.zeros(2, heads, ring_len, device=device)
    # Probe storage must not alias either trajectory's state or ring.
    probe = torch.empty_like(replay)
    probe_rings = tuple(torch.zeros_like(t) for t in (xc, bc, dc))
    zeros = torch.zeros(2, dtype=torch.int32, device=device)
    index = torch.tensor([0], dtype=torch.int32, device=device)
    pages = torch.arange(tokens, dtype=torch.int32, device=device)[None]
    cu_seqlens = torch.tensor([0, tokens], dtype=torch.int32, device=device)
    a = -torch.logspace(-3, 1, heads, device=device)
    a_expanded = a.as_strided((heads, dim, dstate), (1, 0, 0))
    bias = torch.full((heads, 1), -4.0, dtype=torch.bfloat16, device=device)
    d = randn(heads, 1).bfloat16().expand(heads, dim)
    scratch = allocate_checkpointing_ssu_scratch(
        1, heads, tokens, window, torch.bfloat16, device
    )
    stats = {}
    ablation = PrecisionAblationStats() if args.precision_ablation else None
    start = committed = accepted_total = checkpoints = prefixes = 0
    previous_accepted = 1
    worst_dump = None
    worst_dump_score = -1.0
    case = {
        "state_dtype": str(state_dtype),
        "philox_rounds": rounds,
        "algorithm": algorithm,
        "reference_dtype": args.reference_dtype,
    }

    def record(name, actual, reference, step):
        step_metrics[name] = stats.setdefault(name, ComparisonStats()).update(
            actual, reference, step
        )

    for step in range(args.steps):
        step_metrics = {}
        x = randn(1, tokens, heads, dim).bfloat16()
        dt = randn(1, tokens, heads, 1).bfloat16().expand_as(x)
        b = randn(1, tokens, groups, dstate).bfloat16()
        c = randn(1, tokens, groups, dstate).bfloat16()
        seed = (
            torch.tensor([args.seed + step], dtype=torch.int64, device=device)
            if rounds
            else None
        )
        checkpoint = committed + tokens > window
        accepted = step % tokens + 1
        before = standard[previous_accepted - 1].clone()
        processed_dt = torch.nn.functional.softplus(
            dt[0, :, :, 0].to(reference_dtype) + bias[:, 0].to(reference_dtype)
        )
        reference_out, reference_states = recurrence_scan(
            reference_state,
            x[0],
            processed_dt,
            a,
            b[0],
            c[0],
            d,
            dtype=reference_dtype,
        )
        same_state_reference, _ = recurrence_scan(
            before,
            x[0],
            processed_dt,
            a,
            b[0],
            c[0],
            d,
            dtype=reference_dtype,
        )
        # Preserve pre-call inputs only when a reproducible case was requested.
        snapshot = None
        if args.dump_dir is not None or ablation is not None:
            snapshot = {
                "standard_state": standard.clone(),
                "replay_state": replay.clone(),
                "x_cache": xc.clone(),
                "b_cache": bc.clone(),
                "dt_cache": dc.clone(),
                "reference_state": reference_state,
                "x": x,
                "dt": dt,
                "a": a,
                "b": b,
                "c": c,
                "d": d,
                "dt_bias": bias,
            }
        ordinary_out = torch.empty_like(x)
        selective_state_update(
            standard,
            x[0],
            dt[0],
            a_expanded,
            b[0],
            c[0],
            D=d,
            dt_bias=bias.expand(heads, dim),
            dt_softplus=True,
            state_batch_indices=pages,
            dst_state_batch_indices=pages,
            num_accepted_tokens=torch.tensor(
                [previous_accepted], dtype=torch.int32, device=device
            ),
            cu_seqlens=cu_seqlens,
            cache_steps=tokens,
            out=ordinary_out[0],
            rand_seed=seed,
            philox_rounds=rounds,
            algorithm=args.standard_algorithm,
        )
        probe[0].copy_(before)
        outputs = []
        for state, rings, starts, counts in (
            (probe, probe_rings, zeros, zeros),
            (
                replay,
                (xc, bc, dc),
                torch.tensor([start, 0], dtype=torch.int32, device=device),
                torch.tensor([committed, 0], dtype=torch.int32, device=device),
            ),
        ):
            out = torch.empty_like(x)
            checkpointing_ssu(
                state,
                *rings,
                starts,
                counts,
                x,
                dt,
                a_expanded,
                b,
                c,
                out,
                D=d,
                dt_bias=bias.expand(heads, dim),
                dt_softplus=True,
                state_batch_indices=index,
                rand_seed=seed,
                philox_rounds=rounds,
                cb_scaled=scratch[0],
                cumAdt_vec=scratch[1],
                cb_old=scratch[2],
                algorithm=algorithm,
                d_split=1,
            )
            outputs.append(out)
        if ablation is not None:
            ablation.update(
                snapshot,
                outputs[1][0],
                start=start,
                count=committed,
                checkpoint=checkpoint,
                accepted=accepted,
                step=step,
            )
        record("same_state_empty_history_output", outputs[0], ordinary_out, step)
        # Two references answer different questions: actual local input state
        # isolates arithmetic; the shared trajectory also includes cache drift.
        for name, actual in (("standard", ordinary_out), ("replay", outputs[0])):
            record(
                f"{name}_same_state_vs_reference",
                actual[0],
                same_state_reference.to(actual.dtype),
                step,
            )
        for name, actual in (("standard", ordinary_out), ("replay", outputs[1])):
            record(
                f"{name}_accepted_vs_reference",
                actual[0, :accepted],
                reference_out[:accepted].to(actual.dtype),
                step,
            )
            record(
                f"{name}_accepted_vs_unrounded_reference",
                actual[0, :accepted],
                reference_out[:accepted],
                step,
            )
        record(
            "standard_accepted_cache_vs_reference",
            standard[accepted - 1],
            reference_states[accepted - 1],
            step,
        )
        phase = "checkpoint" if checkpoint else "live"
        record(f"trajectory_{phase}_output", outputs[1], ordinary_out, step)
        record(
            "trajectory_accepted_output",
            outputs[1][:, :accepted],
            ordinary_out[:, :accepted],
            step,
        )
        if checkpoint:
            # Both now describe the history BEFORE this verification call.
            record("checkpoint_cache_vs_standard_start", replay[0], before, step)
            record(
                "replay_checkpoint_cache_vs_reference",
                replay[0],
                reference_state,
                step,
            )
        if args.dump_dir is not None:
            score = step_metrics["replay_accepted_vs_unrounded_reference"]
            rank = score["relative_rms"] if score["finite"] else float("inf")
            if rank > worst_dump_score:
                worst_dump_score = rank
                snapshot.update(
                    standard_out=ordinary_out,
                    probe_out=outputs[0],
                    replay_out=outputs[1],
                    reference_out=reference_out,
                    same_state_reference_out=same_state_reference,
                    reference_states=reference_states,
                )
                worst_dump = {
                    "format_version": 1,
                    "case": case,
                    "config": {
                        k: str(v) if isinstance(v, Path) else v
                        for k, v in vars(args).items()
                    },
                    "step": step,
                    "accepted": accepted,
                    "previous_accepted": previous_accepted,
                    "ring_start": start,
                    "committed": committed,
                    "checkpoint": checkpoint,
                    "rand_seed": args.seed + step if rounds else None,
                    "metrics": dict(step_metrics),
                    "strides": {k: list(v.stride()) for k, v in snapshot.items()},
                    "tensors": {
                        k: v.detach().cpu().clone() for k, v in snapshot.items()
                    },
                }
        if checkpoint:
            start = (start + committed) % ring_len
            committed = 0
            checkpoints += 1
        committed += accepted
        accepted_total += accepted
        previous_accepted = accepted
        # Never carry rejected draft states into the independent oracle.
        reference_state = reference_states[accepted - 1].clone()
        if args.prefix_every and (step + 1) % args.prefix_every == 0:
            materialize(
                replay,
                xc,
                bc,
                dc,
                a,
                seed,
                rounds,
                count=committed,
                start=start,
                window=window,
                src=0,
                dst=1,
            )
            record(
                "prefix_cache_vs_standard_accepted",
                replay[1],
                standard[accepted - 1],
                step,
            )
            record(
                "replay_prefix_cache_vs_reference",
                replay[1],
                reference_state,
                step,
            )
            replay[0].copy_(replay[1])
            start = committed = 0
            prefixes += 1
        if args.per_step:
            print(
                json.dumps(
                    {
                        "mode": "comparison-step",
                        **case,
                        "step": step,
                        "accepted": accepted,
                        "checkpoint": checkpoint,
                        "metrics": step_metrics,
                    }
                ),
                flush=True,
            )
    dump_path = None
    if worst_dump is not None:
        dump_path = args.dump_dir / (
            f"{str(state_dtype).split('.')[-1]}-sr{rounds}-{algorithm}.pt"
        )
        flashinfer = sys.modules["flashinfer"]
        worst_dump["runtime"] = {
            "python_executable": sys.executable,
            "torch_version": str(torch.__version__),
            "flashinfer_version": getattr(flashinfer, "__version__", None),
            "flashinfer_path": getattr(flashinfer, "__file__", None),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "source_info": replay_source_info() if device == "cuda" else None,
            "loaded_libraries": loaded_replay_libraries(),
        }
        torch.save(worst_dump, dump_path)
    return {
        "mode": "compare-standard",
        "reference": "flashinfer.mamba.selective_state_update (paged varlen API)",
        "state_dtype": str(state_dtype),
        "philox_rounds": rounds,
        "algorithm": algorithm,
        "standard_algorithm": args.standard_algorithm,
        "independent_reference": f"PyTorch sequential {args.reference_dtype}",
        "seed": args.seed,
        "shape": {
            "batch": 1,
            "heads": heads,
            "groups": groups,
            "dim": dim,
            "dstate": dstate,
            "tokens": tokens,
            "window": window,
        },
        "steps": args.steps,
        "accepted_tokens": accepted_total,
        "checkpoint_calls": checkpoints,
        "prefix_materializations": prefixes,
        "prefix_every": args.prefix_every,
        "worst_case_dump": str(dump_path) if dump_path is not None else None,
        "metrics": {name: value.result() for name, value in stats.items()},
        "precision_ablation": ablation.result() if ablation is not None else None,
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
            and any(
                name in line
                for name in (
                    "checkpointing_ssu_",
                    "replayssm_materialize_",
                    "selective_state_update",
                )
            )
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
        "--precision-ablation",
        action="store_true",
        help="With --compare-standard, isolate three BF16 conversions using emulation",
    )
    parser.add_argument(
        "--ablate-dump",
        type=Path,
        help="Analyze a saved .pt call on CPU; no CUDA or FlashInfer required",
    )
    parser.add_argument(
        "--compare-standard",
        action="store_true",
        help="Compare actual ordinary SSU and ReplaySSM; defaults to 128 steps",
    )
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--groups", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20261011)
    parser.add_argument(
        "--reference-dtype",
        choices=["float32", "float64"],
        default="float32",
        help="Independent recurrence precision for --compare-standard",
    )
    parser.add_argument(
        "--per-step",
        action="store_true",
        help="Print per-call comparison metrics in addition to aggregate metrics",
    )
    parser.add_argument(
        "--dump-dir",
        type=Path,
        help="Save worst-case tensors under a new subdirectory of this directory",
    )
    parser.add_argument(
        "--standard-algorithm",
        choices=["auto", "simple", "vertical", "horizontal"],
        default="auto",
        help="Ordinary FlashInfer SSU algorithm for --compare-standard",
    )
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
    if args.ablate_dump is not None:
        if args.compare_standard or args.reference_only or args.verify_patch:
            parser.error("--ablate-dump is a standalone mode")
        result = ablate_saved_call(args.ablate_dump)
        print(json.dumps(result), flush=True)
        if not all(
            m["finite"] for phase in result["metrics"].values() for m in phase.values()
        ):
            parser.exit(1, "Non-finite values detected in precision ablation\n")
        return
    if args.precision_ablation and (not args.compare_standard or args.reference_only):
        parser.error("--precision-ablation requires --compare-standard")
    if args.steps < 0 or args.prefix_every < 0 or args.tokens > args.window:
        parser.error("steps/prefix-every must be nonnegative and tokens <= window")
    if args.heads <= 0 or args.groups <= 0 or args.heads % args.groups:
        parser.error("heads and groups must be positive, and groups must divide heads")
    if args.compare_standard and args.steps == 0:
        args.steps = 128
    if (args.per_step or args.dump_dir is not None) and not args.compare_standard:
        parser.error("--per-step and --dump-dir require --compare-standard")
    if not 0 <= args.seed < 2**63 - args.steps:
        parser.error("seed and seed + steps must fit a nonnegative int64")
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
    if args.compare_standard:
        if args.dump_dir is not None:
            args.dump_dir.mkdir(parents=True, exist_ok=True)
            args.dump_dir = Path(tempfile.mkdtemp(prefix="ssu-", dir=args.dump_dir))
            print(json.dumps({"dump_directory": str(args.dump_dir)}), flush=True)
        print(
            json.dumps(
                {
                    "comparison_notes": {
                        "same_state_empty_history_output": (
                            "Same cached values and new inputs; no replay history. "
                            "Isolates output arithmetic without accumulated drift."
                        ),
                        "trajectory": (
                            "Independent caches, identical inputs and acceptance "
                            "schedule (1..tokens cyclically). Includes each "
                            "algorithm's native cache-rounding frequency."
                        ),
                        "independent_reference": (
                            f"Sequential PyTorch {args.reference_dtype}, "
                            "shared initial state and inputs, only accepted "
                            "tokens committed; never reset or cache-quantized. "
                            "No FlashInfer kernels or tensor-core matmuls."
                        ),
                        "reference_metrics": (
                            "*_same_state_vs_reference isolates arithmetic from "
                            "the actual ordinary-SSU starting cache. "
                            "*_accepted_vs_reference uses the shared trajectory "
                            "rounded to BF16; *_vs_unrounded_reference retains "
                            "reference precision, including output-rounding error. "
                            "Cache comparisons use unrounded reference states."
                        ),
                        "stochastic_rounding": (
                            "Same seed values do not imply the same Philox draws "
                            "or bitwise cache parity across different kernels."
                        ),
                        "scope": (
                            "Synthetic single-request SSM comparison, not model "
                            "accuracy, scheduling, convolution, or PD transport."
                        ),
                        "precision_ablation": (
                            "FP32 PyTorch factorization, with all eight combinations "
                            "of BF16 readout/CB/reconstruction rounding. CB includes "
                            "both current and historical tokens. Each arm starts "
                            "from identical real pre-call state/history; inherited "
                            "drift and actual prior RN/SR cache values are held fixed. "
                            "Reference is a local FP64 sequential recurrence. "
                            "Output metrics include final BF16 rounding; compare "
                            "output_rounding_floor. Check native_emulation_vs_kernel "
                            "before attributing error: CUDA fast math, MMA reduction "
                            "order and FMA are not reproduced bitwise. Reconstruction "
                            "affects current output only at checkpoints; its state "
                            "metric is BEFORE cache quantization. This does not "
                            "simulate improved future trajectories or SR draws."
                        )
                        if args.precision_ablation
                        else None,
                    }
                }
            ),
            flush=True,
        )
        finite = True
        for algorithm in algorithms:
            for dtype, rounds in (
                (torch.float32, 0),
                (torch.float16, 0),
                (torch.float16, 5),
            ):
                result = run_standard_comparison(args, dtype, rounds, algorithm)
                finite &= all(m["finite"] for m in result["metrics"].values())
                if result["precision_ablation"] is not None:
                    finite &= all(
                        m["finite"]
                        for phase in result["precision_ablation"].values()
                        for m in phase.values()
                    )
                print(json.dumps(result), flush=True)
        print(json.dumps({"loaded_libraries": loaded_replay_libraries()}), flush=True)
        if not finite:
            parser.exit(1, "Non-finite values detected in the SSU comparison\n")
        return
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
