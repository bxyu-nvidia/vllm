# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare active Mamba2 prefill kernels against saved original source files.

Save causal_conv1d.py, ssd_chunk_state.py, and ssd_chunk_scan.py from the baseline
revision into a directory, preserving their names, then run from the repo root:

    .venv/bin/python benchmarks/kernels/mamba/bench_prefill.py \
        --baseline-dir /tmp/before

Uses Nemotron TP1 dimensions, BF16 activations, FP32 SSM state, align-mode
convolution caching, and precomputed serving metadata. Checks exact outputs and
cache contents before alternating baseline/candidate CUPTI rounds. Compilation
and metadata construction are excluded. Reports summed GPU kernel time with
CUDA graphs and cold L2; this is not a model-serving benchmark.
"""

import argparse
import importlib.util
import itertools
import json
import statistics
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

from vllm.model_executor.layers.mamba.ops import causal_conv1d, ssd_combined
from vllm.platforms import current_platform
from vllm.v1.attention.backends.mamba2_attn import compute_varlen_chunk_metadata
from vllm.v1.attention.backends.utils import compute_causal_conv1d_metadata


def load_baseline(name, path):
    module_name = f"vllm.model_executor.layers.mamba.ops._baseline_{name}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def make_inputs(lengths):
    torch.manual_seed(179)
    total = sum(lengths)
    heads, headdim, dstate, groups, chunk = 128, 64, 128, 8, 128
    dim = heads * headdim + 2 * groups * dstate
    x = torch.randn(total, dim + 32, device="cuda", dtype=torch.bfloat16)
    x = x[:, :dim].T
    weight = torch.randn(dim, 4, device="cuda", dtype=x.dtype)
    bias = torch.randn(dim, device="cuda", dtype=x.dtype)
    cu_cpu = torch.tensor([0] + list(itertools.accumulate(lengths)))
    cu = cu_cpu.to(device="cuda", dtype=torch.int32)
    nums, batch, offsets = compute_causal_conv1d_metadata(cu_cpu, device=x.device)
    metadata = SimpleNamespace(
        nums_dict=nums, batch_ptr=batch, token_chunk_offset_ptr=offsets
    )
    last = [(n + chunk - 1) // chunk for n in lengths]
    pages = max(last) + 1
    indices = torch.arange(
        1, len(lengths) * pages + 1, device="cuda", dtype=torch.int32
    ).view(len(lengths), pages)
    cache = torch.randn(indices.numel() + 1, dim, 3, device="cuda", dtype=x.dtype)
    conv = dict(
        query_start_loc=cu,
        cache_indices=indices,
        has_initial_state=torch.arange(len(lengths), device="cuda") % 2 == 0,
        activation="silu",
        metadata=metadata,
        block_idx_first_scheduled_token=torch.ones_like(cu[:-1]),
        block_idx_last_scheduled_token=torch.tensor(last, device="cuda"),
        initial_state_idx=torch.zeros_like(cu[:-1]),
        num_computed_tokens=torch.zeros_like(cu[:-1]),
        block_size_to_align=chunk,
    )
    chunks, last_indices, ids = compute_varlen_chunk_metadata(cu, chunk)
    ssd = dict(
        dt=torch.randn(total, heads, device="cuda") - 4,
        A=-torch.rand(heads, device="cuda") - 1,
        chunk_size=chunk,
        cu_seqlens=cu,
        cu_chunk_seqlens=chunks,
        last_chunk_indices=last_indices,
        seq_idx=ids,
        D=torch.randn(heads, device="cuda", dtype=x.dtype),
        initial_states=torch.randn(
            len(lengths), heads, headdim, dstate, device="cuda", dtype=torch.float32
        ),
        dt_softplus=True,
        state_dtype=torch.float32,
    )
    return x, weight, bias, cache, conv, ssd


def main():
    from flashinfer.testing import bench_gpu_time_with_cupti

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument(
        "--lengths", type=int, nargs="+", default=[1024, 2048, 4096, 9792]
    )
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=30)
    args = parser.parse_args()
    if min(args.lengths) <= 0:
        parser.error("Sequence lengths must be positive")
    torch.set_num_threads(2)
    old_conv = load_baseline("causal_conv1d", args.baseline_dir / "causal_conv1d.py")
    old_state = load_baseline(
        "ssd_chunk_state", args.baseline_dir / "ssd_chunk_state.py"
    )
    old_scan = load_baseline("ssd_chunk_scan", args.baseline_dir / "ssd_chunk_scan.py")
    old_ssd = load_baseline("ssd_combined", ssd_combined.__file__)
    old_ssd._chunk_state_fwd = old_state._chunk_state_fwd
    old_ssd._chunk_scan_fwd = old_scan._chunk_scan_fwd
    x, weight, bias, cache, conv, ssd = make_inputs(args.lengths)
    arms = []
    for conv_module, ssd_module in [
        (old_conv, old_ssd),
        (causal_conv1d, ssd_combined),
    ]:
        state = cache.clone()
        out = torch.empty(x.shape[1], 128, 64, device=x.device, dtype=x.dtype)

        def run_conv(module=conv_module, state=state):
            return module.causal_conv1d_fn(x, weight, bias, conv_states=state, **conv)

        def run_ssd(projected, module=ssd_module, out=out):
            projected = projected.T
            return module.mamba_chunk_scan_combined_varlen(
                x=projected[:, :8192].view(-1, 128, 64),
                B=projected[:, 8192:9216].view(-1, 8, 128),
                C=projected[:, 9216:].view(-1, 8, 128),
                out=out,
                **ssd,
            )

        def run_pipeline(run_conv=run_conv, run_ssd=run_ssd):
            return run_ssd(run_conv())

        arms.append((run_conv, run_pipeline, out, state))
    references = []
    for conv_fn, pipeline, out, state in arms:
        conv_out = conv_fn().clone()
        final = pipeline().clone()
        references.append((conv_out, out.clone(), final, state.clone()))
    torch.accelerator.synchronize()
    for baseline, candidate in zip(*references):
        torch.testing.assert_close(baseline, candidate, atol=0, rtol=0)
    result = dict(
        device=current_platform.get_device_name(),
        lengths=args.lengths,
        activation_dtype="bfloat16",
        state_dtype="float32",
        exact_outputs_and_states=True,
    )
    for index, label in [(0, "convolution"), (1, "convolution_and_ssd")]:
        timings = [[], []]
        for round_idx in range(args.rounds):
            for arm in [0, 1] if round_idx % 2 == 0 else [1, 0]:
                samples = bench_gpu_time_with_cupti(
                    arms[arm][index],
                    dry_run_iters=3,
                    repeat_iters=args.repetitions,
                    use_cuda_graph=True,
                    cold_l2_cache=True,
                    aggregate_op=sum,
                )
                timings[arm].append(statistics.median(samples) * 1000)
        result[label] = dict(
            baseline_us=statistics.median(timings[0]),
            candidate_us=statistics.median(timings[1]),
            rounds_us=timings,
        )
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
