# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure independent DWDP peer pulls and optional BF16 Prims MoE overlap.

Default shapes are Nemotron 3.5 Super's non-gated latent MoE projections.
Inputs are synthetic; this is not an end-to-end model or serving benchmark.
CUDA events include eager weight-manager dispatch and stream waits, excluding
startup, allocation, compilation, tuning, and the per-sample host rendezvous.
Prims compute is graph replay by default; --eager-compute includes its Python
dispatch overhead. Peer copies remain outside these compute-only graphs.
Use --capture-transport to capture the entire pipeline, including the serial
copy fallback; its byte counts describe the captured copy schedule.

Example:
    .venv/bin/python -m benchmarks.kernels.benchmark_dwdp --world-size 4 \
        --compute --tokens 1024 16960 33920 --output /tmp/dwdp.json

"""

import argparse
import json
import os
import statistics
import subprocess
import tempfile
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from vllm.model_executor.offloader.dwdp import DWDPWeightManager, expert_range


def _barrier(store, rank, size, key):
    store.set(f"{key}/{rank}", b"1")
    store.wait([f"{key}/{r}" for r in range(size)])


def _worker(rank, args, path):
    torch.set_num_threads(1)
    torch.accelerator.set_device_index(rank)
    device = torch.device("cuda", rank)
    store = dist.FileStore(path, args.world_size)
    store.set_timeout(timedelta(seconds=1800))
    manager = DWDPWeightManager(
        rank,
        args.world_size,
        device,
        args.slice_bytes,
        merge_free=args.merge_free,
        allow_cuda_graph=args.capture_transport,
    )
    start, end = expert_range(args.experts, rank, args.world_size)
    for layer in range(args.layers):
        weights = []
        for rows, cols in [
            (args.intermediate, args.hidden),
            (args.hidden, args.intermediate),
        ]:
            weight = torch.empty(
                (end - start, cols // 64, rows, 64),
                dtype=torch.bfloat16,
                device=device,
            )
            values = torch.arange(start, end, device=device).float() * 1e-5 + 1e-4
            weight.copy_(values[:, None, None, None])
            weights.append(weight)
        manager.register_layer(str(layer), args.experts, weights)
    manager.initialize(store)
    active = rank < args.active_ranks
    tokens_list = args.tokens if args.compute else [0]
    records = []
    full_weights = None
    if args.compute and active:
        from flashinfer import autotune
        from flashinfer.fused_moe import prims_ts_bf16_routed_moe
        from flashinfer.tllm_enums import ActivationType, WeightLayout

        with manager.layer("0") as weights:
            full_weights = tuple(w.clone() for w in weights)
            if args.merge_free:
                for full, resident in zip(full_weights, manager.layers[0].weights):
                    full[start:end].copy_(resident)

    for tokens in tokens_list:
        if args.compute and active:
            torch.manual_seed(42 + rank)
            x = torch.randn(tokens, args.hidden, dtype=torch.bfloat16, device=device)
            logits = torch.randn(tokens, args.experts, device=device)
            scores, ids = logits.sigmoid().topk(args.top_k, dim=-1)
            route_weights = (scores / scores.sum(-1, keepdim=True) * 5).bfloat16()
            ids = ids.int()
            output = torch.empty_like(x)

            def compute(
                weights,
                resident=None,
                ids=ids,
                route_weights=route_weights,
                x=x,
                output=output,
                tokens=tokens,
            ):
                split = {}
                if resident is not None:
                    split = dict(
                        resident_gemm1_weights=resident[0],
                        resident_gemm2_weights=resident[1],
                        resident_expert_offset=start,
                    )
                return prims_ts_bf16_routed_moe(
                    (ids, route_weights),
                    x,
                    *weights,
                    num_experts=args.experts,
                    top_k=args.top_k,
                    n_group=None,
                    topk_group=None,
                    intermediate_size=args.intermediate,
                    local_expert_offset=0,
                    local_num_experts=args.experts,
                    routed_scaling_factor=5.0,
                    routing_method_type=6,
                    weight_layout=WeightLayout.BlockMajorK,
                    activation_type=ActivationType.Relu2,
                    output=output,
                    tune_max_num_tokens=tokens,
                    **split,
                )

            with autotune(
                tune_mode=True,
                tuning_buckets=(tokens,),
                cache=f"{args.output}.rank{rank}.tuning.json",
            ):
                compute(full_weights)
            reference = output.clone()
        else:

            def compute(weights, resident=None):
                return None

        modes = ["compute", "synchronous", "overlap"] if args.compute else ["copy"]
        for mode in modes:
            manager.prefetch = mode == "overlap"
            compute_graphs = []

            def forward(mode=mode, compute=compute, compute_graphs=compute_graphs):
                for layer in range(args.layers):
                    if mode == "compute":
                        if compute_graphs:
                            compute_graphs[layer].replay()
                        else:
                            compute(full_weights)
                    else:
                        with manager.layer(str(layer)) as weights:
                            if compute_graphs:
                                compute_graphs[layer].replay()
                            else:
                                compute(
                                    weights,
                                    manager.layers[layer].weights
                                    if args.merge_free
                                    else None,
                                )

            if active:
                if args.compute:
                    with autotune(
                        tune_mode=True,
                        tuning_buckets=(tokens,),
                        cache=f"{args.output}.rank{rank}.tuning.json",
                    ):
                        forward()
                forward()
                torch.accelerator.synchronize()
                if (
                    args.compute
                    and not args.eager_compute
                    and not args.capture_transport
                ):
                    for layer in range(args.layers):
                        graph = torch.cuda.CUDAGraph()
                        if mode == "compute":
                            with torch.cuda.graph(graph):
                                compute(full_weights)
                        else:
                            with manager.layer(str(layer)) as weights:
                                torch.accelerator.synchronize()
                                with torch.cuda.graph(graph):
                                    compute(
                                        weights,
                                        manager.layers[layer].weights
                                        if args.merge_free
                                        else None,
                                    )
                        compute_graphs.append(graph)
                    forward()
                    torch.accelerator.synchronize()
                measured_forward = forward
                if args.capture_transport:
                    pipeline_graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(pipeline_graph):
                        forward()
                        if mode != "compute":
                            manager.join_copy_stream(torch.cuda.current_stream())
                    measured_forward = pipeline_graph.replay
                    measured_forward()
                    torch.accelerator.synchronize()
                if args.compute:
                    torch.testing.assert_close(output, reference, rtol=0, atol=0)
                else:
                    with manager.layer(str(args.layers - 1)) as weights:
                        expected = (
                            torch.arange(args.experts, device=device).float() * 1e-5
                            + 1e-4
                        ).bfloat16()
                        for weight in weights:
                            actual = weight[:, 0, 0, 0]
                            wanted = expected
                            if args.merge_free:
                                actual = torch.cat((actual[:start], actual[end:]))
                                wanted = torch.cat((expected[:start], expected[end:]))
                            torch.testing.assert_close(actual, wanted, rtol=0, atol=0)
            timings = []
            remote_bytes = []
            begin, finish = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
            for iteration in range(args.iterations):
                key = f"sample/{tokens}/{mode}/{iteration}"
                torch.accelerator.synchronize()
                _barrier(store, rank, args.world_size, key)
                if active:
                    before = manager.remote_bytes
                    begin.record()
                    measured_forward()
                    finish.record()
                    finish.synchronize()
                    timings.append(begin.elapsed_time(finish))
                    remote_bytes.append(
                        sum(manager._remote_sizes)
                        if args.capture_transport and mode != "compute"
                        else manager.remote_bytes - before
                    )
            if active:
                records.append(
                    dict(
                        mode=mode,
                        tokens=tokens,
                        samples_ms=timings,
                        remote_bytes_per_sample=remote_bytes,
                    )
                )
                Path(f"{args.output}.rank{rank}.partial.json").write_text(
                    json.dumps(dict(config=vars(args), records=records), indent=2)
                    + "\n"
                )
    props = torch.cuda.get_device_properties(rank)
    record = dict(
        rank=rank,
        gpu=props.name,
        uuid=str(props.uuid),
        active=active,
        resident_bytes=manager.resident_bytes,
        staging_bytes=manager.staging_bytes,
        peak_allocated_bytes=torch.accelerator.memory.max_memory_allocated(),
        records=records,
    )
    store.set(f"result/{rank}", json.dumps(record))
    if rank == 0:
        ranks = [json.loads(store.get(f"result/{r}")) for r in range(args.world_size)]
        summary = []
        for i, record in enumerate(ranks[0]["records"]):
            maxima = [
                max(r["records"][i]["samples_ms"][s] for r in ranks if r["active"])
                for s in range(args.iterations)
            ]
            milliseconds = statistics.median(maxima)
            remote = statistics.median(record["remote_bytes_per_sample"])
            summary.append(
                dict(
                    mode=record["mode"],
                    tokens=record["tokens"],
                    median_max_rank_ms=milliseconds,
                    remote_GB_per_s_per_worker=remote / milliseconds / 1e6,
                )
            )
        result = dict(
            config=vars(args),
            torch_version=torch.__version__,
            cuda_version=torch.version.cuda,
            environment={
                name: os.environ[name]
                for name in (
                    "CUDA_VISIBLE_DEVICES",
                    "OMP_NUM_THREADS",
                    "FLASHINFER_WORKSPACE_BASE",
                    "CUTE_DSL_NO_CACHE",
                    "SLURM_JOB_ID",
                )
                if name in os.environ
            },
            revision=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            flashinfer_revision=subprocess.check_output(
                ["git", "-C", "../flashinfer", "rev-parse", "HEAD"], text=True
            ).strip(),
            dirty=subprocess.check_output(["git", "status", "--short"], text=True),
            timing=(
                "CUDA events, full pipeline graph replay with serial captured copies, "
                "host barrier outside timing, synthetic inputs, precomputed routing"
                if args.capture_transport
                else "CUDA events, eager manager dispatch, compute graph replay unless "
                "eager_compute, host barrier outside timing, synthetic inputs, "
                "precomputed routing excluded"
            ),
            summary=summary,
            ranks=ranks,
        )
        Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(summary, indent=2), flush=True)
    manager.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--active-ranks", type=int)
    parser.add_argument("--experts", type=int, default=512)
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--intermediate", type=int, default=2688)
    parser.add_argument("--top-k", type=int, default=22)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--slice-bytes", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--tokens", nargs="+", type=int, default=[1024, 16960, 33920])
    parser.add_argument("--compute", action="store_true")
    parser.add_argument("--merge-free", action="store_true")
    parser.add_argument("--eager-compute", action="store_true")
    parser.add_argument("--capture-transport", action="store_true")
    parser.add_argument("--output", default="/tmp/dwdp-benchmark.json")
    args = parser.parse_args()
    if args.active_ranks is None:
        args.active_ranks = args.world_size
    if (
        args.layers < 4
        or args.iterations < 1
        or not 1 <= args.active_ranks <= args.world_size
        or args.hidden % 64
        or args.intermediate % 64
        or (args.capture_transport and args.eager_compute)
    ):
        parser.error(
            "Require >=4 layers, positive iterations, valid ranks and K % 64=0"
        )
    with tempfile.TemporaryDirectory(prefix="dwdp-bench-") as directory:
        mp.spawn(
            _worker,
            args=(args, str(Path(directory) / "store")),
            nprocs=args.world_size,
            join=True,
        )


if __name__ == "__main__":
    main()
