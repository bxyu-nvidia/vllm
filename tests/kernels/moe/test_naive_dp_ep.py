# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AG/RS monolithic DBO: stream ownership and concurrent collective correctness."""

import json
import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
import torch
import torch.distributed as dist

from tests.utils import ensure_current_vllm_config
from vllm.config import ParallelConfig
from vllm.distributed.device_communicators.all2all import AgRsAll2AllManager
from vllm.distributed.device_communicators.cuda_communicator import CudaCommunicator
from vllm.forward_context import DPMetadata, ForwardContext
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.prepare_finalize import naive_dp_ep as naive
from vllm.v1.worker.gpu import dp_utils
from vllm.v1.worker.gpu.dbo_timing import DBOTimingEvents
from vllm.v1.worker.gpu.dbo_trace import DBOTraceWriter
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.ubatch_utils import create_whole_request_ubatch_slices
from vllm.v1.worker.ubatching import make_ubatch_contexts

from .parallel_utils import ProcessGroupInfo, parallel_launch


@pytest.mark.parametrize("enable_dbo", [False, True])
def test_monolithic_collective_stream_ownership(monkeypatch, enable_dbo):
    """Ordering and allocator lifetime tracking apply only inside a DBO context."""
    events = []
    stream = ["compute"]

    def to_comm():
        events.append("compute->comm")
        stream[0] = "comm"

    def to_compute():
        events.append("yield->compute")
        stream[0] = "compute"

    def gather(x, logits, **kwargs):
        events.append("gather")
        assert stream[0] == ("comm" if enable_dbo else "compute")
        return torch.cat((x, x + 1)), torch.cat((logits, logits + 2))

    def combine(x, **kwargs):
        events.append("scatter")
        assert stream[0] == ("comm" if enable_dbo else "compute")
        return x.chunk(2)[0] + x.chunk(2)[1]

    monkeypatch.setattr(naive, "dbo_enabled", lambda: enable_dbo)
    monkeypatch.setattr(naive, "dbo_switch_to_comm_sync", to_comm)
    monkeypatch.setattr(naive, "dbo_yield_and_switch_from_comm_to_compute", to_compute)
    monkeypatch.setattr(naive, "current_stream", lambda: stream[0])
    monkeypatch.setattr(
        torch.Tensor, "record_stream", lambda self, s: events.append(f"record:{s}")
    )
    monkeypatch.setattr(
        naive,
        "get_ep_group",
        lambda: SimpleNamespace(dispatch_router_logits=gather, combine=combine),
    )
    pf = naive.MoEPrepareAndFinalizeNaiveDPEPMonolithic(num_dispatchers=2)
    x, logits = torch.ones(3, 8), torch.ones(3, 2)
    gathered, scale, routes = pf.prepare(
        x, logits, FusedMoEQuantConfig.make(), defer_input_quant=True
    )
    assert scale is None
    torch.testing.assert_close(routes, torch.cat((logits, logits + 2)))
    result = pf.finalize(gathered * 2)
    torch.testing.assert_close(result, torch.full_like(x, 6))
    assert events == (
        [
            "compute->comm",
            "record:comm",
            "record:comm",
            "gather",
            "yield->compute",
            "record:compute",
            "record:compute",
            "compute->comm",
            "record:comm",
            "scatter",
            "yield->compute",
            "record:compute",
        ]
        if enable_dbo
        else ["gather", "scatter"]
    )


def test_collective_error_restores_compute_stream(monkeypatch):
    events = []
    monkeypatch.setattr(naive, "dbo_enabled", lambda: True)
    monkeypatch.setattr(naive, "dbo_switch_to_comm_sync", lambda: events.append("comm"))
    monkeypatch.setattr(
        naive, "dbo_switch_to_compute_sync", lambda: events.append("compute")
    )

    def fail():
        raise RuntimeError("collective failed")

    with pytest.raises(RuntimeError, match="collective failed"):
        naive._run_dbo_collective(fail, [])
    assert events == ["comm", "compute"]


def _ag_rs_dbo_worker(
    pgi: ProcessGroupInfo,
    sizes: list[list[int]],
    symmetric: bool,
    query_lens: list[list[int]] | None = None,
):
    """Exercise real AG/RS and DBO threads without loading a model or DeepEP."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("VLLM_USE_NCCL_SYMM_MEM", "1" if symmetric else "0")
        patch.setenv("VLLM_BATCH_INVARIANT", "0")
        cpu_group = dist.new_group(backend="gloo")
        if query_lens is not None:
            # Exercise the production DP handshake and request-boundary split
            # before the real NCCL collectives, with different full-batch sizes.
            patch.setattr(
                dp_utils, "get_dp_group", lambda: SimpleNamespace(cpu_group=cpu_group)
            )
            patch.setattr(dp_utils, "should_skip_dp_coordination", lambda: False)
            local_queries = query_lens[pgi.rank]
            local_tokens = sum(local_queries)
            desc, sync = dp_utils.dispatch_cg_and_sync_dp(
                None,
                len(local_queries),
                local_tokens,
                None,
                pgi.world_size,
                pgi.rank,
                need_eager=True,
                parallel_config=ParallelConfig(
                    enable_dbo=True,
                    dbo_prefill_token_threshold=2,
                    dbo_decode_token_threshold=2,
                    all2all_backend="allgather_reducescatter",
                ),
                allow_ubatching=True,
                dbo_whole_requests=True,
            )
            assert desc.num_tokens == local_tokens and desc.num_ubatches == 2
            assert sync is not None
            assert sync.num_tokens_across_dp.tolist() == [sum(q) for q in query_lens]
            batch = cast(
                InputBatch,
                SimpleNamespace(
                    num_reqs=len(local_queries),
                    num_reqs_after_padding=desc.num_reqs,
                    num_tokens=local_tokens,
                    num_computed_tokens_np=np.zeros(len(local_queries), dtype=np.int64),
                    num_tokens_after_padding=desc.num_tokens,
                    query_start_loc_np=np.cumsum([0, *local_queries], dtype=np.int32),
                ),
            )
            slices = create_whole_request_ubatch_slices(batch)
            sizes = dp_utils.sync_ubatch_token_counts(
                [max(1, s.num_tokens) for s in slices], pgi.world_size, pgi.rank
            ).tolist()
            assert [sum(s[r] for s in sizes) for r in range(pgi.world_size)] == [
                sum(q) + (len(q) == 1) for q in query_lens
            ]
        with ensure_current_vllm_config():
            comm = CudaCommunicator(
                cpu_group,
                device=pgi.device,
                device_group=dist.group.WORLD,
                unique_name="dbo_test",
            )
        group = SimpleNamespace(
            world_size=pgi.world_size,
            rank_in_group=pgi.rank,
            all_gatherv=comm.all_gatherv,
            reduce_scatterv=comm.reduce_scatterv,
        )
        manager = object.__new__(AgRsAll2AllManager)
        manager.dp_world_size = pgi.world_size
        manager._get_comm_group = lambda sequence_parallel: group
        patch.setattr(naive, "get_ep_group", lambda: manager)
        pf = naive.MoEPrepareAndFinalizeNaiveDPEPMonolithic(
            num_dispatchers=pgi.world_size
        )
        compute = torch.cuda.current_stream()
        comm_stream = torch.cuda.Stream()
        barrier = threading.Barrier(3, timeout=30)
        contexts = make_ubatch_contexts(
            2,
            compute,
            comm_stream,
            [
                ForwardContext(
                    no_compile_layers={},
                    attn_metadata={},
                    slot_mapping={},
                    dp_metadata=DPMetadata(torch.tensor(s), local_sizes=s),
                )
                for s in sizes
            ],
            barrier,
        )
        timing = None
        if query_lens is not None:
            trace_dir = tempfile.TemporaryDirectory()
            writer = DBOTraceWriter(
                trace_dir.name, dp_rank=pgi.rank, device_index=pgi.device.index
            )
            trace = writer.new_trace(
                batch, slices, [ctx.forward_context for ctx in contexts]
            )
            timing = DBOTimingEvents(trace=trace)
        for context in contexts:
            context.dbo_timing = timing
        outputs, errors = {}, []

        def run(index):
            try:
                torch.accelerator.set_device_index(pgi.device.index)
                with torch.inference_mode(), contexts[index]:
                    for iteration in range(8):
                        shape = (sizes[index][pgi.rank], 128)
                        value = pgi.rank + index + iteration
                        x = torch.full(
                            shape, value, device=pgi.device, dtype=torch.bfloat16
                        )
                        logits = torch.full(
                            (shape[0], 4), 1, device=pgi.device, dtype=torch.bfloat16
                        )
                        gathered, _, routes = pf.prepare(
                            x,
                            logits,
                            FusedMoEQuantConfig.make(),
                            defer_input_quant=True,
                        )
                        del x, logits
                        contribution = (gathered + routes[:, :1]) * (pgi.rank + 1)
                        outputs[index, iteration] = pf.finalize(contribution).clone()
            except BaseException as exc:
                errors.append(exc)

        threads = [
            threading.Thread(target=run, args=(i,), daemon=True) for i in range(2)
        ]
        for thread in threads:
            thread.start()
        barrier.wait()
        contexts[0].cpu_wait_event.set()
        for thread in threads:
            thread.join(timeout=60)
        assert not any(thread.is_alive() for thread in threads), "DBO handoff hung"
        if errors:
            raise errors[0]
        torch.accelerator.synchronize()
        if timing is not None:
            measured = timing.resolve()
            writer.close()
            record = json.loads(next(Path(trace_dir.name).glob("*.jsonl")).read_text())
            assert record["dp_rank"] == pgi.rank
            assert (
                sum(w["query_tokens"] for w in record["microbatches"]) == local_tokens
            )
            for index in range(2):
                windows = [
                    w
                    for w in record["windows"]
                    if w["microbatch"] == index and w["phase"] == "comm"
                ]
                assert [w["ordinal"] for w in windows] == list(range(16))
                assert all(
                    w["gpu_end_offset_seconds"] >= w["gpu_start_offset_seconds"]
                    for w in windows
                )
            assert sum(
                w["compute_seconds"] for w in record["microbatches"]
            ) == pytest.approx(measured.compute_seconds, abs=1e-6)
            trace_dir.cleanup()
            assert measured.compute_seconds > 0 and measured.comm_seconds > 0
            assert (
                0
                <= measured.overlap_seconds
                <= min(measured.compute_seconds, measured.comm_seconds) + 1e-6
            )
            assert (
                measured.wall_seconds
                >= max(measured.compute_seconds, measured.comm_seconds) - 1e-6
            )
        factor = sum(range(1, pgi.world_size + 1))
        for (index, iteration), output in outputs.items():
            expected = torch.full_like(
                output, (pgi.rank + index + iteration + 1) * factor
            )
            torch.testing.assert_close(output, expected, rtol=0, atol=0)
        if symmetric and all(s[0] == s[1] for s in sizes):
            # Prove this parameter actually used separate symmetric allocations.
            keys = comm.__dict__.get("_symm_scratch_bufs", {})
            assert {key[1] for key in keys} == {0, 1}
        comm.destroy()
        dist.destroy_process_group(cpu_group)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.accelerator.device_count() < 2,
    reason="Requires two CUDA GPUs",
)
@pytest.mark.parametrize("symmetric", [False, True])
@pytest.mark.parametrize(
    "sizes", [[[3, 3], [5, 5]], [[3, 7], [9, 2]], [[0, 7], [9, 0]]]
)
def test_monolithic_dbo_ag_rs_gpu(sizes, symmetric):
    """Check repeated overlap, unequal ranks, idle ranks, and scratch reuse."""
    parallel_launch(2, _ag_rs_dbo_worker, sizes, symmetric)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.accelerator.device_count() < 2,
    reason="Requires two CUDA GPUs",
)
@pytest.mark.parametrize("symmetric", [False, True])
@pytest.mark.parametrize(
    "query_lens",
    [
        [[2, 2, 4], [4, 4, 8]],
        [[1, 15], [4, 4, 1]],
        [[16], [4, 4, 8]],
        [[4, 4, 8], [6]],
    ],
)
def test_hybrid_dbo_ag_rs_unpadded_gpu(query_lens, symmetric):
    """Unpadded whole-request dispatch feeds matching variable-size AG/RS calls."""
    parallel_launch(2, _ag_rs_dbo_worker, [], symmetric, query_lens)
