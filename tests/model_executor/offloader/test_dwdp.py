# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import gc
import subprocess
import sys
import time
import weakref
from contextlib import nullcontext
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from vllm.config import DeviceConfig, DWDPConfig, ModelConfig, OffloadConfig, VllmConfig
from vllm.config.compilation import CompilationConfig, CompilationMode, CUDAGraphMode
from vllm.config.weight_transfer import WeightTransferConfig
from vllm.model_executor.offloader.dwdp import (
    DWDPWeightManager,
    expert_range,
    sliced_copy_plan,
)
from vllm.model_executor.offloader.dwdp_health import PeerProcessMonitor


@pytest.mark.parametrize(
    "method,eager,graph_mode,error",
    [
        (None, False, CUDAGraphMode.NONE, None),
        ("mtp", True, CUDAGraphMode.FULL, None),
        ("mtp", False, CUDAGraphMode.NONE, None),
        ("mtp", False, None, "does not support CUDA graphs"),
        ("mtp", False, CUDAGraphMode.FULL, "does not support CUDA graphs"),
        ("mtp", False, CUDAGraphMode.PIECEWISE, "does not support CUDA graphs"),
        ("mtp", False, CUDAGraphMode.FULL_DECODE_ONLY, "does not support CUDA graphs"),
        (
            "mtp",
            False,
            CUDAGraphMode.FULL_AND_PIECEWISE,
            "does not support CUDA graphs",
        ),
        ("eagle", True, CUDAGraphMode.NONE, "only MTP"),
        ("eagle", False, CUDAGraphMode.NONE, "only MTP"),
        ("ngram", True, CUDAGraphMode.NONE, "only MTP"),
    ],
)
@pytest.mark.parametrize("allow_cuda_graph", [False, True])
def test_dwdp_speculative_policy(method, eager, graph_mode, error, allow_cuda_graph):
    spec = SimpleNamespace(method=method) if method is not None else None
    config = DWDPConfig(allow_cuda_graph=allow_cuda_graph)
    if error is not None:
        with pytest.raises(ValueError, match=error):
            config.verify_speculative_config(
                spec, enforce_eager=eager, cudagraph_mode=graph_mode
            )
    else:
        config.verify_speculative_config(
            spec, enforce_eager=eager, cudagraph_mode=graph_mode
        )


@pytest.mark.parametrize("dwdp", [False, True])
@pytest.mark.parametrize("fail_draft", [False, True])
def test_draft_loading_preserves_target_offloader(monkeypatch, dwdp, fail_draft):
    from vllm.model_executor import model_loader
    from vllm.model_executor.offloader import base
    from vllm.model_executor.offloader.dwdp import DWDPWeightOffloader

    offloader = DWDPWeightOffloader(DWDPConfig()) if dwdp else base.NoopOffloader()
    monkeypatch.setattr(base, "_instance", offloader)
    target_config, draft_config = object(), object()
    config = SimpleNamespace(
        load_config=object(),
        model_config=target_config,
        speculative_config=SimpleNamespace(draft_model_config=draft_config),
    )
    loaded = []

    def load_model(*, vllm_config, model_config, prefix):
        active = base.get_offloader()
        if model_config is draft_config and dwdp:
            assert isinstance(active, base.NoopOffloader)
        else:
            assert active is offloader
        if model_config is draft_config and fail_draft:
            raise RuntimeError("draft load failed")
        model = torch.nn.Linear(2, 2)
        # Module construction must execute under the selected offloader too.
        assert active.wrap_modules(iter([model]), prefix) == [model]
        loaded.append(model_config)
        return model

    monkeypatch.setattr(
        model_loader,
        "get_model_loader",
        lambda _: SimpleNamespace(load_model=load_model),
    )
    model_loader.get_model(vllm_config=config)
    if fail_draft:
        with pytest.raises(RuntimeError, match="draft load failed"):
            model_loader.get_model(vllm_config=config, model_config=draft_config)
    else:
        model_loader.get_model(vllm_config=config, model_config=draft_config)
    assert base.get_offloader() is offloader
    model_loader.get_model(vllm_config=config, model_config=target_config)
    assert loaded == (
        [target_config, target_config]
        if fail_draft
        else [target_config, draft_config, target_config]
    )
    if dwdp:
        assert offloader._manager is None


@pytest.mark.parametrize("prims", [False, True])
def test_empty_modular_moe_skips_kernel_and_keeps_output(prims):
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.experts.prims_bf16_moe import (
        PrimsBf16Experts,
    )
    from vllm.model_executor.layers.fused_moe.experts.trtllm_bf16_moe import (
        TrtLlmBf16ExpertsModular,
    )

    cls = PrimsBf16Experts if prims else TrtLlmBf16ExpertsModular
    experts = cls.__new__(cls)
    experts._routed_moe = lambda **kwargs: pytest.fail("Empty input launched a kernel")
    hidden = torch.empty(0, 128, dtype=torch.bfloat16)
    output = torch.empty_like(hidden)
    experts.apply(
        output,
        hidden,
        None,
        None,
        torch.empty(0, 2),
        torch.empty(0, 2, dtype=torch.int64),
        MoEActivation.RELU2_NO_MUL,
        4,
        None,
        None,
        None,
        None,
        None,
        None,
        False,
    )
    assert output.shape == hidden.shape
    assert output.dtype == hidden.dtype


def test_empty_monolithic_moe_skips_finalize_policy_and_kernel():
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.experts.trtllm_bf16_moe import (
        TrtLlmBf16ExpertsMonolithic,
    )

    experts = TrtLlmBf16ExpertsMonolithic.__new__(TrtLlmBf16ExpertsMonolithic)
    experts.moe_config = SimpleNamespace(
        should_defer_moe_finalize=lambda n: pytest.fail("Empty input reached finalize")
    )
    hidden = torch.empty(0, 128, dtype=torch.bfloat16)
    result = experts.apply(
        hidden,
        None,
        None,
        torch.empty(0, 4),
        MoEActivation.RELU2_NO_MUL,
        4,
        None,
        None,
        False,
    )
    assert result.shape == hidden.shape
    assert result.dtype == hidden.dtype
    assert result.device == hidden.device


def test_prims_routed_dispatch_keeps_resident_and_staging_weights(monkeypatch):
    from types import ModuleType
    from unittest.mock import Mock

    from vllm.model_executor.layers.fused_moe.experts import prims_bf16_moe

    resident = (torch.ones(1, 1, 128, 128), torch.ones(1, 1, 128, 128))
    experts = prims_bf16_moe.PrimsBf16Experts.__new__(prims_bf16_moe.PrimsBf16Experts)
    experts._resident_weights = resident
    experts._resident_expert_offset = 2
    experts.moe_config = object()
    monkeypatch.setattr(prims_bf16_moe, "fi_moe_largest_bucket", lambda cfg: 16960)
    moe = ModuleType("flashinfer.fused_moe")
    routed_moe = Mock(return_value=object())
    monkeypatch.setattr(moe, "prims_ts_bf16_routed_moe", routed_moe, raising=False)
    monkeypatch.setitem(sys.modules, "flashinfer", ModuleType("flashinfer"))
    monkeypatch.setitem(sys.modules, "flashinfer.fused_moe", moe)
    for _ in range(2):
        w1, w2 = torch.empty(4, 1, 128, 128), torch.empty(4, 1, 128, 128)
        result = experts._routed_moe(gemm1_weights=w1, gemm2_weights=w2)
        args = routed_moe.call_args.kwargs
        assert result is routed_moe.return_value
        assert args["gemm1_weights"] is w1 and args["gemm2_weights"] is w2
        assert args["resident_gemm1_weights"] is resident[0]
        assert args["resident_gemm2_weights"] is resident[1]
        assert args["resident_expert_offset"] == 2
        assert args["tune_max_num_tokens"] == 16960


def test_resident_views_outlive_weights_without_retaining_the_source_layer():
    """Cached packed views must not retain parameter weight-loader callbacks."""
    from vllm.model_executor.layers.fused_moe.experts.prims_bf16_moe import (
        PrimsBf16Experts,
    )

    layer = torch.nn.Module()
    for name in ("w13_weight", "w2_weight"):
        layer.register_parameter(
            name,
            torch.nn.Parameter(
                torch.ones(1, 64, 64, dtype=torch.bfloat16), requires_grad=False
            ),
        )
        getattr(layer, name).weight_loader = layer.load_state_dict
    layer.dwdp_manager = SimpleNamespace(merge_free=True)
    layer.dwdp_expert_range = (0, 1)
    experts = PrimsBf16Experts.__new__(PrimsBf16Experts)
    experts.process_weights_after_loading(layer)
    reference = weakref.ref(layer)
    del layer
    gc.collect()

    assert reference() is None
    assert experts._resident_weights is not None
    for resident in experts._resident_weights:
        torch.testing.assert_close(resident, torch.ones_like(resident))


def test_idle_weight_owner_exit_is_detected_without_forward_collectives():
    owner = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    monitor = PeerProcessMonitor({7: owner.pid})
    try:
        monitor.check()
        owner.terminate()
        owner.wait(timeout=5)
        deadline = time.monotonic() + 5
        while monitor.failed_peer is None and time.monotonic() < deadline:
            time.sleep(0.01)
        with pytest.raises(RuntimeError, match="weight owner 7 exited"):
            monitor.check()
    finally:
        monitor.close()
        if owner.poll() is None:
            owner.kill()
            owner.wait(timeout=5)


def test_sliced_pulls_interleave_peers_and_cover_unequal_tails_once():
    pulls = [(1000, 100, 5), (2000, 200, 8), (3000, 300, 2)]
    plan = sliced_copy_plan(pulls, 3)
    assert [src for _, src, _ in plan[:3]] == [100, 200, 300]
    expected = sorted((dest + i, src + i) for dest, src, n in pulls for i in range(n))
    actual = sorted((dest + i, src + i) for dest, src, n in plan for i in range(n))
    assert actual == expected


def test_dwdp_config_rejects_cpu_offload_and_missing_rendezvous():
    with pytest.raises(ValueError, match="requires dwdp"):
        OffloadConfig(offload_backend="dwdp")
    with pytest.raises(ValueError, match="CPU weight offloading"):
        OffloadConfig(offload_backend="uva", dwdp=DWDPConfig())
    with pytest.raises(ValueError, match="rank"):
        DWDPConfig(rank=4, world_size=4)


@pytest.fixture
def tiny_model_path(tmp_path):
    from transformers import OPTConfig

    OPTConfig(
        architectures=["OPTForCausalLM"],
        hidden_size=128,
        num_attention_heads=4,
        num_hidden_layers=1,
        ffn_dim=256,
        vocab_size=128,
    ).save_pretrained(tmp_path)
    return str(tmp_path)


@pytest.mark.parametrize(
    "mode,graph_mode,allow_cuda_graph,error",
    [
        (3, "NONE", False, None),
        (0, "NONE", False, None),
        (3, "PIECEWISE", False, "requires cudagraph_mode=NONE"),
        (3, "PIECEWISE", True, None),
        (0, "PIECEWISE", True, "supports vLLM compilation"),
        (1, "NONE", False, "supports vLLM compilation"),
        (2, "NONE", False, "supports vLLM compilation"),
    ],
)
def test_dwdp_compilation_policy(
    tiny_model_path, mode, graph_mode, allow_cuda_graph, error
):
    """Compilation with graphs disabled needs no graph-capture opt-in."""
    kwargs = dict(
        model_config=ModelConfig(model=tiny_model_path, enforce_eager=False),
        device_config=DeviceConfig(device="cpu"),
        offload_config=OffloadConfig(
            dwdp=DWDPConfig(allow_cuda_graph=allow_cuda_graph)
        ),
        compilation_config=CompilationConfig(mode=mode, cudagraph_mode=graph_mode),
    )
    if error is not None:
        with pytest.raises(ValueError, match=error):
            VllmConfig(**kwargs)
    else:
        config = VllmConfig(**kwargs)
        assert config.compilation_config.mode == CompilationMode(mode)
        assert config.compilation_config.cudagraph_mode == CUDAGraphMode[graph_mode]
        assert not config.model_config.enforce_eager


@pytest.mark.parametrize("feature", ["sleep", "weight_transfer"])
def test_dwdp_rejects_features_that_can_invalidate_live_sources(
    tiny_model_path, monkeypatch, feature
):
    from vllm.platforms import current_platform

    monkeypatch.setattr(current_platform, "is_sleep_mode_available", lambda: True)
    model = ModelConfig(
        model=tiny_model_path, enforce_eager=True, enable_sleep_mode=feature == "sleep"
    )
    transfer = WeightTransferConfig() if feature == "weight_transfer" else None
    with pytest.raises(ValueError, match="immutable resident weights"):
        VllmConfig(
            model_config=model,
            device_config=DeviceConfig(device="cpu"),
            offload_config=OffloadConfig(dwdp=DWDPConfig()),
            weight_transfer_config=transfer,
        )


@pytest.mark.parametrize("experts,world_size", [(7, 4), (8, 2), (4, 4)])
def test_expert_ownership_covers_uneven_shards(experts, world_size):
    ranges = [expert_range(experts, r, world_size) for r in range(world_size)]
    assert [e for start, end in ranges for e in range(start, end)] == list(
        range(experts)
    )
    sizes = [end - start for start, end in ranges]
    assert max(sizes) - min(sizes) <= 1


@pytest.mark.parametrize("experts,rank,size", [(3, 0, 4), (4, 4, 4), (4, -1, 2)])
def test_expert_ownership_rejects_empty_or_invalid_shards(experts, rank, size):
    with pytest.raises(ValueError, match="rank"):
        expert_range(experts, rank, size)


def _ipc_worker(
    rank: int, world_size: int, path: str, slice_bytes: int, use_graph: bool | str
):
    torch.accelerator.set_device_index(rank)
    device = torch.device("cuda", rank)
    store = dist.FileStore(path, world_size)
    store.set_timeout(timedelta(seconds=90))
    manager = DWDPWeightManager(
        rank,
        world_size,
        device,
        slice_bytes,
        collect_metrics=not use_graph,
        allow_cuda_graph=bool(use_graph),
    )
    experts = 7
    start, end = expert_range(experts, rank, world_size)
    references = []
    for layer in range(5):
        # Vary packed shapes to exercise shared capacity and view boundaries.
        weights = tuple(
            torch.arange(experts * 2 * (8 + p * 8), device=device, dtype=torch.float32)
            .view(experts, 2, 8 + p * 8)
            .add_(1000 * layer)
            for p in range(2)
        )
        references.append(weights)
        manager.register_layer(
            str(layer), experts, [w[start:end].clone() for w in weights]
        )
    manager.initialize(store)
    snapshots = []
    # Rank zero stays idle, others run different numbers of forwards. Only
    # initialization and teardown rendezvous; inference must need no peer work.
    for _ in range(rank * 2):
        for layer in range(5):
            with manager.layer(str(layer), num_tokens=3) as weights:
                snapshots.append((layer, tuple(w.clone() for w in weights)))
    if use_graph and rank:
        with torch.cuda.stream(torch.cuda.Stream()):
            for layer in range(5):
                with manager.layer(str(layer)) as weights:
                    snapshots.append((layer, tuple(w.clone() for w in weights)))
        graphs = []
        for _ in range(2):
            groups = (
                [[layer] for layer in range(5)]
                if use_graph == "fragmented"
                else [list(range(5))]
            )
            pieces = []
            for group in groups:
                graph = torch.cuda.CUDAGraph()
                torch.accelerator.synchronize()
                captured = []
                with torch.cuda.graph(graph):
                    for layer in group:
                        with manager.layer(str(layer)) as weights:
                            captured.append((layer, tuple(w.clone() for w in weights)))
                    torch.cuda.current_stream().wait_stream(manager._copy_stream)
                pieces.append((graph, captured))
            graphs.append(pieces)
        for step in range(2 + rank):
            for graph, captured in graphs[step % 2]:
                graph.replay()
                snapshots.extend(captured)
        for layer in range(5):
            with manager.layer(str(layer)) as weights:
                snapshots.append((layer, tuple(w.clone() for w in weights)))
    torch.accelerator.synchronize()
    for layer, weights in snapshots:
        for actual, expected in zip(weights, references[layer]):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert manager.resident_bytes > 0
    assert manager.staging_bytes > manager.resident_bytes / 5
    assert bool(manager.remote_bytes) == bool(rank)
    metrics = manager.metrics()
    assert metrics["forward_enqueues"] == rank * 2 + (2 if use_graph and rank else 0)
    assert metrics["token_enqueues"] == rank * 6
    assert metrics["remote_bytes_per_forward"] > 0
    assert len(metrics["layers"]) == (5 if rank and not use_graph else 0)
    assert all(
        value >= 0 for layer in metrics["layers"].values() for value in layer.values()
    )
    manager.close()
    manager.close()


@pytest.mark.parametrize(
    "world_size,slice_bytes,use_graph",
    [
        (2, 0, False),
        (4, 60, False),
        (2, 60, True),
        (4, 60, True),
        (2, 60, "fragmented"),
    ],
)
def test_ipc_prefetch_preserves_weights_with_idle_and_unequal_workers(
    tmp_path, world_size, slice_bytes, use_graph
):
    """Copies must wait for prior readers across repeated two-buffer reuse."""
    if not torch.cuda.is_available() or torch.accelerator.device_count() < world_size:
        pytest.skip(f"Requires {world_size} CUDA GPUs")
    mp.spawn(
        _ipc_worker,
        args=(world_size, str(tmp_path / "store"), slice_bytes, use_graph),
        nprocs=world_size,
        join=True,
    )


def _moe_worker(rank, path, backend, merge_free):
    from tests.kernels.moe.utils import make_dummy_moe_config
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.expert_map_manager import ExpertMapManager
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
    from vllm.model_executor.offloader import set_offloader
    from vllm.model_executor.offloader.dwdp import DWDPWeightOffloader
    from vllm.v1.worker.workspace import init_workspace_manager

    torch.set_num_threads(1)
    torch.accelerator.set_device_index(rank)
    device = torch.device("cuda", rank)
    init_workspace_manager(device)
    offloader = DWDPWeightOffloader(
        DWDPConfig(rank=rank, world_size=2, merge_free=merge_free)
    )
    num_experts = 8 if backend == "flashinfer_prims" else 7
    sharded: list[RoutedExperts] = []
    baselines: list[RoutedExperts] = []
    with set_current_vllm_config(VllmConfig()), torch.device(device):
        for layer_id in range(4):
            torch.manual_seed(42 + layer_id)
            full_w13 = torch.randn(num_experts, 256, 128, dtype=torch.bfloat16) * 0.01
            full_w2 = torch.randn(num_experts, 128, 256, dtype=torch.bfloat16) * 0.01
            for use_dwdp, layers in [(False, baselines), (True, sharded)]:
                set_offloader(offloader if use_dwdp else None)
                config = make_dummy_moe_config(
                    num_experts=num_experts,
                    hidden_dim=128,
                    intermediate_size=256,
                    experts_per_token=2,
                    activation=MoEActivation.RELU2_NO_MUL,
                )
                config.moe_backend = backend
                mapping = ExpertMapManager(
                    max_num_batched_tokens=128,
                    top_k=2,
                    global_num_experts=num_experts,
                    num_redundant_experts=0,
                    num_expert_group=None,
                    moe_parallel_config=config.moe_parallel_config,
                    placement_strategy="linear",
                    enable_eplb=False,
                )
                layer = RoutedExperts(
                    str(layer_id),
                    torch.bfloat16,
                    config,
                    None,
                    mapping,
                    ckpt_gate_proj_name="up_proj",
                    ckpt_up_proj_name=None,
                )
                for weight, name, shard in [
                    (full_w13, "w13_weight", "w1"),
                    (full_w2, "w2_weight", "w2"),
                ]:
                    param = getattr(layer, name)
                    if use_dwdp:
                        # Exercise direct fused checkpoint loading as well as
                        # per-expert loading through the same public loader.
                        layer.weight_loader(param, weight, name, shard, 0, True)
                    else:
                        for expert, value in enumerate(weight.unbind()):
                            layer.weight_loader(param, value, name, shard, expert, True)
                if use_dwdp:
                    start, end = expert_range(num_experts, rank, 2)
                    assert layer.w13_weight.shape[0] == end - start
                    assert layer.local_num_experts == num_experts
                    assert layer.expert_map is None
                layer.quant_method.process_weights_after_loading(layer)
                layers.append(layer)
        store = dist.FileStore(path, 2)
        store.set_timeout(timedelta(seconds=90))
        offloader.manager.initialize(store)
        for tokens in [1, 17, 65, 1]:
            x = torch.randn(tokens, 128, dtype=torch.bfloat16)
            ids = (torch.arange(tokens * 2).view(tokens, 2) % num_experts).int()
            scores = torch.full((tokens, 2), 0.5, dtype=torch.float32)
            for actual, baseline in zip(sharded, baselines):
                expected = baseline.forward_modular(x.clone(), scores, ids)
                result = actual.forward_modular(x.clone(), scores, ids)
                torch.testing.assert_close(result, expected, rtol=0, atol=0)
        if merge_free:
            assert offloader.manager.local_copy_bytes == 0
        with pytest.raises(RuntimeError, match="immutable"):
            sharded[0].weight_loader(
                sharded[0].w2_weight, full_w2[0], "w2_weight", "w2", 0, True
            )
        offloader.shutdown()
        set_offloader(None)


@pytest.mark.parametrize(
    "backend,merge_free", [("triton", False), ("flashinfer_prims", True)]
)
def test_sharded_loading_and_moe_outputs_match_replicated_weights(
    tmp_path, backend, merge_free
):
    """Storage ownership must not restrict which experts a worker can execute."""
    if not torch.cuda.is_available() or torch.accelerator.device_count() < 2:
        pytest.skip("Requires two CUDA GPUs")
    if backend == "flashinfer_prims" and torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("Prims requires SM100")
    mp.spawn(
        _moe_worker,
        args=(str(tmp_path / "moe-store"), backend, merge_free),
        nprocs=2,
        join=True,
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize(
    "ids,local_used,remote_used",
    [
        ([[0, 0, 2], [7, 4, 5]], 2, 3),
        ([[2, 3, 4], [4, 3, 2]], 3, 0),
        ([[0, 1, 5], [6, 7, 7]], 0, 5),
        ([[-1, 8, 9]], 0, 0),
        ([], 0, 0),
    ],
)
def test_expert_usage_counts_union_with_uneven_ownership(
    device, ids, local_used, remote_used
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    manager = DWDPWeightManager(1, 3, torch.device("cuda:0"), collect_expert_usage=True)
    # Counting routing needs ownership metadata but no IPC allocations.
    manager.layers = [SimpleNamespace(name="layer0", num_experts=8)]
    manager._indices = {"layer0": 0}
    manager.forward_enqueues = 6
    topk_ids = torch.tensor(ids, dtype=torch.int64, device=device).reshape(-1, 3)
    original_ids = topk_ids.clone()
    manager.record_expert_usage("layer0", topk_ids)
    usage = manager.metrics()["expert_usage"]["layer0"]
    assert usage == {
        "forward_enqueue": 7,
        "tokens": len(ids),
        "local_experts_used": local_used,
        "local_experts_available": 3,
        "remote_experts_used": remote_used,
        "remote_experts_available": 5,
    }
    torch.testing.assert_close(topk_ids, original_ids)
    # RPC consumers cannot mutate the retained snapshot, and the next batch
    # replaces rather than unions with the previous batch's expert set.
    usage["remote_experts_used"] = 100
    assert (
        manager.metrics()["expert_usage"]["layer0"]["remote_experts_used"]
        == remote_used
    )
    manager.forward_enqueues = 7
    manager.record_expert_usage("layer0", topk_ids[:0])
    latest = manager.metrics()["expert_usage"]["layer0"]
    assert latest["forward_enqueue"] == 8
    assert latest["remote_experts_used"] == latest["local_experts_used"] == 0


def test_expert_usage_disabled_does_not_inspect_routing(monkeypatch):
    manager = DWDPWeightManager(0, 4, torch.device("cuda:0"))
    monkeypatch.setattr(torch, "unique", lambda *_: pytest.fail("Collected routing"))
    manager.record_expert_usage("unregistered", None)
    assert manager.metrics()["expert_usage"] == {}


@pytest.mark.parametrize("enabled", [False, True])
def test_expert_usage_observes_existing_routing_before_weight_wait(enabled):
    from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
        UnquantizedFusedMoEMethod,
    )

    calls = []
    ids = torch.tensor([[0, 2], [2, 3]])
    routing_weights = torch.ones_like(ids, dtype=torch.float32)
    x = torch.ones(2, 4)
    w1, w2 = object(), object()

    def record(name, topk_ids):
        assert name == "experts" and topk_ids is ids
        calls.append("count")

    def weights(name, tokens):
        assert name == "experts" and tokens == 2
        calls.append("weights")
        return nullcontext((w1, w2))

    def apply(**kwargs):
        assert kwargs["topk_ids"] is ids
        assert kwargs["topk_weights"] is routing_weights
        assert kwargs["w1"] is w1 and kwargs["w2"] is w2
        calls.append("compute")
        return x

    layer = SimpleNamespace(
        dwdp_manager=SimpleNamespace(
            collect_expert_usage=enabled, record_expert_usage=record, layer=weights
        ),
        layer_name="experts",
        activation=None,
        apply_router_weight_on_input=False,
        global_num_experts=4,
        expert_map=None,
    )
    method = SimpleNamespace(moe_kernel=SimpleNamespace(apply=apply))
    result = UnquantizedFusedMoEMethod.forward_native(
        method, layer, x, routing_weights, ids, None, None
    )
    assert result is x
    assert calls == (["count"] if enabled else []) + ["weights", "compute"]


def test_expert_usage_rejects_graph_capture_configuration():
    with pytest.raises(ValueError, match="expert usage.*CUDA graphs"):
        DWDPConfig(collect_expert_usage=True, allow_cuda_graph=True)
    with pytest.raises(ValueError, match="expert usage.*CUDA graphs"):
        DWDPWeightManager(
            0,
            4,
            torch.device("cuda:0"),
            collect_expert_usage=True,
            allow_cuda_graph=True,
        )


def test_expert_usage_rejects_hidden_monolithic_routing(monkeypatch):
    from vllm.model_executor.layers.fused_moe import unquantized_fused_moe_method as mod
    from vllm.model_executor.offloader.dwdp import DWDPWeightOffloader

    offloader = DWDPWeightOffloader(DWDPConfig(collect_expert_usage=True))
    monkeypatch.setattr(mod, "get_offloader", lambda: offloader)
    with pytest.raises(ValueError, match="externally routed experts"):
        mod.UnquantizedFusedMoEMethod.create_weights(
            SimpleNamespace(is_monolithic=True), None, 8, 4, 4, torch.bfloat16
        )
