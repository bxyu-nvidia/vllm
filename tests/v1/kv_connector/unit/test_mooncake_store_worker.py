# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib
import ctypes
import itertools
import json
import logging
import math
import queue
import sys
import threading
import types
from concurrent.futures import Future, ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from tests.v1.attention.utils import dense_kv_cache_views
from vllm.distributed import mooncake_store
from vllm.distributed.kv_events import KVEventAggregator
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake import (
    rdma_utils,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store import (
    worker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store import (
    worker as mooncake_store_worker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.data import (
    BlobBlockHashes,
    ChunkedTokenDatabase,
    KeyMetadata,
    LBHNCStoreLayout,
    LBNHCStoreLayout,
    LoadSpec,
    MooncakeLookupResult,
    PoolKey,
    RankLocalStoreLayout,
    ReqMeta,
    TailKeyBoundary,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.metrics import (
    MooncakeStoreConnectorStats,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.pipeline import (
    PIPELINE_KEY,
    PipelineTracker,
)
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID
from vllm.v1.core.kv_cache_utils import BlockHash, maybe_convert_block_hash
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
    RSWASpec,
)
from vllm.v1.kv_cache_layout import KVCacheLayout

_TP_SHARED_NAMESPACE = "@store_tp:4@store_pp:1@store_format:tp_shared_lbhnc"
_TP_SHARED_LBNHC_NAMESPACE = "@store_tp:4@store_pp:1@store_format:tp_shared_lbnhc"


def _tp_shared_prefix(tp_rank: int) -> str:
    return (
        f"test-model{_TP_SHARED_NAMESPACE}@tp_rank:{tp_rank}"
        "@pcp0@dcp0@pp_rank:0@group:0"
    )


class _RecordingBlockHashes:
    def __init__(self, values: list[bytes]):
        self.values = values
        self.accessed: list[int] = []

    def __len__(self):
        return len(self.values)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return [self[i] for i in range(*idx.indices(len(self)))]
        if idx < 0:
            idx += len(self.values)
        if not 0 <= idx < len(self.values):
            raise IndexError(idx)
        self.accessed.append(idx)
        return self.values[idx]


def _default_send_coord() -> mooncake_store_worker.MooncakeStoreCoordinator:
    from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheGroupSpec

    spec = FullAttentionSpec(block_size=16, num_kv_heads=8, head_size=64, dtype=None)
    return mooncake_store_worker.MooncakeStoreCoordinator(
        [KVCacheGroupSpec(["layer0"], spec)],
        scheduler_block_size=16,
        hash_block_size=16,
    )


def _make_tp_shared_db(
    tp_rank: int = 0,
) -> tuple[ChunkedTokenDatabase, torch.Tensor]:
    num_blocks = 2
    local_heads = 4
    block_size = 16
    content_size = 4
    strides = (
        local_heads * block_size * content_size,
        block_size * content_size,
        content_size,
        1,
    )
    tensor = torch.empty_strided(
        (num_blocks, local_heads, block_size, content_size),
        strides,
        dtype=torch.float16,
    )
    metadata = KeyMetadata(
        "test-model",
        tp_rank,
        0,
        0,
        0,
        store_namespace=_TP_SHARED_NAMESPACE,
    )
    layout = LBHNCStoreLayout(
        metadata,
        block_size,
        block_size,
        local_tp_size=2,
        store_tp_size=4,
        tp_rank=tp_rank,
        num_kv_heads=8,
    )
    db = ChunkedTokenDatabase(metadata, block_size, store_layout=layout)
    layout.register_kv_caches([tensor], num_blocks)
    return db, tensor


def _make_store_sending_thread(
    store: MagicMock,
    *,
    coord: mooncake_store_worker.MooncakeStoreCoordinator | None = None,
    token_databases: list[ChunkedTokenDatabase] | None = None,
    block_size: int = 16,
    tp_rank: int = 0,
    put_step: int = 1,
    replicate_config: object | None = None,
    enable_group_semantics: bool = False,
    supports_group_ids: bool = False,
    external_cache_only: bool = False,
    lookup_key_prefixes: list[tuple[str, ...]] | None = None,
) -> mooncake_store_worker.KVCacheStoreSendingThread:
    if coord is None:
        coord = _default_send_coord()
    if token_databases is None:
        db = ChunkedTokenDatabase(KeyMetadata("test-model", 0, 0, 0, 0), block_size=16)
        db.set_kv_caches_base_addr([0x1000])
        db.set_block_len([256])
        token_databases = [db]
    thread = mooncake_store_worker.KVCacheStoreSendingThread(
        store=store,
        token_databases=token_databases,
        block_size=block_size,
        coord=coord,
        tp_rank=tp_rank,
        group_put_steps=[put_step] * len(token_databases),
        kv_role="kv_producer",
        ready_event=threading.Event(),
        replicate_config=replicate_config,
        enable_group_semantics=enable_group_semantics,
        supports_group_ids=supports_group_ids,
        external_cache_only=external_cache_only,
        lookup_key_prefixes=lookup_key_prefixes,
    )
    thread.request_queue.task_done = MagicMock()
    return thread


def _make_store_recving_thread(
    store: MagicMock,
    *,
    tp_rank: int = 0,
    disk_offload_buffer_budget_bytes: int | None = None,
    is_hma_required: bool = False,
) -> mooncake_store_worker.KVCacheStoreRecvingThread:
    from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheGroupSpec

    token_database = ChunkedTokenDatabase(
        KeyMetadata("test-model", 0, 0, 0, 0), block_size=16
    )
    token_database.set_kv_caches_base_addr([0x1000])
    token_database.set_block_len([256])
    spec = FullAttentionSpec(block_size=16, num_kv_heads=8, head_size=64, dtype=None)
    coord = mooncake_store_worker.MooncakeStoreCoordinator(
        [KVCacheGroupSpec(["layer0"], spec)],
        scheduler_block_size=16,
        hash_block_size=16,
    )
    thread = mooncake_store_worker.KVCacheStoreRecvingThread(
        store=store,
        token_databases=[token_database],
        block_size=16,
        tp_rank=tp_rank,
        ready_event=threading.Event(),
        coord=coord,
        disk_offload_buffer_budget_bytes=disk_offload_buffer_budget_bytes,
        is_hma_required=is_hma_required,
    )
    thread.request_queue.task_done = MagicMock()
    return thread


def _make_load_req(
    req_id: str,
    block_hashes: list[bytes],
    *,
    token_len: int,
    vllm_cached_tokens: int = 0,
    tail_key_boundaries: tuple[TailKeyBoundary, ...] = (),
) -> ReqMeta:
    return ReqMeta(
        req_id=req_id,
        token_len_chunk=token_len,
        block_ids=(list(range(len(block_hashes))),),
        block_hashes=block_hashes,
        load_spec=LoadSpec(
            vllm_cached_tokens=vllm_cached_tokens,
            kvpool_cached_tokens=token_len,
            can_load=True,
            token_len=token_len,
            tail_key_boundaries=tail_key_boundaries,
        ),
    )


_TEST_SAVE_SEQ = itertools.count(1)


def _run_store_req(thread, req_meta: ReqMeta) -> None:
    """Register, enqueue and run a store job the way the worker does."""
    if req_meta.store_job_id is None:
        req_meta.store_job_id = next(_TEST_SAVE_SEQ)
    thread.add_request(req_meta)
    thread._handle_request(thread.request_queue.get())


def _make_store_req(req_id: str, block_hashes: list[bytes]) -> ReqMeta:
    # 1-based: block 0 is the reserved null block and the save skips it.
    return ReqMeta(
        req_id=req_id,
        token_len_chunk=32,
        block_ids=([1, 2],),
        block_hashes=block_hashes,
        can_save=True,
    )


def _make_multi_group_store_req(req_id: str, block_hashes: list[bytes]) -> ReqMeta:
    return ReqMeta(
        req_id=req_id,
        token_len_chunk=32,
        block_ids=([1, 2], [3, 4]),
        block_hashes=block_hashes,
        can_save=True,
    )


_DISK_OFFLOAD_SINGLE_KEY_BYTES = worker._estimate_disk_offload_staging_bytes([256])
_DISK_OFFLOAD_USABLE_BUDGET_RATIO = 0.9
_DISK_OFFLOAD_BUDGET_FOR_THREE_KEYS = 4 * _DISK_OFFLOAD_SINGLE_KEY_BYTES
_DISK_OFFLOAD_BUDGET_FOR_SPLIT = math.ceil(
    2 * _DISK_OFFLOAD_SINGLE_KEY_BYTES / _DISK_OFFLOAD_USABLE_BUDGET_RATIO
)  # Allows two 256-byte chunks but not the third.
_DISK_OFFLOAD_BUDGET_TOO_SMALL = (
    _DISK_OFFLOAD_SINGLE_KEY_BYTES - 1
)  # Smaller than a single 256-byte chunk.


class _FakeKVTransferConfig:
    def __init__(
        self,
        *,
        kv_role: str = "kv_both",
        extra_config: dict[str, object] | None = None,
    ) -> None:
        self.kv_role = kv_role
        self.kv_connector_extra_config = extra_config or {}

    def get_from_extra_config(self, key: str, default: object) -> object:
        return self.kv_connector_extra_config.get(key, default)


class _FakeModelConfig:
    model = "test-model"
    use_mla = False
    enable_sleep_mode = False
    enable_cumem_allocator = False

    def get_num_layers(self, parallel_config) -> int:
        return 1

    def get_total_num_kv_heads(self) -> int:
        return 1


def _make_vllm_config(
    *,
    extra_config: dict[str, object] | None = None,
    rank: int = 0,
    decode_context_parallel_size: int = 1,
    kv_role: str = "kv_both",
    pipeline_parallel_size: int = 1,
    kv_cache_layout: KVCacheLayout = KVCacheLayout.LBHNC,
    disable_hybrid_kv_cache_manager: bool = True,
) -> SimpleNamespace:
    cache_config = SimpleNamespace(block_size=16, num_gpu_blocks=10)
    cache_config.get_resolved_kv_cache_layout = lambda: kv_cache_layout
    return SimpleNamespace(
        model_config=_FakeModelConfig(),
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=pipeline_parallel_size,
            rank=rank,
            data_parallel_index=0,
            decode_context_parallel_size=decode_context_parallel_size,
            prefill_context_parallel_size=1,
        ),
        kv_transfer_config=_FakeKVTransferConfig(
            kv_role=kv_role, extra_config=extra_config
        ),
        cache_config=cache_config,
        scheduler_config=SimpleNamespace(
            disable_hybrid_kv_cache_manager=disable_hybrid_kv_cache_manager
        ),
        kv_events_config=SimpleNamespace(enable_kv_cache_events=False),
        speculative_config=None,
    )


def _make_kv_cache_config(
    *, block_size: int = 16, prefix_cache_retention_interval: int | None = 0
) -> object:
    """Minimal single-group KVCacheConfig for topology tests."""
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        KVCacheConfig,
        KVCacheGroupSpec,
    )

    spec = FullAttentionSpec(
        block_size=block_size, num_kv_heads=8, head_size=64, dtype=None
    )
    return KVCacheConfig(
        num_blocks=10,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(["layer0"], spec)],
        prefix_cache_retention_interval=prefix_cache_retention_interval,
    )


def _make_qsa_hybrid_kv_cache_config():
    return KVCacheConfig(
        num_blocks=10,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["full"],
                FullAttentionSpec(
                    block_size=800,
                    num_kv_heads=8,
                    head_size=64,
                    dtype=None,
                ),
            ),
            KVCacheGroupSpec(
                ["qsa"],
                CircularBufferSpec(
                    block_size=8,
                    num_kv_heads=1,
                    head_size=64,
                    head_size_v=0,
                    dtype=torch.float16,
                ),
            ),
            KVCacheGroupSpec(
                ["mamba"],
                MambaSpec(
                    block_size=800,
                    shapes=((1, 1),),
                    dtypes=(torch.float32,),
                    mamba_cache_mode="align",
                ),
            ),
        ],
    )


def _write_mooncake_config(tmp_path, config: dict[str, object]) -> str:
    config_path = tmp_path / "mooncake_config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return str(config_path)


def _install_fake_mooncake(monkeypatch, store_instance: MagicMock):
    class FakeReplicateConfig:
        def __init__(self) -> None:
            self.preferred_segment = ""

    fake_store_module = types.ModuleType("mooncake.store")
    fake_store_module.MooncakeDistributedStore = lambda: store_instance  # type: ignore[attr-defined]
    fake_store_module.ReplicateConfig = FakeReplicateConfig  # type: ignore[attr-defined]
    fake_store_module.ObjectDataType = SimpleNamespace(TENSOR=1)  # type: ignore[attr-defined]
    fake_mooncake_module = types.ModuleType("mooncake")
    fake_mooncake_module.store = fake_store_module  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mooncake", fake_mooncake_module)
    monkeypatch.setitem(sys.modules, "mooncake.store", fake_store_module)
    return FakeReplicateConfig


def _patch_worker_runtime(
    monkeypatch,
    *,
    local_ip: str = "10.0.0.7",
    tp_rank: int = 0,
    tp_size: int = 1,
    pcp_size: int = 1,
    dcp_size: int = 1,
) -> None:
    pcp_group = SimpleNamespace(world_size=pcp_size, rank_in_group=0)
    # DCP groups are contiguous splits of the TP group (see
    # parallel_state.py), so dcp_rank == tp_rank % dcp_size.
    dcp_group = SimpleNamespace(world_size=dcp_size, rank_in_group=tp_rank % dcp_size)
    monkeypatch.setattr(worker, "get_tensor_model_parallel_rank", lambda: tp_rank)
    monkeypatch.setattr(worker, "get_tensor_model_parallel_world_size", lambda: tp_size)
    monkeypatch.setattr(worker, "get_pcp_group", lambda: pcp_group)
    monkeypatch.setattr(worker, "get_dcp_group", lambda: dcp_group)
    monkeypatch.setattr(worker, "get_ip", lambda: local_ip)
    monkeypatch.setattr(worker, "LookupKeyServer", MagicMock())


def test_create_mem_pool_returns_none_when_unconfigured():
    assert worker.MooncakeStoreWorker._create_mem_pool({}, _FakeModelConfig()) is None


def test_create_mem_pool_rejects_unknown_pool():
    with pytest.raises(
        ValueError,
        match="Unsupported custom_mem_pool='UNKNOWN'",
    ):
        worker.MooncakeStoreWorker._create_mem_pool(
            {"custom_mem_pool": "unknown"}, _FakeModelConfig()
        )


@pytest.mark.parametrize(
    "conflicting_option", ["enable_sleep_mode", "enable_cumem_allocator"]
)
def test_create_mem_pool_rejects_cumem_conflicts(conflicting_option):
    model_config = _FakeModelConfig()
    setattr(model_config, conflicting_option, True)

    with pytest.raises(ValueError, match="custom_mem_pool is incompatible"):
        worker.MooncakeStoreWorker._create_mem_pool(
            {"custom_mem_pool": "NVLINK"}, model_config
        )


@pytest.mark.parametrize(
    ("pool_type", "allocator_name"),
    [("nvlink", "NVLinkAllocator"), ("barex", "BarexAllocator")],
)
def test_create_mem_pool_uses_mooncake_allocator(
    monkeypatch, pool_type, allocator_name
):
    allocator_cls = MagicMock()
    allocator = allocator_cls.get_allocator.return_value
    allocator_module = types.ModuleType("mooncake.allocator")
    setattr(allocator_module, allocator_name, allocator_cls)
    mooncake_module = types.ModuleType("mooncake")
    mooncake_module.allocator = allocator_module  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mooncake", mooncake_module)
    monkeypatch.setitem(sys.modules, "mooncake.allocator", allocator_module)

    monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: 2)
    mem_pool = MagicMock()
    mem_pool_ctor = MagicMock(return_value=mem_pool)
    monkeypatch.setattr(torch.cuda, "MemPool", mem_pool_ctor)

    result = worker.MooncakeStoreWorker._create_mem_pool(
        {"custom_mem_pool": pool_type}, _FakeModelConfig()
    )

    assert result is mem_pool
    allocator_cls.get_allocator.assert_called_once_with(torch.device("cuda", 2))
    allocator.allocator.assert_called_once_with()
    mem_pool_ctor.assert_called_once_with(allocator.allocator.return_value)


def test_pool_key_to_string_without_prefix_is_unchanged():
    """Default (empty) cache_prefix keeps keys byte-identical to the
    historical unprefixed format so existing deployments keep their hits."""
    key = PoolKey(KeyMetadata("test-model", 0, 0, 0, 0), "deadbeef")
    assert (
        key.to_string() == "test-model@tp_rank:0@pcp0@dcp0@pp_rank:0@group:0@deadbeef"
    )


def test_pool_key_store_tp_namespace():
    key = PoolKey(
        KeyMetadata(
            "test-model",
            2,
            0,
            0,
            0,
            store_namespace=_TP_SHARED_NAMESPACE,
        ),
        "deadbeef",
    )

    assert key.to_string() == f"{_tp_shared_prefix(2)}@deadbeef"


def test_tp_shared_sending_writes_each_store_shard():
    store = MagicMock()
    store.batch_is_exist.return_value = [0, 0, 0, 0]
    store.batch_put_from_multi_buffers.return_value = [1, 1, 1, 1]
    db, _ = _make_tp_shared_db()
    thread = _make_store_sending_thread(
        store,
        token_databases=[db],
        replicate_config=SimpleNamespace(),
    )
    req = _make_store_req("req", [b"h0", b"h1"])
    _run_store_req(thread, req)

    keys, addrs, sizes, _ = store.batch_put_from_multi_buffers.call_args.args
    assert keys == [
        f"{_tp_shared_prefix(0)}@6830",
        f"{_tp_shared_prefix(1)}@6830",
        f"{_tp_shared_prefix(0)}@6831",
        f"{_tp_shared_prefix(1)}@6831",
    ]
    assert len(addrs) == len(sizes) == 4


def test_tp_shared_kv_events_aggregate_once_per_logical_block():
    aggregator = KVEventAggregator(num_workers=2)

    for tp_rank in range(2):
        store = MagicMock()
        store.batch_is_exist.return_value = [0, 0]
        store.batch_put_from_multi_buffers.return_value = [256, 256]
        db, _ = _make_tp_shared_db(tp_rank)
        thread = _make_store_sending_thread(
            store,
            token_databases=[db],
            tp_rank=tp_rank,
            replicate_config=SimpleNamespace(),
        )
        thread.enable_kv_event = True

        _run_store_req(thread, _make_event_store_req(16))

        events = thread.get_kv_events()
        assert len(events) == 1
        aggregator.add_events(events)

    assert len(aggregator.get_common_events()) == 1


def test_tp_shared_kv_event_waits_for_all_missing_shards():
    store = MagicMock()
    store.batch_is_exist.return_value = [0, 0]
    store.batch_put_from_multi_buffers.return_value = [256, -1]
    db, _ = _make_tp_shared_db()
    thread = _make_store_sending_thread(
        store,
        token_databases=[db],
        replicate_config=SimpleNamespace(),
    )
    thread.enable_kv_event = True

    _run_store_req(thread, _make_event_store_req(16))

    assert thread.get_kv_events() == []


def test_tp_shared_kv_event_counts_existing_shards_as_satisfied():
    store = MagicMock()
    store.batch_is_exist.return_value = [1, 0]
    store.batch_put_from_multi_buffers.return_value = [256]
    db, _ = _make_tp_shared_db()
    thread = _make_store_sending_thread(
        store,
        token_databases=[db],
        replicate_config=SimpleNamespace(),
    )
    thread.enable_kv_event = True

    _run_store_req(thread, _make_event_store_req(16))

    assert len(thread.get_kv_events()) == 1


def test_tp_shared_receiving_reads_each_local_store_shard():
    from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheGroupSpec

    store = MagicMock()
    store.batch_get_into_multi_buffers.return_value = [1, 1, 1, 1]
    db, _ = _make_tp_shared_db()
    spec = FullAttentionSpec(block_size=16, num_kv_heads=8, head_size=64, dtype=None)
    coord = mooncake_store_worker.MooncakeStoreCoordinator(
        [KVCacheGroupSpec(["layer0"], spec)],
        scheduler_block_size=16,
        hash_block_size=16,
    )
    thread = mooncake_store_worker.KVCacheStoreRecvingThread(
        store=store,
        token_databases=[db],
        block_size=16,
        tp_rank=0,
        ready_event=threading.Event(),
        coord=coord,
    )
    thread.request_queue.task_done = MagicMock()

    thread._handle_request(_make_load_req("req", [b"h0", b"h1"], token_len=32))

    keys, addrs, sizes = store.batch_get_into_multi_buffers.call_args.args
    assert keys == [
        f"{_tp_shared_prefix(0)}@6830",
        f"{_tp_shared_prefix(1)}@6830",
        f"{_tp_shared_prefix(0)}@6831",
        f"{_tp_shared_prefix(1)}@6831",
    ]
    assert len(addrs) == len(sizes) == 4


def test_pool_key_cache_prefix_namespaces_and_disambiguates():
    """A non-empty cache_prefix is prepended, and two instances with
    different prefixes never collide on identical block hashes."""
    md_a = KeyMetadata("test-model", 0, 0, 0, 0, cache_prefix="depA")
    md_b = KeyMetadata("test-model", 0, 0, 0, 0, cache_prefix="depB")

    key_a = PoolKey(md_a, "deadbeef")
    key_b = PoolKey(md_b, "deadbeef")

    assert key_a.to_string() == (
        "depA@test-model@tp_rank:0@pcp0@dcp0@pp_rank:0@group:0@deadbeef"
    )
    assert key_a.to_string() != key_b.to_string()
    assert hash(key_a) != hash(key_b)


def test_default_local_buffer_size_matches_pr40900():
    """PR-40900 shipped a 4 GiB default for local_buffer_size; the dual-mode
    patch preserves it (and the JSON key) so unchanged PR-40900 configs work."""
    assert mooncake_store.DEFAULT_LOCAL_BUFFER_SIZE == 4 * 1024**3


def test_get_requester_local_hostname_prefers_override(monkeypatch):
    monkeypatch.setenv("MOONCAKE_REQUESTER_LOCAL_HOSTNAME", "worker-a:50053")

    assert rdma_utils.get_requester_local_hostname("10.0.0.7") == "worker-a:50053"


def test_get_configured_preferred_segment_returns_explicit_override():
    assert (
        rdma_utils.get_configured_preferred_segment(
            {"preferred_segment": "10.0.0.7:50053"}
        )
        == "10.0.0.7:50053"
    )


def test_get_configured_preferred_segment_prefers_explicit_over_env(monkeypatch):
    monkeypatch.setenv("MOONCAKE_PREFERRED_SEGMENT", "10.0.0.8:50053")

    assert (
        rdma_utils.get_configured_preferred_segment(
            {"preferred_segment": "10.0.0.7:50053"}
        )
        == "10.0.0.7:50053"
    )


def test_get_configured_preferred_segment_returns_env_override(monkeypatch):
    monkeypatch.setenv("MOONCAKE_PREFERRED_SEGMENT", "10.0.0.8:50053")

    assert rdma_utils.get_configured_preferred_segment({}) == "10.0.0.8:50053"


def test_get_configured_preferred_segment_rejects_empty_override():
    with pytest.raises(ValueError, match="preferred_segment"):
        rdma_utils.get_configured_preferred_segment({"preferred_segment": "  "})


class _ReplicaDesc:
    def __init__(self, tier: str):
        self.tier = tier

    def is_memory_replica(self) -> bool:
        return self.tier == "memory"

    def is_disk_replica(self) -> bool:
        return self.tier == "disk"

    def is_local_disk_replica(self) -> bool:
        return self.tier == "disk"


def test_store_sending_thread_skips_request_during_cpu_pressure():
    store = MagicMock()
    store.batch_is_exist.side_effect = lambda keys: [0] * len(keys)
    store.batch_put_from_multi_buffers.side_effect = [
        [-200, -200],
        [256, 256],
        [256, 256],
    ]
    thread = _make_store_sending_thread(store)

    _run_store_req(thread, _make_store_req("req-a", [b"a0", b"a1"]))

    assert thread._store_pressure_active is True
    assert "req-a" in thread._skip_store_requests
    assert store.batch_put_from_multi_buffers.call_count == 1

    _run_store_req(thread, _make_store_req("req-a", [b"a2", b"a3"]))

    assert store.batch_put_from_multi_buffers.call_count == 1

    _run_store_req(thread, _make_store_req("req-b", [b"b0", b"b1"]))

    assert thread._store_pressure_active is False
    assert "req-a" not in thread._skip_store_requests
    assert store.batch_put_from_multi_buffers.call_count == 2

    _run_store_req(thread, _make_store_req("req-a", [b"a4", b"a5"]))

    assert store.batch_put_from_multi_buffers.call_count == 3


def test_store_sending_thread_records_mooncake_metrics():
    store = MagicMock()
    store.batch_is_exist.return_value = [0, 0]
    store.batch_put_from_multi_buffers.return_value = [256, 256]
    thread = _make_store_sending_thread(store)
    stats = MooncakeStoreConnectorStats()
    thread._record_operation_cb = stats.record_operation

    _run_store_req(thread, _make_store_req("req-a", [b"a0", b"a1"]))

    assert len(stats.data["save_exists"]) == 1
    assert stats.data["save_exists"][0]["num_keys"] == 2
    assert len(stats.data["save_put"]) == 1
    assert stats.data["save_put"][0]["num_bytes"] == 512
    assert stats.data["save_put"][0]["status"] == "ok"


def test_process_tokens_uses_mask_num_as_start_chunk():
    db = ChunkedTokenDatabase(
        KeyMetadata("test-model", 0, 0, 0, 0),
        block_size=32,
        hash_block_size=8,
    )
    block_hashes = _RecordingBlockHashes([bytes([i]) for i in range(16)])

    results = list(
        db.process_tokens(
            token_len=96,
            block_hashes=block_hashes,
            mask_num=64,
        )
    )

    assert block_hashes.accessed == [11]
    assert results == [(64, 96, bytes([11]))]


def test_process_tokens_applies_chunk_mask_before_hash_access():
    db = ChunkedTokenDatabase(
        KeyMetadata("test-model", 0, 0, 0, 0),
        block_size=32,
        hash_block_size=8,
    )
    block_hashes = _RecordingBlockHashes([bytes([i]) for i in range(16)])

    results = list(
        db.process_tokens(
            token_len=128,
            block_hashes=block_hashes,
            mask_num=64,
            chunk_mask=[False, True],
        )
    )

    assert block_hashes.accessed == [15]
    assert results == [(96, 128, bytes([15]))]


def test_process_tokens_applies_stride_before_hash_access():
    db = ChunkedTokenDatabase(
        KeyMetadata("test-model", 0, 0, 0, 0),
        block_size=32,
        hash_block_size=8,
    )
    block_hashes = _RecordingBlockHashes([bytes([i]) for i in range(16)])

    results = list(
        db.process_tokens(
            token_len=128,
            block_hashes=block_hashes,
            mask_num=64,
            put_step=2,
            put_step_rank=1,
        )
    )

    assert block_hashes.accessed == [15]
    assert results == [(96, 128, bytes([15]))]


def test_store_sending_thread_delta_saves_only_new_full_attention_chunks():
    store = MagicMock()
    store.batch_is_exist.side_effect = lambda keys: [0] * len(keys)
    store.batch_put_from_multi_buffers.return_value = [256, 256]
    thread = _make_store_sending_thread(store)

    thread._saved_offset["req-a"] = 32
    _run_store_req(
        thread,
        ReqMeta(
            req_id="req-a",
            token_len_chunk=64,
            block_ids=([1, 2, 3, 4],),
            block_hashes=[b"a0", b"a1", b"a2", b"a3"],
            can_save=True,
        ),
    )

    keys = store.batch_is_exist.call_args.args[0]
    assert keys == [
        "test-model@tp_rank:0@pcp0@dcp0@pp_rank:0@group:0@6132",
        "test-model@tp_rank:0@pcp0@dcp0@pp_rank:0@group:0@6133",
    ]
    assert store.batch_put_from_multi_buffers.call_args.args[0] == keys


def test_store_sending_thread_delta_strides_with_local_phase():
    store = MagicMock()
    store.batch_is_exist.side_effect = lambda keys: [0] * len(keys)
    store.batch_put_from_multi_buffers.return_value = [256]
    thread = _make_store_sending_thread(store, tp_rank=0, put_step=2)

    thread._saved_offset["req-a"] = 16
    _run_store_req(
        thread,
        ReqMeta(
            req_id="req-a",
            token_len_chunk=64,
            block_ids=([1, 2, 3, 4],),
            block_hashes=[b"a0", b"a1", b"a2", b"a3"],
            can_save=True,
        ),
    )

    keys = store.batch_is_exist.call_args.args[0]
    assert keys == [
        "test-model@tp_rank:0@pcp0@dcp0@pp_rank:0@group:0@6132",
    ]
    assert store.batch_put_from_multi_buffers.call_args.args[0] == keys


def test_tp_sharded_group_saves_every_block_on_every_rank():
    """Sharded ranks must write every block because peers hold different bytes."""
    store = MagicMock()
    store.batch_is_exist.side_effect = lambda keys: [0] * len(keys)
    store.batch_put_from_multi_buffers.side_effect = lambda keys, *a: [256] * len(keys)
    thread = _make_store_sending_thread(store, tp_rank=0, put_step=2)
    thread.group_put_steps = [1]

    _run_store_req(
        thread,
        ReqMeta(
            req_id="req-a",
            token_len_chunk=64,
            block_ids=([1, 2, 3, 4],),
            block_hashes=[b"a0", b"a1", b"a2", b"a3"],
            can_save=True,
        ),
    )

    keys = store.batch_is_exist.call_args.args[0]
    assert len(keys) == 4


def test_store_sending_thread_retries_skipped_range_after_pressure():
    store = MagicMock()
    store.batch_is_exist.side_effect = lambda keys: [0] * len(keys)
    store.batch_put_from_multi_buffers.side_effect = lambda keys, *a: [256] * len(keys)
    thread = _make_store_sending_thread(store, tp_rank=0, put_step=2)

    # Under pressure the request is skipped without persisting anything, so the
    # saved offset stays at 0.
    thread._store_pressure_active = True
    thread._skip_store_requests.add("req-a")

    _run_store_req(
        thread,
        ReqMeta(
            req_id="req-a",
            token_len_chunk=16,
            block_ids=([1],),
            block_hashes=[b"a0"],
            can_save=True,
        ),
    )

    store.batch_is_exist.assert_not_called()
    assert thread._saved_offset.get("req-a", 0) == 0

    thread._store_pressure_active = False
    thread._skip_store_requests.clear()

    # The next batch resumes from offset 0, re-covering the chunk skipped under
    # pressure (chunk 0) rather than losing it.
    _run_store_req(
        thread,
        ReqMeta(
            req_id="req-a",
            token_len_chunk=64,
            block_ids=([1, 2, 3, 4],),
            block_hashes=[b"a0", b"a1", b"a2", b"a3"],
            can_save=True,
        ),
    )

    keys = store.batch_is_exist.call_args.args[0]
    assert keys == [
        "test-model@tp_rank:0@pcp0@dcp0@pp_rank:0@group:0@6130",
        "test-model@tp_rank:0@pcp0@dcp0@pp_rank:0@group:0@6132",
    ]
    assert store.batch_put_from_multi_buffers.call_args.args[0] == keys


def _make_partial_tail_send_thread(
    store,
    *,
    replicate_config=None,
    enable_group_semantics=False,
    supports_group_ids=False,
):
    coord = SimpleNamespace(
        enable_partial_hash_hits=True,
        hash_block_size=4,
        lcm_block_size=16,
        mamba_group_ids={1},
    )
    db = ChunkedTokenDatabase(
        KeyMetadata("test-model", 0, 0, 0, 0),
        block_size=4,
        hash_block_size=4,
    )
    db.set_kv_caches_base_addr([0x1000])
    db.set_block_len([256])
    # Group 1 models the mamba "align" group the hand-offs reference.
    db_mamba = ChunkedTokenDatabase(
        KeyMetadata("test-model", 0, 0, 0, 0, group_id=1),
        block_size=16,
        hash_block_size=4,
    )
    db_mamba.set_kv_caches_base_addr([0x2000])
    db_mamba.set_block_len([256])
    return _make_store_sending_thread(
        store,
        coord=coord,
        token_databases=[db, db_mamba],
        replicate_config=replicate_config,
        enable_group_semantics=enable_group_semantics,
        supports_group_ids=supports_group_ids,
    )


def _make_partial_tail_req(block_ids: list[int]) -> ReqMeta:
    return ReqMeta(
        req_id="req-a",
        token_len_chunk=0,
        block_ids=(block_ids, [1]),
        block_hashes=[b"a0", b"a1", b"a2"],
        can_save=True,
        boundary_state_offloads=[(1, 7, 12)],
    )


def test_partial_tail_offload_skips_null_source_blocks():
    store = MagicMock()
    store.batch_is_exist.side_effect = lambda keys: [0] * len(keys)
    store.batch_put_from_multi_buffers.side_effect = lambda keys, *a: [256] * len(keys)
    thread = _make_partial_tail_send_thread(store)

    assert thread._maybe_offload_boundary_states(_make_partial_tail_req([0, 2, 3]))

    keys, addrs, _sizes, _replicate_config = (
        store.batch_put_from_multi_buffers.call_args.args
    )
    assert keys == [
        "test-model@tp_rank:0@pcp0@dcp0@pp_rank:0@group:0@6131",
        "test-model@tp_rank:0@pcp0@dcp0@pp_rank:0@group:0@6132",
        "test-model@tp_rank:0@pcp0@dcp0@pp_rank:0@group:1@6132",
    ]
    # FA reads blocks 2 and 3 (block 0 is null and skipped); the mamba
    # boundary block reads the core-provided CoW block 7, not block_ids.
    assert addrs == [
        [0x1000 + 2 * 256],
        [0x1000 + 3 * 256],
        [0x2000 + 7 * 256],
    ]


def test_store_sending_thread_skips_null_sparse_group_blocks():
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        KVCacheGroupSpec,
    )

    store = MagicMock()
    store.batch_is_exist.side_effect = lambda keys: [0] * len(keys)
    store.batch_put_from_multi_buffers.side_effect = (
        lambda keys, addrs, sizes, replicate_config: [256] * len(keys)
    )
    full = FullAttentionSpec(block_size=16, num_kv_heads=8, head_size=64, dtype=None)
    sparse = FullAttentionSpec(block_size=16, num_kv_heads=8, head_size=64, dtype=None)
    coord = mooncake_store_worker.MooncakeStoreCoordinator(
        [
            KVCacheGroupSpec(["full"], full),
            KVCacheGroupSpec(["sparse"], sparse),
        ],
        scheduler_block_size=16,
        hash_block_size=16,
    )

    db_full = ChunkedTokenDatabase(
        KeyMetadata("test-model", 0, 0, 0, 0, group_id=0),
        block_size=16,
    )
    db_full.set_kv_caches_base_addr([0x1000])
    db_full.set_block_len([256])
    db_sparse = ChunkedTokenDatabase(
        KeyMetadata("test-model", 0, 0, 0, 0, group_id=1),
        block_size=16,
    )
    db_sparse.set_kv_caches_base_addr([0x2000])
    db_sparse.set_block_len([512])

    thread = _make_store_sending_thread(
        store,
        coord=coord,
        token_databases=[db_full, db_sparse],
    )
    _run_store_req(
        thread,
        ReqMeta(
            req_id="req-a",
            token_len_chunk=48,
            block_ids=([1, 2, 3], [NULL_BLOCK_ID, 4, NULL_BLOCK_ID]),
            block_hashes=[b"a0", b"a1", b"a2"],
            can_save=True,
        ),
    )

    keys, addrs, _, _ = store.batch_put_from_multi_buffers.call_args.args
    sparse_keys = [key for key in keys if "@group:1" in key]
    assert [key.rsplit("@", 1)[-1] for key in sparse_keys] == [b"a1".hex()]
    assert all(addr[0] >= 0x2000 for key, addr in zip(keys, addrs) if "@group:1" in key)


def test_partial_tail_offload_skips_cap_omitted_mamba_group():
    """A Mamba group omitted by the handoff cap must not fall back to the
    connector's positional block table."""
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        KVCacheGroupSpec,
        MambaSpec,
    )

    store = MagicMock()
    store.batch_is_exist.side_effect = lambda keys: [0] * len(keys)
    store.batch_put_from_multi_buffers.side_effect = lambda keys, *a: [256] * len(keys)

    full_spec = FullAttentionSpec(
        block_size=4, num_kv_heads=8, head_size=64, dtype=torch.float32
    )
    mamba_spec = MambaSpec(
        block_size=16,
        shapes=((1, 1),),
        dtypes=(torch.float32,),
        mamba_cache_mode="align",
    )
    coord = mooncake_store_worker.MooncakeStoreCoordinator(
        [
            KVCacheGroupSpec(["full"], full_spec),
            KVCacheGroupSpec(["mamba-1"], mamba_spec),
            KVCacheGroupSpec(["mamba-2"], mamba_spec),
        ],
        scheduler_block_size=16,
        hash_block_size=4,
    )

    def make_db(group_id: int, block_size: int, base_addr: int):
        db = ChunkedTokenDatabase(
            KeyMetadata("test-model", 0, 0, 0, 0, group_id=group_id),
            block_size=block_size,
            hash_block_size=4,
        )
        db.set_kv_caches_base_addr([base_addr])
        db.set_block_len([256])
        return db

    db_full = make_db(0, 4, 0x1000)
    db_mamba_1 = make_db(1, 16, 0x2000)
    db_mamba_2 = make_db(2, 16, 0x3000)
    thread = _make_store_sending_thread(
        store,
        coord=coord,
        token_databases=[db_full, db_mamba_1, db_mamba_2],
    )

    req = ReqMeta(
        req_id="req-a",
        token_len_chunk=0,
        block_ids=([1, 2, 3], [5], [9]),
        block_hashes=[b"a0", b"a1", b"a2"],
        can_save=True,
        # The scheduler accepted group 1 and omitted group 2 at boundary 12.
        boundary_state_offloads=[(1, 7, 12)],
    )
    assert thread._maybe_offload_boundary_states(req)

    keys, addrs, _sizes, _replicate_config = (
        store.batch_put_from_multi_buffers.call_args.args
    )
    puts = list(zip(keys, addrs, strict=True))
    boundary_hash = BlockHash(b"a2")
    assert (db_mamba_1.key_for(boundary_hash), [0x2000 + 7 * 256]) in puts
    assert (db_mamba_2.key_for(boundary_hash), [0x3000 + 9 * 256]) not in puts


def test_partial_tail_offload_replaces_stale_group_ids_after_filtering():
    store = MagicMock()
    store.batch_is_exist.return_value = [1, 0, 0, 0]
    store.batch_put_from_multi_buffers.return_value = [256, 256, 256]
    replicate_config = SimpleNamespace(group_ids=["stale"])
    thread = _make_partial_tail_send_thread(
        store,
        replicate_config=replicate_config,
        enable_group_semantics=True,
        supports_group_ids=True,
    )

    assert thread._maybe_offload_boundary_states(_make_partial_tail_req([1, 2, 3]))

    keys, _addrs, _sizes, config = store.batch_put_from_multi_buffers.call_args.args
    assert keys == [
        "test-model@tp_rank:0@pcp0@dcp0@pp_rank:0@group:0@6131",
        "test-model@tp_rank:0@pcp0@dcp0@pp_rank:0@group:0@6132",
        "test-model@tp_rank:0@pcp0@dcp0@pp_rank:0@group:1@6132",
    ]
    assert config is replicate_config
    assert config.group_ids == [
        "vllm-mooncake-store:test-model@6131",
        "vllm-mooncake-store:test-model@6132",
        "vllm-mooncake-store:test-model@6132",
    ]


def test_partial_tail_offload_honors_active_pressure_gate():
    store = MagicMock()
    thread = _make_partial_tail_send_thread(store)
    thread._store_pressure_active = True
    thread._skip_store_requests.add("req-a")

    _run_store_req(thread, _make_partial_tail_req([1, 2, 3]))

    store.batch_is_exist.assert_not_called()
    store.batch_put_from_multi_buffers.assert_not_called()


def test_partial_tail_put_failure_activates_pressure_gate():
    store = MagicMock()
    store.batch_is_exist.side_effect = lambda keys: [0] * len(keys)
    store.batch_put_from_multi_buffers.return_value = [256, -200, 256]
    thread = _make_partial_tail_send_thread(store)

    _run_store_req(thread, _make_partial_tail_req([1, 2, 3]))

    assert thread._store_pressure_active is True
    assert thread._skip_store_requests == {"req-a"}
    assert thread._saved_offset.get("req-a", 0) == 0

    _run_store_req(thread, _make_partial_tail_req([1, 2, 3]))
    assert store.batch_put_from_multi_buffers.call_count == 1


def test_normal_save_excludes_mamba_group_and_null_blocks():
    """The positional normal save must never cover mamba chunks (align-mode
    block tables are not append-only) and must skip chunks whose block is the
    reserved null block, for any group."""
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        KVCacheGroupSpec,
        MambaSpec,
    )

    store = MagicMock()
    store.batch_is_exist.side_effect = lambda keys: [0] * len(keys)
    store.batch_put_from_multi_buffers.side_effect = lambda keys, *a: [256] * len(keys)

    full_spec = FullAttentionSpec(
        block_size=16, num_kv_heads=8, head_size=64, dtype=None
    )
    mamba_spec = MambaSpec(
        block_size=16,
        shapes=((1, 1),),
        dtypes=(torch.float32,),
        mamba_cache_mode="align",
    )
    coord = mooncake_store_worker.MooncakeStoreCoordinator(
        [KVCacheGroupSpec(["L0"], full_spec), KVCacheGroupSpec(["L1"], mamba_spec)],
        scheduler_block_size=16,
        hash_block_size=16,
    )
    db_full = ChunkedTokenDatabase(
        KeyMetadata("test-model", 0, 0, 0, 0, group_id=0), block_size=16
    )
    db_full.set_kv_caches_base_addr([0x1000])
    db_full.set_block_len([256])
    db_mamba = ChunkedTokenDatabase(
        KeyMetadata("test-model", 0, 0, 0, 0, group_id=1), block_size=16
    )
    db_mamba.set_kv_caches_base_addr([0x2000])
    db_mamba.set_block_len([256])
    thread = _make_store_sending_thread(
        store, coord=coord, token_databases=[db_full, db_mamba]
    )

    _run_store_req(
        thread,
        ReqMeta(
            req_id="r0",
            token_len_chunk=64,
            # FA has a null at chunk 1; the mamba table is align-mode shaped
            # (interior nulls + one state block) and must not be read at all.
            block_ids=([1, 0, 3, 4], [0, 0, 0, 9]),
            block_hashes=[b"a0", b"a1", b"a2", b"a3"],
            can_save=True,
        ),
    )

    keys, addrs, _sizes, _ = store.batch_put_from_multi_buffers.call_args.args
    assert all("@group:0@" in k for k in keys)
    assert [k.rsplit("@", 1)[-1] for k in keys] == ["6130", "6132", "6133"]
    assert addrs == [[0x1000 + 256], [0x1000 + 3 * 256], [0x1000 + 4 * 256]]


def test_block_aligned_snapshot_offload_uses_provided_block():
    """A block-aligned hand-off (sparse-retention mamba checkpoint) uploads
    exactly the core-pinned block under the boundary-end hash key; the
    positional block table entry for that chunk must be ignored."""
    store = MagicMock()
    store.batch_is_exist.side_effect = lambda keys: [0] * len(keys)
    store.batch_put_from_multi_buffers.side_effect = lambda keys, *a: [256] * len(keys)
    thread = _make_partial_tail_send_thread(store)

    hs = [bytes([i + 1]) * 4 for i in range(8)]  # 8 hash units = 32 tokens
    req = ReqMeta(
        req_id="req-a",
        token_len_chunk=0,
        block_ids=([1, 2, 3], [5]),
        block_hashes=hs,
        can_save=True,
        boundary_state_offloads=[(1, 7, 32)],
    )
    assert thread._maybe_offload_boundary_states(req)

    keys, addrs, _sizes, _ = store.batch_put_from_multi_buffers.call_args.args
    # boundary 32 is block-aligned for the mamba group (block 16): one key,
    # keyed by hs[32 // 4 - 1], read from the handed-off block 7 — not from
    # block_ids[1] and with no FA gap coverage.
    assert keys == [thread.token_databases[1].key_for(BlockHash(hs[7]))]
    assert addrs == [[0x2000 + 7 * 256]]


def test_mixed_snapshot_and_sub_block_offloads():
    """A retention snapshot and the prompt-end sub-block CoW tail can arrive
    in one hand-off; the sub-block path covers FA gap blocks but reads the
    mamba boundary only from the CoW block, never positionally."""
    store = MagicMock()
    store.batch_is_exist.side_effect = lambda keys: [0] * len(keys)
    store.batch_put_from_multi_buffers.side_effect = lambda keys, *a: [256] * len(keys)
    thread = _make_partial_tail_send_thread(store)

    hs = [bytes([i + 1]) * 4 for i in range(11)]  # 11 hash units = 44 tokens
    req = ReqMeta(
        req_id="req-a",
        token_len_chunk=0,
        block_ids=([1, 2, 3], [0, 0, 0]),
        block_hashes=hs,
        can_save=True,
        boundary_state_offloads=[(1, 9, 32), (1, 7, 44)],
    )
    assert thread._maybe_offload_boundary_states(req)

    keys, addrs, _sizes, _ = store.batch_put_from_multi_buffers.call_args.args
    db_full, db_mamba = thread.token_databases
    assert keys == [
        # Aligned snapshot: boundary 32 from the handed-off block 9.
        db_mamba.key_for(BlockHash(hs[7])),
        # Sub-block boundary 44: FA gap blocks ending at 4, 8, 12.
        db_full.key_for(BlockHash(hs[0])),
        db_full.key_for(BlockHash(hs[1])),
        db_full.key_for(BlockHash(hs[2])),
        # Mamba boundary block from the CoW hand-off (block 7).
        db_mamba.key_for(BlockHash(hs[10])),
    ]
    assert addrs == [
        [0x2000 + 9 * 256],
        [0x1000 + 1 * 256],
        [0x1000 + 2 * 256],
        [0x1000 + 3 * 256],
        [0x2000 + 7 * 256],
    ]


def test_snapshot_offload_skips_null_handoff_block():
    """A hand-off the core could not materialize (null block) carries no
    committed state; persisting it would poison the boundary key."""
    store = MagicMock()
    thread = _make_partial_tail_send_thread(store)

    hs = [bytes([i + 1]) * 4 for i in range(8)]
    assert thread._maybe_offload_boundary_states(
        ReqMeta(
            req_id="req-a",
            token_len_chunk=0,
            block_ids=([1, 2, 3], [5]),
            block_hashes=hs,
            can_save=True,
            boundary_state_offloads=[(1, NULL_BLOCK_ID, 32)],
        )
    )
    store.batch_is_exist.assert_not_called()
    store.batch_put_from_multi_buffers.assert_not_called()


def test_worker_meta_is_absent_before_any_completed_job():
    send_thread = _make_partial_tail_send_thread(MagicMock())
    store_worker = object.__new__(mooncake_store_worker.MooncakeStoreWorker)
    store_worker.kv_send_thread = send_thread

    assert store_worker.build_connector_worker_meta() is None


def test_store_sending_thread_delta_start_rank_saves_second_local_chunk():
    store = MagicMock()
    store.batch_is_exist.side_effect = lambda keys: [0] * len(keys)
    store.batch_put_from_multi_buffers.return_value = [256, 256]
    thread = _make_store_sending_thread(store, tp_rank=1, put_step=2)

    thread._saved_offset["req-a"] = 16
    _run_store_req(
        thread,
        ReqMeta(
            req_id="req-a",
            token_len_chunk=64,
            block_ids=([1, 2, 3, 4],),
            block_hashes=[b"a0", b"a1", b"a2", b"a3"],
            can_save=True,
        ),
    )

    keys = store.batch_is_exist.call_args.args[0]
    assert keys == [
        "test-model@tp_rank:0@pcp0@dcp0@pp_rank:0@group:0@6131",
        "test-model@tp_rank:0@pcp0@dcp0@pp_rank:0@group:0@6133",
    ]
    assert store.batch_put_from_multi_buffers.call_args.args[0] == keys


def test_store_sending_thread_delta_saves_only_new_masked_chunks():
    store = MagicMock()
    store.batch_is_exist.side_effect = lambda keys: [0] * len(keys)
    store.batch_put_from_multi_buffers.side_effect = (
        lambda keys, addrs, sizes, replicate_config: [256] * len(keys)
    )
    coord = SimpleNamespace(
        lcm_block_size=16,
        store_mask=lambda token_len, start_token, num_prompt_tokens=None: (
            None,
            [True, False],
        ),
    )

    db_full = ChunkedTokenDatabase(
        KeyMetadata("test-model", 0, 0, 0, 0, group_id=0),
        block_size=16,
    )
    db_full.set_kv_caches_base_addr([0x1000])
    db_full.set_block_len([256])
    db_masked = ChunkedTokenDatabase(
        KeyMetadata("test-model", 0, 0, 0, 0, group_id=1),
        block_size=16,
    )
    db_masked.set_kv_caches_base_addr([0x2000])
    db_masked.set_block_len([256])

    thread = _make_store_sending_thread(
        store,
        coord=coord,
        token_databases=[db_full, db_masked],
    )

    thread._saved_offset["req-a"] = 32
    _run_store_req(
        thread,
        ReqMeta(
            req_id="req-a",
            token_len_chunk=64,
            block_ids=([0, 1, 2, 3], [0, 1, 2, 3]),
            block_hashes=[b"a0", b"a1", b"a2", b"a3"],
            can_save=True,
        ),
    )

    keys = store.batch_is_exist.call_args.args[0]
    full_hashes = [k.rsplit("@", 1)[-1] for k in keys if "@group:0" in k]
    masked_hashes = [k.rsplit("@", 1)[-1] for k in keys if "@group:1" in k]

    assert full_hashes == [b"a2".hex(), b"a3".hex()]
    assert masked_hashes == [b"a2".hex()]


def test_store_sending_thread_prepares_missing_chunks_once_per_group():
    store = MagicMock()
    store.batch_is_exist.return_value = [0, 1, 0, 1, 0, 0]
    store.batch_put_from_multi_buffers.return_value = [256, 256, 512, 512]
    coord = SimpleNamespace(
        lcm_block_size=16,
        store_mask=lambda token_len, start_token, num_prompt_tokens=None: (
            None,
            None,
        ),
    )

    db0 = ChunkedTokenDatabase(
        KeyMetadata("test-model", 0, 0, 0, 0, group_id=0),
        block_size=16,
    )
    db0.set_kv_caches_base_addr([0x1000])
    db0.set_block_len([256])
    db0.store_layout.prepare_values = MagicMock(wraps=db0.store_layout.prepare_values)

    db1 = ChunkedTokenDatabase(
        KeyMetadata("test-model", 0, 0, 0, 0, group_id=1),
        block_size=16,
    )
    db1.set_kv_caches_base_addr([0x2000])
    db1.set_block_len([512])
    db1.store_layout.prepare_values = MagicMock(wraps=db1.store_layout.prepare_values)

    thread = _make_store_sending_thread(
        store,
        coord=coord,
        token_databases=[db0, db1],
    )
    _run_store_req(
        thread,
        ReqMeta(
            req_id="req-a",
            token_len_chunk=48,
            block_ids=([1, 2, 3], [3, 2, 1]),
            block_hashes=[b"a0", b"a1", b"a2"],
            can_save=True,
        ),
    )

    db0.store_layout.prepare_values.assert_called_once_with(
        [(0, 16), (32, 48)], [1, 2, 3], [0, 0]
    )
    db1.store_layout.prepare_values.assert_called_once_with(
        [(16, 32), (32, 48)], [3, 2, 1], [0, 0]
    )

    keys, addrs, sizes, _ = store.batch_put_from_multi_buffers.call_args.args
    assert [key.rsplit("@", 1)[-1] for key in keys] == [
        "6130",
        "6132",
        "6131",
        "6132",
    ]
    assert addrs == [[0x1100], [0x1300], [0x2400], [0x2200]]
    assert sizes == [[256], [256], [512], [512]]


def test_store_sending_thread_only_skips_on_no_available_handle():
    store = MagicMock()
    store.batch_is_exist.side_effect = lambda keys: [0] * len(keys)
    store.batch_put_from_multi_buffers.side_effect = [
        [-500, -500],
        [256, 256],
    ]
    thread = _make_store_sending_thread(store)

    _run_store_req(thread, _make_store_req("req-a", [b"a0", b"a1"]))

    assert thread._store_pressure_active is False
    assert "req-a" not in thread._skip_store_requests
    assert store.batch_put_from_multi_buffers.call_count == 1

    _run_store_req(thread, _make_store_req("req-a", [b"a2", b"a3"]))

    assert store.batch_put_from_multi_buffers.call_count == 2


@pytest.mark.parametrize(
    "failing_call", ["batch_is_exist", "batch_put_from_multi_buffers"]
)
def test_store_sending_thread_reports_job_when_store_raises(failing_call):
    # A store that blows up must still report its job, or the scheduler keeps
    # the job's GPU block references for the rest of the run.
    store = MagicMock()
    store.batch_is_exist.return_value = [0, 0]
    getattr(store, failing_call).side_effect = RuntimeError("mooncake down")
    thread = _make_store_sending_thread(store)
    req = _make_store_req("req-a", [b"a0", b"a1"])

    with contextlib.suppress(RuntimeError):
        _run_store_req(thread, req)

    assert thread.take_completed_saves() == {req.store_job_id: 1}


def test_store_sending_thread_reports_job_when_the_preamble_raises():
    # The report has to survive a failure before the first store call too, so
    # every dequeue leaves through the same exit.
    thread = _make_store_sending_thread(MagicMock())
    req = _make_store_req("req-a", [b"a0", b"a1"])
    req.token_len_chunk = None  # type: ignore[assignment]

    with contextlib.suppress(TypeError):
        _run_store_req(thread, req)

    assert thread.take_completed_saves() == {req.store_job_id: 1}
    thread.request_queue.task_done.assert_called_once()


def test_stale_store_job_cannot_touch_a_reused_request_id():
    # A preempted request resumes under its original id, so a job left over from
    # the retired generation carries a req_id that now belongs to a live one.
    thread = _make_store_sending_thread(MagicMock())
    stale = _make_store_req("req-a", [b"a0", b"a1"])
    stale.store_job_id = 1
    thread.add_request(stale)
    thread._record_saved(stale, 32)
    thread.delete_finished_stored_request("req-a")

    live = _make_store_req("req-a", [b"a0", b"a1"])
    live.store_job_id = 2
    thread.add_request(live)
    thread._retry_token_ids["req-a"] = (32, list(range(32, 64)))

    thread.finish_store_job(stale)
    thread._record_saved(stale, 64)
    thread._mark_request_skipped_for_pressure(stale)
    thread._update_retry_token_ids(stale, False, 0, list(range(32)))
    thread._update_retry_token_ids(stale, True, 0, None)

    assert thread.is_live_store_job(live)
    assert not thread.is_live_store_job(stale)
    assert thread._saved_offset.get("req-a") is None
    assert "req-a" not in thread._skip_store_requests
    assert thread._get_retry_token_ids(stale) is None
    assert thread._get_retry_token_ids(live) == (32, list(range(32, 64)))


def test_store_recving_thread_reports_failed_block_ids():
    store = MagicMock()
    store.batch_get_into_multi_buffers.return_value = [256, -5, -7]
    thread = _make_store_recving_thread(store)

    thread._handle_request(
        _make_load_req(
            "req-a",
            [b"a0", b"a1", b"a2"],
            token_len=48,
        )
    )

    assert thread.get_and_clear_finished_requests() == {"req-a"}
    assert thread.get_and_clear_block_ids_with_load_errors() == {1, 2}
    assert thread.get_and_clear_block_ids_with_load_errors() == set()


def test_store_recving_thread_reports_failed_block_ids_after_rotation():
    store = MagicMock()
    store.batch_get_into_multi_buffers.return_value = [256, -5, 256]
    thread = _make_store_recving_thread(store, tp_rank=1)

    thread._handle_request(
        _make_load_req(
            "req-a",
            [b"a0", b"a1", b"a2"],
            token_len=48,
        )
    )

    assert thread.get_and_clear_block_ids_with_load_errors() == {2}


def test_store_recving_thread_reports_all_attempted_blocks_on_exception():
    store = MagicMock()
    store.batch_get_into_multi_buffers.side_effect = RuntimeError("boom")
    thread = _make_store_recving_thread(store)

    thread._handle_request(
        _make_load_req(
            "req-a",
            [b"a0", b"a1", b"a2"],
            token_len=48,
        )
    )

    assert thread.get_and_clear_finished_requests() == {"req-a"}
    assert thread.get_and_clear_block_ids_with_load_errors() == {0, 1, 2}


def test_store_worker_get_block_ids_with_load_errors_delegates_to_recv_thread():
    recv_thread = MagicMock()
    recv_thread.get_and_clear_block_ids_with_load_errors.return_value = {3, 4}
    w = _make_bare_worker()
    w.kv_recv_threads = [recv_thread]

    assert w.get_block_ids_with_load_errors() == {3, 4}
    recv_thread.get_and_clear_block_ids_with_load_errors.assert_called_once_with()


def test_store_sending_thread_passes_replicate_config_when_preferred_segment_set():
    store = MagicMock()
    store.batch_is_exist.side_effect = lambda keys: [0] * len(keys)
    store.batch_put_from_multi_buffers.return_value = [256, 256]
    replicate_config = SimpleNamespace(preferred_segment="10.0.0.7:50053")
    thread = _make_store_sending_thread(store, replicate_config=replicate_config)

    _run_store_req(thread, _make_store_req("req-a", [b"a0", b"a1"]))

    assert store.batch_put_from_multi_buffers.call_count == 1
    call_args = store.batch_put_from_multi_buffers.call_args.args
    assert len(call_args) == 4
    assert call_args[3] is replicate_config


def test_store_sending_thread_passes_default_replicate_config_when_no_preferred_segment():  # noqa: E501
    """Without a preferred_segment the SendingThread still forwards a
    (default-constructed) ReplicateConfig so the C++ side always sees a
    well-defined config object."""
    store = MagicMock()
    store.batch_is_exist.side_effect = lambda keys: [0] * len(keys)
    store.batch_put_from_multi_buffers.return_value = [256, 256]
    replicate_config = SimpleNamespace()
    thread = _make_store_sending_thread(store, replicate_config=replicate_config)

    _run_store_req(thread, _make_store_req("req-a", [b"a0", b"a1"]))

    assert store.batch_put_from_multi_buffers.call_count == 1
    call_args = store.batch_put_from_multi_buffers.call_args.args
    assert len(call_args) == 4
    assert call_args[3] is replicate_config


def test_group_id_detection_uses_class_attribute_without_instantiating():
    class ReplicateConfigWithClassGroupIdsAndRequiredInit:
        group_ids = None

        def __init__(self, required):
            self.group_ids = required

    assert worker._replicate_config_supports_group_ids(
        ReplicateConfigWithClassGroupIdsAndRequiredInit,
        object(),
    )


def test_group_id_detection_uses_instance_attribute_when_class_lacks_attribute():
    class ReplicateConfigWithInstanceGroupIds:
        def __init__(self) -> None:
            self.group_ids = None

    replicate_config = ReplicateConfigWithInstanceGroupIds()

    assert worker._replicate_config_supports_group_ids(
        ReplicateConfigWithInstanceGroupIds,
        replicate_config,
    )


def test_group_id_detection_rejects_old_replicate_config():
    class ReplicateConfigWithoutGroupIds:
        __slots__ = ()

    assert not worker._replicate_config_supports_group_ids(
        ReplicateConfigWithoutGroupIds,
        ReplicateConfigWithoutGroupIds(),
    )


def test_store_sending_thread_sets_group_ids_when_enabled():
    store = MagicMock()
    store.batch_is_exist.side_effect = lambda keys: [0] * len(keys)
    store.batch_put_from_multi_buffers.return_value = [256, 256]
    replicate_config = SimpleNamespace(group_ids=None)
    thread = _make_store_sending_thread(
        store,
        replicate_config=replicate_config,
        enable_group_semantics=True,
        supports_group_ids=True,
    )

    _run_store_req(thread, _make_store_req("req-a", [b"a0", b"a1"]))

    assert store.batch_put_from_multi_buffers.call_count == 1
    keys, _addrs, _sizes, config = store.batch_put_from_multi_buffers.call_args.args
    assert config is replicate_config
    assert config.group_ids == [
        "vllm-mooncake-store:test-model@6130",
        "vllm-mooncake-store:test-model@6131",
    ]
    assert len(config.group_ids) == len(keys)


def test_store_sending_thread_leaves_group_ids_unchanged_when_flag_disabled():
    store = MagicMock()
    store.batch_is_exist.side_effect = lambda keys: [0] * len(keys)
    store.batch_put_from_multi_buffers.return_value = [256, 256]
    replicate_config = SimpleNamespace(group_ids=["existing"])
    thread = _make_store_sending_thread(
        store,
        replicate_config=replicate_config,
        enable_group_semantics=False,
        supports_group_ids=True,
    )

    _run_store_req(thread, _make_store_req("req-a", [b"a0", b"a1"]))

    assert store.batch_put_from_multi_buffers.call_count == 1
    assert store.batch_put_from_multi_buffers.call_args.args[3] is replicate_config
    assert replicate_config.group_ids == ["existing"]


def test_store_sending_thread_leaves_group_ids_unchanged_when_unsupported():
    store = MagicMock()
    store.batch_is_exist.side_effect = lambda keys: [0] * len(keys)
    store.batch_put_from_multi_buffers.return_value = [256, 256]
    replicate_config = SimpleNamespace(group_ids=["existing"])
    thread = _make_store_sending_thread(
        store,
        replicate_config=replicate_config,
        enable_group_semantics=True,
        supports_group_ids=False,
    )

    _run_store_req(thread, _make_store_req("req-a", [b"a0", b"a1"]))

    assert store.batch_put_from_multi_buffers.call_count == 1
    assert store.batch_put_from_multi_buffers.call_args.args[3] is replicate_config
    assert replicate_config.group_ids == ["existing"]


def test_mooncake_group_id_uses_logical_chunk_and_excludes_physical_ranks():
    group_id = worker._make_mooncake_group_id(
        KeyMetadata(
            "test-model",
            tp_rank=3,
            pcp_rank=5,
            dcp_rank=7,
            pp_rank=11,
            group_id=13,
        ),
        "abcdef",
    )

    assert group_id == "vllm-mooncake-store:test-model@abcdef"
    assert "tp_rank" not in group_id
    assert "pp_rank" not in group_id
    assert "pcp" not in group_id
    assert "dcp" not in group_id
    assert "group:" not in group_id


def test_mooncake_group_id_uses_cache_prefix_namespace():
    metadata_a = KeyMetadata("test-model", 0, 0, 0, 0, cache_prefix="deployment-a")
    metadata_b = KeyMetadata("test-model", 0, 0, 0, 0, cache_prefix="deployment-b")

    group_id_a = worker._make_mooncake_group_id(metadata_a, "abcdef")
    group_id_b = worker._make_mooncake_group_id(metadata_b, "abcdef")

    assert group_id_a == "vllm-mooncake-store:deployment-a@test-model@abcdef"
    assert group_id_b == "vllm-mooncake-store:deployment-b@test-model@abcdef"
    assert group_id_a != group_id_b


def test_store_sending_thread_group_id_excludes_physical_sharding():
    store = MagicMock()
    store.batch_is_exist.return_value = [0, 0]
    store.batch_put_from_multi_buffers.return_value = [256, 256]
    replicate_config = SimpleNamespace(group_ids=None)
    db = ChunkedTokenDatabase(
        KeyMetadata(
            "test-model",
            tp_rank=2,
            pcp_rank=0,
            dcp_rank=0,
            pp_rank=0,
            group_id=0,
        ),
        block_size=16,
    )
    db.set_kv_caches_base_addr([0x1000])
    db.set_block_len([256])
    thread = _make_store_sending_thread(
        store,
        token_databases=[db],
        tp_rank=2,
        replicate_config=replicate_config,
        enable_group_semantics=True,
        supports_group_ids=True,
    )

    _run_store_req(thread, _make_store_req("req-a", [b"a0", b"a1"]))

    keys, _addrs, _sizes, config = store.batch_put_from_multi_buffers.call_args.args
    assert keys == [
        "test-model@tp_rank:2@pcp0@dcp0@pp_rank:0@group:0@6130",
        "test-model@tp_rank:2@pcp0@dcp0@pp_rank:0@group:0@6131",
    ]
    assert config.group_ids == [
        "vllm-mooncake-store:test-model@6130",
        "vllm-mooncake-store:test-model@6131",
    ]


def test_store_sending_thread_multiple_segments_share_logical_group_id():
    store = MagicMock()
    store.batch_is_exist.return_value = [0, 0]
    store.batch_put_from_multi_buffers.return_value = [512, 512]
    replicate_config = SimpleNamespace(group_ids=None)
    db = ChunkedTokenDatabase(KeyMetadata("test-model", 0, 0, 0, 0), block_size=16)
    db.set_kv_caches_base_addr([0x1000, 0x2000])
    db.set_block_len([256, 256])
    thread = _make_store_sending_thread(
        store,
        token_databases=[db],
        replicate_config=replicate_config,
        enable_group_semantics=True,
        supports_group_ids=True,
    )

    _run_store_req(thread, _make_store_req("req-a", [b"a0", b"a1"]))

    keys, addrs, sizes, config = store.batch_put_from_multi_buffers.call_args.args
    assert keys == [
        "test-model@tp_rank:0@pcp0@dcp0@pp_rank:0@group:0@6130",
        "test-model@tp_rank:0@pcp0@dcp0@pp_rank:0@group:0@6131",
    ]
    assert addrs == [[0x1100, 0x2100], [0x1200, 0x2200]]
    assert sizes == [[256, 256], [256, 256]]
    assert config.group_ids == [
        "vllm-mooncake-store:test-model@6130",
        "vllm-mooncake-store:test-model@6131",
    ]


def test_store_sending_thread_group_ids_share_across_kv_cache_groups():
    from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheGroupSpec

    store = MagicMock()
    store.batch_is_exist.return_value = [0, 0, 0, 0]
    store.batch_put_from_multi_buffers.return_value = [256, 256, 256, 256]
    replicate_config = SimpleNamespace(group_ids=None)
    spec = FullAttentionSpec(block_size=16, num_kv_heads=8, head_size=64, dtype=None)
    coord = mooncake_store_worker.MooncakeStoreCoordinator(
        [
            KVCacheGroupSpec(["layer0"], spec),
            KVCacheGroupSpec(["layer1"], spec),
        ],
        scheduler_block_size=16,
        hash_block_size=16,
    )
    token_databases = []
    for group_id, base_addr in enumerate([0x1000, 0x3000]):
        db = ChunkedTokenDatabase(
            KeyMetadata("test-model", 0, 0, 0, 0, group_id=group_id),
            block_size=16,
        )
        db.set_kv_caches_base_addr([base_addr])
        db.set_block_len([256])
        token_databases.append(db)
    thread = _make_store_sending_thread(
        store,
        coord=coord,
        token_databases=token_databases,
        replicate_config=replicate_config,
        enable_group_semantics=True,
        supports_group_ids=True,
    )

    _run_store_req(thread, _make_multi_group_store_req("req-a", [b"a0", b"a1"]))

    keys, addrs, sizes, config = store.batch_put_from_multi_buffers.call_args.args
    assert keys == [
        "test-model@tp_rank:0@pcp0@dcp0@pp_rank:0@group:0@6130",
        "test-model@tp_rank:0@pcp0@dcp0@pp_rank:0@group:0@6131",
        "test-model@tp_rank:0@pcp0@dcp0@pp_rank:0@group:1@6130",
        "test-model@tp_rank:0@pcp0@dcp0@pp_rank:0@group:1@6131",
    ]
    assert addrs == [[0x1100], [0x1200], [0x3300], [0x3400]]
    assert sizes == [[256], [256], [256], [256]]
    # Different vLLM KV cache groups for the same prefix chunk share the
    # same Mooncake lifecycle group id.
    assert config.group_ids == [
        "vllm-mooncake-store:test-model@6130",
        "vllm-mooncake-store:test-model@6131",
        "vllm-mooncake-store:test-model@6130",
        "vllm-mooncake-store:test-model@6131",
    ]
    assert config.group_ids[0] == config.group_ids[2]
    assert config.group_ids[1] == config.group_ids[3]


def test_store_sending_thread_group_ids_follow_missing_key_filter():
    store = MagicMock()
    store.batch_is_exist.return_value = [1, 0]
    store.batch_put_from_multi_buffers.return_value = [256]
    replicate_config = SimpleNamespace(group_ids=None)
    thread = _make_store_sending_thread(
        store,
        replicate_config=replicate_config,
        enable_group_semantics=True,
        supports_group_ids=True,
    )

    _run_store_req(thread, _make_store_req("req-a", [b"a0", b"a1"]))

    keys, _addrs, _sizes, config = store.batch_put_from_multi_buffers.call_args.args
    assert keys == ["test-model@tp_rank:0@pcp0@dcp0@pp_rank:0@group:0@6131"]
    assert config.group_ids == ["vllm-mooncake-store:test-model@6131"]


def test_estimate_disk_offload_staging_bytes_sums_multi_segment_sizes():
    assert worker._estimate_disk_offload_staging_bytes([256, 512]) == 12288


def test_recv_thread_uses_single_batch_when_no_disk_offload_budget(monkeypatch):
    monkeypatch.delenv("VLLM_MOONCAKE_STORE_TIER_LOG", raising=False)
    store = MagicMock()
    store.batch_get_into_multi_buffers.return_value = [256, 256, 256]
    thread = _make_store_recving_thread(store, disk_offload_buffer_budget_bytes=None)

    req = _make_load_req(
        "req-a",
        [b"a0", b"a1", b"a2"],
        token_len=48,
    )

    thread._handle_request(req)

    assert store.batch_get_into_multi_buffers.call_count == 1
    keys, addrs, sizes = store.batch_get_into_multi_buffers.call_args.args
    assert keys == [
        "test-model@tp_rank:0@pcp0@dcp0@pp_rank:0@group:0@6130",
        "test-model@tp_rank:0@pcp0@dcp0@pp_rank:0@group:0@6131",
        "test-model@tp_rank:0@pcp0@dcp0@pp_rank:0@group:0@6132",
    ]
    assert sizes == [[256], [256], [256]]
    store.batch_get_replica_desc.assert_not_called()


def test_recv_thread_keys_chunk_by_lookup_selected_boundary():
    store = MagicMock()
    store.batch_get_into_multi_buffers.return_value = [256, 256]
    thread = _make_store_recving_thread(store)

    # 32-token hit over 16-token chunks: chunk 1 would default to the hash at
    # the 32-token boundary (a1); the lookup matched the 48-token one (a2).
    req = _make_load_req(
        "req-a",
        [b"a0", b"a1", b"a2"],
        token_len=32,
        tail_key_boundaries=(TailKeyBoundary(group_id=0, num_tokens=48),),
    )

    thread._handle_request(req)

    keys = store.batch_get_into_multi_buffers.call_args.args[0]
    assert keys == [
        "test-model@tp_rank:0@pcp0@dcp0@pp_rank:0@group:0@6130",
        "test-model@tp_rank:0@pcp0@dcp0@pp_rank:0@group:0@6132",
    ]


def test_recv_thread_logs_tier_summary_when_enabled(monkeypatch, caplog_vllm):
    monkeypatch.setenv("VLLM_MOONCAKE_STORE_TIER_LOG", "1")
    caplog_vllm.set_level(logging.INFO, logger=worker.logger.name)

    store = MagicMock()
    store.batch_get_into_multi_buffers.return_value = [256, 256, -10]
    thread = _make_store_recving_thread(store, disk_offload_buffer_budget_bytes=None)

    req = _make_load_req(
        "req-a",
        [b"a0", b"a1", b"a2"],
        token_len=48,
    )
    expected_keys = [
        "test-model@tp_rank:0@pcp0@dcp0@pp_rank:0@group:0@6130",
        "test-model@tp_rank:0@pcp0@dcp0@pp_rank:0@group:0@6131",
        "test-model@tp_rank:0@pcp0@dcp0@pp_rank:0@group:0@6132",
    ]
    store.batch_get_replica_desc.return_value = {
        expected_keys[0]: [_ReplicaDesc("memory")],
        expected_keys[1]: [_ReplicaDesc("disk")],
        expected_keys[2]: [],
    }

    thread._handle_request(req)

    assert store.batch_get_replica_desc.call_args.args == (expected_keys,)
    assert store.method_calls[0][0] == "batch_get_replica_desc"
    assert store.method_calls[1][0] == "batch_get_into_multi_buffers"

    messages = [record.getMessage() for record in caplog_vllm.records]
    assert any(
        "Mooncake load tier summary" in message
        and "req_id=req-a" in message
        and "batch_keys=3" in message
        and "memory_keys=1" in message
        and "disk_keys=1" in message
        and "unknown_keys=1" in message
        and "success_keys=2" in message
        and "failed_keys=1" in message
        and "bytes_by_tier={'memory': 256, 'disk': 256, 'unknown': 0}" in message
        for message in messages
    )


def test_recv_thread_records_partial_failure_metrics(monkeypatch):
    monkeypatch.delenv("VLLM_MOONCAKE_STORE_TIER_LOG", raising=False)
    store = MagicMock()
    store.batch_get_into_multi_buffers.return_value = [256, -10]
    thread = _make_store_recving_thread(store, disk_offload_buffer_budget_bytes=None)
    stats = MooncakeStoreConnectorStats()
    thread._record_operation_cb = stats.record_operation

    req = _make_load_req(
        "req-a",
        [b"a0", b"a1"],
        token_len=32,
    )

    thread._handle_request(req)

    assert len(stats.data["load_get"]) == 1
    assert stats.data["load_get"][0]["num_keys"] == 2
    assert stats.data["load_get"][0]["num_bytes"] == 512
    assert stats.data["load_get"][0]["status"] == "partial_failure"
    assert stats.data["load_get"][0]["num_failed_keys"] == 1


def test_recv_thread_uses_ratio_scaled_budget_for_first_pass_split():
    store = MagicMock()
    store.batch_get_into_multi_buffers.side_effect = [
        [256],
        [256],
    ]
    thread = _make_store_recving_thread(
        store,
        disk_offload_buffer_budget_bytes=2 * _DISK_OFFLOAD_SINGLE_KEY_BYTES,
    )

    req = _make_load_req(
        "req-a",
        [b"a0", b"a1"],
        token_len=32,
    )

    thread._handle_request(req)

    assert store.batch_get_into_multi_buffers.call_count == 2
    first_keys = store.batch_get_into_multi_buffers.call_args_list[0].args[0]
    second_keys = store.batch_get_into_multi_buffers.call_args_list[1].args[0]
    assert first_keys == [
        "test-model@tp_rank:0@pcp0@dcp0@pp_rank:0@group:0@6130",
    ]
    assert second_keys == [
        "test-model@tp_rank:0@pcp0@dcp0@pp_rank:0@group:0@6131",
    ]


def test_recv_thread_splits_disk_offload_loads_by_budget():
    store = MagicMock()
    store.batch_get_into_multi_buffers.side_effect = [
        [256, 256],
        [256],
    ]
    thread = _make_store_recving_thread(
        store,
        disk_offload_buffer_budget_bytes=_DISK_OFFLOAD_BUDGET_FOR_SPLIT,
    )

    req = _make_load_req(
        "req-a",
        [b"a0", b"a1", b"a2"],
        token_len=48,
    )

    thread._handle_request(req)

    assert store.batch_get_into_multi_buffers.call_count == 2

    first_keys = store.batch_get_into_multi_buffers.call_args_list[0].args[0]
    second_keys = store.batch_get_into_multi_buffers.call_args_list[1].args[0]
    first_addrs = store.batch_get_into_multi_buffers.call_args_list[0].args[1]
    second_addrs = store.batch_get_into_multi_buffers.call_args_list[1].args[1]
    first_sizes = store.batch_get_into_multi_buffers.call_args_list[0].args[2]
    second_sizes = store.batch_get_into_multi_buffers.call_args_list[1].args[2]
    assert first_keys == [
        "test-model@tp_rank:0@pcp0@dcp0@pp_rank:0@group:0@6130",
        "test-model@tp_rank:0@pcp0@dcp0@pp_rank:0@group:0@6131",
    ]
    assert second_keys == [
        "test-model@tp_rank:0@pcp0@dcp0@pp_rank:0@group:0@6132",
    ]
    layout = thread.token_databases[0].store_layout
    assert isinstance(layout, RankLocalStoreLayout)
    base_addr = layout.kv_caches_base_addr[0]
    block_len = layout.block_len[0]
    assert first_addrs == [[base_addr], [base_addr + block_len]]
    assert second_addrs == [[base_addr + 2 * block_len]]
    expected_size = block_len
    assert first_sizes == [[expected_size], [expected_size]]
    assert second_sizes == [[expected_size]]


def test_recv_thread_stops_after_first_failing_disk_offload_sub_batch():
    store = MagicMock()
    store.batch_get_into_multi_buffers.return_value = [-10, -10]
    thread = _make_store_recving_thread(
        store,
        disk_offload_buffer_budget_bytes=_DISK_OFFLOAD_BUDGET_FOR_SPLIT,
    )

    req = _make_load_req(
        "req-a",
        [b"a0", b"a1", b"a2"],
        token_len=48,
    )

    thread._handle_request(req)

    assert store.batch_get_into_multi_buffers.call_count == 1
    assert thread.get_and_clear_block_ids_with_load_errors() == {0, 1}


def test_recv_thread_skips_split_when_budget_holds_all_keys():
    """PR-36 removed the count-based split trigger; with budget for 3 keys,
    all three should be requested in a single call."""
    store = MagicMock()
    store.batch_get_into_multi_buffers.return_value = [256, 256, 256]
    thread = _make_store_recving_thread(
        store,
        disk_offload_buffer_budget_bytes=_DISK_OFFLOAD_BUDGET_FOR_THREE_KEYS,
    )

    req = _make_load_req(
        "req-a",
        [b"a0", b"a1", b"a2"],
        token_len=48,
    )

    thread._handle_request(req)

    assert store.batch_get_into_multi_buffers.call_count == 1
    assert store.batch_get_into_multi_buffers.call_args_list[0].args[0] == [
        "test-model@tp_rank:0@pcp0@dcp0@pp_rank:0@group:0@6130",
        "test-model@tp_rank:0@pcp0@dcp0@pp_rank:0@group:0@6131",
        "test-model@tp_rank:0@pcp0@dcp0@pp_rank:0@group:0@6132",
    ]


def test_recv_thread_reports_unsplittable_key_larger_than_budget():
    # Non-zero tp_rank exercises the rotation: the oversized key may not sit
    # at the original request's first block. Every block must still be marked
    # invalid, since none are loaded.
    store = MagicMock()
    thread = _make_store_recving_thread(
        store,
        tp_rank=2,
        disk_offload_buffer_budget_bytes=_DISK_OFFLOAD_BUDGET_TOO_SMALL,
    )

    req = _make_load_req(
        "req-a",
        [b"a0", b"a1", b"a2"],
        token_len=48,
    )

    thread._handle_request(req)

    assert store.batch_get_into_multi_buffers.call_count == 0
    assert thread.get_and_clear_block_ids_with_load_errors() == {0, 1, 2}


def test_recv_thread_partial_load_failure_reports_block_ids_without_hma():
    store = MagicMock()
    # First key fails, second succeeds.
    store.batch_get_into_multi_buffers.return_value = [-1, 0]
    thread = _make_store_recving_thread(store)

    thread._handle_request(_make_load_req("req-a", [b"h0", b"h1"], token_len=32))

    assert thread.get_and_clear_block_ids_with_load_errors() == {0}
    assert thread.get_and_clear_failed_requests() == set()
    assert thread.get_and_clear_finished_requests() == {"req-a"}


def test_recv_thread_partial_load_failure_reports_request_with_hma():
    store = MagicMock()
    store.batch_get_into_multi_buffers.return_value = [-1, 0]
    thread = _make_store_recving_thread(store, is_hma_required=True)

    thread._handle_request(_make_load_req("req-a", [b"h0", b"h1"], token_len=32))

    assert thread.get_and_clear_block_ids_with_load_errors() == set()
    assert thread.get_and_clear_failed_requests() == {"req-a"}
    assert thread.get_and_clear_finished_requests() == {"req-a"}


def test_recv_thread_oversized_key_reports_request_with_hma():
    store = MagicMock()
    thread = _make_store_recving_thread(
        store,
        tp_rank=2,
        disk_offload_buffer_budget_bytes=_DISK_OFFLOAD_BUDGET_TOO_SMALL,
        is_hma_required=True,
    )

    thread._handle_request(_make_load_req("req-a", [b"a0", b"a1", b"a2"], token_len=48))

    assert store.batch_get_into_multi_buffers.call_count == 0
    assert thread.get_and_clear_block_ids_with_load_errors() == set()
    assert thread.get_and_clear_failed_requests() == {"req-a"}
    assert thread.get_and_clear_finished_requests() == {"req-a"}


def test_worker_init_excludes_nonprefix_cache_groups(monkeypatch):
    store = MagicMock()
    store.setup.return_value = 0
    _install_fake_mooncake(monkeypatch, store)
    _patch_worker_runtime(monkeypatch)
    monkeypatch.setattr(
        worker.MooncakeStoreConfig,
        "load_from_config",
        staticmethod(lambda: _make_config()),
    )
    vllm_config = _make_vllm_config()
    vllm_config.cache_config.block_size = 800
    vllm_config.cache_config.enable_prefix_caching = True
    vllm_config.cache_config.prefix_match_unit = None

    store_worker = worker.MooncakeStoreWorker(
        vllm_config, _make_qsa_hybrid_kv_cache_config()
    )

    assert [
        group.kv_cache_spec.block_size for group in store_worker._kv_cache_groups
    ] == [800, 800]
    assert [db.block_size for db in store_worker.token_dbs] == [800, 800]


def _make_two_full_attention_groups_kv_cache_config():
    """Two full-attention groups with different block sizes: no hybrid
    layers, but block-level failure reporting is still unsupported."""
    return KVCacheConfig(
        num_blocks=10,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["full0"],
                FullAttentionSpec(
                    block_size=800, num_kv_heads=8, head_size=64, dtype=None
                ),
            ),
            KVCacheGroupSpec(
                ["full1"],
                FullAttentionSpec(
                    block_size=1600, num_kv_heads=8, head_size=64, dtype=None
                ),
            ),
        ],
    )


@pytest.mark.parametrize(
    ("make_config", "expected_is_hma_required"),
    [
        (lambda: _make_qsa_hybrid_kv_cache_config(), True),
        (lambda: _make_two_full_attention_groups_kv_cache_config(), True),
        (lambda: _make_kv_cache_config(block_size=800), False),
    ],
    ids=["hybrid", "two-full-attention-groups", "single-group"],
)
def test_worker_is_hma_required_from_kv_cache_groups(
    monkeypatch, make_config, expected_is_hma_required
):
    store = MagicMock()
    store.setup.return_value = 0
    _install_fake_mooncake(monkeypatch, store)
    _patch_worker_runtime(monkeypatch)
    monkeypatch.setattr(
        worker.MooncakeStoreConfig,
        "load_from_config",
        staticmethod(lambda: _make_config()),
    )
    vllm_config = _make_vllm_config()
    vllm_config.cache_config.block_size = 800
    vllm_config.cache_config.enable_prefix_caching = True
    vllm_config.cache_config.prefix_match_unit = None

    store_worker = worker.MooncakeStoreWorker(vllm_config, make_config())

    assert store_worker._is_hma_required is expected_is_hma_required


def test_requester_worker_init_uses_positional_setup(tmp_path, monkeypatch):
    store = MagicMock()
    store.setup.return_value = 0
    _install_fake_mooncake(monkeypatch, store)
    _patch_worker_runtime(monkeypatch)
    monkeypatch.setenv(
        "MOONCAKE_CONFIG_PATH",
        _write_mooncake_config(
            tmp_path,
            {
                "metadata_server": "http://metadata/endpoint",
                "global_segment_size": "4gb",
                "local_buffer_size": "64mb",
                "protocol": "rdma",
                "device_name": "mlx5_0",
                "master_server_address": "10.0.0.7:50051",
                "enable_offload": True,
            },
        ),
    )
    w = worker.MooncakeStoreWorker(_make_vllm_config(), _make_kv_cache_config())

    assert not hasattr(w, "_isolate_offload_resources")
    assert store.setup.call_args.args == (
        "10.0.0.7",
        "http://metadata/endpoint",
        4 * 1024 * 1024 * 1024,  # global_segment_size: "4gb" honored
        64 * 1024 * 1024,
        "rdma",
        "mlx5_0",
        "10.0.0.7:50051",
    )
    assert w.coord.retention_interval == 0


def test_requester_worker_init_prefers_local_hostname_override(
    tmp_path,
    monkeypatch,
):
    store = MagicMock()
    store.setup.return_value = 0
    _install_fake_mooncake(monkeypatch, store)
    _patch_worker_runtime(monkeypatch)
    monkeypatch.setenv("MOONCAKE_REQUESTER_LOCAL_HOSTNAME", "worker-a:50053")
    monkeypatch.setenv(
        "MOONCAKE_CONFIG_PATH",
        _write_mooncake_config(
            tmp_path,
            {
                "metadata_server": "http://metadata/endpoint",
                "local_buffer_size": "64mb",
                "protocol": "tcp",
                "device_name": "",
                "master_server_address": "10.0.0.7:50051",
            },
        ),
    )
    worker.MooncakeStoreWorker(_make_vllm_config(rank=1), _make_kv_cache_config())

    assert store.setup.call_args.args[0] == "worker-a:50053"


def test_requester_worker_init_skips_disk_budget_when_offload_disabled(
    tmp_path,
    monkeypatch,
):
    """enable_offload=False zeroes out the disk budget so we don't generate
    redundant owner GET-RPCs."""
    store = MagicMock()
    store.setup.return_value = 0
    _install_fake_mooncake(monkeypatch, store)
    _patch_worker_runtime(monkeypatch)
    monkeypatch.setenv(
        "MOONCAKE_CONFIG_PATH",
        _write_mooncake_config(
            tmp_path,
            {
                "metadata_server": "http://metadata/endpoint",
                "protocol": "tcp",
                "device_name": "",
                "master_server_address": "10.0.0.7:50051",
                "enable_offload": False,
            },
        ),
    )
    w = worker.MooncakeStoreWorker(_make_vllm_config(), _make_kv_cache_config())

    assert w.disk_offload_buffer_budget_bytes is None


def test_worker_enables_lbhnc_store_tp_layout(tmp_path, monkeypatch):
    store = MagicMock()
    store.setup.return_value = 0
    _install_fake_mooncake(monkeypatch, store)
    _patch_worker_runtime(monkeypatch, tp_size=2)
    monkeypatch.setattr(_FakeModelConfig, "get_total_num_kv_heads", lambda _self: 8)
    monkeypatch.setenv(
        "MOONCAKE_CONFIG_PATH",
        _write_mooncake_config(
            tmp_path,
            {
                "metadata_server": "http://metadata/endpoint",
                "protocol": "tcp",
                "master_server_address": "10.0.0.7:50051",
            },
        ),
    )

    w = worker.MooncakeStoreWorker(
        _make_vllm_config(extra_config={"store_tp_size": 4}),
        _make_kv_cache_config(),
    )

    assert w.store_tp_size == 4
    assert w.token_dbs[0].metadata.store_namespace == _TP_SHARED_NAMESPACE
    assert isinstance(w.token_dbs[0].store_layout, LBHNCStoreLayout)
    assert w.token_dbs[0].store_layout.local_shard_ids == (0, 1)
    assert len(w._lookup_key_prefixes[0]) == 4


def test_worker_enables_lbnhc_store_tp_layout(tmp_path, monkeypatch):
    store = MagicMock()
    store.setup.return_value = 0
    _install_fake_mooncake(monkeypatch, store)
    _patch_worker_runtime(monkeypatch, tp_size=2)
    monkeypatch.setattr(_FakeModelConfig, "get_total_num_kv_heads", lambda _self: 8)
    monkeypatch.setenv(
        "MOONCAKE_CONFIG_PATH",
        _write_mooncake_config(
            tmp_path,
            {
                "metadata_server": "http://metadata/endpoint",
                "protocol": "tcp",
                "master_server_address": "10.0.0.7:50051",
            },
        ),
    )

    with patch.object(worker.logger, "warning_once") as warning_once:
        w = worker.MooncakeStoreWorker(
            _make_vllm_config(
                extra_config={"store_tp_size": 4},
                kv_cache_layout=KVCacheLayout.LBNHC,
            ),
            _make_kv_cache_config(),
        )

    warning_once.assert_called_once()
    assert w.store_tp_size == 4
    assert w.token_dbs[0].metadata.store_namespace == _TP_SHARED_LBNHC_NAMESPACE
    assert isinstance(w.token_dbs[0].store_layout, LBNHCStoreLayout)
    assert w.token_dbs[0].store_layout.local_shard_ids == (0, 1)
    assert len(w._lookup_key_prefixes[0]) == 4


def test_tp_sharded_layout_rejects_full_attention_subclasses():
    spec = RSWASpec(
        block_size=16,
        num_kv_heads=8,
        head_size=64,
        dtype=None,
        rswa_window=32,
    )
    store_worker = object.__new__(worker.MooncakeStoreWorker)
    store_worker.pcp_size = 1
    store_worker.dcp_size = 1
    store_worker._kv_cache_groups = [SimpleNamespace(kv_cache_spec=spec)]

    assert not store_worker._supports_tp_sharded_store_layout(LBHNCStoreLayout, {})


@pytest.mark.parametrize(
    ("extra_config", "expected"),
    [
        ({"store_tp_size": 4}, 4),
        (
            {
                "enable_store_tp_lcm": True,
                "prefill_tp_sizes": [3, 4, 6],
            },
            12,
        ),
        (
            {
                "enable_store_tp_lcm": False,
                "prefill_tp_sizes": [4, 2],
            },
            None,
        ),
        ({"enable_store_tp_lcm": True, "prefill_tp_sizes": []}, None),
        ({"enable_store_tp_lcm": True, "prefill_tp_sizes": [4, 0]}, None),
        ({"enable_store_tp_lcm": True, "prefill_tp_sizes": [4, 2.0]}, None),
        ({"store_tp_size": 0}, None),
        ({"store_tp_size": -1}, None),
        ({"store_tp_size": True}, None),
        ({"store_tp_size": 4.0}, None),
    ],
)
def test_resolve_store_tp_size(extra_config: dict[str, object], expected: int | None):
    assert worker.resolve_store_tp_size(extra_config) == expected


def test_lcm_store_tp_gives_prefill_and_decode_common_namespace(tmp_path, monkeypatch):
    store = MagicMock()
    store.setup.return_value = 0
    _install_fake_mooncake(monkeypatch, store)
    monkeypatch.setattr(_FakeModelConfig, "get_total_num_kv_heads", lambda _self: 8)
    monkeypatch.setenv(
        "MOONCAKE_CONFIG_PATH",
        _write_mooncake_config(
            tmp_path,
            {
                "metadata_server": "http://metadata/endpoint",
                "protocol": "tcp",
                "master_server_address": "10.0.0.7:50051",
            },
        ),
    )
    lcm_config = {
        "enable_store_tp_lcm": True,
        "prefill_tp_sizes": [4, 2],
    }

    workers = []
    topologies = [
        (4, 2, "kv_both", (2,)),
        (2, 1, "kv_both", (2, 3)),
        (2, 1, "kv_consumer", (2, 3)),
    ]
    for tp_size, tp_rank, kv_role, expected_shards in topologies:
        _patch_worker_runtime(
            monkeypatch,
            tp_size=tp_size,
            tp_rank=tp_rank,
        )
        extra_config = dict(lcm_config)
        if kv_role == "kv_consumer":
            extra_config["save_decode_cache"] = True
        store_worker = worker.MooncakeStoreWorker(
            _make_vllm_config(extra_config=extra_config, kv_role=kv_role),
            _make_kv_cache_config(),
        )
        assert store_worker.store_tp_size == 4
        assert isinstance(store_worker.token_dbs[0].store_layout, LBHNCStoreLayout)
        assert store_worker.token_dbs[0].store_layout.local_shard_ids == expected_shards
        workers.append(store_worker)

    shard_keys = [
        store_worker.token_dbs[0].store_layout.key_for(2, BlockHash(b"h"))
        for store_worker in workers
    ]
    assert len(set(shard_keys)) == 1
    assert workers[-1].can_put


@pytest.mark.parametrize(
    ("tp_size", "prefill_tp_sizes"),
    [
        (3, [4, 2]),
        (2, [4, 3]),
        (2, []),
        (2, [4, 0]),
    ],
)
def test_lcm_store_tp_falls_back_when_topology_is_incompatible(
    tmp_path,
    monkeypatch,
    tp_size: int,
    prefill_tp_sizes: list[int],
):
    store = MagicMock()
    store.setup.return_value = 0
    _install_fake_mooncake(monkeypatch, store)
    _patch_worker_runtime(monkeypatch, tp_size=tp_size)
    monkeypatch.setattr(_FakeModelConfig, "get_total_num_kv_heads", lambda _self: 8)
    monkeypatch.setenv(
        "MOONCAKE_CONFIG_PATH",
        _write_mooncake_config(
            tmp_path,
            {
                "metadata_server": "http://metadata/endpoint",
                "protocol": "tcp",
                "master_server_address": "10.0.0.7:50051",
            },
        ),
    )

    store_worker = worker.MooncakeStoreWorker(
        _make_vllm_config(
            extra_config={
                "enable_store_tp_lcm": True,
                "prefill_tp_sizes": prefill_tp_sizes,
            }
        ),
        _make_kv_cache_config(),
    )

    assert store_worker.store_tp_size is None
    assert type(store_worker.token_dbs[0]) is ChunkedTokenDatabase
    assert "@store_format:rank_local_tp" in store_worker.token_dbs[0].key_for(
        BlockHash(b"h")
    )


@pytest.mark.parametrize("store_tp_size", [1, 3, 6])
def test_worker_keeps_rank_local_layout_for_unsupported_store_tp(
    tmp_path,
    monkeypatch,
    store_tp_size: int,
):
    store = MagicMock()
    store.setup.return_value = 0
    _install_fake_mooncake(monkeypatch, store)
    _patch_worker_runtime(monkeypatch, tp_size=2)
    monkeypatch.setattr(_FakeModelConfig, "get_total_num_kv_heads", lambda _self: 8)
    monkeypatch.setenv(
        "MOONCAKE_CONFIG_PATH",
        _write_mooncake_config(
            tmp_path,
            {
                "metadata_server": "http://metadata/endpoint",
                "protocol": "tcp",
                "master_server_address": "10.0.0.7:50051",
            },
        ),
    )

    w = worker.MooncakeStoreWorker(
        _make_vllm_config(
            extra_config={"store_tp_size": store_tp_size},
            kv_cache_layout=KVCacheLayout.LBNHC,
        ),
        _make_kv_cache_config(),
    )

    assert w.store_tp_size is None
    assert type(w.token_dbs[0]) is ChunkedTokenDatabase
    key = w.token_dbs[0].key_for(BlockHash(b"h"))
    assert "@store_tp:" not in key
    assert "@store_pp:1" in key
    assert "@store_format:rank_local_tp2_layout_LBNHC" in key


def test_store_tp_opt_in_isolates_different_pp_sizes(tmp_path, monkeypatch):
    store = MagicMock()
    store.setup.return_value = 0
    _install_fake_mooncake(monkeypatch, store)
    _patch_worker_runtime(monkeypatch, tp_size=2)
    monkeypatch.setattr(_FakeModelConfig, "get_total_num_kv_heads", lambda _self: 8)
    monkeypatch.setenv(
        "MOONCAKE_CONFIG_PATH",
        _write_mooncake_config(
            tmp_path,
            {
                "metadata_server": "http://metadata/endpoint",
                "protocol": "tcp",
                "master_server_address": "10.0.0.7:50051",
            },
        ),
    )

    keys = []
    for pp_size in (1, 2):
        w = worker.MooncakeStoreWorker(
            _make_vllm_config(
                extra_config={"store_tp_size": 4},
                pipeline_parallel_size=pp_size,
            ),
            _make_kv_cache_config(),
        )
        layout = w.token_dbs[0].store_layout
        keys.append(layout.key_for(layout.local_shard_ids[0], BlockHash(b"h")))

    assert "@store_pp:1" in keys[0]
    assert "@store_pp:2" in keys[1]
    assert keys[0] != keys[1]


def test_save_decode_cache_keeps_transfer_path_enabled(tmp_path, monkeypatch):
    store = MagicMock()
    store.setup.return_value = 0
    _install_fake_mooncake(monkeypatch, store)
    _patch_worker_runtime(monkeypatch)
    monkeypatch.setenv(
        "MOONCAKE_CONFIG_PATH",
        _write_mooncake_config(
            tmp_path,
            {
                "metadata_server": "http://metadata/endpoint",
                "protocol": "tcp",
                "device_name": "",
                "master_server_address": "10.0.0.7:50051",
            },
        ),
    )

    w = worker.MooncakeStoreWorker(
        _make_vllm_config(
            kv_role="kv_consumer",
            extra_config={
                "enable_lookup": False,
                "save_decode_cache": True,
            },
        ),
        _make_kv_cache_config(),
    )

    assert w.can_put is True
    assert w._capacity_only is False


def test_requester_worker_init_builds_replicate_config_for_preferred_segment(
    tmp_path,
    monkeypatch,
):
    store = MagicMock()
    store.setup.return_value = 0
    fake_replicate_config_cls = _install_fake_mooncake(monkeypatch, store)
    _patch_worker_runtime(monkeypatch)
    monkeypatch.setenv(
        "MOONCAKE_CONFIG_PATH",
        _write_mooncake_config(
            tmp_path,
            {
                "metadata_server": "http://metadata/endpoint",
                "protocol": "tcp",
                "device_name": "",
                "master_server_address": "10.0.0.7:50051",
            },
        ),
    )
    w = worker.MooncakeStoreWorker(
        _make_vllm_config(
            extra_config={
                "preferred_segment": "10.0.0.7:50053",
            }
        ),
        _make_kv_cache_config(),
    )

    assert isinstance(w.store_replicate_config, fake_replicate_config_cls)
    assert w.store_replicate_config.preferred_segment == "10.0.0.7:50053"


def test_worker_scales_uniform_attention_group_under_dcp(tmp_path, monkeypatch):
    from vllm.v1.kv_cache_interface import (
        KVCacheConfig,
        UniformTypeKVCacheSpecs,
    )

    store = MagicMock()
    store.setup.return_value = 0
    _install_fake_mooncake(monkeypatch, store)
    _patch_worker_runtime(monkeypatch, tp_size=4, dcp_size=4)
    monkeypatch.setenv(
        "MOONCAKE_CONFIG_PATH",
        _write_mooncake_config(
            tmp_path,
            {
                "metadata_server": "http://metadata/endpoint",
                "protocol": "tcp",
                "device_name": "",
                "master_server_address": "10.0.0.7:50051",
            },
        ),
    )
    inner_specs = {
        "layer0": FullAttentionSpec(
            block_size=16, num_kv_heads=8, head_size=64, dtype=None
        ),
        "layer1": FullAttentionSpec(
            block_size=16, num_kv_heads=4, head_size=64, dtype=None
        ),
    }
    uniform_spec = UniformTypeKVCacheSpecs(
        block_size=16,
        kv_cache_specs=inner_specs,
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=10,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(list(inner_specs), uniform_spec)],
    )

    w = worker.MooncakeStoreWorker(
        _make_vllm_config(decode_context_parallel_size=4),
        kv_cache_config,
    )

    scaled_spec = w._kv_cache_groups[0].kv_cache_spec
    assert isinstance(scaled_spec, UniformTypeKVCacheSpecs)
    assert scaled_spec.block_size == 64
    assert all(spec.block_size == 64 for spec in scaled_spec.kv_cache_specs.values())
    assert w.coord.attention_groups[0].spec.block_size == 64
    assert w.token_dbs[0].block_size == 64


@pytest.mark.parametrize("dcp_size", [1, 4])
def test_worker_put_striding_covers_every_rank_get_namespace(
    tmp_path, monkeypatch, dcp_size
):
    """Every key a rank GETs must have been PUT by some rank.

    When num_kv_head < tp_size, ranks holding the same KV heads stripe
    their PUTs across one shared key namespace. That dedup is only valid
    when those ranks really share a namespace: with DCP > 1 each rank GETs
    every key from its own ``@dcpN`` namespace, so striding must be
    disabled.
    """
    tp_size = 4
    store = MagicMock()
    store.setup.return_value = 0
    _install_fake_mooncake(monkeypatch, store)
    monkeypatch.setenv(
        "MOONCAKE_CONFIG_PATH",
        _write_mooncake_config(
            tmp_path,
            {
                "metadata_server": "http://metadata/endpoint",
                "protocol": "tcp",
                "device_name": "",
                "master_server_address": "10.0.0.7:50051",
            },
        ),
    )

    # _FakeModelConfig has num_kv_head=1 < tp_size, which enables striding.
    block_hashes = [f"hash-{i}".encode() for i in range(4)]
    put_keys: set[str] = set()
    get_keys_per_rank: dict[int, set[str]] = {}
    for tp_rank in range(tp_size):
        _patch_worker_runtime(
            monkeypatch, tp_rank=tp_rank, tp_size=tp_size, dcp_size=dcp_size
        )
        w = worker.MooncakeStoreWorker(
            _make_vllm_config(rank=tp_rank, decode_context_parallel_size=dcp_size),
            _make_kv_cache_config(),
        )
        db = w.token_dbs[0]
        token_len = len(block_hashes) * db.block_size
        keys = [
            PoolKey(db.metadata, block_hash.hex()).to_string()
            for _, _, block_hash in db.process_tokens(token_len, block_hashes)
        ]
        assert len(keys) == len(block_hashes)
        # PUT side: mirrors KVCacheStoreSendingThread's striding slice.
        put_step = w._group_tp_replication_factors[0]
        put_keys.update(keys[w.tp_rank % put_step :: put_step])
        # GET side: KVCacheStoreRecvingThread fetches every key.
        get_keys_per_rank[tp_rank] = set(keys)

    for tp_rank, rank_keys in get_keys_per_rank.items():
        missing = rank_keys - put_keys
        assert not missing, (
            f"tp_rank={tp_rank} would GET {len(missing)}/{len(rank_keys)} keys "
            f"that no rank PUT (Mooncake OBJECT_NOT_FOUND): {sorted(missing)}"
        )


def test_mqa_p4_to_d2_uses_shared_rank_zero_namespace(tmp_path, monkeypatch):
    store = MagicMock()
    store.setup.return_value = 0
    _install_fake_mooncake(monkeypatch, store)
    monkeypatch.setenv(
        "MOONCAKE_CONFIG_PATH",
        _write_mooncake_config(
            tmp_path,
            {
                "metadata_server": "http://metadata/endpoint",
                "protocol": "tcp",
                "master_server_address": "10.0.0.7:50051",
            },
        ),
    )

    block_hash = BlockHash(b"hash")
    put_keys: set[str] = set()
    get_keys: set[str] = set()
    for tp_size, target in ((4, put_keys), (2, get_keys)):
        for tp_rank in range(tp_size):
            _patch_worker_runtime(monkeypatch, tp_rank=tp_rank, tp_size=tp_size)
            w = worker.MooncakeStoreWorker(
                _make_vllm_config(
                    rank=tp_rank,
                    extra_config={"store_tp_size": 4},
                ),
                _make_kv_cache_config(),
            )

            assert w.store_tp_size is None
            assert w.token_dbs[0].metadata.tp_rank == 0
            key = w.token_dbs[0].key_for(block_hash)
            if tp_size == 4:
                put_step = w._group_tp_replication_factors[0]
                if tp_rank % put_step == 0:
                    target.add(key)
            else:
                target.add(key)

    assert (
        put_keys
        == get_keys
        == {
            (
                "test-model@store_pp:1@store_format:tp_shared_mqa"
                "@tp_rank:0@pcp0@dcp0@pp_rank:0@group:0@68617368"
            )
        }
    )


def test_requester_worker_group_semantics_falls_back_without_group_ids(
    tmp_path,
    monkeypatch,
):
    store = MagicMock()
    store.setup.return_value = 0
    _install_fake_mooncake(monkeypatch, store)
    _patch_worker_runtime(monkeypatch)
    warning = MagicMock()
    monkeypatch.setattr(worker.logger, "warning", warning)
    monkeypatch.setenv(
        "MOONCAKE_CONFIG_PATH",
        _write_mooncake_config(
            tmp_path,
            {
                "metadata_server": "http://metadata/endpoint",
                "protocol": "tcp",
                "device_name": "",
                "master_server_address": "10.0.0.7:50051",
            },
        ),
    )

    w = worker.MooncakeStoreWorker(
        _make_vllm_config(
            extra_config={
                "enable_group_semantics": True,
            }
        ),
        _make_kv_cache_config(),
    )

    assert w.enable_group_semantics is True
    assert w._supports_group_ids is False
    assert any(
        "does not support ReplicateConfig.group_ids" in call.args[0]
        for call in warning.call_args_list
    )


def test_requester_worker_group_semantics_string_false_does_not_enable(
    tmp_path,
    monkeypatch,
):
    store = MagicMock()
    store.setup.return_value = 0
    _install_fake_mooncake(monkeypatch, store)
    _patch_worker_runtime(monkeypatch)
    warning = MagicMock()
    monkeypatch.setattr(worker.logger, "warning", warning)
    monkeypatch.setenv(
        "MOONCAKE_CONFIG_PATH",
        _write_mooncake_config(
            tmp_path,
            {
                "metadata_server": "http://metadata/endpoint",
                "protocol": "tcp",
                "device_name": "",
                "master_server_address": "10.0.0.7:50051",
            },
        ),
    )

    w = worker.MooncakeStoreWorker(
        _make_vllm_config(
            extra_config={
                "enable_group_semantics": "false",
            }
        ),
        _make_kv_cache_config(),
    )

    assert w.enable_group_semantics is False
    assert not any(
        "does not support ReplicateConfig.group_ids" in call.args[0]
        for call in warning.call_args_list
    )


def test_requester_worker_group_semantics_string_true_enables(
    tmp_path,
    monkeypatch,
):
    store = MagicMock()
    store.setup.return_value = 0

    class FakeReplicateConfig:
        def __init__(self) -> None:
            self.group_ids = None

    fake_store_module = types.ModuleType("mooncake.store")
    fake_store_module.MooncakeDistributedStore = lambda: store  # type: ignore[attr-defined]
    fake_store_module.ReplicateConfig = FakeReplicateConfig  # type: ignore[attr-defined]
    fake_store_module.ObjectDataType = SimpleNamespace(TENSOR=1)  # type: ignore[attr-defined]
    fake_mooncake_module = types.ModuleType("mooncake")
    fake_mooncake_module.store = fake_store_module  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mooncake", fake_mooncake_module)
    monkeypatch.setitem(sys.modules, "mooncake.store", fake_store_module)
    _patch_worker_runtime(monkeypatch)
    monkeypatch.setenv(
        "MOONCAKE_CONFIG_PATH",
        _write_mooncake_config(
            tmp_path,
            {
                "metadata_server": "http://metadata/endpoint",
                "protocol": "tcp",
                "device_name": "",
                "master_server_address": "10.0.0.7:50051",
            },
        ),
    )

    w = worker.MooncakeStoreWorker(
        _make_vllm_config(
            extra_config={
                "enable_group_semantics": "true",
            }
        ),
        _make_kv_cache_config(),
    )

    assert w.enable_group_semantics is True
    assert w._supports_group_ids is True


# ---------------------------------------------------------------------------
# Helpers for register_kv_caches tests
# ---------------------------------------------------------------------------


def test_store_sending_thread_clamps_token_len_to_lcm():
    """Partial chunks past the last lcm boundary aren't stored — cache hits
    are always lcm-aligned (mirrors HybridKVCacheCoordinator)."""
    store = MagicMock()
    store.batch_is_exist.return_value = [0, 0]
    store.batch_put_from_multi_buffers.return_value = [256, 256]
    # Default coord: single full-attn block_size=16, lcm=16.
    # token_len_chunk=33 clamps to 32 → 2 chunks (not 3 with a partial 1-token chunk).
    thread = _make_store_sending_thread(store)

    _run_store_req(
        thread,
        ReqMeta(
            req_id="r0",
            token_len_chunk=33,
            block_ids=([1, 2, 3],),
            block_hashes=[b"a0", b"a1", b"a2"],
            can_save=True,
        ),
    )

    keys = store.batch_put_from_multi_buffers.call_args.args[0]
    assert len(keys) == 2


def test_store_sending_thread_skips_when_token_len_below_lcm():
    """Requests shorter than lcm_block_size cannot produce any aligned chunk,
    so neither the existence check nor the put should be issued."""
    from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheGroupSpec

    store = MagicMock()
    # lcm=64 via single full-attn block_size=64.
    spec = FullAttentionSpec(block_size=64, num_kv_heads=8, head_size=64, dtype=None)
    coord = mooncake_store_worker.MooncakeStoreCoordinator(
        [KVCacheGroupSpec(["L"], spec)],
        scheduler_block_size=64,
        hash_block_size=64,
    )
    db = ChunkedTokenDatabase(
        KeyMetadata("test-model", 0, 0, 0, 0, group_id=0),
        block_size=64,
        hash_block_size=64,
    )
    db.set_kv_caches_base_addr([0x1000])
    db.set_block_len([1024])
    thread = _make_store_sending_thread(
        store, coord=coord, token_databases=[db], block_size=64
    )

    _run_store_req(
        thread,
        ReqMeta(
            req_id="r0",
            token_len_chunk=32,
            block_ids=([1, 2],),
            block_hashes=[b"a0", b"a1"],
            can_save=True,
        ),
    )

    store.batch_is_exist.assert_not_called()
    store.batch_put_from_multi_buffers.assert_not_called()


def test_store_sending_thread_only_stores_swa_blocks_in_window():
    """For SWA groups, only blocks within ``sliding_window`` of an
    lcm-aligned boundary are stored — a block outside every such window
    can never participate in any future hit (mirrors recv-side masking).

    Setup: full-attn (block_size=32) + SWA (block_size=8, sliding_window=8)
    → lcm=32. With token_len=64 there are two LCM boundaries (32, 64);
    the SWA group should store exactly one block per boundary: the block
    ending at 32 and the block ending at 64.
    """
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        KVCacheGroupSpec,
        SlidingWindowSpec,
    )

    store = MagicMock()
    store.batch_is_exist.side_effect = lambda keys: [0] * len(keys)

    def successful_put(keys, addrs, sizes, *_args):
        return [256] * len(keys)

    store.batch_put_from_multi_buffers.side_effect = successful_put

    full_spec = FullAttentionSpec(
        block_size=32, num_kv_heads=8, head_size=64, dtype=None
    )
    swa_spec = SlidingWindowSpec(
        block_size=8,
        num_kv_heads=8,
        head_size=64,
        dtype=None,
        sliding_window=8,
    )
    coord = mooncake_store_worker.MooncakeStoreCoordinator(
        [KVCacheGroupSpec(["L0"], full_spec), KVCacheGroupSpec(["L1"], swa_spec)],
        scheduler_block_size=32,
        hash_block_size=8,
    )

    db_full = ChunkedTokenDatabase(
        KeyMetadata("test-model", 0, 0, 0, 0, group_id=0),
        block_size=32,
        hash_block_size=8,
    )
    db_full.set_kv_caches_base_addr([0x1000])
    db_full.set_block_len([512])
    db_swa = ChunkedTokenDatabase(
        KeyMetadata("test-model", 0, 0, 0, 0, group_id=1),
        block_size=8,
        hash_block_size=8,
    )
    db_swa.set_kv_caches_base_addr([0x2000])
    db_swa.set_block_len([128])

    thread = _make_store_sending_thread(
        store,
        coord=coord,
        token_databases=[db_full, db_swa],
        block_size=32,
    )

    hs = [bytes([i + 1]) * 4 for i in range(8)]
    _run_store_req(
        thread,
        ReqMeta(
            req_id="r0",
            token_len_chunk=64,
            block_ids=([1, 2], list(range(1, 9))),
            block_hashes=hs,
            can_save=True,
        ),
    )

    keys = store.batch_put_from_multi_buffers.call_args.args[0]
    full_keys = [k for k in keys if "@group:0" in k]
    swa_keys = [k for k in keys if "@group:1" in k]
    # Full-attn: 2 blocks (chunks ending at 32 and 64).
    assert len(full_keys) == 2
    # SWA: only the two blocks ending at lcm boundaries (32 and 64), i.e.
    # blocks covering tokens [24,32) and [56,64) — hashes hs[3] and hs[7].
    assert len(swa_keys) == 2
    swa_hashes = {k.rsplit("@", 1)[-1] for k in swa_keys}
    assert swa_hashes == {hs[3].hex(), hs[7].hex()}


def test_store_sending_thread_delta_saves_only_new_swa_boundary_chunks():
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        KVCacheGroupSpec,
        SlidingWindowSpec,
    )

    store = MagicMock()
    store.batch_is_exist.side_effect = lambda keys: [0] * len(keys)
    store.batch_put_from_multi_buffers.side_effect = (
        lambda keys, addrs, sizes, replicate_config: [256] * len(keys)
    )

    full_spec = FullAttentionSpec(
        block_size=32, num_kv_heads=8, head_size=64, dtype=None
    )
    swa_spec = SlidingWindowSpec(
        block_size=8,
        num_kv_heads=8,
        head_size=64,
        dtype=None,
        sliding_window=8,
    )
    coord = mooncake_store_worker.MooncakeStoreCoordinator(
        [KVCacheGroupSpec(["L0"], full_spec), KVCacheGroupSpec(["L1"], swa_spec)],
        scheduler_block_size=32,
        hash_block_size=8,
    )

    db_full = ChunkedTokenDatabase(
        KeyMetadata("test-model", 0, 0, 0, 0, group_id=0),
        block_size=32,
        hash_block_size=8,
    )
    db_full.set_kv_caches_base_addr([0x1000])
    db_full.set_block_len([512])
    db_swa = ChunkedTokenDatabase(
        KeyMetadata("test-model", 0, 0, 0, 0, group_id=1),
        block_size=8,
        hash_block_size=8,
    )
    db_swa.set_kv_caches_base_addr([0x2000])
    db_swa.set_block_len([128])

    thread = _make_store_sending_thread(
        store,
        coord=coord,
        token_databases=[db_full, db_swa],
        block_size=32,
    )

    hs = [bytes([i + 1]) * 4 for i in range(8)]
    thread._saved_offset["r0"] = 32
    _run_store_req(
        thread,
        ReqMeta(
            req_id="r0",
            token_len_chunk=64,
            block_ids=([1, 2], list(range(1, 9))),
            block_hashes=hs,
            can_save=True,
        ),
    )

    keys = store.batch_put_from_multi_buffers.call_args.args[0]
    full_hashes = [k.rsplit("@", 1)[-1] for k in keys if "@group:0" in k]
    swa_hashes = [k.rsplit("@", 1)[-1] for k in keys if "@group:1" in k]
    assert full_hashes == [hs[7].hex()]
    assert swa_hashes == [hs[7].hex()]


def test_store_sending_thread_kv_events_use_group_chunk_metadata():
    from vllm.v1.core.kv_cache_utils import BlockHash, maybe_convert_block_hash
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        KVCacheGroupSpec,
        SlidingWindowSpec,
    )

    store = MagicMock()
    store.batch_is_exist.side_effect = lambda keys: [0] * len(keys)
    store.batch_put_from_multi_buffers.return_value = [256, 256]

    full_spec = FullAttentionSpec(
        block_size=32, num_kv_heads=8, head_size=64, dtype=None
    )
    swa_spec = SlidingWindowSpec(
        block_size=8,
        num_kv_heads=8,
        head_size=64,
        dtype=None,
        sliding_window=8,
    )
    coord = mooncake_store_worker.MooncakeStoreCoordinator(
        [KVCacheGroupSpec(["L0"], full_spec), KVCacheGroupSpec(["L1"], swa_spec)],
        scheduler_block_size=32,
        hash_block_size=8,
    )

    db_full = ChunkedTokenDatabase(
        KeyMetadata("test-model", 0, 0, 0, 0, group_id=0),
        block_size=32,
        hash_block_size=8,
    )
    db_full.set_kv_caches_base_addr([0x1000])
    db_full.set_block_len([512])
    db_swa = ChunkedTokenDatabase(
        KeyMetadata("test-model", 0, 0, 0, 0, group_id=1),
        block_size=8,
        hash_block_size=8,
    )
    db_swa.set_kv_caches_base_addr([0x2000])
    db_swa.set_block_len([128])

    thread = _make_store_sending_thread(
        store,
        coord=coord,
        token_databases=[db_full, db_swa],
        block_size=32,
    )
    thread.enable_kv_event = True

    hs = [bytes([i + 1]) * 4 for i in range(4)]
    _run_store_req(
        thread,
        ReqMeta(
            req_id="r0",
            token_len_chunk=32,
            block_ids=([1], list(range(1, 5))),
            block_hashes=hs,
            can_save=True,
            token_ids=list(range(32)),
        ),
    )

    full_event, swa_event = thread.get_kv_events()
    assert full_event.group_idx == 0
    assert full_event.block_size == 32
    assert full_event.token_ids == list(range(32))
    # block_size=32 over hash_block_size=8 (scale 4): the chunk is keyed by its
    # last sub-hash, not the concatenation of all four.
    assert full_event.block_hashes == [maybe_convert_block_hash(BlockHash(hs[3]))]

    assert swa_event.group_idx == 1
    assert swa_event.block_size == 8
    assert swa_event.token_ids == list(range(24, 32))
    assert swa_event.block_hashes == [maybe_convert_block_hash(BlockHash(hs[3]))]


def _make_event_store_req(token_len: int, token_ids_start: int = 0) -> ReqMeta:
    num_blocks = token_len // 16
    return ReqMeta(
        req_id="r0",
        token_len_chunk=token_len,
        block_ids=(list(range(1, num_blocks + 1)),),
        block_hashes=[f"a{i}".encode() for i in range(num_blocks)],
        can_save=True,
        token_ids=list(range(token_ids_start, token_len)),
        token_ids_start=token_ids_start,
    )


@pytest.mark.parametrize(
    ("saved_offset", "put_step", "exists", "stored_indices", "parent_indices"),
    [
        pytest.param(16, 1, [0, 0, 0], [1, 2, 3], [0, 1, 2], id="suffix"),
        pytest.param(0, 1, [1, 0, 1, 0], [1, 3], [0, 2], id="dedup-holes"),
        pytest.param(0, 2, [0, 0], [0, 2], [None, 1], id="tp-stride"),
    ],
)
def test_store_sending_thread_kv_events_use_request_chain_parents(
    saved_offset, put_step, exists, stored_indices, parent_indices
):
    store = MagicMock()
    store.batch_is_exist.return_value = exists
    store.batch_put_from_multi_buffers.return_value = [256] * len(stored_indices)
    thread = _make_store_sending_thread(store, put_step=put_step)
    thread.enable_kv_event = True

    thread._saved_offset["r0"] = saved_offset
    _run_store_req(thread, _make_event_store_req(64, saved_offset))

    events = thread.get_kv_events()
    assert [event.block_hashes for event in events] == [
        [maybe_convert_block_hash(BlockHash(f"a{i}".encode()))] for i in stored_indices
    ]
    assert [event.parent_block_hash for event in events] == [
        (None if i is None else maybe_convert_block_hash(BlockHash(f"a{i}".encode())))
        for i in parent_indices
    ]
    assert thread._retry_token_ids == {}


def test_store_sending_thread_kv_events_retry_without_covered_tokens():
    store = MagicMock()
    store.batch_is_exist.return_value = [0, 0]
    store.batch_put_from_multi_buffers.return_value = [256, 256]
    thread = _make_store_sending_thread(store)
    thread.enable_kv_event = True

    _run_store_req(thread, _make_event_store_req(32, 16))

    retry_event, suffix_event = thread.get_kv_events()
    assert retry_event.token_ids == []
    assert suffix_event.token_ids == list(range(16, 32))


def test_store_sending_thread_kv_events_recover_suffix_after_put_failure():
    store = MagicMock()
    store.batch_is_exist.side_effect = ([0], [0, 0])
    store.batch_put_from_multi_buffers.side_effect = ([-1], [256, 256])
    thread = _make_store_sending_thread(store)
    thread.enable_kv_event = True

    _run_store_req(thread, _make_event_store_req(16))

    assert thread.get_kv_events() == []
    assert thread._retry_token_ids["r0"] == (0, list(range(16)))

    _run_store_req(thread, _make_event_store_req(32, 16))

    retry_event, suffix_event = thread.get_kv_events()
    assert retry_event.token_ids == list(range(16))
    assert suffix_event.token_ids == list(range(16, 32))
    assert thread._retry_token_ids == {}
    assert thread._saved_offset["r0"] == 32


def _auto_set_ready_event(*args, **kwargs):
    """Side effect for mocked thread constructors that auto-sets ready_event."""
    for arg in args:
        if isinstance(arg, threading.Event):
            arg.set()
    for val in kwargs.values():
        if isinstance(val, threading.Event):
            val.set()
    return MagicMock()


def _register_with_mocked_threads(
    worker: mooncake_store_worker.MooncakeStoreWorker,
    kv_caches: dict[str, torch.Tensor],
) -> None:
    """Call register_kv_caches with the I/O transfer threads mocked out."""
    prefix = "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.worker."
    with (
        patch(prefix + "KVCacheStoreSendingThread", side_effect=_auto_set_ready_event),
        patch(prefix + "KVCacheStoreRecvingThread", side_effect=_auto_set_ready_event),
    ):
        worker.register_kv_caches(kv_caches)


def _refresh_group_tp_replication_factors(
    worker: mooncake_store_worker.MooncakeStoreWorker,
) -> None:
    worker._group_tp_replication_factors = (
        worker._compute_group_tp_replication_factors()
    )
    worker._init_lookup_key_prefixes()


def _make_bare_worker(
    *,
    num_gpu_blocks: int = 10,
    block_size: int = 16,
    kv_role: str = "kv_both",
    save_decode_cache: bool = False,
) -> mooncake_store_worker.MooncakeStoreWorker:
    """Construct a MooncakeStoreWorker via __new__, bypassing __init__.

    Sets only the attributes that register_kv_caches() reads so we can
    test the stride-based layout detection without a real
    MooncakeDistributedStore.
    """
    worker = object.__new__(mooncake_store_worker.MooncakeStoreWorker)
    worker.cache_config = MagicMock()
    worker.cache_config.num_gpu_blocks = num_gpu_blocks
    worker.store = MagicMock()
    worker.store.register_buffer.return_value = 0
    worker.kv_role = kv_role
    worker.can_put = kv_role in ("kv_producer", "kv_both") or save_decode_cache
    worker.external_cache_only = False
    worker.retire_mamba_checkpoints = False
    worker.checkpoint_retirer = None
    worker._is_checkpoint_retirement_rank = True
    worker._capacity_only = False
    worker.block_size = block_size
    worker._is_hma_required = False
    worker.tp_rank = 0
    worker.enable_kv_events = False
    worker.load_async = True
    worker.kv_send_thread = None
    worker.kv_recv_threads = []
    worker.num_recv_threads = 1
    worker.recv_request_queue = queue.Queue()
    worker.load_pipeline = PipelineTracker("load", 0)
    worker.finished_store_req = set()
    worker.tp_size = 1
    worker.store_tp_size = None
    worker.num_kv_head = 1
    worker.pp_size = 1
    # Minimal single-full-attention-group config so the coordinator-based
    # lookup path works (the connector no longer carries a legacy single-group
    # path; everything flows through the coordinator).
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        KVCacheConfig,
        KVCacheGroupSpec,
        KVCacheTensor,
    )

    worker.disk_offload_buffer_budget_bytes = None
    worker.store_replicate_config = SimpleNamespace()
    worker.enable_group_semantics = False
    worker._supports_group_ids = False
    worker._kv_connector_stats_lock = threading.Lock()
    worker.kv_connector_stats = MooncakeStoreConnectorStats()

    spec = FullAttentionSpec(
        block_size=block_size, num_kv_heads=8, head_size=64, dtype=None
    )
    group = KVCacheGroupSpec(["layer0", "__cross_layer__"], spec)
    worker._kv_cache_config = KVCacheConfig(
        num_blocks=num_gpu_blocks,
        kv_cache_tensors=[
            KVCacheTensor(
                size=0,
                layers=["layer0", "__cross_layer__"],
                layer_stride=0,
                block_stride=0,
            )
        ],
        kv_cache_groups=[group],
    )
    worker._kv_cache_groups = [group]
    worker.pcp_size = 1
    worker.dcp_size = 1
    worker.hash_block_size = block_size
    # Pre-build a single-group token_dbs so lookup-only tests don't have to
    # call register_kv_caches.
    worker.token_dbs = [
        ChunkedTokenDatabase(
            KeyMetadata("test-model", 0, 0, 0, 0, group_id=0),
            block_size=block_size,
            hash_block_size=block_size,
        )
    ]
    worker.coord = mooncake_store_worker.MooncakeStoreCoordinator(
        worker._kv_cache_groups,
        scheduler_block_size=block_size,
        hash_block_size=block_size,
    )
    _refresh_group_tp_replication_factors(worker)
    return worker


def test_lookup_key_prefixes_cover_dcp_rank_namespaces():
    worker = _make_bare_worker()
    worker.tp_size = 4
    worker.num_kv_head = 1
    worker.dcp_size = 4
    _refresh_group_tp_replication_factors(worker)

    assert worker._lookup_key_prefixes[0] == (
        "test-model@tp_rank:0@pcp0@dcp0@pp_rank:0@group:0",
        "test-model@tp_rank:1@pcp0@dcp1@pp_rank:0@group:0",
        "test-model@tp_rank:2@pcp0@dcp2@pp_rank:0@group:0",
        "test-model@tp_rank:3@pcp0@dcp3@pp_rank:0@group:0",
    )


def test_lookup_key_prefixes_cover_store_tp_shards():
    worker = _make_bare_worker()
    worker.store_tp_size = 4
    metadata = KeyMetadata(
        "test-model",
        0,
        0,
        0,
        0,
        group_id=0,
        store_namespace=_TP_SHARED_NAMESPACE,
    )
    worker.token_dbs = [
        ChunkedTokenDatabase(
            metadata,
            block_size=16,
            store_layout=LBHNCStoreLayout(
                metadata,
                block_size=16,
                hash_block_size=16,
                local_tp_size=2,
                store_tp_size=4,
                tp_rank=0,
                num_kv_heads=8,
            ),
        )
    ]
    worker._init_lookup_key_prefixes()

    assert worker._lookup_key_prefixes[0] == (
        *(_tp_shared_prefix(tp_rank) for tp_rank in range(4)),
    )


def test_lookup_key_prefixes_cover_pcp_rank_namespaces():
    worker = _make_bare_worker()
    worker.tp_size = 4
    worker.num_kv_head = 1
    worker.pcp_size = 2
    worker.dcp_size = 1
    _refresh_group_tp_replication_factors(worker)

    assert worker._lookup_key_prefixes[0] == (
        "test-model@tp_rank:0@pcp0@dcp0@pp_rank:0@group:0",
        "test-model@tp_rank:0@pcp1@dcp0@pp_rank:0@group:0",
    )


def test_lookup_key_prefixes_expand_tp_sharded_groups_per_rank():
    """Replicated attention needs one namespace; sharded Mamba needs every rank."""
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        KVCacheGroupSpec,
        MambaSpec,
    )

    worker = _make_bare_worker(block_size=16)
    worker.tp_size = 2
    worker.num_kv_head = 1
    fa = FullAttentionSpec(block_size=16, num_kv_heads=8, head_size=64, dtype=None)
    mamba = MambaSpec(
        block_size=16,
        shapes=((1, 1),),
        dtypes=(torch.float32,),
        mamba_cache_mode="align",
    )
    worker._kv_cache_groups = [
        KVCacheGroupSpec(["l0"], fa),
        KVCacheGroupSpec(["l1"], mamba),
    ]
    worker.token_dbs = [
        ChunkedTokenDatabase(
            KeyMetadata("test-model", 0, 0, 0, 0, group_id=0), block_size=16
        ),
        ChunkedTokenDatabase(
            KeyMetadata("test-model", 1, 0, 0, 0, group_id=1), block_size=16
        ),
    ]
    _refresh_group_tp_replication_factors(worker)

    assert worker._lookup_key_prefixes[0] == (
        "test-model@tp_rank:0@pcp0@dcp0@pp_rank:0@group:0",
    )
    assert worker._lookup_key_prefixes[1] == (
        "test-model@tp_rank:0@pcp0@dcp0@pp_rank:0@group:1",
        "test-model@tp_rank:1@pcp0@dcp0@pp_rank:0@group:1",
    )


def test_group_tp_replication_factors_mixed_mla_gqa_mamba():
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        KVCacheGroupSpec,
        MambaSpec,
        MLAAttentionSpec,
    )

    worker = _make_bare_worker(block_size=16)
    worker.tp_size = 4
    worker.num_kv_head = 2
    mla = MLAAttentionSpec(block_size=16, num_kv_heads=1, head_size=64, dtype=None)
    gqa = FullAttentionSpec(block_size=16, num_kv_heads=8, head_size=64, dtype=None)
    mamba = MambaSpec(
        block_size=16,
        shapes=((1, 1),),
        dtypes=(torch.float32,),
        mamba_cache_mode="align",
    )
    worker._kv_cache_groups = [
        KVCacheGroupSpec(["l0"], mla),
        KVCacheGroupSpec(["l1"], gqa),
        KVCacheGroupSpec(["l2"], mamba),
    ]
    worker.token_dbs = [
        ChunkedTokenDatabase(
            KeyMetadata("test-model", 0, 0, 0, 0, group_id=g_idx), block_size=16
        )
        for g_idx in range(3)
    ]

    _refresh_group_tp_replication_factors(worker)
    assert worker._group_tp_replication_factors == (4, 2, 1)
    assert worker._lookup_key_prefixes[0] == (
        "test-model@tp_rank:0@pcp0@dcp0@pp_rank:0@group:0",
    )
    assert worker._lookup_key_prefixes[1] == (
        "test-model@tp_rank:0@pcp0@dcp0@pp_rank:0@group:1",
        "test-model@tp_rank:1@pcp0@dcp0@pp_rank:0@group:1",
    )
    assert worker._lookup_key_prefixes[2] == (
        "test-model@tp_rank:0@pcp0@dcp0@pp_rank:0@group:2",
        "test-model@tp_rank:1@pcp0@dcp0@pp_rank:0@group:2",
        "test-model@tp_rank:2@pcp0@dcp0@pp_rank:0@group:2",
        "test-model@tp_rank:3@pcp0@dcp0@pp_rank:0@group:2",
    )


@pytest.mark.parametrize("spec_order", [("mla", "gqa"), ("gqa", "mla")])
def test_uniform_group_uses_common_inner_replication_factor(spec_order):
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        KVCacheGroupSpec,
        MLAAttentionSpec,
        UniformTypeKVCacheSpecs,
    )

    worker = _make_bare_worker(block_size=16)
    worker.tp_size = 4
    worker.num_kv_head = 2
    specs_by_name = {
        "mla": MLAAttentionSpec(
            block_size=16, num_kv_heads=1, head_size=64, dtype=None
        ),
        "gqa": FullAttentionSpec(
            block_size=16, num_kv_heads=1, head_size=64, dtype=None
        ),
    }
    inner_specs = {name: specs_by_name[name] for name in spec_order}
    uniform_spec = UniformTypeKVCacheSpecs(
        block_size=16,
        kv_cache_specs=inner_specs,
    )
    worker._kv_cache_groups = [
        KVCacheGroupSpec(list(inner_specs), uniform_spec),
    ]

    _refresh_group_tp_replication_factors(worker)

    assert worker._group_tp_replication_factors == (2,)
    assert worker._lookup_key_prefixes[0] == (
        "test-model@tp_rank:0@pcp0@dcp0@pp_rank:0@group:0",
        "test-model@tp_rank:1@pcp0@dcp0@pp_rank:0@group:0",
    )


def _make_hybrid_lookup_worker(*, hash_block_size=16):
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        KVCacheGroupSpec,
        MambaSpec,
    )

    worker = _make_bare_worker(block_size=16)
    worker.hash_block_size = hash_block_size
    worker.tp_size = 2
    worker.num_kv_head = 1
    fa = FullAttentionSpec(block_size=16, num_kv_heads=8, head_size=64, dtype=None)
    mamba = MambaSpec(
        block_size=16,
        shapes=((1, 1),),
        dtypes=(torch.float32,),
        mamba_cache_mode="align",
    )
    worker._kv_cache_groups = [
        KVCacheGroupSpec(["l0"], fa),
        KVCacheGroupSpec(["l1"], mamba),
    ]
    worker.token_dbs = [
        ChunkedTokenDatabase(
            KeyMetadata("test-model", 0, 0, 0, 0, group_id=0),
            block_size=16,
            hash_block_size=hash_block_size,
        ),
        ChunkedTokenDatabase(
            KeyMetadata("test-model", 1, 0, 0, 0, group_id=1),
            block_size=16,
            hash_block_size=hash_block_size,
        ),
    ]
    worker.coord = mooncake_store_worker.MooncakeStoreCoordinator(
        worker._kv_cache_groups,
        scheduler_block_size=16,
        hash_block_size=hash_block_size,
    )
    _refresh_group_tp_replication_factors(worker)

    return worker


def test_lookup_rejects_boundary_missing_one_mamba_shard():
    worker = _make_hybrid_lookup_worker()
    # 33 tokens for two 16-token blocks: the hit stops below the request end,
    # so the full-hit re-derivation stays out of the shard accounting.
    worker.store.batch_is_exist.side_effect = lambda keys: [1] * len(keys)
    assert worker.lookup(33, [b"h0", b"h1"]).hit_length == 32

    worker.store.batch_is_exist.side_effect = lambda keys: [
        0 if "tp_rank:1" in k and "group:1" in k else 1 for k in keys
    ]
    assert worker.lookup(33, [b"h0", b"h1"]).hit_length == 0


def test_lookup_requires_all_dcp_rank_namespaces():
    worker = _make_bare_worker(block_size=16)
    worker.tp_size = 4
    worker.num_kv_head = 1
    worker.dcp_size = 4
    _refresh_group_tp_replication_factors(worker)
    worker.store.batch_is_exist.return_value = [1, 1, 0, 1]

    assert worker.lookup(16, [b"a0"]).hit_length == 0
    assert worker.store.batch_is_exist.call_args.args[0] == [
        "test-model@tp_rank:0@pcp0@dcp0@pp_rank:0@group:0@6130",
        "test-model@tp_rank:1@pcp0@dcp1@pp_rank:0@group:0@6130",
        "test-model@tp_rank:2@pcp0@dcp2@pp_rank:0@group:0@6130",
        "test-model@tp_rank:3@pcp0@dcp3@pp_rank:0@group:0@6130",
    ]


def test_lookup_partial_prefix_returns_first_hit_length():
    worker = _make_bare_worker()
    worker.store.batch_is_exist.return_value = [1, 1, 0]
    assert worker.lookup(48, [b"a0", b"a1", b"a2"]).hit_length == 32


def test_lookup_partial_tail_uses_hash_alignment():
    """A stored sub-block tail can serve a request extending past it."""
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        KVCacheGroupSpec,
        MambaSpec,
    )

    worker = _make_bare_worker(block_size=16)
    full = FullAttentionSpec(block_size=16, num_kv_heads=8, head_size=64, dtype=None)
    mamba = MambaSpec(
        block_size=16,
        shapes=((1, 1),),
        dtypes=(torch.float32,),
        mamba_cache_mode="align",
    )
    worker._kv_cache_groups = [
        KVCacheGroupSpec(["full"], full),
        KVCacheGroupSpec(["mamba"], mamba),
    ]
    worker.hash_block_size = 4
    worker.token_dbs = [
        ChunkedTokenDatabase(
            KeyMetadata("test-model", 0, 0, 0, 0, group_id=group_id),
            block_size=16,
            hash_block_size=4,
        )
        for group_id in range(2)
    ]
    worker.coord = mooncake_store_worker.MooncakeStoreCoordinator(
        worker._kv_cache_groups,
        scheduler_block_size=16,
        hash_block_size=4,
    )
    _refresh_group_tp_replication_factors(worker)
    worker.store.batch_is_exist.return_value = [0, 0, 1, 0, 0, 1]

    assert worker.lookup(13, [b"h0", b"h1", b"h2"]).hit_length == 12


def test_lookup_full_hit_reuses_existing_boundary():
    """A full hit is re-derived below the request end without another RPC."""
    worker = _make_bare_worker(block_size=16)
    worker.store.batch_is_exist.return_value = [1, 1]

    assert worker.lookup(32, [b"h0", b"h1"]) == MooncakeLookupResult(
        hit_length=16,
        tail_key_boundaries=(TailKeyBoundary(group_id=0, num_tokens=16),),
    )
    assert worker.store.batch_is_exist.call_count == 1


def test_lookup_full_hit_with_eagle_pops_once_not_twice():
    """Eagle already leaves the last block for the drafter, so a
    full-prompt re-derivation must never fire for eagle-governed hits:
    firing would anchor the search one block lower and pop a second
    block, regressing the hit by an extra producer boundary."""
    worker = _make_bare_worker(block_size=16)
    worker.coord = mooncake_store_worker.MooncakeStoreCoordinator(
        worker._kv_cache_groups,
        scheduler_block_size=16,
        hash_block_size=16,
        use_eagle=True,
    )
    worker.store.batch_is_exist.return_value = [1, 1, 1, 1]

    # 64-token exact-multiple prompt, all 4 blocks stored: one eagle pop
    # gives 48; a spurious re-derivation (anchored at 48) would pop again
    # and return 32.
    assert worker.lookup(64, [b"h0", b"h1", b"h2", b"h3"]).hit_length == 48
    assert worker.store.batch_is_exist.call_count == 1


def test_lookup_plan_resolves_group_tail_keys_from_existing_hashes():
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        KVCacheGroupSpec,
        MambaSpec,
    )

    worker = _make_bare_worker(block_size=16)
    full = FullAttentionSpec(block_size=16, num_kv_heads=8, head_size=64, dtype=None)
    mamba = MambaSpec(
        block_size=16,
        shapes=((1, 1),),
        dtypes=(torch.float32,),
        mamba_cache_mode="align",
    )
    worker._kv_cache_groups = [
        KVCacheGroupSpec(["full"], full),
        KVCacheGroupSpec(["mamba"], mamba),
    ]
    worker.hash_block_size = 4
    worker.token_dbs = [
        ChunkedTokenDatabase(
            KeyMetadata("test-model", 0, 0, 0, 0, group_id=group_id),
            block_size=16,
            hash_block_size=4,
        )
        for group_id in range(2)
    ]
    worker.coord = mooncake_store_worker.MooncakeStoreCoordinator(
        worker._kv_cache_groups,
        scheduler_block_size=16,
        hash_block_size=4,
        use_eagle=True,
    )
    _refresh_group_tp_replication_factors(worker)
    hashes = [BlockHash(f"h{i}".encode()) for i in range(6)]
    present = {
        (0, bytes(hashes[3])),
        (0, bytes(hashes[5])),
        (1, bytes(hashes[4])),
        # A later Mamba state also exists, but its exact hit-boundary hash
        # must win and therefore needs no override.
        (1, bytes(hashes[5])),
    }

    def exists(keys):
        return [
            int(
                any(
                    f"@group:{group_id}@{block_hash.hex()}" in key
                    for group_id, block_hash in present
                )
            )
            for key in keys
        ]

    worker.store.batch_is_exist.side_effect = exists

    result = worker.lookup(25, hashes)

    # The eagle drop trims the hit to 20 tokens, but the block that survives
    # truncation is the one keyed at the 24-token boundary (hashes[5]), which
    # the load path cannot derive from hit_length.
    assert result.hit_length == 20
    assert result.tail_key_boundaries == (
        TailKeyBoundary(group_id=0, num_tokens=24),
        TailKeyBoundary(group_id=1, num_tokens=20),
    )


def test_lookup_plan_recovers_tail_key_after_multi_chunk_convergence():
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        KVCacheGroupSpec,
        MambaSpec,
    )

    worker = _make_bare_worker(block_size=16)
    full = FullAttentionSpec(block_size=16, num_kv_heads=8, head_size=64, dtype=None)
    mamba = MambaSpec(
        block_size=16,
        shapes=((1, 1),),
        dtypes=(torch.float32,),
        mamba_cache_mode="align",
    )
    worker._kv_cache_groups = [
        KVCacheGroupSpec(["full"], full),
        KVCacheGroupSpec(["mamba"], mamba),
    ]
    worker.hash_block_size = 4
    worker.token_dbs = [
        ChunkedTokenDatabase(
            KeyMetadata("test-model", 0, 0, 0, 0, group_id=group_id),
            block_size=16,
            hash_block_size=4,
        )
        for group_id in range(2)
    ]
    worker.coord = mooncake_store_worker.MooncakeStoreCoordinator(
        worker._kv_cache_groups,
        scheduler_block_size=16,
        hash_block_size=4,
    )
    _refresh_group_tp_replication_factors(worker)
    hashes = [BlockHash(f"h{i}".encode()) for i in range(12)]
    present = {
        (0, bytes(hashes[3])),
        (0, bytes(hashes[7])),
        (0, bytes(hashes[11])),
        (1, bytes(hashes[4])),
    }

    def exists(keys):
        return [
            int(
                any(
                    f"@group:{group_id}@{block_hash.hex()}" in key
                    for group_id, block_hash in present
                )
            )
            for key in keys
        ]

    worker.store.batch_is_exist.side_effect = exists

    result = worker.lookup(49, hashes)

    assert result.hit_length == 20
    assert result.tail_key_boundaries == (
        TailKeyBoundary(group_id=0, num_tokens=32),
        TailKeyBoundary(group_id=1, num_tokens=20),
    )


def test_lookup_full_hit_swa_degrades_when_no_stored_boundary_is_usable():
    """The motivating livelock: the producer of a 64-token prompt stored
    only its SWA tail window (blocks 2-3). The old arithmetic clamp turned
    the full hit into 48, whose SWA window needs the never-written block 1,
    so every load failed and the recompute re-entered the same lookup. The
    re-derivation must report that no stored boundary below the request end
    is usable."""
    from vllm.v1.kv_cache_interface import KVCacheGroupSpec, SlidingWindowSpec

    worker = _make_bare_worker(block_size=16)
    swa = SlidingWindowSpec(
        block_size=16, num_kv_heads=8, head_size=64, dtype=None, sliding_window=32
    )
    worker._kv_cache_groups = [KVCacheGroupSpec(["layer0"], swa)]
    worker.coord = mooncake_store_worker.MooncakeStoreCoordinator(
        worker._kv_cache_groups,
        scheduler_block_size=worker.hash_block_size,
        hash_block_size=worker.hash_block_size,
    )
    worker.store.batch_is_exist.return_value = [0, 0, 1, 1]

    assert worker.lookup(64, [b"h0", b"h1", b"h2", b"h3"]).hit_length == 0
    assert worker.store.batch_is_exist.call_count == 1


def test_lookup_swa_single_group_returns_full_when_tail_window_present():
    """Single-SWA, sliding_window=32 (= 2 blocks): producer stored only the
    tail. Coordinator-driven lookup returns full prefix even though the
    pre-window blocks are absent."""
    from vllm.v1.kv_cache_interface import KVCacheGroupSpec, SlidingWindowSpec

    worker = _make_bare_worker(block_size=16)
    swa = SlidingWindowSpec(
        block_size=16, num_kv_heads=8, head_size=64, dtype=None, sliding_window=32
    )
    worker._kv_cache_groups = [KVCacheGroupSpec(["layer0"], swa)]
    worker.coord = mooncake_store_worker.MooncakeStoreCoordinator(
        worker._kv_cache_groups,
        scheduler_block_size=worker.hash_block_size,
        hash_block_size=worker.hash_block_size,
    )
    worker.store.batch_is_exist.return_value = [0, 0, 1, 1]
    assert worker.lookup(65, [b"h0", b"h1", b"h2", b"h3"]).hit_length == 64


def test_lookup_checks_all_potential_swa_hit_boundaries():
    """Lookup should skip SWA chunks that can never validate a hit, but still
    check earlier aligned boundaries when sparse retention stores only the
    current request's replay boundary.
    """
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        KVCacheGroupSpec,
        SlidingWindowSpec,
    )

    worker = _make_bare_worker(block_size=8)
    full = FullAttentionSpec(block_size=32, num_kv_heads=8, head_size=64, dtype=None)
    swa = SlidingWindowSpec(
        block_size=8, num_kv_heads=8, head_size=64, dtype=None, sliding_window=8
    )
    worker._kv_cache_groups = [
        KVCacheGroupSpec(["full"], full),
        KVCacheGroupSpec(["swa"], swa),
    ]
    worker.token_dbs = [
        ChunkedTokenDatabase(
            KeyMetadata("test-model", 0, 0, 0, 0, group_id=0),
            block_size=32,
            hash_block_size=8,
        ),
        ChunkedTokenDatabase(
            KeyMetadata("test-model", 0, 0, 0, 0, group_id=1),
            block_size=8,
            hash_block_size=8,
        ),
    ]
    worker.coord = mooncake_store_worker.MooncakeStoreCoordinator(
        worker._kv_cache_groups,
        scheduler_block_size=32,
        hash_block_size=8,
        retention_interval=0,
    )
    _refresh_group_tp_replication_factors(worker)
    # Candidate order: 3 full-attention chunks, then SWA chunks 3, 7, 11.
    # Only the first full chunk and the SWA chunk ending at token 32 exist, so
    # lookup should recover a 32-token external prefix hit. A sparse
    # prompt-specific store mask for num_prompt_tokens=96 would only check SWA
    # chunk 7 and miss this earlier reusable prefix.
    worker.store.batch_is_exist.return_value = [1, 0, 0, 1, 0, 0]

    result = worker.lookup(
        96,
        [f"h{i}".encode() for i in range(12)],
    )

    assert result.hit_length == 32
    keys = worker.store.batch_is_exist.call_args.args[0]
    assert len(keys) == 6
    swa_keys = [key for key in keys if "@group:1@" in key]
    assert swa_keys == [
        "test-model@tp_rank:0@pcp0@dcp0@pp_rank:0@group:1@6833",
        "test-model@tp_rank:0@pcp0@dcp0@pp_rank:0@group:1@6837",
        "test-model@tp_rank:0@pcp0@dcp0@pp_rank:0@group:1@683131",
    ]


def test_lookup_applies_swa_mask_before_accessing_hashes():
    """Lookup must apply the sparse SWA mask before touching group hashes, so
    false-mask chunks pay neither the hash access nor the key-construction cost.
    """
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        KVCacheGroupSpec,
        SlidingWindowSpec,
    )

    worker = _make_bare_worker(block_size=8)
    full = FullAttentionSpec(block_size=32, num_kv_heads=8, head_size=64, dtype=None)
    swa = SlidingWindowSpec(
        block_size=8, num_kv_heads=8, head_size=64, dtype=None, sliding_window=8
    )
    worker._kv_cache_groups = [
        KVCacheGroupSpec(["full"], full),
        KVCacheGroupSpec(["swa"], swa),
    ]
    worker.token_dbs = [
        ChunkedTokenDatabase(
            KeyMetadata("test-model", 0, 0, 0, 0, group_id=0),
            block_size=32,
            hash_block_size=8,
        ),
        ChunkedTokenDatabase(
            KeyMetadata("test-model", 0, 0, 0, 0, group_id=1),
            block_size=8,
            hash_block_size=8,
        ),
    ]
    worker.coord = mooncake_store_worker.MooncakeStoreCoordinator(
        worker._kv_cache_groups,
        scheduler_block_size=32,
        hash_block_size=8,
        retention_interval=0,
    )
    _refresh_group_tp_replication_factors(worker)

    block_hashes = _RecordingBlockHashes([f"h{i}".encode() for i in range(12)])
    accessed_before_rpc: list[int] = []

    def exists(keys):
        accessed_before_rpc.extend(block_hashes.accessed)
        return [0] * len(keys)

    worker.store.batch_is_exist.side_effect = exists

    worker.lookup(96, block_hashes)

    # Full-attention chunks (compact hashes at 3, 7, 11) plus only the reachable
    # SWA boundary tails (chunks 3, 7, 11) are hash-accessed. Every masked SWA
    # chunk in between is skipped before both the hash access and key build.
    assert accessed_before_rpc == [3, 7, 11, 3, 7, 11]


# ---------------------------------------------------------------------------
# register_kv_caches tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("save_decode_cache", [False, True])
def test_consumer_starts_send_thread_only_when_put_is_enabled(save_decode_cache):
    num_blocks = 10
    tensor = torch.zeros(num_blocks, 64, dtype=torch.float16)

    worker = _make_bare_worker(
        kv_role="kv_consumer", save_decode_cache=save_decode_cache
    )
    _register_with_mocked_threads(worker, {"layer0": tensor})
    assert (worker.kv_send_thread is not None) == save_decode_cache


def test_start_load_kv_queues_async_load():
    w = _make_bare_worker()
    req = _make_load_req(
        "load-req",
        [b"h0", b"h1"],
        token_len=32,
    )
    assert req.load_spec is not None
    req.load_spec.token_len = 0
    meta = mooncake_store_worker.MooncakeStoreConnectorMetadata(set(), set())
    meta.add_request(req)

    w.start_load_kv(meta)

    assert req.load_spec.token_len == 32
    queued, ticket = w.recv_request_queue.get_nowait()
    assert queued is req
    assert w.load_pipeline.snapshot()["queued"] == 1
    w.load_pipeline.start(ticket)
    w.load_pipeline.finish(ticket)


def test_putting_consumer_queues_decode_save():
    w = _make_bare_worker(kv_role="kv_consumer", save_decode_cache=True)
    send_thread = MagicMock()
    w.kv_send_thread = send_thread
    req = _make_store_req("decode-req", [b"h0", b"h1"])
    req.store_job_id = 1
    meta = mooncake_store_worker.MooncakeStoreConnectorMetadata(set(), set())
    meta.add_request(req)

    with patch.object(torch.cuda, "Event") as event_cls:
        event = event_cls.return_value
        w.wait_for_save(meta)

    event.record.assert_called_once_with()
    send_thread.add_request.assert_called_once_with(req)
    assert req.current_event is event


@pytest.mark.parametrize("layout", list(KVCacheLayout))
def test_register_kv_caches_shared_storage(layout: KVCacheLayout):
    num_blocks = 10
    num_layers = 2
    worker = _make_bare_worker(num_gpu_blocks=num_blocks)
    spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=2,
        head_size=8,
        dtype=torch.float16,
    )
    raw = torch.zeros(
        num_blocks * num_layers * spec.page_size_bytes,
        dtype=torch.int8,
    )
    caches = dense_kv_cache_views(raw, spec, num_blocks, num_layers, layout)

    _register_with_mocked_threads(
        worker,
        {"layer0": caches[0], "__cross_layer__": caches[1]},
    )

    db = worker.token_dbs[0]
    if not layout.is_block_compact:
        # Each head group is its own region, so a block spans L*H regions.
        head_cache = caches[0][:, 0]
        assert db.kv_caches_base_addr == [
            cache[:, head_idx].data_ptr()
            for cache in caches
            for head_idx in range(cache.shape[1])
        ]
        assert db.block_len == [head_cache.stride(0) * head_cache.element_size()] * (
            num_layers * caches[0].shape[1]
        )
    elif layout.is_layer_compact:
        assert db.kv_caches_base_addr == [cache.data_ptr() for cache in caches]
        assert db.block_len == [spec.page_size_bytes] * num_layers
    else:
        assert db.kv_caches_base_addr == [raw.data_ptr()]
        assert db.block_len == [num_layers * spec.page_size_bytes]
    worker.store.register_buffer.assert_called_once_with(raw.data_ptr(), raw.nbytes)


def test_register_kv_caches_uses_transfer_group_memory_domain():
    """Derived GPU caches must not change host-source transfer addresses."""
    from vllm.v1.kv_cache_interface import (
        KVCacheConfig,
        KVCacheGroupRole,
        KVCacheGroupSpec,
        KVCacheTensor,
    )

    host_num_blocks = 4
    gpu_num_blocks = 2
    page_size = 32
    source_spec = FullAttentionSpec(
        block_size=16, num_kv_heads=1, head_size=1, dtype=torch.float16
    )
    indexer_spec = FullAttentionSpec(
        block_size=16, num_kv_heads=1, head_size=1, dtype=torch.float16
    )
    source_group = KVCacheGroupSpec(
        ["source"],
        source_spec,
        host_resident=True,
    )
    indexer_group = KVCacheGroupSpec(
        ["indexer"],
        indexer_spec,
        enable_kv_transfer=False,
        role=KVCacheGroupRole.HISPARSE_INDEXER,
    )
    config = KVCacheConfig(
        num_blocks=gpu_num_blocks,
        kv_cache_tensors=[
            KVCacheTensor(
                size=host_num_blocks * page_size,
                layers=["source"],
                layer_stride=host_num_blocks * page_size,
                block_stride=page_size,
                host_resident=True,
            ),
            KVCacheTensor(
                size=gpu_num_blocks * page_size,
                layers=["indexer"],
                layer_stride=gpu_num_blocks * page_size,
                block_stride=page_size,
            ),
        ],
        kv_cache_groups=[source_group, indexer_group],
        hisparse_host_num_blocks=host_num_blocks,
    )
    worker = _make_bare_worker(num_gpu_blocks=gpu_num_blocks)
    worker._kv_cache_config = config
    worker._kv_cache_groups = list(config.transfer_groups)
    worker.token_dbs = [
        ChunkedTokenDatabase(KeyMetadata("test-model", 0, 0, 0, 0), block_size=16)
    ]
    source = torch.zeros(host_num_blocks, page_size, dtype=torch.uint8)
    indexer = torch.zeros(gpu_num_blocks, page_size, dtype=torch.uint8)

    _register_with_mocked_threads(
        worker,
        {"source": source, "indexer": indexer},
    )

    db = worker.token_dbs[0]
    assert db.kv_caches_base_addr == [source.data_ptr()]
    assert db.block_len == [page_size]
    worker.store.register_buffer.assert_called_once_with(
        source.data_ptr(), source.nbytes
    )


def test_register_kv_caches_excludes_nonprefix_cacheable_layers():
    num_blocks = 2
    worker = _make_bare_worker(num_gpu_blocks=num_blocks)
    store_layout = MagicMock()
    worker.token_dbs[0].store_layout = store_layout
    full_cache = torch.zeros(num_blocks, 4, dtype=torch.float16)
    qsa_cache = torch.zeros(num_blocks, 4, dtype=torch.float16)

    _register_with_mocked_threads(
        worker,
        {"layer0": full_cache, "qsa": qsa_cache},
    )

    store_layout.register_kv_caches.assert_called_once()
    registered_caches, registered_num_blocks = (
        store_layout.register_kv_caches.call_args.args
    )
    assert registered_num_blocks == num_blocks
    assert len(registered_caches) == 1
    assert registered_caches[0] is full_cache
    worker.store.register_buffer.assert_called_once_with(
        full_cache.data_ptr(), full_cache.untyped_storage().nbytes()
    )


def test_register_kv_caches_separate_head_groups():
    # LHBNC gives each head group its own region; the K/V split doubles heads.
    layout = KVCacheLayout.LHBNC
    num_blocks = 3
    num_layers = 2
    worker = _make_bare_worker(num_gpu_blocks=num_blocks)
    spec = FullAttentionSpec(
        block_size=4,
        num_kv_heads=2,
        head_size=8,
        dtype=torch.float16,
        num_head_slots=2,
        state_content_bytes=2 * 8 * 2,
    )
    layer_names = ["layer0", "__cross_layer__"]
    worker._kv_cache_groups = [KVCacheGroupSpec(layer_names, spec)]
    raw = torch.zeros(
        num_blocks * num_layers * spec.page_size_bytes,
        dtype=torch.int8,
    )
    caches = dense_kv_cache_views(raw, spec, num_blocks, num_layers, layout)

    _register_with_mocked_threads(worker, dict(zip(layer_names, caches)))

    head_block_bytes = caches[0].stride(0) * caches[0].element_size()
    expected_addrs = [
        cache[:, head_idx].data_ptr()
        for cache in caches
        for head_idx in range(cache.shape[1])
    ]
    db = worker.token_dbs[0]
    assert db.kv_caches_base_addr == expected_addrs
    assert db.block_len == [head_block_bytes] * len(expected_addrs)
    worker.store.register_buffer.assert_called_once_with(raw.data_ptr(), raw.nbytes)


# ---------------------------------------------------------------------------
# Dual-mode (embedded / standalone-store) config validation tests
# ---------------------------------------------------------------------------


def _make_config(**overrides):
    """Build a MooncakeStoreConfig with sensible defaults for validation tests.

    Required dataclass fields are populated; callers override only the field
    under test.
    """
    base = dict(
        metadata_server="http://metadata/endpoint",
        master_server_address="10.0.0.7:50051",
        protocol="rdma",
        device_name="mlx5_0",
    )
    base.update(overrides)
    return worker.MooncakeStoreConfig(**base)


def test_config_defaults_to_embedded():
    """A JSON without explicit mode parses as embedded with 4 GiB segment."""
    cfg = _make_config()
    assert cfg.mode == "embedded"
    assert cfg.global_segment_size == mooncake_store.DEFAULT_GLOBAL_SEGMENT_SIZE
    assert cfg.local_buffer_size == mooncake_store.DEFAULT_LOCAL_BUFFER_SIZE
    assert cfg.enable_offload is False
    assert cfg.tenant_id == mooncake_store.DEFAULT_TENANT_ID


def test_config_pr40900_unchanged(tmp_path):
    """A literal PR-40900 config (no mode, no enable_offload, no preferred_segment)
    parses without raising and resolves to embedded mode."""
    config_path = _write_mooncake_config(
        tmp_path,
        {
            "metadata_server": "http://metadata/endpoint",
            "global_segment_size": "4GB",
            "local_buffer_size": "4GB",
            "protocol": "rdma",
            "device_name": "mlx5_0",
            "master_server_address": "10.0.0.7:50051",
        },
    )
    cfg = worker.MooncakeStoreConfig.from_file(config_path)
    assert cfg.mode == "embedded"
    assert cfg.global_segment_size == 4 * 1024**3
    assert cfg.local_buffer_size == 4 * 1024**3
    assert cfg.enable_offload is False
    assert cfg.tenant_id == mooncake_store.DEFAULT_TENANT_ID


def test_config_from_file_normalizes_tenant_id(tmp_path):
    config_path = _write_mooncake_config(
        tmp_path,
        {
            "metadata_server": "http://metadata/endpoint",
            "master_server_address": "10.0.0.7:50051",
            "tenant_id": "  tenant-a  ",
        },
    )

    cfg = worker.MooncakeStoreConfig.from_file(config_path)

    assert cfg.tenant_id == "tenant-a"


def test_config_from_file_normalizes_empty_tenant_id_to_default(tmp_path):
    config_path = _write_mooncake_config(
        tmp_path,
        {
            "metadata_server": "http://metadata/endpoint",
            "master_server_address": "10.0.0.7:50051",
            "tenant_id": "   ",
        },
    )

    cfg = worker.MooncakeStoreConfig.from_file(config_path)

    assert cfg.tenant_id == mooncake_store.DEFAULT_TENANT_ID


def test_config_from_file_rejects_non_string_tenant_id(tmp_path):
    config_path = _write_mooncake_config(
        tmp_path,
        {
            "metadata_server": "http://metadata/endpoint",
            "master_server_address": "10.0.0.7:50051",
            "tenant_id": False,
        },
    )

    with pytest.raises(TypeError, match="tenant_id must be a string or null"):
        worker.MooncakeStoreConfig.from_file(config_path)


def test_config_embedded_rejects_zero_segment():
    with pytest.raises(
        ValueError, match=r"embedded mode requires global_segment_size > 0"
    ):
        _make_config(mode="embedded", global_segment_size=0)


def test_config_standalone_store_rejects_nonzero_segment():
    with pytest.raises(
        ValueError,
        match=r"standalone-store mode requires global_segment_size == 0",
    ):
        _make_config(mode="standalone-store", global_segment_size=4 * 1024**3)


def test_config_standalone_store_accepts_zero_segment():
    cfg = _make_config(mode="standalone-store", global_segment_size=0)
    assert cfg.mode == "standalone-store"
    assert cfg.global_segment_size == 0


def test_config_unknown_mode():
    with pytest.raises(ValueError, match=r"unknown Mooncake mode"):
        _make_config(mode="something-else")


def test_config_zero_local_buffer():
    with pytest.raises(ValueError, match=r"local_buffer_size must be > 0"):
        _make_config(local_buffer_size=0)


# ---------------------------------------------------------------------------
# End-to-end topology tests
# Covers the two supported recipes:
#   (A) standalone-store mode + disk offload (mode="standalone-store",
#       segment=0, enable_offload=true, preferred_segment set)
#   (B) embedded mode + CPU only      (mode default, segment>0,
#       enable_offload=false, no preferred_segment)
# ---------------------------------------------------------------------------


def test_topology_standalone_store_with_disk_offload(tmp_path, monkeypatch):
    """standalone-store + disk: global_segment_size=0, enable_offload=True,
    preferred_segment set. Assert setup() positional args, ReplicateConfig
    wiring, and that the disk-offload buffer budget is allocated."""
    store = MagicMock()
    store.setup.return_value = 0
    fake_replicate_config_cls = _install_fake_mooncake(monkeypatch, store)
    _patch_worker_runtime(monkeypatch)
    monkeypatch.setenv(
        "MOONCAKE_CONFIG_PATH",
        _write_mooncake_config(
            tmp_path,
            {
                "mode": "standalone-store",
                "metadata_server": "http://metadata/endpoint",
                "global_segment_size": 0,
                "local_buffer_size": "1GB",
                "protocol": "rdma",
                "device_name": "mlx5_0",
                "master_server_address": "10.0.0.7:50051",
                "enable_offload": True,
            },
        ),
    )

    w = worker.MooncakeStoreWorker(
        _make_vllm_config(extra_config={"preferred_segment": "10.0.0.7:50053"}),
        _make_kv_cache_config(),
    )

    # setup() receives global_segment_size=0 and the configured local buffer.
    assert store.setup.call_args.args == (
        "10.0.0.7",
        "http://metadata/endpoint",
        0,
        1024 * 1024 * 1024,
        "rdma",
        "mlx5_0",
        "10.0.0.7:50051",
    )
    assert store.setup.call_args.kwargs == {}
    # ReplicateConfig is built and carries the preferred_segment.
    assert isinstance(w.store_replicate_config, fake_replicate_config_cls)
    assert w.store_replicate_config.preferred_segment == "10.0.0.7:50053"
    # Disk-offload staging budget is allocated (enable_offload=True).
    assert w.disk_offload_buffer_budget_bytes is not None
    assert w.disk_offload_buffer_budget_bytes > 0


def test_topology_embedded_cpu_only(tmp_path, monkeypatch):
    """Embedded + CPU-only: no mode key (defaults to embedded),
    global_segment_size>0, enable_offload absent, no preferred_segment.
    This is the PR-40900 baseline recipe."""
    store = MagicMock()
    store.setup.return_value = 0
    fake_replicate_config_cls = _install_fake_mooncake(monkeypatch, store)
    _patch_worker_runtime(monkeypatch)
    monkeypatch.setenv(
        "MOONCAKE_CONFIG_PATH",
        _write_mooncake_config(
            tmp_path,
            {
                "metadata_server": "http://metadata/endpoint",
                "global_segment_size": "4GB",
                "local_buffer_size": "4GB",
                "protocol": "rdma",
                "device_name": "mlx5_0",
                "master_server_address": "10.0.0.7:50051",
            },
        ),
    )

    w = worker.MooncakeStoreWorker(_make_vllm_config(), _make_kv_cache_config())

    # setup() receives global_segment_size=4 GiB (rank contributes a segment).
    assert store.setup.call_args.args == (
        "10.0.0.7",
        "http://metadata/endpoint",
        4 * 1024 * 1024 * 1024,
        4 * 1024 * 1024 * 1024,
        "rdma",
        "mlx5_0",
        "10.0.0.7:50051",
    )
    assert store.setup.call_args.kwargs == {}
    # No preferred_segment — ReplicateConfig is default-constructed (so the
    # preferred_segment field keeps its default value).
    assert w.preferred_segment is None
    assert isinstance(w.store_replicate_config, fake_replicate_config_cls)
    assert w.store_replicate_config.preferred_segment == ""
    # No disk budget — enable_offload was absent (defaults to False).
    assert w.disk_offload_buffer_budget_bytes is None


def _create_store_client_for_tenant_test(consumer):
    if consumer == "ec":
        from vllm.distributed.ec_transfer.ec_connector.mooncake_store_embedding.store_client import (  # noqa: E501
            create_mooncake_embedding_store_client,
        )

        return create_mooncake_embedding_store_client()
    return worker.MooncakeStoreWorker(
        _make_vllm_config(rank=1), _make_kv_cache_config()
    )


@pytest.mark.parametrize("consumer", ["kv", "ec"])
def test_topology_forwards_non_default_tenant_id(tmp_path, monkeypatch, consumer):
    store = MagicMock()
    store.setup.return_value = 0
    _install_fake_mooncake(monkeypatch, store)
    _patch_worker_runtime(monkeypatch)
    monkeypatch.setenv(
        "MOONCAKE_CONFIG_PATH",
        _write_mooncake_config(
            tmp_path,
            {
                "metadata_server": "http://metadata/endpoint",
                "global_segment_size": "4GB",
                "local_buffer_size": "4GB",
                "protocol": "rdma",
                "device_name": "mlx5_0",
                "master_server_address": "10.0.0.7:50051",
                "tenant_id": "  tenant-a  ",
            },
        ),
    )

    _create_store_client_for_tenant_test(consumer)

    assert store.setup.call_args.kwargs == {"tenant_id": "tenant-a"}


@pytest.mark.parametrize("consumer", ["kv", "ec"])
def test_non_default_tenant_preserves_setup_type_error(tmp_path, monkeypatch, consumer):
    store = MagicMock()
    setup_error = TypeError(
        "setup(): incompatible function arguments; "
        "supported signature includes tenant_id"
    )
    store.setup.side_effect = setup_error
    _install_fake_mooncake(monkeypatch, store)
    _patch_worker_runtime(monkeypatch)
    monkeypatch.setenv(
        "MOONCAKE_CONFIG_PATH",
        _write_mooncake_config(
            tmp_path,
            {
                "metadata_server": "http://metadata/endpoint",
                "global_segment_size": "4GB",
                "local_buffer_size": "4GB",
                "protocol": "rdma",
                "device_name": "mlx5_0",
                "master_server_address": "10.0.0.7:50051",
                "tenant_id": "tenant-a",
            },
        ),
    )

    with pytest.raises(TypeError) as exc_info:
        _create_store_client_for_tenant_test(consumer)

    assert exc_info.value is setup_error


# ---------------------------------------------------------------------------
# Stats/metrics tests (PR-35 port)
# ---------------------------------------------------------------------------


def test_mooncake_store_stats_aggregate_reduce():
    stats = MooncakeStoreConnectorStats()
    stats.record_operation("save_put", 0.01, 2, num_bytes=128)
    other = MooncakeStoreConnectorStats()
    other.record_operation(
        "save_put",
        0.03,
        1,
        num_bytes=64,
        status="error",
        num_failed_keys=1,
    )

    reduced = stats.aggregate(other).reduce()

    assert reduced["save_put_count"] == 2
    assert reduced["save_put_total_keys"] == 3
    assert reduced["save_put_total_bytes"] == 192
    assert reduced["save_put_failed_keys"] == 1
    assert reduced["save_put_error_count"] == 1


def test_worker_get_kv_connector_stats_resets_after_read():
    worker = _make_bare_worker()
    worker._record_kv_connector_operation(
        "save_put",
        0.01,
        2,
        num_bytes=128,
    )

    stats = worker.get_kv_connector_stats()

    assert isinstance(stats, MooncakeStoreConnectorStats)
    assert stats.data["save_put"][0]["num_bytes"] == 128
    next_stats = worker.get_kv_connector_stats()
    assert set(next_stats.data) == {PIPELINE_KEY}
    assert next_stats.data[PIPELINE_KEY][0]["queued"] == 0
    assert next_stats.data[PIPELINE_KEY][0]["active"] == 0


def test_lookup_records_mooncake_metrics():
    worker = _make_bare_worker()
    worker.store.batch_is_exist.return_value = [1, 1]

    result = worker.lookup(33, [b"a0", b"a1"])
    stats = worker.get_kv_connector_stats()

    assert result.hit_length == 32
    assert isinstance(stats, MooncakeStoreConnectorStats)
    assert len(stats.data["lookup_exists"]) == 1
    assert stats.data["lookup_exists"][0]["num_keys"] == 2


def test_store_worker_close_releases_store():
    worker = _make_bare_worker()
    store = worker.store

    worker.close()

    store.close.assert_called_once_with()
    assert worker.store is None


def test_store_worker_close_is_idempotent():
    worker = _make_bare_worker()
    store = worker.store

    worker.close()
    worker.close()

    # Second call short-circuits because store was already released.
    store.close.assert_called_once_with()


def test_store_worker_close_swallows_store_errors():
    worker = _make_bare_worker()
    worker.store.close.side_effect = RuntimeError("boom")

    # A failure tearing down the store must not propagate out of close().
    worker.close()

    assert worker.store is None


def test_blob_block_hashes_wire_roundtrip():
    """The lookup wire format sends a ``hash_len`` frame plus the raw hashes
    concatenated back-to-back; the server rebuilds them through a zero-copy
    ``BlobBlockHashes`` view over the frame buffer."""
    hashes = [BlockHash(bytes([i]) * 16) for i in range(5)]
    hash_len = len(hashes[0])

    # Client side (LookupKeyClient._lookup): flat payload frame.
    blob = b"".join(hashes)

    # Server side (LookupKeyServer): view over the frame buffer (a memoryview),
    # never materializing the full hash list upfront.
    view = BlobBlockHashes(memoryview(blob), hash_len)

    assert len(view) == 5
    assert list(view) == hashes  # default Sequence iter terminates via IndexError
    assert [bytes(h) for h in view] == hashes
    assert bytes(view[-1]) == hashes[-1]
    assert [bytes(h) for h in view[1:3]] == hashes[1:3]
    with pytest.raises(IndexError):
        _ = view[5]


def test_blob_block_hashes_empty():
    """Empty lookups send hash_len=0 and an empty payload."""
    view = BlobBlockHashes(memoryview(b""), 0)
    assert len(view) == 0
    assert list(view) == []


def test_pipeline_tracks_queue_execution_and_oldest_age_without_completed_jobs():
    now = [10.0]
    tracker = PipelineTracker("lookup", -1, clock=lambda: now[0])
    first = tracker.enqueue()
    now[0] = 12.0
    tracker.start(first)
    second = tracker.enqueue()
    now[0] = 15.0
    pending = tracker.snapshot()
    assert (pending["queued"], pending["active"]) == (1, 1)
    assert pending["oldest_pending_seconds"] == 5.0
    assert pending["queue_wait_seconds"] == [2.0]
    assert pending["execution_seconds"] == []
    tracker.finish(second)  # Cancel a job before it starts.
    tracker.finish(first)
    done = tracker.snapshot()
    assert (done["queued"], done["active"], done["oldest_pending_seconds"]) == (0, 0, 0)
    assert done["execution_seconds"] == [3.0]
    assert done["queue_wait_seconds"] == []
    assert tracker.snapshot()["execution_seconds"] == []


@pytest.mark.parametrize("discard", [False, True])
def test_lookup_completion_wait_excludes_execution_and_drains_once(discard):
    """A ready result can wait for the scheduler after the pipeline is idle."""
    now = [10.0]
    client = worker.LookupKeyClient.__new__(worker.LookupKeyClient)
    client.pipeline = PipelineTracker("lookup", -1, clock=lambda: now[0])
    client.futures = {}
    future: Future[tuple[MooncakeLookupResult, float]] = Future()
    client.executor = MagicMock()
    client.executor.submit.return_value = future

    def lookup(num_tokens, hashes):
        now[0] = 13.0
        return MooncakeLookupResult(128)

    client._lookup = lookup
    assert client.lookup("req", 256, [], non_block=True) is None
    assert client.lookup("req", 256, [], non_block=True) is None
    client.executor.submit.assert_called_once()
    now[0] = 12.0
    future.set_running_or_notify_cancel()
    if discard:
        client.discard("req")  # An already-running lookup cannot be cancelled.
    fn, *args = client.executor.submit.call_args.args
    future.set_result(fn(*args))
    completed = client.pipeline.snapshot()
    assert (completed["queued"], completed["active"]) == (0, 0)
    assert completed["queue_wait_seconds"] == [2.0]
    assert completed["execution_seconds"] == [1.0]
    assert completed["consumption_wait_seconds"] == []
    stats = MooncakeStoreConnectorStats(data={PIPELINE_KEY: [completed]})
    now[0] = 18.0
    if not discard:
        assert client.lookup("req", 256, [], non_block=True).hit_length == 128
    assert not client.futures
    stats.aggregate(
        MooncakeStoreConnectorStats(data={PIPELINE_KEY: [client.pipeline.snapshot()]})
    )
    stats.aggregate(
        MooncakeStoreConnectorStats(data={PIPELINE_KEY: [client.pipeline.snapshot()]})
    )
    assert stats.data[PIPELINE_KEY][0]["consumption_wait_seconds"] == (
        [] if discard else [5.0]
    )
    if not discard:
        assert stats.reduce()["lookup_rank-1_consumption_wait_seconds_avg"] == 5.0


def test_lookup_pipeline_cleans_up_cancelled_and_failed_jobs():
    client = worker.LookupKeyClient.__new__(worker.LookupKeyClient)
    client.pipeline = PipelineTracker("lookup", -1)
    client.futures = {}
    entered, release = threading.Event(), threading.Event()

    def lookup(num_tokens, hashes):
        if num_tokens == 1:
            entered.set()
            assert release.wait(timeout=5)
            return MooncakeLookupResult(1)
        raise RuntimeError("lookup failed")

    client._lookup = lookup
    with ThreadPoolExecutor(max_workers=1) as executor:
        client.executor = executor
        try:
            assert client.lookup("first", 1, [], non_block=True) is None
            assert entered.wait(timeout=5)
            assert client.lookup("cancelled", 2, [], non_block=True) is None
            pending = client.pipeline.snapshot()
            assert (pending["queued"], pending["active"]) == (1, 1)
            client.discard("cancelled")
            assert client.pipeline.snapshot()["queued"] == 0
        finally:
            release.set()
        assert client.lookup("first", 1, []).hit_length == 1
        assert client.lookup("failed", 2, []).hit_length == 0
    done = client.pipeline.snapshot()
    assert (done["queued"], done["active"]) == (0, 0)
    assert len(done["execution_seconds"]) == 2
    assert len(done["consumption_wait_seconds"]) == 1
    assert not client.futures


def test_load_pipeline_cleans_up_on_handler_exception():
    tracker = PipelineTracker("load", 2)
    transfer = worker.KVTransferThread(
        MagicMock(), [], 16, 2, threading.Event(), "test-load"
    )
    transfer.pipeline_tracker = tracker
    ticket = tracker.enqueue()
    transfer.request_queue = MagicMock()
    transfer.request_queue.get.side_effect = [
        (SimpleNamespace(req_id="load"), ticket),
        KeyboardInterrupt(),
    ]
    transfer._handle_request = MagicMock(side_effect=RuntimeError("load failed"))
    with pytest.raises(KeyboardInterrupt):
        transfer.run()
    done = tracker.snapshot()
    assert (done["queued"], done["active"]) == (0, 0)
    assert len(done["execution_seconds"]) == 1


def test_pipeline_aggregation_preserves_ranks_and_replaces_live_gauges():
    now = [0.0]
    rank0 = PipelineTracker("load", 0, clock=lambda: now[0])
    rank1 = PipelineTracker("load", 1, clock=lambda: now[0])
    ticket = rank0.enqueue()
    now[0] = 2.0
    rank0.start(ticket)
    stats = MooncakeStoreConnectorStats(data={PIPELINE_KEY: [rank0.snapshot()]})
    stats.aggregate(
        MooncakeStoreConnectorStats(data={PIPELINE_KEY: [rank1.snapshot()]})
    )
    now[0] = 5.0
    rank0.finish(ticket)
    stats.aggregate(
        MooncakeStoreConnectorStats(data={PIPELINE_KEY: [rank0.snapshot()]})
    )
    snapshots = {s["rank"]: s for s in stats.data[PIPELINE_KEY]}
    assert set(snapshots) == {0, 1}
    assert snapshots[0]["active"] == 0
    assert snapshots[0]["queue_wait_seconds"] == [2.0]
    assert snapshots[0]["execution_seconds"] == [3.0]
    assert stats.reduce()["load_rank0_active"] == 0


def test_pipeline_prometheus_exports_backlog_and_resets_to_zero():
    from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.metrics import (
        MooncakeStorePromMetrics,
    )

    registry = CollectorRegistry()
    from functools import partial

    metrics = MooncakeStorePromMetrics(
        SimpleNamespace(kv_transfer_config=None),
        {cls: partial(cls, registry=registry) for cls in (Counter, Gauge, Histogram)},
        ["engine"],
        {0: ["0"]},
    )
    now = [10.0]
    tracker = PipelineTracker("lookup", -1, clock=lambda: now[0])
    ticket = tracker.enqueue()
    tracker.start(ticket)
    metrics.observe({PIPELINE_KEY: [tracker.snapshot()]})
    labels = {"engine": "0", "stage": "lookup", "rank": "-1"}
    assert registry.get_sample_value("vllm:mooncake_pipeline_active", labels) == 1
    completed_at = tracker.finish(ticket)
    now[0] = 12.0
    tracker.record_consumption(completed_at)
    metrics.observe({PIPELINE_KEY: [tracker.snapshot()]})
    metrics.observe({PIPELINE_KEY: [tracker.snapshot()]})
    assert registry.get_sample_value("vllm:mooncake_pipeline_active", labels) == 0
    assert (
        registry.get_sample_value(
            "vllm:mooncake_pipeline_execution_seconds_count", labels
        )
        == 1
    )

    assert (
        registry.get_sample_value(
            "vllm:mooncake_pipeline_consumption_wait_seconds_count", labels
        )
        == 1
    )
    assert (
        registry.get_sample_value(
            "vllm:mooncake_pipeline_consumption_wait_seconds_sum", labels
        )
        == 2.0
    )


@pytest.mark.parametrize(
    "failure", ["lookup", "put", "partial", "pressure", "boundary"]
)
def test_external_only_save_failure_is_fatal_and_does_not_release_blocks(failure):
    store = MagicMock()
    store.batch_is_exist.return_value = [0, 0]
    store.batch_put_from_multi_buffers.return_value = [0, 0]
    if failure in ("lookup", "put"):
        method = (
            "batch_is_exist" if failure == "lookup" else "batch_put_from_multi_buffers"
        )
        getattr(store, method).side_effect = RuntimeError("store unavailable")
    elif failure == "partial":
        store.batch_put_from_multi_buffers.return_value = [0, -500]
    elif failure == "pressure":
        store.batch_put_from_multi_buffers.return_value = [
            mooncake_store_worker.MOONCAKE_NO_AVAILABLE_HANDLE
        ] * 2
    thread = _make_store_sending_thread(store, external_cache_only=True)
    req = _make_store_req("req-a", [b"a0", b"a1"])
    if failure == "boundary":
        req.boundary_state_offloads = [(0, 1, 16)]
        thread._maybe_offload_boundary_states = MagicMock(return_value=False)
    with contextlib.suppress(RuntimeError):
        _run_store_req(thread, req)
    assert thread.is_live_store_job(req)
    assert thread._completed_saves == {}
    with pytest.raises(RuntimeError, match="external_cache_only save failed"):
        thread.take_completed_saves()


@pytest.mark.parametrize("already_stored", [False, True])
def test_external_only_success_or_dedup_releases_job(already_stored):
    store = MagicMock()
    store.batch_is_exist.return_value = [int(already_stored)] * 2
    store.batch_put_from_multi_buffers.return_value = [0, 0]
    thread = _make_store_sending_thread(store, external_cache_only=True)
    req = _make_store_req("req-a", [b"a0", b"a1"])
    # External-only mode must never silently skip a live save under pressure.
    thread._store_pressure_active = True
    thread._skip_store_requests.add(req.req_id)
    _run_store_req(thread, req)
    assert thread.take_completed_saves() == {req.store_job_id: 1}
    assert store.batch_put_from_multi_buffers.call_count == int(not already_stored)


def test_external_only_empty_job_is_not_a_store_failure():
    thread = _make_store_sending_thread(MagicMock(), external_cache_only=True)
    req = _make_store_req("short", [])
    req.token_len_chunk = 0
    _run_store_req(thread, req)
    assert thread.take_completed_saves() == {req.store_job_id: 1}


def test_external_only_saved_kv_survives_local_reuse_and_reloads():
    """Exercise real key/transfer logic with byte-copying CPU store I/O."""
    from vllm.v1.core.block_pool import BlockPool

    data = torch.arange(3 * 256, dtype=torch.int64).to(torch.uint8)
    expected = data[256:].clone()
    stored = {}
    store = MagicMock()

    def put(keys, addresses, sizes, *_args):
        for key, addrs, lengths in zip(keys, addresses, sizes):
            stored[key] = [
                ctypes.string_at(addr, length) for addr, length in zip(addrs, lengths)
            ]
        return [0] * len(keys)

    def get(keys, addresses, sizes):
        for key, addrs, lengths in zip(keys, addresses, sizes):
            for addr, length, value in zip(addrs, lengths, stored[key]):
                assert length == len(value)
                ctypes.memmove(addr, value, length)
        return [0] * len(keys)

    store.batch_is_exist.side_effect = lambda keys: [int(k in stored) for k in keys]
    store.batch_put_from_multi_buffers.side_effect = put
    store.batch_get_into_multi_buffers.side_effect = get
    worker = _make_bare_worker(num_gpu_blocks=3)
    worker.store = store
    db = worker.token_dbs[0]
    db.set_kv_caches_base_addr([data.data_ptr()])
    db.set_block_len([256])
    sender = _make_store_sending_thread(
        store,
        token_databases=[db],
        external_cache_only=True,
    )
    pool = BlockPool(3, enable_caching=True, hash_block_size=16)
    pool.enable_local_cache = False
    blocks = pool.get_new_blocks(2)
    req = _make_store_req("first", [b"h0", b"h1"])
    pool.cache_full_blocks(req, blocks, 0, 2, 16, 0)
    pool.touch(blocks)  # Async store ownership.
    pool.free_blocks(blocks)  # The prefill request finished.
    _run_store_req(sender, req)
    assert sender.take_completed_saves() == {req.store_job_id: 1}
    pool.free_blocks(blocks)
    assert pool.get_num_free_blocks() == 2
    assert len(pool.cached_block_hash_to_block) == 0
    assert pool.get_cached_block(b"h0", [0]) is None
    data.zero_()  # Reusing GPU pages cannot destroy the external copy.
    assert worker.lookup(33, req.block_hashes).hit_length == 32
    reused = pool.get_new_blocks(2)
    receiver = _make_store_recving_thread(store)
    receiver.token_databases = [db]
    load = _make_load_req("second", req.block_hashes, token_len=32)
    load.block_ids = ([block.block_id for block in reused],)
    receiver._handle_request(load)
    assert receiver.get_and_clear_finished_requests() == {"second"}
    assert not receiver.get_and_clear_block_ids_with_load_errors()
    for index, block in enumerate(reused):
        torch.testing.assert_close(
            data[block.block_id * 256 : (block.block_id + 1) * 256],
            expected[index * 256 : (index + 1) * 256],
        )


@pytest.mark.parametrize("with_local_entries", [False, True])
def test_external_only_lookup_schedules_missing_mamba_checkpoint(with_local_entries):
    """A remote-only boundary creates a checkpoint without adopting local KV."""
    from tests.v1.core.prefix_cache.test_mamba_eagle_resume_checkpoint import (
        PREFIX,
        _manager,
        _prefill,
        _stub,
    )
    from tests.v1.core.test_prefix_caching import make_request
    from tests.v1.kv_connector.unit.test_mooncake_store_scheduler import (
        _make_bare_scheduler,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.protocol import (
        decode_lookup_response,
        encode_lookup_response,
    )
    from vllm.utils.hashing import sha256
    from vllm.v1.core.sched.scheduler import Scheduler

    block_size, hash_size = 8576, 128
    manager = _manager(
        block_size,
        hash_size,
        eagle_group=0,
        num_prefill_lookahead=1,
        num_speculative_blocks=5,
    )
    stub = _stub(manager, block_size, hash_size)
    scheduler = _make_bare_scheduler(
        hash_block_size=hash_size, enable_partial_hash_hits=True
    )
    scheduler.external_cache_only = True
    scheduler.bind_gpu_block_pool(manager.block_pool)
    if with_local_entries:
        owner = make_request("owner", PREFIX[:16000], hash_size, sha256)
        _prefill(manager, stub, owner)
    request = make_request("sibling", PREFIX[:12000] + [-1] * 4000, hash_size, sha256)
    blocks, local, request.shared_prefix_boundary = manager.get_computed_blocks(request)
    assert local == 0
    assert request.shared_prefix_boundary == 0

    worker = _make_bare_worker(block_size=hash_size)
    worker._kv_cache_groups = manager.kv_cache_config.kv_cache_groups
    worker.coord = mooncake_store_worker.MooncakeStoreCoordinator(
        worker._kv_cache_groups,
        scheduler_block_size=block_size,
        hash_block_size=hash_size,
        use_eagle=True,
    )
    worker.token_dbs = [
        ChunkedTokenDatabase(
            KeyMetadata("test-model", 0, 0, 0, 0, group_id=gid),
            block_size=block_size,
            hash_block_size=hash_size,
        )
        for gid in range(2)
    ]
    _refresh_group_tp_replication_factors(worker)
    # The physical boundary exists in the store, but EAGLE resumes one hash
    # unit earlier, where no Mamba checkpoint has yet been materialized.
    physical_hash = request.block_hashes[block_size // hash_size - 1]
    present = {(gid, bytes(physical_hash)) for gid in range(2)}
    worker.store.batch_is_exist.side_effect = lambda keys: [
        int(any(f"@group:{gid}@{h.hex()}" in key for gid, h in present)) for key in keys
    ]
    scheduler.client = SimpleNamespace(
        lookup=lambda req_id, num_tokens, hashes, **kwargs: decode_lookup_response(
            encode_lookup_response(worker.lookup(num_tokens, hashes))
        )
    )
    assert scheduler.get_num_new_matched_tokens(request, local) == (0, False)
    assert request.shared_prefix_boundary == 8448
    assert not scheduler.load_specs
    assert not manager.block_pool.enable_local_cache

    chunk = Scheduler._mamba_block_aligned_split(stub, request, request.num_tokens)
    assert chunk == 8448
    allocated = manager.allocate_slots(request, chunk, 0, blocks)
    assert allocated is not None
    checkpoint = allocated.blocks[1][0]
    assert checkpoint.block_hash_num_tokens == chunk
    assert manager.get_computed_blocks(request)[1] == 0

    # Advancing the producer preserves the partial checkpoint via CoW and
    # hands its exact block to the connector for offload.
    request.num_computed_tokens = chunk
    manager.new_step_starts()
    next_chunk = Scheduler._mamba_block_aligned_split(
        stub, request, request.num_tokens - chunk
    )
    assert manager.allocate_slots(request, next_chunk) is not None
    offloads = manager.take_boundary_state_offloads()[request.request_id]
    group_id, block_id, boundary = next(
        entry for entry in offloads if entry[0] == 1 and entry[2] == chunk
    )
    assert manager.block_pool.blocks[block_id].block_hash_num_tokens == boundary

    # Simulate publishing the handed-off checkpoint. Full attention still
    # uses its existing physical-boundary key for the next sibling's load.
    present.add((group_id, bytes(request.block_hashes[boundary // hash_size - 1])))
    result = worker.lookup(request.num_tokens, request.block_hashes)
    assert result.hit_length == chunk
    assert result.shared_prefix_boundary == 0


@pytest.mark.parametrize(
    "coverage,expected_reuse",
    [
        ("full", True),
        ("partial", True),
        ("missing", False),
        ("other_chunk", False),
        ("one_rank", False),
        ("different_rank_hash", False),
        ("non_causal", False),
        ("mamba_only", False),
        ("other_prefix", False),
        ("no_future_hash", False),
    ],
)
def test_checkpoint_reuses_only_matching_attention_in_every_namespace(
    coverage, expected_reuse
):
    from functools import partial

    from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.metrics import (
        MooncakeStorePromMetrics,
    )
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        KVCacheGroupSpec,
        MambaSpec,
    )

    store = MagicMock()
    boundary = 8 if coverage in {"partial", "different_rank_hash"} else 12
    hashes = [BlockHash(f"h{i}".encode()) for i in range(8)]
    if coverage == "no_future_hash":
        hashes = hashes[:3]
    groups = [
        KVCacheGroupSpec(
            ["full"],
            FullAttentionSpec(
                block_size=16,
                num_kv_heads=1,
                head_size=1,
                dtype=torch.float32,
                non_causal=coverage == "non_causal",
            ),
        ),
        KVCacheGroupSpec(
            ["mamba"],
            MambaSpec(
                block_size=16,
                shapes=((1, 1),),
                dtypes=(torch.float32,),
                mamba_cache_mode="align",
            ),
        ),
    ]
    coord = mooncake_store_worker.MooncakeStoreCoordinator(
        groups,
        scheduler_block_size=16,
        hash_block_size=4,
    )
    dbs = []
    for gid in range(2):
        db = ChunkedTokenDatabase(
            KeyMetadata("test-model", 0, 0, 0, 0, group_id=gid),
            block_size=16,
            hash_block_size=4,
        )
        db.set_kv_caches_base_addr([0x1000 * (gid + 1)])
        db.set_block_len([256])
        dbs.append(db)
    prefixes = [db.key_for(hashes[0]).rsplit("@", 1)[0] for db in dbs]
    rank1 = prefixes[0].replace("tp_rank:0", "tp_rank:1")
    namespaces = [(prefixes[0], rank1), (prefixes[1],)]
    present = set()
    cover_hash = hashes[2] if coverage == "partial" else BlockHash(b"h3")
    if coverage in {"full", "partial", "one_rank", "non_causal"}:
        present.add(f"{prefixes[0]}@{cover_hash.hex()}")
        if coverage != "one_rank":
            present.add(f"{rank1}@{cover_hash.hex()}")
    elif coverage == "different_rank_hash":
        present.add(f"{prefixes[0]}@{hashes[2].hex()}")
        present.add(f"{rank1}@{hashes[3].hex()}")
    elif coverage == "other_chunk":
        present.update(f"{prefix}@{hashes[7].hex()}" for prefix in namespaces[0])
    elif coverage == "mamba_only":
        present.add(dbs[1].key_for(hashes[3]))
    elif coverage == "other_prefix":
        present.update(f"{prefix}@{b'other'.hex()}" for prefix in namespaces[0])
    store.batch_is_exist.side_effect = lambda keys: [
        int(key in present) for key in keys
    ]
    store.batch_put_from_multi_buffers.side_effect = lambda keys, *args: [0] * len(keys)
    thread = _make_store_sending_thread(
        store,
        coord=coord,
        token_databases=dbs,
        lookup_key_prefixes=namespaces,
    )
    stats = MooncakeStoreConnectorStats()
    thread._record_operation_cb = stats.record_operation
    req = ReqMeta(
        req_id="checkpoint",
        token_len_chunk=0,
        block_ids=([1, 2], [0]),
        block_hashes=hashes,
        can_save=True,
        boundary_state_offloads=[(1, 7, boundary)],
    )
    assert thread._maybe_offload_boundary_states(req)
    written = store.batch_put_from_multi_buffers.call_args.args[0]
    boundary_hash = hashes[boundary // 4 - 1]
    assert (dbs[0].key_for(boundary_hash) not in written) == expected_reuse
    assert dbs[1].key_for(boundary_hash) in written
    assert stats.data["checkpoint_mamba_put"][0]["num_bytes"] == 256
    assert stats.data["save_put"][0]["num_bytes"] == (256 if expected_reuse else 512)
    if expected_reuse:
        assert stats.data["checkpoint_attention_reuse"][0]["num_bytes"] == 256
        assert "checkpoint_attention_put" not in stats.data
        # Check that lookup can recover this state via the original attention key.
        present.update(written)
        worker = _make_bare_worker(block_size=4)
        worker.coord = coord
        worker._kv_cache_groups = groups
        worker.token_dbs = dbs
        worker._lookup_key_prefixes = namespaces
        worker.store = store
        result = worker.lookup(33, hashes)
        assert result.hit_length == boundary
        assert result.tail_key_boundaries[0].num_tokens == (
            12 if coverage == "partial" else 16
        )
        # Recheck each save: evicting the covering key restores normal uploads.
        present.clear()
        assert thread._maybe_offload_boundary_states(req)
        assert len(store.batch_put_from_multi_buffers.call_args.args[0]) == 2
    else:
        assert "checkpoint_attention_reuse" not in stats.data
        assert stats.data["checkpoint_attention_put"][0]["num_bytes"] == 256

    registry = CollectorRegistry()
    metrics = MooncakeStorePromMetrics(
        SimpleNamespace(kv_transfer_config=None),
        {cls: partial(cls, registry=registry) for cls in (Counter, Gauge, Histogram)},
        ["engine"],
        {0: ["0"]},
    )
    metrics.observe(stats.data)
    assert registry.get_sample_value(
        "vllm:mooncake_store_operation_bytes_total",
        {"engine": "0", "operation": "checkpoint_mamba_put", "status": "ok"},
    ) == (512 if expected_reuse else 256)


@pytest.mark.parametrize("failure", ["exception", "partial"])
def test_checkpoint_write_metrics_record_failed_keys(failure):
    store = MagicMock()
    store.batch_is_exist.side_effect = lambda keys: [0] * len(keys)
    if failure == "exception":
        store.batch_put_from_multi_buffers.side_effect = RuntimeError("failed")
    else:
        store.batch_put_from_multi_buffers.side_effect = lambda keys, *args: [0] * (
            len(keys) - 1
        ) + [-1]
    thread = _make_partial_tail_send_thread(store)
    stats = MooncakeStoreConnectorStats()
    thread._record_operation_cb = stats.record_operation
    assert not thread._maybe_offload_boundary_states(_make_partial_tail_req([1, 2, 3]))
    mamba = stats.data["checkpoint_mamba_put"][0]
    attention = stats.data["checkpoint_attention_put"][0]
    assert mamba["num_failed_keys"] == 1
    assert mamba["status"] == ("error" if failure == "exception" else "partial_failure")
    assert attention["num_failed_keys"] == (3 if failure == "exception" else 0)
    assert attention["num_bytes"] == 768


def test_inventory_layout_reports_expected_shards_without_store_reads(monkeypatch):
    monkeypatch.setenv("VLLM_CACHE_INVENTORY_INTERVAL", "10")
    worker = _make_bare_worker()
    worker.load_pipeline = mooncake_store_worker.PipelineTracker("load", 0)
    worker._lookup_key_prefixes = [
        ("model@tp_rank:0@group:0", "model@tp_rank:1@group:0")
    ]
    first = worker.get_kv_connector_stats()
    assert first is not None
    assert first.data["_cache_layout"] == [
        {"scope": "model", "prefix": "model@tp_rank:0@group:0", "kind": "attention"},
        {"scope": "model", "prefix": "model@tp_rank:1@group:0", "kind": "attention"},
    ]
    first.aggregate(first)  # Several workers may advertise the same layout.
    assert len(first.data["_cache_layout"]) == 2
    assert not any("cache_layout" in key for key in first.reduce())
    second = worker.get_kv_connector_stats()
    assert "_cache_layout" not in second.data
    worker.store.batch_is_exist.assert_not_called()


# Retirement tests exercise observable store contents through the policy API.
# A dict-backed Store catches deletion, completeness and retry errors without
# requiring a GPU or a running Mooncake service.
def _retirement_store():
    store = MagicMock()
    del store.batch_retire
    objects: dict[str, bytes] = {}
    leased: set[str] = set()
    store.batch_is_exist.side_effect = lambda keys: [int(k in objects) for k in keys]

    def put(key, value):
        objects.setdefault(key, value)
        return 0

    def remove(keys, *, force):
        assert force is False
        result = []
        for key in keys:
            if key in leased:
                result.append(-706)
            else:
                result.append(0 if objects.pop(key, None) is not None else -704)
        return result

    store.put.side_effect = put
    store.batch_remove.side_effect = remove
    return store, objects, leased


def test_mamba_retirement_keeps_anchor_and_tip_across_replicas():
    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.checkpoint_retirement import (  # noqa: E501
        MambaCheckpointRetirer,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.data import (
        MambaCheckpointCommit,
    )

    store, objects, _ = _retirement_store()
    prefixes = [("attention",), ("mamba-rank0", "mamba-rank1"), ("mamba-group2",)]
    original = {f"{p}@{h}" for group in prefixes for p in group for h in "abcd"}
    objects.update(dict.fromkeys(original, b"state"))
    # Each turn lands on a fresh replica: lineage cannot rely on Python memory.
    for parent, child in [(None, "a"), ("a", "b"), ("b", "c"), ("c", "d")]:
        policy = MambaCheckpointRetirer(store, prefixes, {1, 2})
        policy.commit(MambaCheckpointCommit(parent, child))
    assert original & objects.keys() == {
        *(f"attention@{h}" for h in "abcd"),
        *(f"{p}@{h}" for group in prefixes[1:] for p in group for h in "ad"),
    }
    # A fork/retry may retire the same parent twice; absent keys are harmless.
    policy.commit(MambaCheckpointCommit("b", "d"))
    assert "mamba-rank0@a" in objects
    assert "mamba-rank0@d" in objects


@pytest.mark.parametrize("failure", ["missing_shard", "lookup", "marker_put"])
def test_mamba_retirement_preserves_parent_without_a_complete_replacement(failure):
    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.checkpoint_retirement import (  # noqa: E501
        MambaCheckpointRetirer,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.data import (
        MambaCheckpointCommit,
    )

    store, objects, _ = _retirement_store()
    prefixes = [("m0", "m1")]
    objects.update({f"{p}@{h}": b"state" for p in prefixes[0] for h in "abc"})
    policy = MambaCheckpointRetirer(store, prefixes, {0})
    policy.commit(MambaCheckpointCommit("a", "b"))
    if failure == "missing_shard":
        del objects["m1@c"]
    elif failure == "lookup":
        store.batch_is_exist.side_effect = RuntimeError("unavailable")
    else:
        store.put.side_effect = lambda *_: -200
    policy.commit(MambaCheckpointCommit("b", "c"))
    assert "m0@b" in objects and "m1@b" in objects
    store.batch_remove.assert_not_called()


def test_mamba_retirement_retries_leased_keys_without_renewing_them():
    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.checkpoint_retirement import (  # noqa: E501
        MambaCheckpointRetirer,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.data import (
        MambaCheckpointCommit,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.metrics import (
        MooncakeStoreConnectorStats,
    )

    store, objects, leased = _retirement_store()
    objects.update({f"{p}@{h}": b"state" for p in ("m0", "m1") for h in "abc"})
    stats = MooncakeStoreConnectorStats()
    policy = MambaCheckpointRetirer(
        store, [("m0", "m1")], {0}, record_operation=stats.record_operation
    )
    policy.commit(MambaCheckpointCommit("a", "b"))
    leased.add("m1@b")
    policy.commit(MambaCheckpointCommit("b", "c"))
    assert "m0@b" not in objects and "m1@b" in objects
    store.batch_is_exist.reset_mock()
    policy.retry()
    assert "m1@b" in objects
    leased.clear()
    policy.retry()
    assert "m1@b" not in objects
    store.batch_remove.assert_called_with(["m1@b"], force=False)
    store.batch_is_exist.assert_not_called()
    assert [r["status"] for r in stats.data["checkpoint_retire"]] == [
        "deferred",
        "deferred",
        "ok",
    ]


@pytest.mark.parametrize("success", [True, False])
def test_checkpoint_completion_reports_success_separately_from_block_release(success):
    store = MagicMock()
    thread = _make_store_sending_thread(store)
    thread.report_save_success = True
    req = ReqMeta(
        req_id="r",
        token_len_chunk=0,
        block_ids=([1],),
        block_hashes=[b"a"],
        can_save=True,
        store_job_id=99,
        boundary_state_offloads=[(0, 1, 16)],
    )
    thread._maybe_offload_boundary_states = MagicMock(return_value=success)
    thread.add_request(req)
    thread._handle_request(req)
    result = thread.take_save_results()
    assert result.completed_saves == {99: 1}
    assert result.successful_saves == ({99: 1} if success else {})


def test_mamba_retirement_reset_discards_queued_and_leased_work():
    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.checkpoint_retirement import (  # noqa: E501
        MambaCheckpointRetirer,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.data import (
        MambaCheckpointCommit,
    )

    store, objects, leased = _retirement_store()
    store.remove_all.side_effect = lambda *, force: objects.clear()
    objects.update({f"m@{h}": b"state" for h in "abc"})
    policy = MambaCheckpointRetirer(store, [("m",)], {0})
    policy.commit(MambaCheckpointCommit("a", "b"))
    leased.add("m@b")
    policy.commit(MambaCheckpointCommit("b", "c"))
    policy.submit(MambaCheckpointCommit("b", "c"))
    policy.reset_store()
    # A new run may reuse the same hashes; stale retries must not remove them.
    objects["m@b"] = b"new state"
    leased.clear()
    policy.retry()
    assert objects == {"m@b": b"new state"}
    assert policy._commits.empty()


def test_mamba_retirement_bounds_queue_and_retry_work(monkeypatch):
    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store import (
        checkpoint_retirement as retirement,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.data import (
        MambaCheckpointCommit,
    )

    monkeypatch.setattr(retirement, "_MAX_PENDING", 2)
    monkeypatch.setattr(retirement, "_RETRY_BATCH_SIZE", 1)
    store, objects, leased = _retirement_store()
    objects.update({f"m@{h}": b"state" for h in "abcde"})
    leased.update(objects)
    policy = retirement.MambaCheckpointRetirer(store, [("m",)], {0})
    for parent, child in zip("abcd", "bcde", strict=True):
        commit = MambaCheckpointCommit(parent, child)
        policy.submit(commit)
        policy.commit(commit)
    assert policy._commits.qsize() == 2
    assert len(policy._pending) == 2
    store.batch_remove.reset_mock()
    policy.retry()
    store.batch_remove.assert_called_once_with(["m@c"], force=False)
    policy.retry()
    assert store.batch_remove.call_args.args == (["m@d"],)


def test_mamba_retirement_shutdown_stops_background_retries():
    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.checkpoint_retirement import (  # noqa: E501
        MambaCheckpointRetirer,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.data import (
        MambaCheckpointCommit,
    )

    store, _, _ = _retirement_store()
    policy = MambaCheckpointRetirer(store, [("m",)], {0})
    policy.start()
    policy.close()
    policy.submit(MambaCheckpointCommit("a", "b"))
    assert not policy.is_alive()
    store.batch_is_exist.assert_not_called()
    store.batch_remove.assert_not_called()


@pytest.mark.parametrize("enabled", [False, True])
def test_retired_checkpoint_lookup_stops_lease_renewal_across_replicas(enabled):
    """Repeated candidate lookups must let a retired payload's lease expire."""
    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.checkpoint_retirement import (  # noqa: E501
        MambaCheckpointRetirer,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.data import (
        MambaCheckpointCommit,
    )

    worker = _make_hybrid_lookup_worker(hash_block_size=8)
    worker.retire_mamba_checkpoints = enabled
    store, objects, leased = _retirement_store()
    worker.store = store
    hashes = [b"a", b"b", b"c"]
    prefixes = worker._lookup_key_prefixes
    objects.update(
        {
            f"{p}@{h.hex()}": b"state"
            for group in prefixes
            for p in group
            for h in hashes
        }
    )
    now = [0]
    expiries = {}

    def exists(keys):
        for key in keys:
            if key in objects:
                expiries[key] = now[0] + 180
                leased.add(key)
        return [int(key in objects) for key in keys]

    store.batch_is_exist.side_effect = exists
    # Writer is a separate replica; the lookup worker has no local retirer.
    writer = MambaCheckpointRetirer(store, prefixes, {1})
    writer.commit(MambaCheckpointCommit(b"a".hex(), b"b".hex()))
    writer.commit(MambaCheckpointCommit(b"b".hex(), b"c".hex()))
    retired_keys = {f"{p}@{b'b'.hex()}" for p in prefixes[1]}
    assert retired_keys <= objects.keys()  # Existing readers are still protected.
    store.batch_is_exist.reset_mock()
    for tick in (60, 120, 181):
        now[0] = tick
        assert worker.lookup(25, hashes).hit_length == 24
        leased.difference_update(k for k, expiry in expiries.items() if expiry <= tick)
        writer.retry()
    probes = {
        key for call in store.batch_is_exist.call_args_list for key in call.args[0]
    }
    assert bool(retired_keys & probes) is (not enabled)
    assert bool(retired_keys & objects.keys()) is (not enabled)
    # Attention at the retired boundary remains necessary for the longer hit.
    assert all(f"{p}@{b'b'.hex()}" in probes for p in prefixes[0])
    if enabled:
        assert "lookup_retired_skip" in worker.kv_connector_stats.data
    else:
        assert "lookup_retired_exists" not in worker.kv_connector_stats.data


@pytest.mark.parametrize("failure", ["exception", "short", "negative"])
def test_retirement_marker_lookup_error_does_not_probe_payloads(failure):
    worker = _make_hybrid_lookup_worker(hash_block_size=8)
    worker.retire_mamba_checkpoints = True
    del worker.store.batch_retire
    if failure == "exception":
        worker.store.batch_is_exist.side_effect = RuntimeError("unavailable")
    else:
        worker.store.batch_is_exist.return_value = (
            [] if failure == "short" else [-706, 0]
        )
    assert worker.lookup(17, [b"a", b"b"]).hit_length == 0
    worker.store.batch_is_exist.assert_called_once()
    assert all(
        key.startswith("vllm-mamba-retired-v1@")
        for key in worker.store.batch_is_exist.call_args.args[0]
    )
    assert (
        worker.kv_connector_stats.data["lookup_retired_exists"][0]["status"] == "error"
    )


def test_retirement_marker_failure_preserves_parent():
    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.checkpoint_retirement import (  # noqa: E501
        MambaCheckpointRetirer,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.data import (
        MambaCheckpointCommit,
    )

    store, objects, _ = _retirement_store()
    objects.update({f"m@{h}": b"state" for h in "abc"})
    writer = MambaCheckpointRetirer(store, [("m",)], {0})
    writer.commit(MambaCheckpointCommit("a", "b"))
    put = store.put.side_effect
    store.put.side_effect = lambda key, value: (
        -200 if key.startswith("vllm-mamba-retired-v1@") else put(key, value)
    )
    writer.commit(MambaCheckpointCommit("b", "c"))
    assert "m@b" in objects
    store.batch_remove.assert_not_called()


def test_retirement_lookup_observes_new_markers_and_cache_reset():
    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.checkpoint_retirement import (  # noqa: E501
        MambaCheckpointRetirer,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.data import (
        MambaCheckpointCommit,
    )

    worker = _make_hybrid_lookup_worker(hash_block_size=8)
    worker.retire_mamba_checkpoints = True
    store, objects, leased = _retirement_store()
    worker.store = store
    prefixes = worker._lookup_key_prefixes
    payloads = {
        f"{p}@{h.hex()}": b"state"
        for group in prefixes
        for p in group
        for h in (b"a", b"b", b"c")
    }
    objects.update(payloads)
    assert worker.lookup(17, [b"a", b"b"]).hit_length == 16
    writer = MambaCheckpointRetirer(store, prefixes, {1})
    writer.commit(MambaCheckpointCommit(b"a".hex(), b"b".hex()))
    leased.update(payloads)
    writer.commit(MambaCheckpointCommit(b"b".hex(), b"c".hex()))
    assert worker.lookup(17, [b"a", b"b"]).hit_length < 16
    assert payloads.keys() <= objects.keys()
    store.remove_all.side_effect = lambda *, force: objects.clear()
    writer.reset_store()
    objects.update(payloads)
    assert worker.lookup(17, [b"a", b"b"]).hit_length == 16


@pytest.mark.parametrize("interval", ["0", "10"])
def test_inventory_marker_namespaces_match_retirement_writes(monkeypatch, interval):
    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.checkpoint_retirement import (  # noqa: E501
        MambaCheckpointRetirer,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.data import (
        MambaCheckpointCommit,
    )

    monkeypatch.setenv("VLLM_CACHE_INVENTORY_INTERVAL", interval)
    worker = _make_hybrid_lookup_worker()
    store, objects, _ = _retirement_store()
    worker.store = store
    objects.update(
        {
            f"{p}@{h}": b"state"
            for prefixes in worker._lookup_key_prefixes
            for p in prefixes
            for h in "abc"
        }
    )
    writer = MambaCheckpointRetirer(store, worker._lookup_key_prefixes, {1})
    writer.commit(MambaCheckpointCommit("a", "b"))
    writer.commit(MambaCheckpointCommit("b", "c"))
    store.reset_mock()
    stats = worker.get_kv_connector_stats()
    store.batch_is_exist.assert_not_called()
    store.put.assert_not_called()
    if interval == "0":
        assert "_cache_layout" not in stats.data
        return
    layout = stats.data["_cache_layout"]
    markers = {
        entry["kind"]: entry
        for entry in layout
        if entry["kind"] in ("mamba_lineage", "mamba_retired")
    }
    assert set(markers) == {"mamba_lineage", "mamba_retired"}
    for entry in markers.values():
        assert entry["scope"] == "test-model"
        assert entry["prefix"] + "@b" in objects
    # Metadata namespaces do not change the expected number of Mamba shards.
    assert sum(entry["kind"] == "mamba" for entry in layout) == 2


@pytest.mark.parametrize("failure", [False, True])
def test_master_retirement_acceptance_stops_client_deletion_retries(failure):
    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.checkpoint_retirement import (  # noqa: E501
        MambaCheckpointRetirer,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.data import (
        MambaCheckpointCommit,
    )

    store, objects, leased = _retirement_store()
    store.batch_retire = MagicMock(return_value=[-900 if failure else 0])
    objects.update({f"m@{h}": b"state" for h in "abc"})
    leased.update(objects)
    stats = MooncakeStoreConnectorStats()
    writer = MambaCheckpointRetirer(
        store,
        [("m",)],
        {0},
        use_master_retirement=True,
        record_operation=stats.record_operation,
    )
    writer.commit(MambaCheckpointCommit("a", "b"))
    writer.commit(MambaCheckpointCommit("b", "c"))
    store.batch_retire.assert_called_once_with(["m@b"])
    assert "m@b" in objects  # Accepted means hidden/draining, not freed.
    assert not any(key.startswith("vllm-mamba-retired-v1@") for key in objects)
    store.batch_remove.assert_not_called()
    store.batch_retire.return_value = [0]
    writer.retry()
    assert store.batch_retire.call_count == (2 if failure else 1)
    assert not writer._pending
    assert stats.data["checkpoint_retire_request"][-1]["status"] == "ok"
    assert "checkpoint_retire" not in stats.data


def test_master_retirement_lookup_relies_on_master_without_marker_probes():
    worker = _make_hybrid_lookup_worker(hash_block_size=8)
    worker.retire_mamba_checkpoints = True
    # A new client exposes batch_retire; the master hides retired Mamba keys.
    worker.store.batch_is_exist.side_effect = lambda keys: [
        0 if "group:1" in key and key.endswith("@62") else 1 for key in keys
    ]
    assert worker.lookup(17, [b"a", b"b"]).hit_length < 16
    assert worker.lookup(25, [b"a", b"b", b"c"]).hit_length == 24
    assert worker.store.batch_is_exist.call_count == 2
    assert all(
        not key.startswith("vllm-mamba-retired-v1@")
        for call in worker.store.batch_is_exist.call_args_list
        for key in call.args[0]
    )
    assert "lookup_retired_exists" not in worker.kv_connector_stats.data
