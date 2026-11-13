# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Independent peer-weight pulls for distributed weight data parallelism.

The store coordinates initialization, paused weight updates, and shutdown.
Inference uses copy engines and local CUDA events; a peer need not execute a
forward to serve its weights.
"""

import gc
import os
import socket
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING

import msgspec
import torch
from torch.multiprocessing.reductions import rebuild_cuda_tensor, reduce_tensor

from vllm.logger import init_logger
from vllm.model_executor.offloader.base import BaseOffloader
from vllm.model_executor.offloader.dwdp_health import PeerProcessMonitor

if TYPE_CHECKING:
    from vllm.config.offload import DWDPConfig

logger = init_logger(__name__)


def expert_range(num_experts: int, rank: int, world_size: int) -> tuple[int, int]:
    """Return a balanced contiguous ownership range, including uneven shards."""
    if not 1 <= world_size <= num_experts or not 0 <= rank < world_size:
        raise ValueError("DWDP requires 0 <= rank < world_size <= num_experts")
    return num_experts * rank // world_size, num_experts * (rank + 1) // world_size


def sliced_copy_plan(
    pulls: Sequence[tuple[int, int, int]], slice_bytes: int
) -> list[tuple[int, int, int]]:
    """Interleave (destination, source, size) pulls by slice, then by peer."""
    if slice_bytes < 0 or any(size <= 0 for _, _, size in pulls):
        raise ValueError("DWDP copy sizes must be positive and slice_bytes nonnegative")
    maximum = max((size for _, _, size in pulls), default=0)
    step = slice_bytes or maximum or 1
    return [
        (dest + offset, src + offset, min(step, size - offset))
        for offset in range(0, maximum, step)
        for dest, src, size in pulls
        if offset < size
    ]


@dataclass
class _Layer:
    name: str
    num_experts: int
    weights: tuple[torch.Tensor, ...]
    replicated: dict[int, torch.Tensor]


@dataclass
class _Slot:
    storage: tuple[torch.Tensor, ...]
    ready: torch.cuda.Event
    consumed: torch.cuda.Event
    layer: int = -1
    has_consumer: bool = False
    ready_capture_id: int | None = None


@dataclass
class _Timings:
    copy_start: torch.cuda.Event
    copy_end: torch.cuda.Event
    wait_start: torch.cuda.Event
    wait_end: torch.cuda.Event
    compute_end: torch.cuda.Event
    recorded: bool = False


class DWDPWeightManager:
    """Own expert shards and gather one layer into two reusable weight buffers.

    All ranks must register the same layers in execution order, with weights
    whose leading dimension is the rank's owned expert count. The remaining
    dimensions may contain any expert-major packed layout, including BlockMajorK.
    Sources remain immutable during inference; coordinated refits preserve
    their addresses.
    Forwards must execute in model order without concurrent consumption.
    Ordered stream handoffs and optional CUDA graph capture are supported.

    Args:
        rank: Rank in the weight-sharing group, independent of vLLM DP/EP ranks.
        world_size: Number of independent workers sharing expert weights.
        device: Local CUDA device. Workers must be on distinct GPUs on one host.
        slice_bytes: Maximum DMA request size; zero issues one copy per shard.
        prefetch: Fetch the next layer while computing the current layer.
        collect_metrics: Enable CUDA timing events for local pipeline diagnostics.
        merge_free: Leave resident expert regions uncopied for split-weight kernels.
        allow_cuda_graph: Enable graph capture with the serial-copy fallback.

    """

    def __init__(
        self,
        rank: int,
        world_size: int,
        device: torch.device,
        slice_bytes: int = 0,
        prefetch: bool = True,
        collect_metrics: bool = False,
        merge_free: bool = False,
        allow_cuda_graph: bool = False,
    ) -> None:
        if world_size < 2 or not 0 <= rank < world_size:
            raise ValueError("DWDP requires world_size >= 2 and a valid rank")
        if slice_bytes < 0:
            raise ValueError("slice_bytes must be nonnegative")
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device(device)
        if self.device.type != "cuda" or self.device.index is None:
            raise ValueError("DWDP requires an indexed CUDA device")
        self.slice_bytes = slice_bytes
        self.prefetch = prefetch
        self.collect_metrics = collect_metrics
        self.merge_free = merge_free
        self.allow_cuda_graph = allow_cuda_graph
        self._capture_id: int | None = None
        self._capture_copy_plans: list[list[tuple[int, int, int]]] = []
        self._timings: list[_Timings] = []
        self.layers: list[_Layer] = []
        self._indices: dict[str, int] = {}
        self._declared_layers: dict[str, int] = {}
        self._peers: list[list[tuple[torch.Tensor, ...]]] = []
        self._slots: list[_Slot] = []
        self._store: torch.distributed.Store | None = None
        self._monitor: PeerProcessMonitor | None = None
        self._active = False
        self._closed = False
        self._initialized = False
        self._failed = False
        self._refitting = False
        self._refit_epoch = 0
        self._refit_layers: set[str] = set()
        self._compute_stream: int | None = None
        self._weight_views: list[tuple[torch.Tensor, ...]] = []
        self._copy_plans: list[tuple[torch.Tensor, ...]] = []
        self._remote_sizes: list[int] = []
        self._local_sizes: list[int] = []
        self.remote_bytes = 0
        self.local_copy_bytes = 0
        self.staging_bytes = 0
        self.resident_bytes = 0
        self.forward_enqueues = 0
        self.token_enqueues = 0

    def declare_layer(self, name: str) -> None:
        """Record execution order before asynchronous checkpoint processing."""
        if self._store is not None or self._closed or name in self._declared_layers:
            raise ValueError("DWDP layers must be declared once before initialization")
        self._declared_layers[name] = len(self._declared_layers)

    def register_layer(
        self,
        name: str,
        num_experts: int,
        weights: Sequence[torch.Tensor],
        replicated_indices: Sequence[int] = (),
    ) -> dict[int, torch.Tensor]:
        """Retain packed local tensors and optionally replicate scalar metadata.

        ``replicated_indices`` identifies expert-major scalar vectors gathered
        once during initialization. Returned buffers are filled before profiling
        or inference; other tensors remain sharded throughout execution.
        """
        if self._refitting:
            return self._update_layer(name, num_experts, weights, replicated_indices)
        if self._store is not None or self._closed:
            raise RuntimeError("Cannot register weights after DWDP initialization")
        if name in self._indices or not weights:
            raise ValueError("DWDP layer names must be unique and weights nonempty")
        start, end = expert_range(num_experts, self.rank, self.world_size)
        for weight in weights:
            if (
                weight.device != self.device
                or weight.ndim < 2
                or weight.shape[0] != end - start
                or weight.numel() == 0
                or not weight.is_contiguous()
                or weight.requires_grad
            ):
                raise ValueError(
                    "DWDP weights must be contiguous, detached local CUDA expert shards"
                )
        if any(
            not 0 <= index < len(weights) or weights[index].shape[1:] != (1,)
            for index in replicated_indices
        ):
            raise ValueError("DWDP replicated metadata must be expert-major scalars")
        self._indices[name] = len(self.layers)
        replicated = {
            index: torch.empty(
                (num_experts, *weights[index].shape[1:]),
                device=self.device,
                dtype=weights[index].dtype,
            )
            for index in replicated_indices
        }
        self.layers.append(_Layer(name, num_experts, tuple(weights), replicated))
        self.resident_bytes += sum(w.nbytes for w in weights)
        self.resident_bytes += sum(w.nbytes for w in replicated.values())
        return replicated

    def initialize(self, store: torch.distributed.Store) -> None:
        """Exchange IPC mappings once, before profiling, warmup or inference.

        Use a fresh store or a unique PrefixStore for each worker generation.
        The caller sets the store's timeout to bound startup and shutdown waits.
        This method does not initialize or modify any default process group.
        """
        if self._store is not None or self._closed or not self.layers:
            raise RuntimeError("DWDP initialization requires uninitialized layers")
        from vllm._custom_ops import swap_blocks_batch

        if self._declared_layers:
            if set(self._declared_layers) != set(self._indices):
                raise ValueError("DWDP declared layers must all register their weights")
            self.layers.sort(key=lambda layer: self._declared_layers[layer.name])
            self._indices = {
                layer.name: index for index, layer in enumerate(self.layers)
            }
        self._copy = swap_blocks_batch
        self._store = store
        with torch.accelerator.device_index(self.device.index):
            self._copy_stream = torch.cuda.Stream(device=self.device)
            self._synchronous_fork = torch.cuda.Event()
            self._compute_fence = torch.cuda.Event()
            self._copy_join = torch.cuda.Event()
            self._has_compute_fence = False
            if self.allow_cuda_graph:
                from cuda.bindings import runtime

                self._runtime = runtime
                self._capture_fork = torch.cuda.Event()
            # A peer may start immediately after receiving its descriptors.
            torch.cuda.current_stream(self.device).synchronize()
            props = torch.cuda.get_device_properties(self.device)
            manifest = (
                socket.gethostname(),
                str(props.uuid),
                [
                    (
                        layer.name,
                        layer.num_experts,
                        [
                            (tuple(w.shape[1:]), str(w.dtype), i in layer.replicated)
                            for i, w in enumerate(layer.weights)
                        ],
                    )
                    for layer in self.layers
                ],
                os.getpid(),
                os.stat("/proc/self/ns/pid").st_ino,
            )
            store.set(f"manifest/{self.rank}", msgspec.msgpack.encode(manifest))
            manifests = [
                msgspec.msgpack.decode(store.get(f"manifest/{rank}"))
                for rank in range(self.world_size)
            ]
            if any(
                m[0] != manifest[0] or m[2] != manifests[self.rank][2]
                for m in manifests
            ):
                raise ValueError("DWDP requires matching layers on one host")
            if len({m[1] for m in manifests}) != self.world_size:
                raise ValueError("DWDP workers must use distinct physical GPUs")
            if any(m[4] != manifest[4] for m in manifests):
                raise ValueError("DWDP weight owners must share a PID namespace")

            # Each consumer needs its own PyTorch IPC reference-counter token.
            # Sending one reduce_tensor result to multiple consumers is unsafe.
            for rank in range(self.world_size):
                if rank == self.rank:
                    continue
                descriptors = []
                for layer in self.layers:
                    projections = []
                    for weight in layer.weights:
                        args = list(reduce_tensor(weight)[1])
                        # The receiver supplies fixed tensor/storage classes and
                        # its validated dtype; only IPC data crosses the store.
                        args[0] = args[4] = args[5] = None
                        projections.append(args)
                    descriptors.append(projections)
                store.set(
                    f"weights/{self.rank}/{rank}", msgspec.msgpack.encode(descriptors)
                )

            self._peers = [[] for _ in self.layers]
            for rank in range(self.world_size):
                if rank == self.rank:
                    weights = [layer.weights for layer in self.layers]
                else:
                    descriptors = msgspec.msgpack.decode(
                        store.get(f"weights/{rank}/{self.rank}")
                    )
                    weights = []
                    for layer, layer_args in zip(self.layers, descriptors):
                        tensors = []
                        for weight, args in zip(layer.weights, layer_args):
                            args[0] = torch.Tensor
                            args[4] = torch.storage.TypedStorage
                            args[5] = weight.dtype
                            # CUDA IPC maps the peer's storage into this context.
                            args[6] = self.device.index
                            tensors.append(rebuild_cuda_tensor(*args))
                        weights.append(tuple(tensors))
                for layer_peers, peer_weights in zip(self._peers, weights):
                    layer_peers.append(peer_weights)

            # Immutable per-expert metadata is replicated once. Packed weights
            # and block scales continue to use merge-free per-layer prefetch.
            for index, layer in enumerate(self.layers):
                for projection, dest in layer.replicated.items():
                    for rank in range(self.world_size):
                        start, end = expert_range(
                            layer.num_experts, rank, self.world_size
                        )
                        dest[start:end].copy_(self._peers[index][rank][projection])
            torch.cuda.current_stream(self.device).synchronize()

            counts = {len(layer.weights) for layer in self.layers}
            if len(counts) != 1:
                raise ValueError("DWDP layers must have the same number of projections")
            capacities = []
            for projection in range(len(self.layers[0].weights)):
                capacities.append(
                    max(
                        0
                        if projection in layer.replicated
                        else layer.weights[projection][0].numel()
                        * layer.num_experts
                        * layer.weights[projection].element_size()
                        for layer in self.layers
                    )
                )
            for _ in range(2):
                storage = tuple(
                    torch.empty(n, dtype=torch.uint8, device=self.device)
                    for n in capacities
                )
                self._slots.append(
                    _Slot(storage, torch.cuda.Event(), torch.cuda.Event())
                )
            self.staging_bytes = 2 * sum(capacities)
            for index, layer in enumerate(self.layers):
                if self.collect_metrics:
                    self._timings.append(
                        _Timings(
                            *[torch.cuda.Event(enable_timing=True) for _ in range(5)]
                        )
                    )
                destinations = self._views(index)
                self._weight_views.append(destinations)
                plan = []
                remote_size = local_size = 0
                for projection, dest in enumerate(destinations):
                    if projection in layer.replicated:
                        continue
                    pulls = []
                    for offset in range(self.world_size):
                        rank = (self.rank + offset) % self.world_size
                        start, end = expert_range(
                            layer.num_experts, rank, self.world_size
                        )
                        src = self._peers[index][rank][projection]
                        pull = (dest[start:end].data_ptr(), src.data_ptr(), src.nbytes)
                        if rank == self.rank:
                            if not self.merge_free:
                                plan.append(pull)
                                local_size += src.nbytes
                        else:
                            pulls.append(pull)
                            remote_size += src.nbytes
                    plan.extend(sliced_copy_plan(pulls, self.slice_bytes))
                self._copy_plans.append(
                    tuple(
                        torch.tensor(values, dtype=torch.int64, device="cpu")
                        for values in zip(*plan)
                    )
                )
                if self.allow_cuda_graph:
                    self._capture_copy_plans.append(plan)
                self._remote_sizes.append(remote_size)
                self._local_sizes.append(local_size)

            logger.info(
                "Initialized DWDP rank %d/%d: %d MoE layers, %.3f GiB resident, "
                "%.3f GiB staging, prefetch=%s",
                self.rank,
                self.world_size,
                len(self.layers),
                self.resident_bytes / 2**30,
                self.staging_bytes / 2**30,
                self.prefetch,
            )
            self._initialized = True
            self._monitor = PeerProcessMonitor(
                {
                    rank: metadata[3]
                    for rank, metadata in enumerate(manifests)
                    if rank != self.rank
                }
            )
            self.check_peer_health()

    @property
    def refitting(self) -> bool:
        """Whether all owners are inside a coordinated weight update."""
        return self._refitting

    def local_weights(self, name: str) -> tuple[torch.Tensor, ...]:
        """Return the resident tensors whose storage is exported to peers."""
        return self.layers[self._indices[name]].weights

    def _update_layer(
        self,
        name: str,
        num_experts: int,
        weights: Sequence[torch.Tensor],
        replicated_indices: Sequence[int],
    ) -> dict[int, torch.Tensor]:
        if self._failed or name not in self._indices or name in self._refit_layers:
            raise RuntimeError(f"Invalid or duplicate DWDP refit layer: {name}")
        layer = self.layers[self._indices[name]]
        if (
            num_experts != layer.num_experts
            or len(weights) != len(layer.weights)
            or set(replicated_indices) != set(layer.replicated)
            or any(
                src.shape != dest.shape
                or src.dtype != dest.dtype
                or src.device != dest.device
                or not src.is_contiguous()
                or src.requires_grad
                for src, dest in zip(weights, layer.weights)
            )
        ):
            raise ValueError(f"DWDP refit must preserve the storage layout of {name}")
        for dest, src in zip(layer.weights, weights):
            dest.copy_(src)
        self._refit_layers.add(name)
        return layer.replicated

    def _refit_barrier(self, phase: str) -> None:
        assert self._store is not None
        self.check_peer_health()
        prefix = f"refit/{self._refit_epoch}/{phase}"
        self._store.set(f"{prefix}/{self.rank}", b"1")
        self._store.wait([f"{prefix}/{rank}" for rank in range(self.world_size)])
        self.check_peer_health()

    @contextmanager
    def refit(self) -> Iterator[None]:
        """Update all owners in place after the caller pauses every scheduler.

        All ranks must enter once per update. The drain barrier fences remote
        reads before any owner overwrites IPC-exported storage. An exception
        poisons this manager; the caller must shut down the serving fleet.
        """
        if not self.initialized or self._active or self._refitting or self._failed:
            raise RuntimeError("DWDP refit requires an idle, healthy manager")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("DWDP cannot refit during CUDA graph capture")
        self._refit_epoch += 1
        self._refit_layers.clear()
        self._refitting = True
        try:
            torch.accelerator.synchronize(self.device)
            self._refit_barrier("drained")
            yield
            missing = self._indices.keys() - self._refit_layers
            if missing:
                raise RuntimeError(
                    f"DWDP refit did not update layers: {sorted(missing)}"
                )
            torch.accelerator.synchronize(self.device)
            self._refit_barrier("written")
            for index, layer in enumerate(self.layers):
                for projection, dest in layer.replicated.items():
                    for rank in range(self.world_size):
                        start, end = expert_range(
                            layer.num_experts, rank, self.world_size
                        )
                        dest[start:end].copy_(self._peers[index][rank][projection])
            torch.accelerator.synchronize(self.device)
            self._refit_barrier("replicated")
            for slot in self._slots:
                slot.layer = -1
                slot.has_consumer = False
                slot.ready_capture_id = None
            for timing in self._timings:
                timing.recorded = False
            self._capture_id = None
            self._compute_stream = None
            self._has_compute_fence = False
        except BaseException:
            self._failed = True
            raise
        finally:
            self._refitting = False

    def check_peer_health(self) -> None:
        if self._monitor is not None:
            self._monitor.check()

    @property
    def initialized(self) -> bool:
        return self._initialized and not self._closed

    def _views(self, index: int) -> tuple[torch.Tensor, ...]:
        layer = self.layers[index]
        slot = self._slots[index % 2]
        return tuple(
            layer.replicated[projection]
            if projection in layer.replicated
            else storage[: w[0].nbytes * layer.num_experts]
            .view(w.dtype)
            .view(layer.num_experts, *w.shape[1:])
            for projection, (storage, w) in enumerate(zip(slot.storage, layer.weights))
        )

    def _captured_copy(self, index: int) -> None:
        for dest, source, size in self._capture_copy_plans[index]:
            (error,) = self._runtime.cudaMemcpyAsync(
                dest,
                source,
                size,
                self._runtime.cudaMemcpyKind.cudaMemcpyDeviceToDevice,
                self._copy_stream.cuda_stream,
            )
            if error != self._runtime.cudaError_t.cudaSuccess:
                raise RuntimeError(f"DWDP captured copy failed: {error}")

    def _prefetch(self, index: int) -> None:
        slot = self._slots[index % 2]
        if slot.layer == index:
            return
        with torch.cuda.stream(self._copy_stream):
            if slot.has_consumer:
                self._copy_stream.wait_event(slot.consumed)
            if self.collect_metrics:
                self._timings[index].recorded = False
                self._timings[index].copy_start.record(self._copy_stream)
            dests, sources, sizes = self._copy_plans[index]
            if torch.cuda.is_current_stream_capturing():
                # cuMemcpyBatchAsync is not capture-compatible on all drivers.
                self._captured_copy(index)
            else:
                self._copy(sources, dests, sizes, is_src_access_order_any=True)
            self.local_copy_bytes += self._local_sizes[index]
            self.remote_bytes += self._remote_sizes[index]
            if self.collect_metrics:
                self._timings[index].copy_end.record(self._copy_stream)
            slot.ready.record(self._copy_stream)
            slot.ready_capture_id = (
                self._capture_id if torch.cuda.is_current_stream_capturing() else None
            )
        slot.layer = index

    def _prepare_capture(self, stream: torch.cuda.Stream, index: int) -> bool:
        capturing = torch.cuda.is_current_stream_capturing()
        if capturing:
            if not self.allow_cuda_graph or self.collect_metrics:
                raise RuntimeError(
                    "DWDP capture requires allow_cuda_graph=True "
                    "and collect_metrics=False"
                )
            error, _, capture_id, *_ = self._runtime.cudaStreamGetCaptureInfo(
                stream.cuda_stream
            )
            if error != self._runtime.cudaError_t.cudaSuccess:
                raise RuntimeError(f"DWDP cannot inspect stream capture: {error}")
            if capture_id != self._capture_id:
                self._capture_id = capture_id
                # Never wait on event records from before this capture.
                for slot in self._slots:
                    slot.layer = -1
                    slot.has_consumer = False
                    slot.ready_capture_id = None
                # Previous MoE graphs join their lookahead copy before ending.
                # Replays follow model order, so later pieces inherit that data.
                if self.prefetch and index > 0:
                    self._slots[index % 2].layer = index
                self._capture_fork.record(stream)
                self._copy_stream.wait_event(self._capture_fork)
        elif self._capture_id is not None:
            self._capture_id = None
            self._capture_fork.record(stream)
            self._copy_stream.wait_event(self._capture_fork)
            for slot in self._slots:
                slot.layer = -1
                slot.has_consumer = False
        return capturing

    def join_copy_stream(self, stream: torch.cuda.Stream) -> None:
        """Join pending copies without allocating a new CUDA event."""
        self._copy_join.record(self._copy_stream)
        stream.wait_event(self._copy_join)

    @contextmanager
    def layer(
        self, name: str, num_tokens: int = 0
    ) -> Iterator[tuple[torch.Tensor, ...]]:
        """Wait for this layer, prefetch the next, then mark weights consumed."""
        self.check_peer_health()
        if not self.initialized or self._failed or self._refitting:
            raise RuntimeError("DWDP manager is not initialized")
        if self._active:
            raise RuntimeError("DWDP does not support concurrent layer consumption")
        index = self._indices[name]
        stream = torch.cuda.current_stream(self.device)
        capturing = self._prepare_capture(stream, index)
        if index == 0 and not capturing:
            self.forward_enqueues += 1
            self.token_enqueues += num_tokens
        if (
            not capturing
            and self._compute_stream is not None
            and stream.cuda_stream != self._compute_stream
        ):
            stream.wait_event(self._compute_fence)
        if not capturing:
            self._compute_stream = stream.cuda_stream
        self._active = True
        slot = self._slots[index % 2]
        try:
            if self.collect_metrics:
                self._timings[index].wait_start.record(stream)
            if not self.prefetch:
                self._synchronous_fork.record(stream)
                self._copy_stream.wait_event(self._synchronous_fork)
            self._prefetch(index)
            if not capturing or slot.ready_capture_id == self._capture_id:
                stream.wait_event(slot.ready)
            if self.collect_metrics:
                self._timings[index].wait_end.record(stream)
            if self.prefetch and index + 1 < len(self.layers):
                self._prefetch(index + 1)
            yield self._weight_views[index]
        except BaseException:
            self._failed = True
            raise
        finally:
            slot.consumed.record(stream)
            if not capturing:
                self._compute_fence.record(stream)
                self._has_compute_fence = True
            if self.collect_metrics:
                self._timings[index].compute_end.record(stream)
                self._timings[index].recorded = True
            slot.has_consumer = True
            self._active = False

    def metrics(self) -> dict:
        """Return counters and the latest completed layer timings without waiting.

        Timing samples are diagnostic snapshots, not a running average. Event
        intervals include exposed eager dispatch gaps. No cross-rank operation
        or host/device synchronization is performed. Byte counters count host
        enqueues, including capture, but exclude CUDA graph replays.
        """
        timings = {}
        for layer, events in zip(self.layers, self._timings):
            if (
                not self._active
                and not self._failed
                and events.recorded
                and events.compute_end.query()
            ):
                timings[layer.name] = {
                    "prefetch_ms": events.copy_start.elapsed_time(events.copy_end),
                    "weight_wait_ms": events.wait_start.elapsed_time(events.wait_end),
                    "moe_ms": events.wait_end.elapsed_time(events.compute_end),
                }
        return {
            "remote_bytes": self.remote_bytes,
            "local_copy_bytes": self.local_copy_bytes,
            "resident_bytes": self.resident_bytes,
            "staging_bytes": self.staging_bytes,
            "remote_bytes_per_forward": sum(self._remote_sizes),
            "local_copy_bytes_per_forward": sum(self._local_sizes),
            "forward_enqueues": self.forward_enqueues,
            "token_enqueues": self.token_enqueues,
            "layers": timings,
        }

    def close(self) -> None:
        """Release imports before their owners release immutable source tensors.

        All workers must call this after their last forward. If a peer fails,
        the store timeout raises and sources remain retained until process exit.
        """
        if self._closed:
            return
        if self._active:
            raise RuntimeError("Cannot close DWDP during layer consumption")
        if self._monitor is not None:
            self._monitor.close()
            self._monitor.check()
        if self._store is not None:
            torch.accelerator.synchronize(self.device)
            self._peers.clear()
            gc.collect()
            self._store.set(f"closed/{self.rank}", b"1")
            self._store.wait([f"closed/{r}" for r in range(self.world_size)])
            torch.cuda.ipc_collect()
        self.layers.clear()
        self._declared_layers.clear()
        self._weight_views.clear()
        self._slots.clear()
        self._copy_plans.clear()
        self._capture_copy_plans.clear()
        self._timings.clear()
        self._closed = True


class DWDPWeightOffloader(BaseOffloader):
    """Integrate owned expert loading and startup IPC with vLLM offloading."""

    def __init__(self, config: "DWDPConfig") -> None:
        self.config = config
        self._manager: DWDPWeightManager | None = None

    @property
    def manager(self) -> DWDPWeightManager:
        if self._manager is None:
            self._manager = DWDPWeightManager(
                self.config.rank,
                self.config.world_size,
                torch.device("cuda", torch.accelerator.current_device_index()),
                self.config.slice_bytes,
                self.config.prefetch,
                self.config.collect_metrics,
                self.config.merge_free,
                self.config.allow_cuda_graph,
            )
        return self._manager

    def wrap_modules(self, modules_generator, prefix=""):
        return list(modules_generator)

    def post_init(self) -> None:
        if self._manager is None:
            raise ValueError("DWDP found no supported routed expert layers")
        store = torch.distributed.TCPStore(
            self.config.host,
            self.config.port,
            self.config.world_size,
            self.config.rank == 0,
            timedelta(seconds=self.config.timeout_seconds),
        )
        self.manager.initialize(store)

    def shutdown(self) -> None:
        if self._manager is not None:
            self._manager.close()

    def sync_prev_onload(self) -> None:
        if self._manager is not None and self._manager.initialized:
            self._manager.check_peer_health()
            if self._manager._has_compute_fence:
                torch.cuda.current_stream().wait_event(self._manager._compute_fence)
            self._manager.join_copy_stream(torch.cuda.current_stream())

    def join_after_forward(self) -> None:
        if self._manager is not None and self._manager.initialized:
            if torch.cuda.is_current_stream_capturing():
                if not self._manager.allow_cuda_graph:
                    return
                runtime = self._manager._runtime
                error, _, capture_id, *_ = runtime.cudaStreamGetCaptureInfo(
                    torch.cuda.current_stream().cuda_stream
                )
                if error != runtime.cudaError_t.cudaSuccess:
                    raise RuntimeError(f"DWDP cannot inspect stream capture: {error}")
                if capture_id != self._manager._capture_id:
                    return
            self._manager.join_copy_stream(torch.cuda.current_stream())
