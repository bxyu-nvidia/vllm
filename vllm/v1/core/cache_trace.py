# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in Mamba checkpoint provenance. No tensor reads or device synchronization.

Records are JSON after the ``MAMBA_CACHE_TRACE `` log marker. Timestamps are
observer timestamps, not kernel timestamps. Counts belong to a residency, not
to every copy of a hash in the cluster. Request token records are append-only
and referenced by (instance_id, request_id, token range).
"""

import os
import socket
import threading
import time
import uuid
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_utils import KVCacheBlock
    from vllm.v1.request import Request

logger = init_logger(__name__)


def trace_enabled() -> bool:
    return os.getenv("VLLM_MAMBA_CACHE_TRACE", "0").lower() in ("1", "true")


@dataclass
class Checkpoint:
    residency_id: str
    prefix_hash: str
    cache_hash: str
    group_id: int
    layers: list[str]
    prefix_length: int
    block_id: int
    generation: int
    producer_request_id: str | None
    origin: str
    registered_ns: int
    registered_monotonic_ns: int
    checkpoint_created_ns: int | None = None
    request_released_ns: int | None = None
    retired_ns: int | None = None
    eviction_eligible_ns: int | None = None
    hash_removed_ns: int | None = None
    storage_reclaimed_ns: int | None = None
    lookup_hit_count: int = 0
    successful_load_count: int = 0
    checkpoint_resume_count: int = 0
    consumers: list[dict[str, object]] = field(default_factory=list)
    consumers_truncated: bool = False


class CacheTrace:
    """Scheduler-owned checkpoint histories, bounded by slots and live requests."""

    def __init__(
        self, *, sink: Callable[[dict[str, object]], None] | None = None
    ) -> None:
        self.instance_id = f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex}"
        self.run_id = os.getenv("VLLM_MAMBA_CACHE_TRACE_RUN_ID")
        self.context: dict[str, object] = {}
        self.groups: dict[int, list[str]] = {}
        self.generations: dict[int, int] = {}
        self.entries: dict[tuple[int, bytes], Checkpoint] = {}
        self.by_block: dict[int, dict[bytes, Checkpoint]] = {}
        self.unconfirmed: dict[str, list[Checkpoint]] = {}
        self.removed: dict[int, list[Checkpoint]] = {}
        self.pending: dict[str, list[tuple[Checkpoint, int]]] = {}
        self.active: dict[str, list[tuple[Checkpoint, dict[str, object]]]] = {}
        self.external: dict[str, int] = {}
        self.token_offsets: dict[str, int] = {}
        self._sink = sink
        self._sequence = 0
        self._lock = threading.Lock()

    def emit(self, event: str, **fields: object) -> None:
        with self._lock:
            self._sequence += 1
            sequence = self._sequence
        record = {
            "schema_version": 1,
            "event": event,
            "instance_id": self.instance_id,
            "run_id": self.run_id,
            "sequence": sequence,
            "timestamp_ns": time.time_ns(),
            "monotonic_ns": time.monotonic_ns(),
            **self.context,
            **fields,
        }
        if self._sink is not None:
            self._sink(record)
        else:
            import json

            logger.info(
                "MAMBA_CACHE_TRACE %s", json.dumps(record, separators=(",", ":"))
            )

    def summary(self, event: str, entry: Checkpoint, **fields: object) -> None:
        self.emit(
            event,
            **asdict(entry),
            observed_lifetime_ns=time.monotonic_ns() - entry.registered_monotonic_ns,
            token_reference={
                "instance_id": self.instance_id,
                "request_id": entry.producer_request_id,
                "start": 0,
                "end": entry.prefix_length,
            },
            **fields,
        )

    def tokens(self, request: "Request") -> None:
        offset = self.token_offsets.get(request.request_id, 0)
        end = len(request.all_token_ids)
        if end > offset:
            self.emit(
                "request_tokens",
                request_id=request.request_id,
                start=offset,
                end=end,
                prompt_length=request.num_prompt_tokens,
                token_ids=list(request.all_token_ids[offset:end]),
            )
            self.token_offsets[request.request_id] = end

    def register(
        self,
        block: "KVCacheBlock",
        key: bytes,
        group_id: int,
        num_tokens: int,
        request: "Request",
        hash_block_size: int,
    ) -> None:
        if group_id not in self.groups or (block.block_id, key) in self.entries:
            return
        self.tokens(request)
        external = num_tokens <= self.external.get(request.request_id, 0)
        entry = Checkpoint(
            residency_id=uuid.uuid4().hex,
            prefix_hash=request.block_hashes[num_tokens // hash_block_size - 1].hex(),
            cache_hash=key[:-4].hex(),
            group_id=group_id,
            layers=self.groups[group_id],
            prefix_length=num_tokens,
            block_id=block.block_id,
            generation=self.generations.get(block.block_id, 0),
            producer_request_id=None if external else request.request_id,
            origin="external" if external else "computed",
            registered_ns=time.time_ns(),
            registered_monotonic_ns=time.monotonic_ns(),
        )
        if not external and (
            num_tokens <= request.num_computed_tokens - request.num_in_flight_tokens
        ):
            entry.checkpoint_created_ns = time.time_ns()
        self.entries[block.block_id, key] = entry
        self.by_block.setdefault(block.block_id, {})[key] = entry
        if entry.checkpoint_created_ns is None and not external:
            self.unconfirmed.setdefault(request.request_id, []).append(entry)
        elif entry.checkpoint_created_ns is not None:
            self._reclaim_removed(block.block_id, "checkpoint_overwritten")
        self.summary(
            "checkpoint_registered", entry, adopting_request_id=request.request_id
        )

    def lookup(self, block_id: int, key: bytes) -> None:
        entry = self.entries.get((block_id, key))
        if entry is not None:
            entry.lookup_hit_count += 1

    def adopt(
        self, request_id: str, blocks: Iterable["KVCacheBlock"], resume_position: int
    ) -> None:
        for block in blocks:
            for entry in self.by_block.get(block.block_id, {}).values():
                if entry.prefix_length == resume_position:
                    self.pending.setdefault(request_id, []).append(
                        (entry, resume_position)
                    )

    def expect_external(self, request_id: str, num_tokens: int) -> None:
        self.external[request_id] = num_tokens

    def loaded(self, request: "Request", blocks: Iterable["KVCacheBlock"]) -> None:
        boundary = self.external.pop(request.request_id, None)
        if boundary is None:
            return
        for block in blocks:
            for entry in self.by_block.get(block.block_id, {}).values():
                if entry.prefix_length == boundary:
                    entry.successful_load_count += 1
                    entry.checkpoint_created_ns = time.time_ns()
                    self._reclaim_removed(block.block_id, "checkpoint_loaded_overwrite")
                    self.pending.setdefault(request.request_id, []).append(
                        (entry, boundary)
                    )
                    self.summary(
                        "checkpoint_loaded", entry, request_id=request.request_id
                    )

    def progress(self, request: "Request") -> None:
        self.tokens(request)
        completed = max(0, request.num_computed_tokens - request.num_in_flight_tokens)
        unconfirmed = []
        for entry in self.unconfirmed.pop(request.request_id, []):
            if entry.hash_removed_ns is not None:
                continue
            if entry.prefix_length <= completed:
                entry.checkpoint_created_ns = time.time_ns()
                self._reclaim_removed(entry.block_id, "checkpoint_overwritten")
                self.summary("checkpoint_created", entry)
            else:
                unconfirmed.append(entry)
        if unconfirmed:
            self.unconfirmed[request.request_id] = unconfirmed
        waiting = []
        for entry, position in self.pending.pop(request.request_id, []):
            if completed <= position:
                waiting.append((entry, position))
                continue
            entry.checkpoint_resume_count += 1
            consumer: dict[str, object] = {
                "request_id": request.request_id,
                "resume_position": position,
                "computed_start": position,
                "computed_end": completed,
                "resumed_ns": time.time_ns(),
            }
            if len(entry.consumers) < 32:
                entry.consumers.append(consumer)
            else:
                entry.consumers_truncated = True
            self.active.setdefault(request.request_id, []).append((entry, consumer))
            self.summary("checkpoint_resumed", entry, consumer=consumer)
        if waiting:
            self.pending[request.request_id] = waiting
        for entry, consumer in self.active.get(request.request_id, []):
            consumer["computed_end"] = completed
            self.emit(
                "checkpoint_compute",
                residency_id=entry.residency_id,
                prefix_hash=entry.prefix_hash,
                group_id=entry.group_id,
                **consumer,
            )

    def release(
        self, request_id: str, blocks: Iterable["KVCacheBlock"], *, retired: bool
    ) -> None:
        ids = {block.block_id for block in blocks if not block.is_null}
        now = time.time_ns()
        for block_id in ids:
            for entry in self.by_block.get(block_id, {}).values():
                entry.request_released_ns = now
                if retired:
                    entry.retired_ns = now
                self.summary(
                    "checkpoint_retired" if retired else "request_released",
                    entry,
                    request_id=request_id,
                )

    def eligible(self, block_id: int) -> None:
        for entry in self.by_block.get(block_id, {}).values():
            entry.eviction_eligible_ns = time.time_ns()
            self.summary("eviction_eligible", entry)

    def remove(self, block_id: int, keys: Iterable[bytes], reason: str) -> None:
        for key in keys:
            entry = self.entries.pop((block_id, key), None)
            if entry is not None:
                del self.by_block[block_id][key]
                if not self.by_block[block_id]:
                    del self.by_block[block_id]
                entry.hash_removed_ns = time.time_ns()
                self.removed.setdefault(block_id, []).append(entry)
                self.summary("hash_removed", entry, reason=reason)

    def allocate(self, block_id: int) -> None:
        self._reclaim_removed(block_id, "slot_reallocated")
        self.generations[block_id] = self.generations.get(block_id, 0) + 1

    def _reclaim_removed(self, block_id: int, reason: str) -> None:
        for entry in self.removed.pop(block_id, []):
            entry.storage_reclaimed_ns = time.time_ns()
            self.summary("storage_reclaimed", entry, reason=reason)

    def move(self, source_id: int, destination_id: int) -> None:
        for key, entry in self.by_block.pop(source_id, {}).items():
            del self.entries[source_id, key]
            entry.block_id = destination_id
            entry.generation = self.generations.get(destination_id, 0)
            self.entries[destination_id, key] = entry
            self.by_block.setdefault(destination_id, {})[key] = entry
            self.summary(
                "checkpoint_move_scheduled", entry, previous_block_id=source_id
            )

    def detach(self, request_id: str) -> None:
        self.pending.pop(request_id, None)
        self.external.pop(request_id, None)
        self.unconfirmed.pop(request_id, None)
        for entry, consumer in self.active.pop(request_id, []):
            self.emit("consumer_detached", residency_id=entry.residency_id, **consumer)

    def finish(self, request: "Request") -> None:
        self.tokens(request)
        self.emit(
            "request_finished",
            request_id=request.request_id,
            status=request.status.name,
            prompt_length=request.num_prompt_tokens,
            token_count=len(request.all_token_ids),
        )
        self.detach(request.request_id)
        self.token_offsets.pop(request.request_id, None)


_worker_trace: CacheTrace | None = None


def worker_trace() -> CacheTrace | None:
    """Return a process-local event writer for connector transfer facts."""
    global _worker_trace
    if not trace_enabled():
        return None
    if _worker_trace is None:
        _worker_trace = CacheTrace()
    return _worker_trace
