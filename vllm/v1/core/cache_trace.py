# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Keep Mamba accounting in memory; emit one JSON summary on hash removal.

Counts and downstream hashes cover one GPU residency/cache group, up to removal.
Downstream hashes identify confirmed computed prefixes, not necessarily cached
checkpoints. No token IDs, per-step events, or device reads are recorded.
"""

import json
import os
import socket
import time
import uuid
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
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
    prefix_hash: str
    group_id: int
    prefix_length: int
    hash_block_size: int
    block_id: int
    generation: int
    created_ns: int | None = None
    created_monotonic_ns: int | None = None
    retired_ns: int | None = None
    retired_monotonic_ns: int | None = None
    removed: bool = False
    lookup_hit_count: int = 0
    successful_load_count: int = 0
    checkpoint_resume_count: int = 0
    computed_hashes: set[str] = field(default_factory=set)


class CacheTrace:
    """Observe scheduler completions silently until a cached hash is removed."""

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
        self.pending: dict[str, list[tuple[Checkpoint, int]]] = {}
        self.active: dict[str, list[tuple[Checkpoint, int]]] = {}
        self.external: dict[str, int] = {}
        self._sink = sink
        self._sequence = 0

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
        entry = Checkpoint(
            prefix_hash=request.block_hashes[num_tokens // hash_block_size - 1].hex(),
            group_id=group_id,
            prefix_length=num_tokens,
            hash_block_size=hash_block_size,
            block_id=block.block_id,
            generation=self.generations.get(block.block_id, 0),
        )
        self.entries[block.block_id, key] = entry
        self.by_block.setdefault(block.block_id, {})[key] = entry
        if num_tokens > self.external.get(request.request_id, 0):
            if num_tokens <= request.num_computed_tokens - request.num_in_flight_tokens:
                self._created(entry)
            else:
                self.unconfirmed.setdefault(request.request_id, []).append(entry)

    @staticmethod
    def _created(entry: Checkpoint) -> None:
        if entry.created_ns is None:
            entry.created_ns = time.time_ns()
            entry.created_monotonic_ns = time.monotonic_ns()

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
                    self._created(entry)
                    self.pending.setdefault(request.request_id, []).append(
                        (entry, boundary)
                    )

    def progress(self, request: "Request") -> None:
        completed = max(0, request.num_computed_tokens - request.num_in_flight_tokens)
        unconfirmed = []
        for entry in self.unconfirmed.pop(request.request_id, []):
            if entry.removed:
                continue
            if entry.prefix_length <= completed:
                self._created(entry)
            else:
                unconfirmed.append(entry)
        if unconfirmed:
            self.unconfirmed[request.request_id] = unconfirmed
        waiting = []
        for entry, position in self.pending.pop(request.request_id, []):
            if entry.removed:
                continue
            if completed <= position:
                waiting.append((entry, position))
                continue
            entry.checkpoint_resume_count += 1
            self.active.setdefault(request.request_id, []).append((entry, position))
        if waiting:
            self.pending[request.request_id] = waiting
        active = []
        for entry, position in self.active.get(request.request_id, []):
            if entry.removed:
                continue
            end = min(completed // entry.hash_block_size, len(request.block_hashes))
            start = position // entry.hash_block_size
            entry.computed_hashes.update(
                h.hex() for h in request.block_hashes[start:end]
            )
            active.append((entry, max(position, end * entry.hash_block_size)))
        if active:
            self.active[request.request_id] = active
        else:
            self.active.pop(request.request_id, None)

    def release(
        self, request_id: str, blocks: Iterable["KVCacheBlock"], *, retired: bool
    ) -> None:
        if not retired:
            return
        for block_id in {block.block_id for block in blocks if not block.is_null}:
            for entry in self.by_block.get(block_id, {}).values():
                if entry.retired_ns is None:
                    entry.retired_ns = time.time_ns()
                    entry.retired_monotonic_ns = time.monotonic_ns()

    def remove(self, block_id: int, keys: Iterable[bytes], reason: str) -> None:
        for key in keys:
            entry = self.entries.pop((block_id, key), None)
            if entry is None:
                continue
            del self.by_block[block_id][key]
            if not self.by_block[block_id]:
                del self.by_block[block_id]
            entry.removed = True
            self._sequence += 1
            now = time.monotonic_ns()
            birth = entry.created_monotonic_ns
            record = {
                **self.context,
                "schema_version": 2,
                "event": "hash_removed",
                "medium": "GPU",
                "instance_id": self.instance_id,
                "run_id": self.run_id,
                "sequence": self._sequence,
                "prefix_hash": entry.prefix_hash,
                "group_id": entry.group_id,
                "prefix_length": entry.prefix_length,
                "block_id": entry.block_id,
                "generation": entry.generation,
                "reason": reason,
                "lookup_hit_count": entry.lookup_hit_count,
                "successful_load_count": entry.successful_load_count,
                "checkpoint_resume_count": entry.checkpoint_resume_count,
                "created_ns": entry.created_ns,
                "retired_ns": entry.retired_ns,
                "removed_ns": time.time_ns(),
                "lifetime_ns": None if birth is None else now - birth,
                "time_to_retirement_ns": (
                    None
                    if birth is None or entry.retired_monotonic_ns is None
                    else max(0, entry.retired_monotonic_ns - birth)
                ),
                "computed_hashes": sorted(entry.computed_hashes),
            }
            if self._sink is not None:
                self._sink(record)
            else:
                logger.info(
                    "MAMBA_CACHE_TRACE %s", json.dumps(record, separators=(",", ":"))
                )
            entry.computed_hashes.clear()

    def allocate(self, block_id: int) -> None:
        self.generations[block_id] = self.generations.get(block_id, 0) + 1

    def move(self, source_id: int, destination_id: int) -> None:
        for key, entry in self.by_block.pop(source_id, {}).items():
            del self.entries[source_id, key]
            entry.block_id = destination_id
            entry.generation = self.generations.get(destination_id, 0)
            self.entries[destination_id, key] = entry
            self.by_block.setdefault(destination_id, {})[key] = entry

    def detach(self, request_id: str) -> None:
        self.pending.pop(request_id, None)
        self.external.pop(request_id, None)
        self.unconfirmed.pop(request_id, None)
        self.active.pop(request_id, None)

    def finish(self, request: "Request") -> None:
        self.detach(request.request_id)
