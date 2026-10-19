# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Observe Mamba reuse for optional removal summaries and post-use eviction.

Counts and downstream hashes cover one GPU residency/cache group, up to removal.
Downstream hashes identify confirmed computed prefixes, not necessarily cached
checkpoints. Direct children are the first completed hash boundary per resume.
Only distinct counts are emitted. No token IDs, per-step events, or device reads
are recorded.
"""

import json
import os
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
    removed: bool = False
    lookup_hit_count: int = 0
    successful_load_count: int = 0
    checkpoint_resume_count: int = 0
    computed_hashes: set[str] = field(default_factory=set)
    direct_hashes: set[str] = field(default_factory=set)


class CacheTrace:
    """Observe scheduler completions silently until a cached hash is removed."""

    def __init__(
        self,
        *,
        sink: Callable[[dict[str, object]], None] | None = None,
        emit_events: bool = True,
        on_direct_descendant: Callable[[int], None] | None = None,
    ) -> None:
        self.groups: dict[int, list[str]] = {}
        self.generations: dict[int, int] = {}
        self.entries: dict[tuple[int, bytes], Checkpoint] = {}
        self.by_block: dict[int, dict[bytes, Checkpoint]] = {}
        self.pending: dict[str, list[tuple[Checkpoint, int]]] = {}
        self.active: dict[str, list[tuple[Checkpoint, int]]] = {}
        self.external: dict[str, int] = {}
        self._sink = sink
        self._emit_events = emit_events
        self._on_direct_descendant = on_direct_descendant
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
                    self.pending.setdefault(request.request_id, []).append(
                        (entry, boundary)
                    )

    def progress(self, request: "Request") -> None:
        completed = max(0, request.num_computed_tokens - request.num_in_flight_tokens)
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
        evict_block_ids: set[int] = set()
        for entry, position in self.active.get(request.request_id, []):
            if entry.removed:
                continue
            end = min(completed // entry.hash_block_size, len(request.block_hashes))
            start = position // entry.hash_block_size
            # Only the first completed hash boundary after the adopted state
            # is a direct child; later boundaries are descendants.
            if start == entry.prefix_length // entry.hash_block_size and end > start:
                entry.direct_hashes.add(request.block_hashes[start].hex())
                if self._on_direct_descendant is not None:
                    evict_block_ids.add(entry.block_id)
            entry.computed_hashes.update(
                h.hex() for h in request.block_hashes[start:end]
            )
            if entry.block_id not in evict_block_ids:
                active.append((entry, max(position, end * entry.hash_block_size)))
        if active:
            self.active[request.request_id] = active
        else:
            self.active.pop(request.request_id, None)
        # Eviction emits the final counters and may remove aliases of the same
        # state. Do it after completing this request's accounting.
        if self._on_direct_descendant is not None:
            for block_id in evict_block_ids:
                self._on_direct_descendant(block_id)

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
            record = {
                "event": "hash_removed",
                "sequence": self._sequence,
                "prefix_hash": entry.prefix_hash,
                "group_id": entry.group_id,
                "prefix_length": entry.prefix_length,
                "block_id": entry.block_id,
                "generation": entry.generation,
                "lookup_hit_count": entry.lookup_hit_count,
                "successful_load_count": entry.successful_load_count,
                "checkpoint_resume_count": entry.checkpoint_resume_count,
                "computed_hash_count_direct": len(entry.direct_hashes),
                "computed_hash_count_descendants": len(entry.computed_hashes),
            }
            if self._emit_events and self._sink is not None:
                self._sink(record)
            elif self._emit_events:
                logger.info(
                    "MAMBA_CACHE_TRACE %s", json.dumps(record, separators=(",", ":"))
                )
            entry.computed_hashes.clear()
            entry.direct_hashes.clear()

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
        self.active.pop(request_id, None)

    def finish(self, request: "Request") -> None:
        self.detach(request.request_id)
