# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Observe Mamba reuse for optional removal summaries and post-use eviction.

Counts and downstream hashes cover one GPU residency/cache group, up to removal.
Downstream hashes identify confirmed computed prefixes, not necessarily cached
checkpoints. Direct children are the first completed hash boundary per resume.
Only distinct counts are emitted. No token IDs, per-step events, or device reads
are recorded. The optional eviction policy protects prefixes after two completed
direct uses, including identical children, until an explicit cache reset.
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
        # Optional connector observer; report both uses, including completions
        # after the first use removed the GPU residency.
        self.on_direct_use: Callable[[int, str], None] | None = None
        self._sequence = 0
        # Saturated use counts survive residency changes. Pending uses also
        # survive hash removal: an already-admitted reader can finish later.
        self._direct_use_counts: dict[tuple[int, str], int] = {}
        self._pending_direct_uses: dict[str, dict[tuple[int, str], Checkpoint]] = {}

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
                    self._track_direct_use(request_id, entry)
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
                    self._track_direct_use(request.request_id, entry)
                    entry.successful_load_count += 1
                    self.pending.setdefault(request.request_id, []).append(
                        (entry, boundary)
                    )

    def _track_direct_use(self, request_id: str, entry: Checkpoint) -> None:
        key = (entry.group_id, entry.prefix_hash)
        if (
            self._on_direct_descendant is not None
            and self._direct_use_counts.get(key, 0) < 2
        ):
            self._pending_direct_uses.setdefault(request_id, {})[key] = entry

    def _observe_direct_uses(self, request: "Request", completed: int) -> None:
        uses = self._pending_direct_uses.pop(request.request_id, {})
        waiting = {}
        for key, entry in uses.items():
            if self._direct_use_counts.get(key, 0) >= 2:
                continue
            child_index = entry.prefix_length // entry.hash_block_size
            if (
                completed >= entry.prefix_length + entry.hash_block_size
                and child_index < len(request.block_hashes)
            ):
                self._direct_use_counts[key] = self._direct_use_counts.get(key, 0) + 1
                if self.on_direct_use is not None:
                    self.on_direct_use(entry.group_id, entry.prefix_hash)
            else:
                waiting[key] = entry
        if waiting:
            self._pending_direct_uses[request.request_id] = waiting

    def _is_protected(self, entry: Checkpoint) -> bool:
        return self._direct_use_counts.get((entry.group_id, entry.prefix_hash), 0) >= 2

    def clear_eviction_history(self) -> None:
        """Forget protection after a successful explicit prefix-cache reset."""
        self._direct_use_counts.clear()
        self._pending_direct_uses.clear()

    def progress(self, request: "Request") -> None:
        completed = max(0, request.num_computed_tokens - request.num_in_flight_tokens)
        if self._on_direct_descendant is not None:
            self._observe_direct_uses(request, completed)
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
                if self._on_direct_descendant is not None and not self._is_protected(
                    entry
                ):
                    evict_block_ids.add(entry.block_id)
            entry.computed_hashes.update(
                h.hex() for h in request.block_hashes[start:end]
            )
            active.append((entry, max(position, end * entry.hash_block_size)))
        # Eviction emits the final counters and may remove aliases of the same
        # state. Do it after completing this request's accounting.
        if self._on_direct_descendant is not None:
            for block_id in evict_block_ids:
                # Block-level eviction removes every alias of the state.
                if not any(
                    self._is_protected(entry)
                    for entry in self.by_block.get(block_id, {}).values()
                ):
                    self._on_direct_descendant(block_id)
        active = [(entry, position) for entry, position in active if not entry.removed]
        if active:
            self.active[request.request_id] = active
        else:
            self.active.pop(request.request_id, None)

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
        self._pending_direct_uses.pop(request_id, None)
        self.pending.pop(request_id, None)
        self.external.pop(request_id, None)
        self.active.pop(request_id, None)

    def finish(self, request: "Request") -> None:
        self.detach(request.request_id)
