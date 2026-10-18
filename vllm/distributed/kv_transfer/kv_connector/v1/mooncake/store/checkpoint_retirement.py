# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Best-effort retirement of superseded Mooncake Mamba checkpoints."""

import hashlib
import queue
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Sequence
from typing import Any

from vllm.logger import init_logger

from .data import MambaCheckpointCommit

logger = init_logger(__name__)

# Bound both retained metadata and maintenance RPC work. Overflow leaves data
# for normal eviction; retirement must never block inference or grow unbounded.
_MAX_PENDING = 4096
_RETRY_BATCH_SIZE = 128
_RETRY_INTERVAL = 1.0
_OBJECT_NOT_FOUND = -704


def _layout_digest(lookup_key_prefixes: Sequence[Sequence[str]]) -> str:
    layout = "\n".join(sorted({p for group in lookup_key_prefixes for p in group}))
    return hashlib.sha256(layout.encode()).hexdigest()


def retired_checkpoint_prefix(lookup_key_prefixes: Sequence[Sequence[str]]) -> str:
    """Return the shared retirement namespace, independent of DP instance IDs."""
    return f"vllm-mamba-retired-v1@{_layout_digest(lookup_key_prefixes)}"


class MambaCheckpointRetirer(threading.Thread):
    """Keep roots and tips, deleting intermediate Mamba keys after save commits.

    Only the elected worker receives commits, after every rank reports a
    successful save. Tiny derived-checkpoint markers live in the shared store
    so a continuation can move between DP instances. Missing markers (including
    evicted markers) conservatively make their checkpoints anchors.
    """

    def __init__(
        self,
        store: Any,
        lookup_key_prefixes: Sequence[Sequence[str]],
        mamba_group_ids: Sequence[int] | set[int],
        *,
        record_operation: Callable[..., None] | None = None,
    ) -> None:
        super().__init__(daemon=True, name="MambaCheckpointRetirer")
        self.store = store
        self._record_operation = record_operation
        self._mamba_prefixes = tuple(
            dict.fromkeys(
                prefix
                for gid in sorted(mamba_group_ids)
                for prefix in lookup_key_prefixes[gid]
            )
        )
        # Include the full layout/model/cache namespace, never DP instance IDs.
        digest = _layout_digest(lookup_key_prefixes)
        self._marker_prefix = f"vllm-mamba-lineage-v1@{digest}"
        self._retired_prefix = retired_checkpoint_prefix(lookup_key_prefixes)
        self._commits: queue.Queue[tuple[int, MambaCheckpointCommit]] = queue.Queue(
            _MAX_PENDING
        )
        self._maintenance_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._generation = 0
        self._pending: OrderedDict[str, list[str]] = OrderedDict()

    def submit(self, commit: MambaCheckpointCommit) -> None:
        """Queue an acknowledged save without blocking the model thread."""
        if commit.parent_hash is None or self._stop_event.is_set():
            return
        try:
            self._commits.put_nowait((self._generation, commit))
        except queue.Full:
            logger.warning_once(
                "Mamba checkpoint retirement queue full; leaving excess checkpoints "
                "for normal Mooncake eviction"
            )

    def close(self) -> None:
        """Stop issuing maintenance RPCs before the Store handle is closed."""
        self._stop_event.set()
        if self.is_alive():
            self.join(timeout=5.0)
            if self.is_alive():
                logger.warning("Mamba checkpoint retirement RPC still active at close")

    def reset_store(self) -> None:
        """Serialize a caller-requested cache reset against retirement work."""
        with self._maintenance_lock:
            self.store.remove_all(force=True)
            self._generation += 1
            self._pending.clear()
            while True:
                try:
                    self._commits.get_nowait()
                except queue.Empty:
                    break
                self._commits.task_done()

    def _marker(self, checkpoint_hash: str) -> str:
        return f"{self._marker_prefix}@{checkpoint_hash}"

    def _keys(self, checkpoint_hash: str) -> list[str]:
        return [f"{prefix}@{checkpoint_hash}" for prefix in self._mamba_prefixes]

    def _record(
        self,
        operation: str,
        start: float,
        count: int,
        *,
        failed: int = 0,
        error: bool = False,
    ) -> None:
        if self._record_operation is not None:
            self._record_operation(
                operation=operation,
                duration_seconds=time.perf_counter() - start,
                num_keys=count,
                num_failed_keys=failed,
                status="error" if error else "deferred" if failed else "ok",
            )

    def commit(self, commit: MambaCheckpointCommit) -> None:
        """Publish lineage and retire the parent, if it is not an anchor."""
        parent = commit.parent_hash
        child = commit.checkpoint_hash
        if parent is None or parent == child or not self._mamba_prefixes:
            return
        child_keys = self._keys(child)
        probes = [*child_keys, self._marker(parent), self._marker(child)]
        start = time.perf_counter()
        try:
            exists = self.store.batch_is_exist(probes)
            if len(exists) != len(probes) or any(x < 0 for x in exists):
                raise RuntimeError("invalid checkpoint existence result")
        except Exception:
            self._record(
                "checkpoint_retire_exists",
                start,
                len(probes),
                failed=len(probes),
                error=True,
            )
            logger.warning("Mamba checkpoint retirement probe failed", exc_info=True)
            return
        self._record("checkpoint_retire_exists", start, len(probes))
        # A save may have been evicted before its acknowledgment arrived. Only
        # retire after all required Mamba namespaces are still complete.
        if any(found != 1 for found in exists[:-2]):
            return
        if exists[-1] != 1:
            start = time.perf_counter()
            try:
                # Use the default ReplicateConfig, not the mutable config used
                # by tensor PUTs (which may carry their attention group IDs).
                result = self.store.put(self._marker(child), b"1")
                if result != 0:
                    raise RuntimeError(f"lineage PUT returned {result}")
            except Exception:
                self._record("checkpoint_lineage_put", start, 1, failed=1, error=True)
                logger.warning("Mamba checkpoint lineage write failed", exc_info=True)
                return
            self._record("checkpoint_lineage_put", start, 1)
        if exists[-2] != 1:
            return  # No known predecessor for the parent: retain it as an anchor.
        # Publish before removal: new lookups must stop renewing payload leases
        # while existing readers drain. This marker is not in a tensor group.
        start = time.perf_counter()
        try:
            result = self.store.put(f"{self._retired_prefix}@{parent}", b"1")
            if result != 0:
                raise RuntimeError(f"retirement marker PUT returned {result}")
        except Exception:
            self._record("checkpoint_retired_put", start, 1, failed=1, error=True)
            logger.warning(
                "Mamba checkpoint retirement marker write failed", exc_info=True
            )
            return
        self._record("checkpoint_retired_put", start, 1)
        pending = self._remove(self._keys(parent))
        if pending:
            self._pending[parent] = pending
            self._pending.move_to_end(parent)
            if len(self._pending) > _MAX_PENDING:
                self._pending.popitem(last=False)
                logger.warning_once(
                    "Mamba checkpoint retirement retry limit reached; leaving "
                    "excess checkpoints for normal Mooncake eviction"
                )
        else:
            self._pending.pop(parent, None)

    def _remove(self, keys: list[str]) -> list[str]:
        start = time.perf_counter()
        try:
            # Never force: Mooncake leases protect concurrent DMA readers.
            results = self.store.batch_remove(keys, force=False)
            if len(results) != len(keys):
                raise RuntimeError("incomplete checkpoint removal result")
        except Exception:
            self._record(
                "checkpoint_retire", start, len(keys), failed=len(keys), error=True
            )
            logger.warning("Mamba checkpoint removal failed", exc_info=True)
            return keys
        pending = [
            key
            for key, result in zip(keys, results, strict=True)
            if result not in (0, _OBJECT_NOT_FOUND)
        ]
        self._record("checkpoint_retire", start, len(keys), failed=len(pending))
        return pending

    def retry(self) -> None:
        """Retry a bounded, fair batch without looking up or renewing leases."""
        batch = [
            self._pending.popitem(last=False)
            for _ in range(min(_RETRY_BATCH_SIZE, len(self._pending)))
        ]
        if not batch:
            return
        keys = list(dict.fromkeys(key for _, keys in batch for key in keys))
        failed = set(self._remove(keys))
        for parent, keys in batch:
            remaining = [key for key in keys if key in failed]
            if remaining:
                self._pending[parent] = remaining

    def run(self) -> None:
        next_retry = time.monotonic() + _RETRY_INTERVAL
        while not self._stop_event.is_set():
            try:
                generation, commit = self._commits.get(
                    timeout=max(0, next_retry - time.monotonic())
                )
            except queue.Empty:
                pass
            else:
                try:
                    with self._maintenance_lock:
                        if (
                            generation == self._generation
                            and not self._stop_event.is_set()
                        ):
                            self.commit(commit)
                except Exception:
                    logger.exception("Mamba checkpoint retirement failed")
                finally:
                    self._commits.task_done()
            if time.monotonic() >= next_retry:
                with self._maintenance_lock:
                    if not self._stop_event.is_set():
                        self.retry()
                next_retry = time.monotonic() + _RETRY_INTERVAL
