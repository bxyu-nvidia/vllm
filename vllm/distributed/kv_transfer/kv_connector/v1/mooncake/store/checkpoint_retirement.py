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
CHECKPOINT_ANCHOR = 1
CHECKPOINT_LINEAGE = 2
CHECKPOINT_RETIRED = 4
CHECKPOINT_READY = 8


def supports_checkpoint_metadata(store: object) -> bool:
    """Detect the optional pybind API without probing or renewing any leases."""
    return callable(getattr(type(store), "batch_checkpoint_metadata", None))


def _layout_digest(lookup_key_prefixes: Sequence[Sequence[str]]) -> str:
    layout = "\n".join(sorted({p for group in lookup_key_prefixes for p in group}))
    return hashlib.sha256(layout.encode()).hexdigest()


def anchor_checkpoint_prefix(lookup_key_prefixes: Sequence[Sequence[str]]) -> str:
    """Namespace for observed cache-chain roots, not benchmark sample IDs."""
    return f"vllm-mamba-anchor-v1@{_layout_digest(lookup_key_prefixes)}"


def lineage_checkpoint_prefix(lookup_key_prefixes: Sequence[Sequence[str]]) -> str:
    """Return the namespace used to identify derived checkpoints."""
    return f"vllm-mamba-lineage-v1@{_layout_digest(lookup_key_prefixes)}"


def retired_checkpoint_prefix(lookup_key_prefixes: Sequence[Sequence[str]]) -> str:
    """Return the shared retirement namespace, independent of DP instance IDs."""
    return f"vllm-mamba-retired-v1@{_layout_digest(lookup_key_prefixes)}"


class MambaCheckpointRetirer(threading.Thread):
    """Keep roots and tips, deleting intermediate Mamba keys after save commits.

    Only the elected worker receives commits, after every rank reports a
    successful save. New clients attach provenance to the checkpoint payload;
    older clients use separate markers. Missing evidence conservatively retains
    the checkpoint. Neither mode changes Mooncake read visibility or leases.
    """

    def __init__(
        self,
        store: Any,
        lookup_key_prefixes: Sequence[Sequence[str]],
        mamba_group_ids: Sequence[int] | set[int],
        *,
        record_operation: Callable[..., None] | None = None,
        publish_anchors: bool = False,
        use_payload_metadata: bool = False,
    ) -> None:
        super().__init__(daemon=True, name="MambaCheckpointRetirer")
        self.store = store
        self._publish_anchors = publish_anchors
        self._use_payload_metadata = use_payload_metadata
        self._anchor_prefix = anchor_checkpoint_prefix(lookup_key_prefixes)
        self._record_operation = record_operation
        self._mamba_prefixes = tuple(
            dict.fromkeys(
                prefix
                for gid in sorted(mamba_group_ids)
                for prefix in lookup_key_prefixes[gid]
            )
        )
        # Include the full layout/model/cache namespace, never DP instance IDs.
        self._marker_prefix = lineage_checkpoint_prefix(lookup_key_prefixes)
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
        if self._stop_event.is_set() or (
            commit.parent_hash is None and not self._publish_anchors
        ):
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
        if parent == child or not self._mamba_prefixes:
            return
        if self._use_payload_metadata:
            self._commit_payload_metadata(commit)
            return
        if parent is None:
            if self._publish_anchors:
                self._publish_anchor(child)
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
        self._remove_parent(parent)

    def _remove_parent(self, parent: str) -> None:
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

    def _checkpoint_metadata(
        self, keys: list[str], flags: int, operation: str
    ) -> list[int] | None:
        start = time.perf_counter()
        try:
            results = self.store.batch_checkpoint_metadata(keys, flags)
            if len(results) != len(keys) or any(
                value != _OBJECT_NOT_FOUND and not 0 <= value <= 15 for value in results
            ):
                raise RuntimeError("invalid checkpoint metadata result")
        except Exception:
            self._record(operation, start, len(keys), failed=len(keys), error=True)
            logger.warning("Mamba checkpoint metadata operation failed", exc_info=True)
            return None
        self._record(
            operation,
            start,
            len(keys),
            failed=sum(value == _OBJECT_NOT_FOUND for value in results),
        )
        return results

    def _commit_payload_metadata(self, commit: MambaCheckpointCommit) -> None:
        parent = commit.parent_hash
        child_keys = self._keys(commit.checkpoint_hash)
        if parent is None:
            if self._publish_anchors:
                self._checkpoint_metadata(
                    child_keys, CHECKPOINT_ANCHOR, "checkpoint_anchor_put"
                )
            return
        # Attach lineage only to completed, resident payloads. This does not
        # allocate marker objects or refresh leases. Missing children cannot
        # justify retiring their parent.
        child = self._checkpoint_metadata(
            child_keys, CHECKPOINT_LINEAGE, "checkpoint_lineage_put"
        )
        if child is None or any(
            value < 0 or not value & CHECKPOINT_READY for value in child
        ):
            return
        parent_keys = self._keys(parent)
        previous = self._checkpoint_metadata(parent_keys, 0, "checkpoint_retire_exists")
        if previous is None or not any(
            value >= 0 and value & CHECKPOINT_LINEAGE for value in previous
        ):
            return
        # One surviving shard proves this hash has a predecessor. Publish on
        # every remaining shard before removal so other DP workers skip it.
        retired = self._checkpoint_metadata(
            parent_keys, CHECKPOINT_RETIRED, "checkpoint_retired_put"
        )
        if retired is None:
            return
        self._remove_parent(parent)

    def _publish_anchor(self, checkpoint_hash: str) -> None:
        # Commits arrive only after every rank acknowledges the save. Do not
        # probe payloads for this diagnostic: that would renew their leases.
        # If payloads were evicted meanwhile, inventory ignores orphan markers.
        start = time.perf_counter()
        try:
            result = self.store.put(f"{self._anchor_prefix}@{checkpoint_hash}", b"1")
            if result != 0:
                raise RuntimeError(f"anchor PUT returned {result}")
        except Exception:
            self._record("checkpoint_anchor_put", start, 1, failed=1, error=True)
            logger.warning("Mamba checkpoint anchor marker write failed", exc_info=True)
            return
        self._record("checkpoint_anchor_put", start, 1)

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
