# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared expert-route blocks backed by the KV connector's Mooncake client."""

from collections.abc import Iterable, Sequence
from threading import RLock
from typing import Any

from vllm.distributed.aux_output_connector.routed_experts import routed_experts_keys
from vllm.distributed.aux_output_connector.store import (
    BlockObject,
    BlockObjectStore,
    BlockObjectStoreError,
)
from vllm.logger import init_logger

logger = init_logger(__name__)


class MooncakeBlockObjectStore(BlockObjectStore):
    """Local LRU with shared publication and request-pinned KV-load prefetches.

    The Mooncake client belongs to the KV connector. Missing or malformed route
    blocks fail the KV load before it can become a reusable prefix.
    """

    def __init__(
        self,
        *,
        client: Any,
        replicate_config: Any,
        namespace: str,
        max_bytes: int,
        object_nbytes: int,
    ) -> None:
        super().__init__(max_bytes=max_bytes, object_nbytes=object_nbytes)
        self.client = client
        self.replicate_config = replicate_config
        self.namespace = namespace
        self._lock = RLock()
        self._prefetches: dict[str, tuple[str, ...]] = {}

    def keys(self, hashes: Sequence[bytes]) -> list[str]:
        # Local prefix resets can differ between P/D engines. Shared routes
        # follow the shared KV namespace, not an engine-local reset counter.
        return routed_experts_keys(hashes, f"{self.namespace}/0")

    def put(
        self,
        objects: list[BlockObject],
        *,
        retain_keys: Iterable[str] = (),
        release_keys: Iterable[str] = (),
    ) -> None:
        with self._lock:
            super().put(objects, retain_keys=retain_keys, release_keys=release_keys)
        if not objects:
            return
        # Publication runs on the aux writer thread, never the scheduler.
        # A failed cache save must not invalidate routes already captured locally.
        try:
            status = self.client.put_batch(
                [obj.key for obj in objects],
                [obj.payload for obj in objects],
                self.replicate_config,
            )
            if status != 0:
                raise BlockObjectStoreError(f"Mooncake put_batch returned {status}")
        except Exception:
            logger.warning(
                "Failed to publish expert-route blocks; remote KV hits "
                "without routes will be recomputed",
                exc_info=True,
            )

    def _load(self, keys: list[str]) -> None:
        missing = list(dict.fromkeys(key for key in keys if key not in self._lru))
        if not missing:
            return
        values = self.client.get_batch(missing)
        if len(values) != len(missing) or any(
            not isinstance(value, bytes) or len(value) != self.object_nbytes
            for value in values
        ):
            raise BlockObjectStoreError(
                "Mooncake expert-route blocks are missing or have an invalid size"
            )
        super().put(
            [
                BlockObject(key, value)
                for key, value in zip(missing, values, strict=True)
            ]
        )

    def get_concatenated(self, keys: list[str]) -> bytes:
        with self._lock:
            self._retain(keys)
            try:
                self._load(keys)
                return super().get_concatenated(keys)
            finally:
                self._release(keys)

    def reserve_prefetch(self, request_id: str) -> None:
        with self._lock:
            self.release_prefetches([request_id])
            self._prefetches[request_id] = ()

    def prefetch(self, request_id: str, keys: list[str]) -> None:
        with self._lock:
            if request_id not in self._prefetches:
                raise BlockObjectStoreError("Expert-route prefetch was cancelled")
            # Pin before loading so insertion cannot evict another required block.
            self._retain(keys)
            try:
                self._load(keys)
            except Exception:
                self._release(keys)
                raise
            self._release(self._prefetches[request_id])
            self._prefetches[request_id] = tuple(keys)

    def release_prefetches(self, request_ids: Iterable[str]) -> None:
        with self._lock:
            for request_id in request_ids:
                self._release(self._prefetches.pop(request_id, ()))

    def clear(self) -> None:
        """Discard local copies after draining publishers and KV receivers."""
        with self._lock:
            self.release_prefetches(list(self._prefetches))
            super().clear()

    def close(self) -> None:
        with self._lock:
            self.release_prefetches(list(self._prefetches))
            super().close()
