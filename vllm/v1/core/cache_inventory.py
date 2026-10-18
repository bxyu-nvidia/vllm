# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU metadata inventory. Counts group payload bytes, not CUDA allocations."""

import time
from collections.abc import Collection
from typing import TYPE_CHECKING

from vllm.v1.core.kv_cache_utils import (
    BlockHashWithGroupId,
    get_block_hash,
    get_group_id,
)
from vllm.v1.kv_cache_interface import MambaSpec, UniformTypeKVCacheSpecs

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_manager import KVCacheManager


def collect_cache_inventory(manager: "KVCacheManager") -> dict[str, float]:
    """Sample distinct physical blocks; shared request refs and hash aliases count once.

    Called on the scheduler thread. Unhashed blocks held only by internal work
    are reported as unattributed, never guessed to be attention or Mamba.
    Host-resident groups are excluded from the GPU inventory. Snapshots are
    sampled on scheduler stats ticks; timestamps can stay unchanged while idle.
    """
    started = time.monotonic()
    pool = manager.block_pool
    owners: dict[int, int] = {}
    active_requests: set[str] = set()
    groups = manager.kv_cache_config.kv_cache_groups
    for owner_gid, mgr in enumerate(manager.coordinator.single_type_managers):
        if groups[owner_gid].host_resident:
            continue
        for req_id, blocks in mgr.req_to_blocks.items():
            active_requests.add(req_id)
            for block in blocks:
                if not block.is_null:
                    owners[block.block_id] = owner_gid
    result: dict[str, float] = {
        "pool.active_requests": len(active_requests),
        "pool.reclaimable_blocks": pool.get_num_free_blocks(),
        "pool.unattributed_in_use_blocks": 0,
        "pool.unused_blocks": 0,
        "pool.snapshot_timestamp_seconds": time.time(),
    }
    kinds = []
    block_bytes = []
    prefixes: dict[str, set[bytes]] = {"attention": set(), "mamba": set()}
    for group in groups:
        spec = group.kv_cache_spec
        layers = (
            list(spec.kv_cache_specs.values())
            if isinstance(spec, UniformTypeKVCacheSpecs)
            else [spec]
        )
        kind = "mamba" if isinstance(layers[0], MambaSpec) else "attention"
        kinds.append(kind)
        block_bytes.append(
            spec.page_size_bytes
            if isinstance(spec, UniformTypeKVCacheSpecs)
            else spec.page_size_bytes * len(group.layer_names)
        )
        for field in (
            "in_use_blocks",
            "in_use_bytes",
            "idle_cached_blocks",
            "idle_cached_bytes",
            "cached_blocks",
            "distinct_prefixes",
        ):
            result[f"{kind}.{field}"] = 0
    for block in pool.blocks:
        if block.is_null:
            continue
        aliases = pool.cached_block_hashes_by_block.get(block.block_id)
        if not block.ref_cnt and block.block_hash is None and not aliases:
            result["pool.unused_blocks"] += 1
            continue
        # Reuse the alias set: a snapshot must not copy a set for every block.
        hashes: Collection[BlockHashWithGroupId] = aliases or ()
        if block.block_hash is not None and block.block_hash not in hashes:
            hashes = (*hashes, block.block_hash)
        gid = owners.get(block.block_id)
        if gid is None and hashes:
            gid = get_group_id(next(iter(hashes)))
        if gid is None:
            result[
                "pool.unattributed_in_use_blocks"
                if block.ref_cnt
                else "pool.unused_blocks"
            ] += 1
            continue
        kind = kinds[gid]
        state = "in_use" if block.ref_cnt else "idle_cached"
        result[f"{kind}.{state}_blocks"] += 1
        result[f"{kind}.{state}_bytes"] += block_bytes[gid]
        if hashes:
            result[f"{kind}.cached_blocks"] += 1
            prefixes[kind].update(get_block_hash(h) for h in hashes)
    for kind in set(kinds):
        result[f"{kind}.distinct_prefixes"] = len(prefixes[kind])
    result["pool.scan_seconds"] = time.monotonic() - started
    return result
