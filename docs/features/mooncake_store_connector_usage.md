# MooncakeStoreConnector Usage Guide

MooncakeStoreConnector is a KV cache connector that uses [MooncakeDistributedStore](https://github.com/kvcache-ai/Mooncake) as a shared KV cache pool. Unlike `MooncakeConnector` which does direct point-to-point KV transfer between prefiller and decoder, MooncakeStoreConnector enables KV cache offloading to an external distributed store, supporting:

- **CPU/disk offloading**: Extend effective KV cache capacity by offloading to CPU memory or disk via Mooncake's transfer engine.
- **Prefix caching across instances**: Hash-based deduplication allows multiple vLLM instances to share cached KV blocks through the store.
- **Single-node and multi-node deployment**: Works both as a standalone KV cache extension and in disaggregated prefill-decode setups.

## Prerequisites

### Install Mooncake

Install mooncake through pip:

```bash
uv pip install mooncake-transfer-engine
```

Refer to the [Mooncake official repository](https://github.com/kvcache-ai/Mooncake) for more installation instructions and building from source.

### Start the Mooncake Master Server

The Mooncake master manages metadata and coordinates the distributed store. Start it before launching vLLM:

```bash
mooncake_master --port 50051
```

Default ports:

- RPC: 50051

Multiple vLLM instances can share the same master server.

### Configure Mooncake

Create a JSON configuration file (e.g., `mooncake_config.json`):

```json
{
  "mode": "embedded",
  "metadata_server": "P2PHANDSHAKE",
  "master_server_address": "127.0.0.1:50051",
  "global_segment_size": "80GB",
  "local_buffer_size": "4GB",
  "protocol": "rdma",
  "device_name": "",
  "enable_offload": false
}
```

- `mode`: Topology selection. `"embedded"` (default, PR-40900 baseline) has each
  vLLM rank contribute `global_segment_size` to the pool in-process.
  `"standalone-store"` makes ranks pure requesters — an external
  `mooncake_client` process owns the CPU pool and (optionally) the SSD tier.
- `protocol`: Use `"rdma"` for best performance. `"tcp"` works as a fallback.
- `global_segment_size`: CPU memory contributed to the distributed pool (per
  GPU). Must be `> 0` in `embedded` mode and `0` in `standalone-store` mode.
- `local_buffer_size`: Private buffer for this node's own operations (per GPU).
- `enable_offload`: When `true`, vLLM allocates a DirectIO staging buffer so
  large prefills do not exceed the owner's SSD-write budget. Set this together
  with the matching `--enable_offload=true` flag on `mooncake_master` and on
  the external `mooncake_client` (if any).
- `tenant_id`: Optional Mooncake tenant namespace. Producers and consumers
  that should share store data must use the same tenant id. Default:
  `"default"`.

Set the config path via environment variable:

```bash
export MOONCAKE_CONFIG_PATH=/path/to/mooncake_config.json
```

## Intermediate Mamba checkpoint retirement

For hybrid Mamba models, opt in through the MooncakeStoreConnector's
`kv_connector_extra_config`:

```json
{"retire_mamba_checkpoints": true}
```

With `MultiConnector`, put this setting on its MooncakeStoreConnector child.
It defaults to `false`; disabled connectors do no lineage RPCs and start no
retirement thread. With a rebuilt Mooncake client exposing `batch_retire`, vLLM
automatically uses master-owned retirement. Deploy the matching Mooncake master
(and all HA peers) **before** upgrading clients; a new client with an old master
will fail retirement submissions. Rebuild the vLLM Rust frontend for the new
operation metric. No additional retirement flag or shorter global lease is needed.

For a chain `C1 -> C2 -> C3`, keep C1 as the anchor and retire C2 after C3 is
saved successfully by every worker rank. Continuing to C4 retires C3. This
works across requests and prefill replicas: small lineage markers are stored
in Mooncake under a separate model/layout-scoped namespace. Missing or evicted
lineage markers conservatively retain extra anchors; master-owned retirement
does not recover that ancestry. A completed request's final checkpoint remains
subject to normal Mooncake eviction. Lease expiry makes it eligible for eviction;
it does not automatically delete the final checkpoint.

Only the intermediate checkpoint's Mamba objects are retired. Attention blocks
and GPU cache entries are unchanged. Another branch needing the old checkpoint
may have to recompute from an earlier prefix; this policy trades that reuse
for capacity. A lookup racing retirement uses the existing failed-load fallback.

In master-owned mode, `batch_retire(keys)` hides each accepted object from normal
existence and replica queries immediately. The retired flag stays on its metadata,
so evicting a marker cannot make it readable again. Neither direct queries nor
reads of attention objects in the same group can renew its lease. Already-issued
reads keep their existing lease. Replacement writes to a retired object are
rejected until it is deleted. Admin queries and inventory can still inspect it.

The master checks up to 1024 due retirement entries once per second, independent
of memory pressure, and reclaims objects after their existing leases and internal
replica users drain. With `-default_kv_lease_ttl=180000`, allow up to the remaining
three-minute read lease, plus maintenance backlog and internal task completion.
This is not immediate forced deletion. Snapshot and HA metadata retain the flag;
HA promotion conservatively grants a fresh full lease before cleanup. Retirement
acceptance uses the existing asynchronous operation-log durability semantics.

Once accepted, vLLM stops retrying deletion; it retries only failed retirement
submissions. Queue overflow, missing replacement shards, or lineage failures can
leave checkpoints to ordinary eviction. Per prefill instance there are at most
4096 queued commits and 4096 pending candidates, with at most 128 candidates
retried per second. No separate retirement marker writes or marker lookup RPCs
are needed in master-owned mode.

Older clients without `batch_retire` keep the legacy path: publish an evictable
retirement marker, filter it during lookups, and retry `batch_remove(force=False)`.
That path remains best effort: marker eviction can expose checkpoints again and
grouped attention reads can prolong their leases. Deploy consistent client and
master versions across replicas when switching modes.

Existing `vllm:mooncake_store_operation_*` metrics expose operations
`checkpoint_lineage_put` and `checkpoint_retire_exists`. Master-owned submissions
use **`checkpoint_retire_request`**: key counts are attempts, successful results
mean accepted or already absent, and `deferred` counts keys requiring another
submission. Success does **not** mean payload memory has been reclaimed; use the
retired inventory's object and byte counts to observe draining. The legacy path
keeps `checkpoint_retire`, `checkpoint_retired_put`, `lookup_retired_exists`, and
`lookup_retired_skip`. These operations do not transfer checkpoint payloads.

### Debug inventory by checkpoint lifecycle

With `VLLM_CACHE_INVENTORY_INTERVAL=10` and Gym's cache inventory collector
enabled, the Mooncake snapshot also reports:

| Category | Meaning |
| --- | --- |
| `anchor_or_unknown` | No lineage marker or master/legacy retirement indication: an anchor, an untracked save, or a checkpoint whose lineage marker was evicted. |
| `tip` | A readable lineage marker exists with no master retirement flag or legacy retirement marker. This includes intermediate checkpoints awaiting retirement publication, not just final trajectory checkpoints. |
| `retired` | At least one resident Mamba object is retired by the master, or a readable legacy retirement marker exists, including partially deleted checkpoints. |

Each category exports `complete_prefixes`, `incomplete_prefixes`,
`ready_objects`, `ready_bytes`, `pending_objects`, and `pending_bytes` under
`mooncake/total/cache_inventory/mamba/<category>/` in W&B. The category values
sum to the existing Mamba totals. Marker-only entries are excluded, and marker
bytes are not counted as Mamba payload bytes. `retired` describes resident
marked payloads, not the contents of any worker's bounded retry queue.

Classification uses master retirement flags and markers visible during the
approximate metadata scan. Marker eviction can change `anchor_or_unknown` and
`tip`, but cannot undo accepted master retirement. Concurrent writes and deletes
can change the snapshot. A `tip` does not prove that the trajectory is still
active or that its full attention prefix exists.
There are no extra per-request RPCs, payload reads, or lease renewals for this
inventory. It uses the existing opt-in scan interval and default-off behavior.

Deploy the updated vLLM namespace advertisement and rebuild/redeploy the
Mooncake **master** with lifecycle-aware inventory support. The Gym collector
forwards the new fields automatically. Master-owned retirement additionally
requires the rebuilt client with `batch_retire`, as described above. Older
masters reject the new marker namespace kinds; older vLLM workers do not
advertise the markers, so their checkpoints appear as `anchor_or_unknown`.

## Usage

### Single-Node KV Cache Offloading

Use MooncakeStoreConnector to offload KV cache to CPU memory, extending the effective cache size:

```bash
MOONCAKE_CONFIG_PATH=mooncake_config.json \
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --kv-transfer-config '{"kv_connector":"MooncakeStoreConnector","kv_role":"kv_both"}'
```

### External-only prefix caching on a prefiller

Set `external_cache_only` on the prefiller's **MooncakeStoreConnector** to use
Mooncake as the reusable prefix tier:

```json
{
    "kv_connector": "MooncakeStoreConnector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {
        "external_cache_only": true
    }
}
```

With `MultiConnector`, put this option inside the MooncakeStoreConnector child
entry and keep the P2P MooncakeConnector child configured as `kv_producer`.
Keep prefix caching enabled (the default): live block hashes and hybrid-state
snapshots are still needed for safe external storage.

Every computed prefill chunk is submitted to the Store. Existing keys are
reused; missing keys are written. New requests bypass all local prefix hits,
including blocks still held by another request. GPU blocks remain pinned until
all Store ranks finish reading them, and until their owning request and any P2P
transfer finish. When the last reference is released, local hash entries
(including partial-block aliases) are removed and the pages become reusable.
This also applies to prefixes loaded from Mooncake.

On a disaggregated prefiller, finish the request after prefill (`max_tokens=1`
in the usual P/D flow). An instance that continues decoding must keep that
request's active KV until decoding finishes. This option releases pages into
vLLM's preallocated GPU pool; it does not shrink the CUDA allocation reported by
`nvidia-smi`, or overwrite freed bytes with zeros.

Store failures, including capacity pressure, are fatal in this mode: the worker
raises an error on its next completion poll and does not release the failed
save's references. This avoids silently discarding unsaved KV. Resolve the
reported Store/transfer failure before restarting the instance. The default
mode keeps its existing best-effort save behavior.

Reuse follows the connector's existing hash/block boundaries. Uncacheable tail
tokens and the token needed for logits are recomputed. A cold, evicted, or
not-yet-written prefix is an external miss; disabling local hits cannot make
such a prefix an external hit.

To validate a deployment, submit the same long prompt with `max_tokens=1`, wait
for its Store writes and GPU usage to settle, and submit it again. Verify that
`vllm:prefix_cache_hits_total` stays flat while
`vllm:external_prefix_cache_hits_total` increases on the repeat. Check that
`vllm:kv_cache_usage_perc` returns to its idle level after transfers drain.
Repeat with concurrent requests and prompts spanning multiple prefill chunks;
compare deterministic generated tokens against the default configuration.

### Disaggregated Prefill-Decode (XpYd)

In disaggregated prefill-decode mode, use `MultiConnector` to combine `MooncakeConnector` (point-to-point KV transfer) with `MooncakeStoreConnector` (shared KV cache pool). This enables both direct P2P transfer between prefiller and decoder, and cross-instance prefix cache sharing via the distributed store.
**Prefiller Node:**

```bash
MOONCAKE_CONFIG_PATH=mooncake_config.json \
VLLM_MOONCAKE_BOOTSTRAP_PORT=50052 \
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --port 8100 \
    --kv-transfer-config '{
        "kv_connector": "MultiConnector",
        "kv_role": "kv_producer",
        "kv_connector_extra_config": {
            "connectors": [
                {
                    "kv_connector": "MooncakeConnector",
                    "kv_role": "kv_producer"
                },
                {
                    "kv_connector": "MooncakeStoreConnector",
                    "kv_role": "kv_both"
                }
            ]
        }
    }'
```

**Decoder Node:**

```bash
MOONCAKE_CONFIG_PATH=mooncake_config.json \
VLLM_MOONCAKE_BOOTSTRAP_PORT=50053 \
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --port 8200 \
    --kv-transfer-config '{
        "kv_connector": "MultiConnector",
        "kv_role": "kv_consumer",
        "kv_connector_extra_config": {
            "connectors": [
                {
                    "kv_connector": "MooncakeConnector",
                    "kv_role": "kv_consumer"
                },
                {
                    "kv_connector": "MooncakeStoreConnector",
                    "kv_role": "kv_consumer"
                }
            ]
        }
    }'
```

To also offload newly completed decode KV blocks, add the following extra
configuration to the decoder's `MooncakeStoreConnector` entry.

When decode processing starts, the consumer checks the block-aligned prompt
prefix and fills any blocks missing from the Store. Subsequent saves append
newly completed decode blocks. This keeps a complete, reusable prefix in the
Store. It also covers prompt KV delivered directly by `MooncakeConnector`.

```json
{
    "kv_connector_extra_config": {
        "save_decode_cache": true
    }
}
```

#### Sharing one Store across multiple Prefill TP sizes

Heterogeneous-TP sharing normally uses a fixed `store_tp_size`. When several
prefillers use different TP sizes, opt in to a common Store TP derived from
their least common multiple:

```json
{
    "kv_connector_extra_config": {
        "enable_store_tp_lcm": true,
        "prefill_tp_sizes": [4, 2]
    }
}
```

Every prefiller and decoder that shares these entries must use the same list.
The example selects Store TP 4: a TP4 endpoint maps each rank to one Store
shard, while TP2 endpoints map each rank to two Store shards. Runtime TP sizes
remain unchanged. A decoder configured with `"save_decode_cache": true` uses
the same Store TP for decode KV from every prefiller.

The list may contain positive integer TP sizes. Sharing requires a Store TP that
is at least the local TP and divisible by it, an LBHNC or LBNHC local KV cache,
and the existing topology and KV-head constraints. The Store namespace includes
the attention backend's selected layout. Different layouts use separate Store
entries. Malformed lists and unsupported endpoints use an isolated rank-local
key layout. When
`enable_store_tp_lcm` is absent or false, `prefill_tp_sizes` has no effect and
the existing `store_tp_size` behavior is unchanged.

**Proxy:**

A disaggregation proxy routes requests between prefiller and decoder nodes.
When `MooncakeConnector` is also used for direct P2P transfer, refer to its
[usage guide](mooncake_connector_usage.md) for proxy setup details.

### Disk Offloading

Disk offloading is most commonly run in `standalone-store` mode: an external
`mooncake_client` process owns the CPU pool and the SSD tier, and each vLLM
rank is a pure requester. This avoids per-rank duplication of the SSD pool
and keeps DirectIO budget tracking on a single process.

Three things need to be aligned for end-to-end disk offloading:

1. **`mooncake_master`** is started with `--enable_offload=true`.
2. **`mooncake_client`** (the owner) is started with `--enable_offload=true`
   plus an SSD path via `MOONCAKE_OFFLOAD_FILE_STORAGE_PATH`.
3. **vLLM-side** sets `"enable_offload": true` in the JSON config file (this is
   read by the connector and is **not** an environment variable).

Example `mooncake_config.json` for the vLLM side:

```json
{
  "mode": "standalone-store",
  "metadata_server": "P2PHANDSHAKE",
  "master_server_address": "127.0.0.1:50051",
  "global_segment_size": 0,
  "local_buffer_size": "4GB",
  "protocol": "rdma",
  "device_name": "mlx5_0",
  "enable_offload": true
}
```

Steer this rank to the local owner segment with:

```bash
export MOONCAKE_PREFERRED_SEGMENT=127.0.0.1:50053
```

The owner's SSD directory, on-disk eviction policy, and the DirectIO staging
buffer size are controlled on the `mooncake_client` side via the standard
Mooncake environment variables (`MOONCAKE_OFFLOAD_FILE_STORAGE_PATH`,
`MOONCAKE_BUCKET_EVICTION_POLICY`, `MOONCAKE_USE_URING`,
`MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES`,
`MOONCAKE_OFFLOAD_TOTAL_SIZE_LIMIT_BYTES`, etc.). Those are independent of
the vLLM JSON config.

### Tenant Isolation

Set `tenant_id` in the Mooncake JSON config when different vLLM deployments should use separate Mooncake tenant namespaces:

```json
{
  "mode": "embedded",
  "metadata_server": "P2PHANDSHAKE",
  "master_server_address": "127.0.0.1:50051",
  "global_segment_size": "80GB",
  "local_buffer_size": "4GB",
  "protocol": "rdma",
  "device_name": "",
  "enable_offload": false,
  "tenant_id": "tenant-a"
}
```

Strict isolation requires a Mooncake master started with `--enable_multi_tenants=true` and a tenant quota policy that registers each tenant. Non-default `tenant_id` also requires a Mooncake version whose `MooncakeDistributedStore.setup()` accepts the `tenant_id` parameter. In `standalone-store` mode, start the external `mooncake_client` with the matching tenant id because that process owns the real store client.

## Environment Variables

| Variable | Description | Default |
| --- | --- | --- |
| `MOONCAKE_CONFIG_PATH` | Path to Mooncake JSON config file | (required) |
| `VLLM_MOONCAKE_BOOTSTRAP_PORT` | Bootstrap port for MooncakeConnector P2P transfer (disagg mode only) | 8998 |
| `MOONCAKE_PREFERRED_SEGMENT` | Pin this rank's replicas to a specific owner segment (`host:port`); used in `standalone-store` mode | — |
| `MOONCAKE_REQUESTER_LOCAL_HOSTNAME` | Override the hostname the vLLM rank registers with Mooncake as a requester. Defaults to the rank's resolved IP. | — |
| `VLLM_MOONCAKE_STORE_TIER_LOG` | When `1`, logs a per-batch tier summary (memory vs disk hits) for observability | disabled |
| `VLLM_MOONCAKE_DISK_STAGING_USABLE_RATIO` | Fraction of the owner's DirectIO staging buffer that the requester will fill in a single `batch_get_into_multi_buffers` call. Lower → more conservative pre-split, more round trips. | 0.9 |

## KV Transfer Config

### KV Role Options

- **kv_producer**: For instances that store KV caches to the pool.
- **kv_consumer**: For instances that load KV caches from the pool.
- **kv_both**: The instance both stores and loads KV caches. Use this for single-node CPU offloading or prefiller instances.

### kv_connector_extra_config

- `retire_mamba_checkpoints` (bool): Retire intermediate Mamba states after their replacements are saved by all ranks, keeping anchors and tips. Default: `false`. See [checkpoint retirement](#intermediate-mamba-checkpoint-retirement).
- `external_cache_only` (bool): Disable local prefix reuse and evict idle GPU cache entries after Store/request/P2P references drain. Requires `kv_role="kv_both"` and lookup enabled. Store write failures are fatal in this mode. Default: `false`. See [external-only prefix caching](#external-only-prefix-caching-on-a-prefiller).
- `load_async` (bool): Enable asynchronous loading for better compute-I/O overlap. Default: `true`.
- `lookup_async` (bool): Run the external prefix-cache lookup on a background thread so it never blocks the scheduler step. The request is held until the in-flight lookup completes, then resumed on a later step. Default: `false`.
- `lookup_rpc_port` (int): Custom port for the ZMQ lookup RPC socket. Default: `0`.
- `cache_prefix` (str): Namespace prepended to every store key. Lets separate deployments share one Mooncake master without polluting each other — instances configured with different prefixes never see each other's cached blocks, even for identical prompts. All instances that should share a prefix cache must use the same value. Default: `""` (no prefix; keys are byte-identical to the unprefixed format).
- `save_decode_cache` (bool): Enable offloading decode tokens' KV cache. A `kv_consumer` does not save during prefill; when decode starts, it fills any missing block-aligned prompt prefix before appending completed decode blocks. Default: `false`.
- `store_tp_size` (int): Common Store TP for endpoints with different local TP sizes. It supports LBHNC and LBNHC local KV caches, with `store_tp_size >= local_tp_size` and `store_tp_size % local_tp_size == 0`. The current topology is one full-attention cache group, PCP/DCP disabled, and cross-layer blocks disabled. For GQA and MHA, the total KV-head count must be divisible by `store_tp_size`. Store shards contain fixed global KV-head ranges in the local layout. Shared endpoints use the same KV cache layout, pipeline-parallel size, and Store TP. The Store namespace includes the layout and PP size. Unsupported configurations use a topology-specific rank-local namespace.

LBHNC/HND is strongly recommended for TP-sharded Store when supported. LBNHC/NHD
creates many transfer segments and may significantly reduce PUT/GET performance.

For example, with prefill TP 4, decode TP 2, and eight KV heads, set
`store_tp_size` to 4 on both instances. Each decode rank reads and writes two
of the four Store shards.

MQA with one total KV head uses a replicated-head layout. For the supported
prefill TP 4 to decode TP 2 case, every rank uses the
same rank-0 key namespace. The four prefill replicas stripe block PUTs so each
object is stored once, while both decode ranks GET every block into their local
KV replica. `store_tp_size` does not appear in MQA keys, so identical MQA
objects written at different store TP sizes share the same pool entry when PP
sizes match.

Tensor-parallel collectives and low-precision arithmetic are not bitwise
invariant across TP sizes, so heterogeneous-TP reuse does not guarantee the
same greedy output as recomputing the prefix at the decode TP size.

## Diagnosing lookup and load queues

Both the Python and Rust frontends expose pipeline metrics on `/metrics` when
engine statistics are enabled. No extra Mooncake flag is required. The Rust
frontend must be rebuilt together with these Python changes.

All `vllm:mooncake_pipeline_*` metrics have `stage` (`lookup` or `load`) and
`rank` labels in addition to the normal model and engine labels. Lookup uses
`rank="-1"` and counts work once per engine. Load uses the worker's global rank:
a logical TP4 load can produce four worker jobs. Do not sum worker jobs and
interpret that sum as distinct requests.

| Metric | Meaning |
| --- | --- |
| `vllm:mooncake_pipeline_queued` | Jobs submitted but not yet executing. |
| `vllm:mooncake_pipeline_active` | Jobs currently executing. |
| `vllm:mooncake_pipeline_oldest_pending_seconds` | Age since enqueue of the oldest queued or active job; zero when idle. |
| `vllm:mooncake_pipeline_queue_wait_seconds` | Histogram from enqueue to execution start. |
| `vllm:mooncake_pipeline_execution_seconds` | Histogram from execution start to return, including failed executions. |
| `vllm:mooncake_pipeline_consumption_wait_seconds` | Histogram from successful lookup completion to the scheduler consuming its result (`stage="lookup"`, `rank="-1"` only). |

Lookup execution includes serialization, ZMQ round-trip, and the lookup server's
work; it is not a pure Mooncake RPC timer. Load execution covers the receive
handler, including preparation and any split read batches. Existing
`vllm:mooncake_store_operation_time_seconds` measurements (`lookup_exists` and
`load_get`) cover the narrower store operations. Cancelled jobs that never start
are removed from backlog counts without recording an execution sample.

Consumption wait is recorded once when the scheduler retrieves a successful
lookup result, including a cache miss. It excludes queue wait and execution,
and does not measure subsequent GPU admission delay. Failed, cancelled, or
aborted lookups do not contribute a consumption sample. Completed lookups still
awaiting consumption have not contributed a sample yet; they are also absent
from the queued and active gauges. A large consumption wait explains how
`external_cache_lookup` deferrals can remain high after lookup work finishes.

Gauges are snapshots delivered with engine statistics, not an independent
heartbeat. An entirely stalled engine can leave its last snapshot unchanged.
Histograms contain observations only when the measured interval ends; use live
backlog and oldest age to investigate operations still in progress.

`vllm:num_requests_deferred_by_reason` counts logical requests once per engine,
with reasons `external_cache_lookup`, `remote_kv_load`, `grammar`,
`streaming_input`, `lora_limit`, `encoder_cache_load`, and `other`. It reports the
last observed scheduler dependency; `remote_kv_load` includes any KV connector,
not only Mooncake. The existing waiting-by-reason metric remains unchanged.

For example, compare pending lookup and load work:

```promql
sum by (engine, stage, rank) (vllm:mooncake_pipeline_queued)
```

And inspect p95 queue time without mixing engines or worker ranks:

```promql
histogram_quantile(0.95,
  sum by (le, engine, stage, rank) (
    rate(vllm:mooncake_pipeline_queue_wait_seconds_bucket[5m])
  )
)
```

## Notes

### Reproducible Block Hashes Across Processes

The `MooncakeStoreConnector` relies on consistent block hashes across all vLLM processes sharing the distributed store. Block hashes chain from `NONE_HASH`, which is derived from a fixed default seed, so identical prompts produce identical block hashes across processes by default — enabling cross-process prefix cache hits without extra configuration.

The exception is the non-cryptographic `xxhash`/`xxhash_cbor` values of `--prefix-caching-hash-algo`, which seed `NONE_HASH` randomly per process; sharing a store with those requires `PYTHONHASHSEED`.

To use a custom shared seed, set the same `PYTHONHASHSEED` on every instance that shares the store (DP ranks, separate prefiller/decoder nodes, and any other vLLM process pointed at the same Mooncake store):

```bash
PYTHONHASHSEED=<shared-value> vllm serve ...
```
