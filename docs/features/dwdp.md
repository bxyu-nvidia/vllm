# Distributed weight data parallelism (experimental)

DWDP keeps each worker's requests, activations and computation independent,
while storing routed expert weights across peer GPUs. Each worker executes all
experts for its own requests. Dense weights, routers, latent projections and
separate shared experts remain replicated. See the
[DWDP paper](https://arxiv.org/abs/2604.01621).

The current implementation provides an experimental BF16 implementation for GPUs on one
host. It gathers all experts for the next MoE layer using CUDA IPC peer mappings
and a copy stream. Two reusable full-layer buffers have copy-ready and
compute-consumed events to prevent reading incomplete copies or overwriting
weights still in use. Startup exchanges metadata through a dedicated TCP store;
inference does not perform collectives between weight-sharing workers.

## Launch independent workers

Start one TP1/PP1/DP1 engine per GPU, with a distinct API port and weight-sharing
rank. All workers must use the same checkpoint and layer configuration, and
must start together. For example, run these processes concurrently, substituting
the model path:

```bash
for rank in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$rank vllm serve /path/to/model \
        --port "$((8000 + rank))" \
        --enforce-eager \
        --kernel-config '{"moe_backend":"flashinfer_prims"}' \
        --dwdp-config "{\"rank\":$rank,\"world_size\":4,\"port\":29590}" \
        >"dwdp-$rank.log" 2>&1 &
done
wait
```

`flashinfer_prims` requires the local FlashInfer Prims BF16 APIs and SM100 GPUs.
It uses vLLM's router and passes selected experts to Prims grouped GEMMs.
`triton` can be selected to check the transport independently of Prims.

Route prefill requests to the independent engines. Configure decode engines
separately without DWDP. This prototype does not switch weight ownership based
on individual batches; every forward of a configured engine uses DWDP.

For Nemotron MTP, add the following to each engine:

```bash
--speculative-config '{"method":"mtp","num_speculative_tokens":5}'
```

With NIXL
prefill/decode disaggregation, configure the same speculative width on both
sides. Prefill can use `"num_speculative_tokens_per_batch_size":[[1,1024,0]]`
to disable draft proposals while retaining the matching cache configuration.
Decode engines without DWDP can retain their existing CUDA graph settings.
This integration still requires GPU correctness and end-to-end validation.

## Constraints and lifecycle

- Requires unquantized BF16 experts without bias, at least one owned expert per
  worker and separate shared experts. TP, EP, DP, context
  parallelism, DBO, EPLB and LoRA are currently rejected.
- MTP speculative decoding is supported with `--enforce-eager`. Only target
  expert weights use DWDP; the drafter, including its MoE experts, is replicated
  on each GPU. Budget for the additional draft weights and speculative KV/state
  cache. Other speculative methods and DWDP+MTP CUDA graphs are rejected.
- Owned experts occupy balanced contiguous ranges. Counts need not divide the
  group size. Ownership affects loading only; execution sees all expert IDs.
- CUDA IPC handles and source allocations are process-specific and must stay
  alive for the weight-sharing group's lifetime. Restart the whole group after
  a worker exits or fails. Linux pidfds detect exited owners without forward
  collectives, including idle owners. Workers must share a PID namespace.
  This is not a multi-node NVLink fabric transport, and an in-flight CUDA error
  after owner failure requires restarting the group.
- Source weights are immutable after startup. Weight reloads and preprocessed
  IPC weight-cache loading are unsupported. Sleep mode and weight-transfer
  configurations are rejected because they can replace or unmap live sources.
- Graceful teardown waits for all workers to release peer mappings before
  releasing sources. Startup and shutdown waits are bounded by
  `timeout_seconds` (default 600).
- Set `"merge_free":true` with `flashinfer_prims` to read resident experts
  directly through a separate TMA descriptor, eliminating local merge copies.
  Staging still reserves their unused regions and costs two complete MoE layers.
- Full-model Prims validation has shown rank-dependent logits and occasional
  prediction differences despite exact layer-level parity. Validate accuracy
  for the chosen model and graph settings before using this experimental backend.
- Eager execution is the default validation path. `"allow_cuda_graph":true`
  opts into experimental graph capture. Captured copies use individual
  `cudaMemcpyAsync` calls because the batch-copy driver operation rejects
  capture on the tested driver. Forward execution must retain model layer order;
  concurrent forwards and weight mutation are unsupported.

The `DWDPWeightManager` exposes cumulative `remote_bytes`, `local_copy_bytes`,
`resident_bytes` and `staging_bytes`. Transfer counters count host enqueues,
including graph capture, and exclude graph replays. They are diagnostic counters,
not serving metrics. KV-cache profiling includes the allocated staging buffers.
`remote_bytes_per_forward` and `local_copy_bytes_per_forward` report the complete
layer schedule's traffic. `forward_enqueues` and `token_enqueues` count eager
forwards, including warmup, and exclude capture and graph replay.

Set `"collect_metrics":true` to enable per-layer prefetch, exposed weight wait
and MoE timing events. `manager.metrics()` returns completed local timing
snapshots without synchronization. Timing diagnostics require eager execution.
These include eager dispatch gaps; they are
not running averages or exported Prometheus metrics.

Set `"collect_expert_usage":true` in `--dwdp-config` to log the number of distinct
local and remote experts selected by each batch for each target MoE layer:

```text
DWDP expert usage rank=0 forward=12 layer=... tokens=16960 remote=380/384 local=128/128
```

The count is the union of valid routed expert IDs across tokens and top-k slots,
not the number of token/expert assignments. `remote=380/384` means that only four
remote experts could have been omitted from this layer's weight transfer.
Ownership uses the DWDP rank's contiguous expert range, including uneven shards.
Padding IDs outside `[0, num_experts)` are ignored. `tokens` is the number of
rows in the routing tensor; `forward` is the manager's forward enqueue counter.
Warmup/dummy forwards are included; this does not measure completed requests.

Collection defaults to off and is independent of `collect_metrics`. It uses the
existing routing decisions without changing them, but synchronizes routing to
CPU and logs every layer, so **disable it for latency/throughput profiling**.
It supports Prims/Triton externally routed MoE and compilation with CUDA graphs
disabled; monolithic routing and `allow_cuda_graph=true` are rejected.
`manager.metrics()["expert_usage"]` returns the latest snapshot for each layer
(including forward counter, token count and used/available local/remote experts).
It does not keep unbounded per-batch history or export Prometheus metrics.

For an offline `LLM` engine, retrieve these diagnostics through the named worker
RPC:

```python
metrics = llm.collective_rpc("get_dwdp_metrics")
```

This RPC addresses only that engine's workers; it does not synchronize the
independent weight-sharing group.

## Validate and measure

```bash
.venv/bin/python -m pytest tests/model_executor/offloader/test_dwdp.py -v

.venv/bin/python -m benchmarks.kernels.benchmark_dwdp \
    --world-size 4 --output /tmp/dwdp-transfers.json

.venv/bin/python -m benchmarks.kernels.benchmark_dwdp \
    --world-size 4 --compute --tokens 1024 16960 33920 \
    --output /tmp/dwdp-overlap.json
```

The benchmark uses synthetic activations and expert weights with the target
model's packed dimensions. Its compute measurements use precomputed routing.
By default, compute kernels replay from per-layer CUDA graphs while transport
is enqueued eagerly. Use `--eager-compute` to include Python compute dispatch.
Use `--capture-transport` to measure the full captured pipeline, including its
different copy implementation. Compare both paths before choosing graph settings.
It compares compute alone, synchronous fetch, and overlapped fetch, and reports
the median of per-sample maximum rank latencies. Use `--active-ranks 1` to
measure a single puller, `--slice-bytes` to sweep DMA request sizes, or
`"prefetch":false` in the serving DWDP config for a synchronous baseline.
Sliced pulls follow the paper's offset-first, peer-round-robin copy plan and
are submitted through vLLM's batch-copy operation. CUDA may reorder copies
within a batch; this is not a strict source-side fairness guarantee.

Layer parity and transport throughput do not establish end-to-end accuracy or
benefit. Before deployment, compare model logits and evaluations, chunked
prefill, prefix-cache hits and Mamba state handoff, then measure TTFT,
tokens/second/GPU and KV capacity against EP at equal hardware and load.
