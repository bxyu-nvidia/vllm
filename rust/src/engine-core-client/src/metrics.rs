// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::time::{SystemTime, UNIX_EPOCH};

use vllm_metrics::{
    EngineLabels, EnginePositionLabels, F64Gauge, Family, HistogramMetric, LoraAdapterNames,
    LoraInfoLabels, MooncakeOperationCounterFamily, MooncakeOperationHistogramFamily,
    MooncakeOperationLabels, RequestMetrics, SchedulerLogStatsAccumulator, SchedulerMetrics,
    U64Counter, U64Gauge, WaitingReasonLabels,
};

use vllm_metrics::{
    CacheInventoryLabels, DEFERRED_REASONS, MooncakeCacheLayoutLabels,
    MooncakePipelineHistogramFamily, MooncakePipelineLabels,
};

use crate::protocol::stats::{
    KvConnectorStats, MooncakeStats, MultiConnectorStats, NixlStats, SchedulerStats,
};
use crate::transport::ConnectedEngine;

const WAITING_REASON_CAPACITY: &str = "capacity";
const WAITING_REASON_DEFERRED: &str = "deferred";

/// Cached output-batch metric handles for one model and engine index.
pub(crate) struct IterationMetricHandles {
    pub iteration_tokens_total: HistogramMetric,
}

impl IterationMetricHandles {
    pub(crate) fn new(metrics: &RequestMetrics, model_name: &str, engine: u32) -> Self {
        let labels = EngineLabels {
            model_name: model_name.to_string(),
            engine,
        };
        Self {
            iteration_tokens_total: metrics.iteration_tokens_total.get_or_create_owned(&labels),
        }
    }
}

/// Cached scheduler-stats metric handles for all engines connected to one
/// frontend client.
pub(crate) struct SchedulerStatsRecorder {
    engines: BTreeMap<u32, SchedulerStatsHandles>,
}

/// Per-engine cached metric handles used while recording `SchedulerStats`.
struct SchedulerStatsHandles {
    // Base labels reused for dynamic child labels.
    labels: EngineLabels,

    // Scheduler state gauges.
    scheduler_running: U64Gauge,
    scheduler_waiting: U64Gauge,
    scheduler_waiting_capacity: U64Gauge,
    scheduler_waiting_deferred: U64Gauge,
    scheduler_deferred_by_reason: Family<WaitingReasonLabels, U64Gauge>,
    kv_cache_usage: F64Gauge,

    // Prefix-cache counters, including the connector-backed external cache path.
    prefix_cache_queries: U64Counter,
    prefix_cache_hits: U64Counter,
    external_prefix_cache_queries: U64Counter,
    external_prefix_cache_hits: U64Counter,

    // Speculative decoding counters.
    spec_decode_num_drafts: U64Counter,
    spec_decode_num_draft_tokens: U64Counter,
    spec_decode_num_accepted_tokens: U64Counter,
    spec_decode_num_accepted_tokens_per_pos: Family<EnginePositionLabels, U64Counter>,

    // Per-engine performance / MFU counters.
    estimated_flops_per_gpu: U64Counter,
    estimated_read_bytes_per_gpu: U64Counter,
    estimated_write_bytes_per_gpu: U64Counter,

    // Sampled KV-cache residency histograms.
    kv_block_lifetime_seconds: HistogramMetric,
    kv_block_idle_before_evict_seconds: HistogramMetric,
    kv_block_reuse_gap_seconds: HistogramMetric,

    // Mooncake store connector telemetry, decoded from `kv_connector_stats`.
    // Kept as `Family` (not pre-resolved) because `operation`/`status` are
    // dynamic per-record labels.
    mooncake_operation_time_seconds: MooncakeOperationHistogramFamily,
    mooncake_operation_total: MooncakeOperationCounterFamily,
    mooncake_operation_keys_total: MooncakeOperationCounterFamily,
    mooncake_operation_bytes_total: MooncakeOperationCounterFamily,
    mooncake_operation_failed_keys_total: MooncakeOperationCounterFamily,

    cache_inventory: Family<CacheInventoryLabels, F64Gauge>,
    mooncake_cache_layout: Family<MooncakeCacheLayoutLabels, U64Gauge>,
    mooncake_pipeline_queued: Family<MooncakePipelineLabels, U64Gauge>,
    mooncake_pipeline_active: Family<MooncakePipelineLabels, U64Gauge>,
    mooncake_pipeline_oldest_pending_seconds: Family<MooncakePipelineLabels, F64Gauge>,
    mooncake_pipeline_queue_wait_seconds: MooncakePipelineHistogramFamily,
    mooncake_pipeline_execution_seconds: MooncakePipelineHistogramFamily,
    mooncake_pipeline_consumption_wait_seconds: MooncakePipelineHistogramFamily,

    // NIXL connector telemetry, decoded from `kv_connector_stats`.
    nixl_xfer_time_seconds: HistogramMetric,
    nixl_post_time_seconds: HistogramMetric,
    nixl_bytes_transferred: HistogramMetric,
    nixl_num_descriptors: HistogramMetric,
    nixl_num_failed_transfers: U64Counter,
    nixl_num_failed_notifications: U64Counter,
    nixl_num_kv_expired_reqs: U64Counter,

    // Non-Prometheus interval accumulator for periodic text-log helpers.
    log_stats: SchedulerLogStatsAccumulator,
}

impl SchedulerStatsRecorder {
    /// Resolve the fixed-label metric handles for the connected engines.
    pub(crate) fn new(
        metrics: &SchedulerMetrics,
        model_name: &str,
        engines: &[ConnectedEngine],
    ) -> Self {
        let engines = engines
            .iter()
            .filter_map(|engine| {
                let engine = engine.engine_id.engine_index()?;
                Some((
                    engine,
                    resolve_scheduler_stats_handles(metrics, model_name, engine),
                ))
            })
            .collect();

        Self { engines }
    }

    /// Record one scheduler-stats payload for the given engine index.
    pub(crate) fn record(&self, engine_index: u32, stats: &SchedulerStats) {
        if let Some(handles) = self.engines.get(&engine_index) {
            record_scheduler_stats_with_handles(handles, stats);
        }
    }
}

/// Resolve all fixed-label scheduler metrics for one engine.
fn resolve_scheduler_stats_handles(
    metrics: &SchedulerMetrics,
    model_name: &str,
    engine: u32,
) -> SchedulerStatsHandles {
    let labels = EngineLabels {
        model_name: model_name.to_string(),
        engine,
    };
    let capacity = WaitingReasonLabels {
        model_name: model_name.to_string(),
        engine,
        reason: WAITING_REASON_CAPACITY,
    };
    let deferred = WaitingReasonLabels {
        model_name: model_name.to_string(),
        engine,
        reason: WAITING_REASON_DEFERRED,
    };

    SchedulerStatsHandles {
        scheduler_deferred_by_reason: metrics.scheduler_deferred_by_reason.clone(),
        scheduler_running: metrics.scheduler_running.get_or_create_owned(&labels),
        scheduler_waiting: metrics.scheduler_waiting.get_or_create_owned(&labels),
        scheduler_waiting_capacity: metrics
            .scheduler_waiting_by_reason
            .get_or_create_owned(&capacity),
        scheduler_waiting_deferred: metrics
            .scheduler_waiting_by_reason
            .get_or_create_owned(&deferred),
        kv_cache_usage: metrics.kv_cache_usage.get_or_create_owned(&labels),
        prefix_cache_queries: metrics.prefix_cache_queries.get_or_create_owned(&labels),
        prefix_cache_hits: metrics.prefix_cache_hits.get_or_create_owned(&labels),
        external_prefix_cache_queries: metrics
            .external_prefix_cache_queries
            .get_or_create_owned(&labels),
        external_prefix_cache_hits: metrics.external_prefix_cache_hits.get_or_create_owned(&labels),
        spec_decode_num_drafts: metrics.spec_decode_num_drafts.get_or_create_owned(&labels),
        spec_decode_num_draft_tokens: metrics
            .spec_decode_num_draft_tokens
            .get_or_create_owned(&labels),
        spec_decode_num_accepted_tokens: metrics
            .spec_decode_num_accepted_tokens
            .get_or_create_owned(&labels),
        spec_decode_num_accepted_tokens_per_pos: metrics
            .spec_decode_num_accepted_tokens_per_pos
            .clone(),
        log_stats: metrics.log_stats.get_or_create_owned(&labels),
        estimated_flops_per_gpu: metrics.estimated_flops_per_gpu.get_or_create_owned(&labels),
        estimated_read_bytes_per_gpu: metrics
            .estimated_read_bytes_per_gpu
            .get_or_create_owned(&labels),
        estimated_write_bytes_per_gpu: metrics
            .estimated_write_bytes_per_gpu
            .get_or_create_owned(&labels),
        kv_block_lifetime_seconds: metrics.kv_block_lifetime_seconds.get_or_create_owned(&labels),
        kv_block_idle_before_evict_seconds: metrics
            .kv_block_idle_before_evict_seconds
            .get_or_create_owned(&labels),
        kv_block_reuse_gap_seconds: metrics.kv_block_reuse_gap_seconds.get_or_create_owned(&labels),
        mooncake_operation_time_seconds: metrics.mooncake_operation_time_seconds.clone(),
        mooncake_operation_total: metrics.mooncake_operation_total.clone(),
        mooncake_operation_keys_total: metrics.mooncake_operation_keys_total.clone(),
        mooncake_operation_bytes_total: metrics.mooncake_operation_bytes_total.clone(),
        mooncake_operation_failed_keys_total: metrics.mooncake_operation_failed_keys_total.clone(),
        cache_inventory: metrics.cache_inventory.clone(),
        mooncake_cache_layout: metrics.mooncake_cache_layout.clone(),
        mooncake_pipeline_queued: metrics.mooncake_pipeline_queued.clone(),
        mooncake_pipeline_active: metrics.mooncake_pipeline_active.clone(),
        mooncake_pipeline_oldest_pending_seconds: metrics
            .mooncake_pipeline_oldest_pending_seconds
            .clone(),
        mooncake_pipeline_queue_wait_seconds: metrics.mooncake_pipeline_queue_wait_seconds.clone(),
        mooncake_pipeline_execution_seconds: metrics.mooncake_pipeline_execution_seconds.clone(),
        mooncake_pipeline_consumption_wait_seconds: metrics
            .mooncake_pipeline_consumption_wait_seconds
            .clone(),
        nixl_xfer_time_seconds: metrics.nixl_xfer_time_seconds.get_or_create_owned(&labels),
        nixl_post_time_seconds: metrics.nixl_post_time_seconds.get_or_create_owned(&labels),
        nixl_bytes_transferred: metrics.nixl_bytes_transferred.get_or_create_owned(&labels),
        nixl_num_descriptors: metrics.nixl_num_descriptors.get_or_create_owned(&labels),
        nixl_num_failed_transfers: metrics.nixl_num_failed_transfers.get_or_create_owned(&labels),
        nixl_num_failed_notifications: metrics
            .nixl_num_failed_notifications
            .get_or_create_owned(&labels),
        nixl_num_kv_expired_reqs: metrics.nixl_num_kv_expired_reqs.get_or_create_owned(&labels),
        labels,
    }
}

/// Record scheduler-stats values through pre-resolved metric handles.
fn record_scheduler_stats_with_handles(handles: &SchedulerStatsHandles, stats: &SchedulerStats) {
    // Scheduler state gauges.
    handles.scheduler_running.set(stats.num_running_reqs);
    handles
        .scheduler_waiting
        .set(stats.num_waiting_reqs + stats.num_skipped_waiting_reqs);
    handles.scheduler_waiting_capacity.set(stats.num_waiting_reqs);
    handles.scheduler_waiting_deferred.set(stats.num_skipped_waiting_reqs);
    for reason in DEFERRED_REASONS {
        handles
            .scheduler_deferred_by_reason
            .get_or_create(&WaitingReasonLabels {
                model_name: handles.labels.model_name.clone(),
                engine: handles.labels.engine,
                reason,
            })
            .set(stats.deferred_waiting_reasons.get(reason).copied().unwrap_or(0));
    }
    handles.kv_cache_usage.set(stats.kv_cache_usage);
    if let Some(inventory) = &stats.cache_inventory {
        for (key, value) in inventory {
            if let Some((kind, measure)) = key.split_once('.') {
                handles
                    .cache_inventory
                    .get_or_create(&CacheInventoryLabels {
                        model_name: handles.labels.model_name.clone(),
                        engine: handles.labels.engine,
                        kind: kind.to_string(),
                        measure: measure.to_string(),
                    })
                    .set(*value);
            }
        }
    }

    // Prefix-cache counters, including the connector-backed external cache path.
    handles.prefix_cache_queries.inc_by(stats.prefix_cache_stats.base.queries);
    handles.prefix_cache_hits.inc_by(stats.prefix_cache_stats.base.hits);

    if let Some(connector_prefix_cache_stats) = &stats.connector_prefix_cache_stats {
        handles
            .external_prefix_cache_queries
            .inc_by(connector_prefix_cache_stats.base.queries);
        handles
            .external_prefix_cache_hits
            .inc_by(connector_prefix_cache_stats.base.hits);
    }

    // Speculative decoding counters.
    if let Some(spec_decoding_stats) = &stats.spec_decoding_stats {
        handles.spec_decode_num_drafts.inc_by(spec_decoding_stats.num_drafts);
        handles
            .spec_decode_num_draft_tokens
            .inc_by(spec_decoding_stats.num_draft_tokens);
        handles
            .spec_decode_num_accepted_tokens
            .inc_by(spec_decoding_stats.num_accepted_tokens);
        handles.log_stats.observe_spec_decode(
            spec_decoding_stats.num_drafts,
            &spec_decoding_stats.num_accepted_tokens_per_pos,
        );

        for (position, accepted_tokens) in
            spec_decoding_stats.num_accepted_tokens_per_pos.iter().copied().enumerate()
        {
            handles
                .spec_decode_num_accepted_tokens_per_pos
                .get_or_create(&EnginePositionLabels {
                    model_name: handles.labels.model_name.clone(),
                    engine: handles.labels.engine,
                    position: position as u32,
                })
                .inc_by(accepted_tokens);
        }
    }

    // Per-engine performance / MFU counters.
    if let Some(perf_stats) = &stats.perf_stats
        && (perf_stats.num_flops_per_gpu != 0
            || perf_stats.num_read_bytes_per_gpu != 0
            || perf_stats.num_write_bytes_per_gpu != 0)
    {
        handles.estimated_flops_per_gpu.inc_by(perf_stats.num_flops_per_gpu);
        handles.estimated_read_bytes_per_gpu.inc_by(perf_stats.num_read_bytes_per_gpu);
        handles.estimated_write_bytes_per_gpu.inc_by(perf_stats.num_write_bytes_per_gpu);
    }

    if let Some(cudagraph_stats) = &stats.cudagraph_stats {
        handles.log_stats.observe_cudagraph(
            cudagraph_stats.num_unpadded_tokens,
            cudagraph_stats.num_padded_tokens,
            cudagraph_stats.num_paddings,
            &cudagraph_stats.runtime_mode,
        );
    }

    // Sampled KV-cache residency histograms.
    if !stats.kv_cache_eviction_events.is_empty() {
        for event in &stats.kv_cache_eviction_events {
            handles.kv_block_lifetime_seconds.observe(event.lifetime_seconds);
            handles.kv_block_idle_before_evict_seconds.observe(event.idle_seconds);
            for reuse_gap_seconds in &event.reuse_gaps_seconds {
                handles.kv_block_reuse_gap_seconds.observe(*reuse_gap_seconds);
            }
        }
    }

    // Connector-specific KV transfer stats. A bare connector reports its own
    // flat payload; MultiConnector reports connector class name -> flat child
    // payload.
    if let Some(kv_connector_stats) = &stats.kv_connector_stats {
        match kv_connector_stats {
            KvConnectorStats::Nixl(stats) => record_nixl_stats(handles, stats),
            KvConnectorStats::Mooncake(stats) => record_mooncake_stats(handles, stats),
            KvConnectorStats::Multi(stats) => record_multi_connector_stats(handles, stats),
            KvConnectorStats::Other(_) => {}
        }
    }
}

fn record_multi_connector_stats(handles: &SchedulerStatsHandles, stats: &MultiConnectorStats) {
    for nixl in [&stats.nixl, &stats.nixl_pull, &stats.nixl_push].into_iter().flatten() {
        record_nixl_stats(handles, nixl);
    }
    if let Some(mooncake) = &stats.mooncake {
        record_mooncake_stats(handles, mooncake);
    }
}

fn record_mooncake_stats(handles: &SchedulerStatsHandles, stats: &MooncakeStats) {
    for row in &stats.cache_layout {
        handles
            .mooncake_cache_layout
            .get_or_create(&MooncakeCacheLayoutLabels {
                model_name: handles.labels.model_name.clone(),
                engine: handles.labels.engine,
                scope: row.scope.clone(),
                prefix: row.prefix.clone(),
                kind: row.kind.clone(),
            })
            .set(1);
    }
    for snapshot in &stats.pipeline {
        let labels = MooncakePipelineLabels {
            model_name: handles.labels.model_name.clone(),
            engine: handles.labels.engine,
            stage: snapshot.stage.clone(),
            rank: snapshot.rank,
        };
        handles.mooncake_pipeline_queued.get_or_create(&labels).set(snapshot.queued);
        handles.mooncake_pipeline_active.get_or_create(&labels).set(snapshot.active);
        handles
            .mooncake_pipeline_oldest_pending_seconds
            .get_or_create(&labels)
            .set(snapshot.oldest_pending_seconds);
        for value in &snapshot.queue_wait_seconds {
            handles
                .mooncake_pipeline_queue_wait_seconds
                .get_or_create(&labels)
                .observe(*value);
        }
        for value in &snapshot.execution_seconds {
            handles
                .mooncake_pipeline_execution_seconds
                .get_or_create(&labels)
                .observe(*value);
        }
        for value in &snapshot.consumption_wait_seconds {
            handles
                .mooncake_pipeline_consumption_wait_seconds
                .get_or_create(&labels)
                .observe(*value);
        }
    }
    for (operation, records) in &stats.operations {
        for record in records {
            let labels = MooncakeOperationLabels {
                model_name: handles.labels.model_name.clone(),
                engine: handles.labels.engine,
                operation: operation.as_str().to_string(),
                status: record.status.as_str().to_string(),
            };
            handles
                .mooncake_operation_time_seconds
                .get_or_create(&labels)
                .observe(record.duration_seconds);
            handles.mooncake_operation_total.get_or_create(&labels).inc();
            handles
                .mooncake_operation_keys_total
                .get_or_create(&labels)
                .inc_by(record.num_keys);
            handles
                .mooncake_operation_bytes_total
                .get_or_create(&labels)
                .inc_by(record.num_bytes);
            handles
                .mooncake_operation_failed_keys_total
                .get_or_create(&labels)
                .inc_by(record.num_failed_keys);
        }
    }
}

fn record_nixl_stats(handles: &SchedulerStatsHandles, stats: &NixlStats) {
    for value in &stats.transfer_duration {
        handles.nixl_xfer_time_seconds.observe(*value);
    }
    for value in &stats.post_duration {
        handles.nixl_post_time_seconds.observe(*value);
    }
    for value in &stats.bytes_transferred {
        handles.nixl_bytes_transferred.observe(*value as f64);
    }
    for value in &stats.num_descriptors {
        handles.nixl_num_descriptors.observe(*value as f64);
    }
    handles
        .nixl_num_failed_transfers
        .inc_by(stats.num_failed_transfers.iter().sum());
    handles
        .nixl_num_failed_notifications
        .inc_by(stats.num_failed_notifications.iter().sum());
    handles.nixl_num_kv_expired_reqs.inc_by(stats.num_kv_expired_reqs.iter().sum());
}

/// Exports `vllm:lora_requests_info` as a single series covering all LoRA
/// requests tracked by this client across every engine in the replica.
///
/// The engine's `SchedulerStats` never carries adapter names: the Python
/// frontend fills them in from per-request lifecycle events tracked by
/// `LoRARequestStates` in `vllm/v1/engine/output_processor.py`. The Rust
/// frontend mirrors that, deriving the sets from the request registry.
#[derive(Default)]
pub(crate) struct LoraInfoExporter {
    current: Option<LoraInfoLabels>,
}

impl LoraInfoExporter {
    pub(crate) fn update(
        &mut self,
        metrics: &SchedulerMetrics,
        running: BTreeSet<String>,
        waiting: BTreeSet<String>,
    ) {
        let next = (!running.is_empty() || !waiting.is_empty()).then_some(LoraInfoLabels {
            running_lora_adapters: LoraAdapterNames(running),
            waiting_lora_adapters: LoraAdapterNames(waiting),
        });

        if self.current != next
            && let Some(prev) = &self.current
        {
            metrics.lora_info.remove(prev);
        }

        // Python sets this gauge to the current time on every record.
        if let Some(labels) = &next {
            metrics.lora_info.get_or_create(labels).set(now_unix_secs());
        }

        self.current = next;
    }
}

fn now_unix_secs() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, BTreeSet};

    use expect_test::expect;
    use vllm_metrics::Metrics;

    use crate::metrics::LoraInfoExporter;
    use crate::protocol::stats::{
        KvConnectorStats, MooncakeOperation, MooncakeRecord, MooncakeStats, MooncakeStatus,
        MultiConnectorStats, NixlStats, SchedulerStats,
    };

    fn names(values: &[&str]) -> BTreeSet<String> {
        values.iter().map(|name| (*name).to_string()).collect()
    }

    /// The `lora_requests_info` series with the non-deterministic timestamp
    /// value replaced by `<ts>`, one line per series.
    fn lora_series(rendered: &str) -> String {
        rendered
            .lines()
            .filter(|l| l.starts_with("vllm:lora_requests_info{"))
            .map(|l| match l.rsplit_once("} ") {
                Some((labels, _value)) => format!("{labels}}} <ts>"),
                None => l.to_string(),
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[test]
    fn lora_info_emits_clears_stale_and_drains() {
        let metrics = Metrics::new();
        let mut exporter = LoraInfoExporter::default();

        // No adapters: nothing emitted.
        exporter.update(&metrics.scheduler, names(&[]), names(&[]));
        expect![[""]].assert_eq(&lora_series(&metrics.render().unwrap()));

        // Two running (sorted), one waiting.
        exporter.update(&metrics.scheduler, names(&["b", "a"]), names(&["c"]));
        expect![[
            r#"vllm:lora_requests_info{running_lora_adapters="a,b",waiting_lora_adapters="c"} <ts>"#
        ]]
        .assert_eq(&lora_series(&metrics.render().unwrap()));

        // "c" gets scheduled and "d" arrives: the stale series is replaced.
        exporter.update(&metrics.scheduler, names(&["a", "b", "c"]), names(&["d"]));
        expect![[
            r#"vllm:lora_requests_info{running_lora_adapters="a,b,c",waiting_lora_adapters="d"} <ts>"#
        ]]
        .assert_eq(&lora_series(&metrics.render().unwrap()));

        // Everything but "d" finishes.
        exporter.update(&metrics.scheduler, names(&["d"]), names(&[]));
        expect![[
            r#"vllm:lora_requests_info{running_lora_adapters="d",waiting_lora_adapters=""} <ts>"#
        ]]
        .assert_eq(&lora_series(&metrics.render().unwrap()));

        // All requests done: series removed entirely.
        exporter.update(&metrics.scheduler, names(&[]), names(&[]));
        expect![[""]].assert_eq(&lora_series(&metrics.render().unwrap()));
    }

    #[test]
    fn mooncake_pipeline_payload_exports_live_backlog_and_clears_it() {
        let metrics = Metrics::new();
        let handles = super::resolve_scheduler_stats_handles(&metrics.scheduler, "model", 0);
        let payload = serde_json::json!({
            "MooncakeStoreConnector": {
                "_pipeline": [
                    {"stage": "lookup", "rank": -1, "queued": 2, "active": 1,
                     "oldest_pending_seconds": 7.0,
                     "queue_wait_seconds": [2.0], "execution_seconds": [3.0],
                     "consumption_wait_seconds": [0.25]},
                    {"stage": "load", "rank": 0, "queued": 0, "active": 1,
                     "oldest_pending_seconds": 4.0,
                     "queue_wait_seconds": [], "execution_seconds": []},
                    {"stage": "load", "rank": 1, "queued": 1, "active": 0,
                     "oldest_pending_seconds": 4.0,
                     "queue_wait_seconds": [], "execution_seconds": []}
                ],
                "load_get": [{"duration_seconds": 0.05, "num_keys": 3,
                              "num_bytes": 1024, "status": "ok", "num_failed_keys": 0}]
            }
        });
        let connector: KvConnectorStats = serde_json::from_value(payload).unwrap();
        let KvConnectorStats::Multi(ref multi) = connector else {
            panic!("pipeline payload was not decoded as MultiConnector stats");
        };
        assert_eq!(multi.mooncake.as_ref().unwrap().pipeline.len(), 3);
        let stats = SchedulerStats {
            num_skipped_waiting_reqs: 3,
            deferred_waiting_reasons: BTreeMap::from([
                ("external_cache_lookup".to_string(), 2),
                ("remote_kv_load".to_string(), 1),
            ]),
            kv_connector_stats: Some(connector),
            ..Default::default()
        };
        super::record_scheduler_stats_with_handles(&handles, &stats);
        let rendered = metrics.render().unwrap();
        for expected in [
            "vllm:mooncake_pipeline_queued{model_name=\"model\",engine=\"0\",stage=\"lookup\",rank=\"-1\"} 2",
            "vllm:mooncake_pipeline_active{model_name=\"model\",engine=\"0\",stage=\"load\",rank=\"0\"} 1",
            "vllm:mooncake_pipeline_queued{model_name=\"model\",engine=\"0\",stage=\"load\",rank=\"1\"} 1",
            "vllm:mooncake_pipeline_execution_seconds_sum{model_name=\"model\",engine=\"0\",stage=\"lookup\",rank=\"-1\"} 3",
            "vllm:mooncake_pipeline_consumption_wait_seconds_sum{model_name=\"model\",engine=\"0\",stage=\"lookup\",rank=\"-1\"} 0.25",
            "vllm:num_requests_deferred_by_reason{model_name=\"model\",engine=\"0\",reason=\"external_cache_lookup\"} 2",
        ] {
            assert!(rendered.contains(expected), "missing metric: {expected}");
        }
        let mut drained = stats.kv_connector_stats.unwrap();
        let KvConnectorStats::Multi(ref mut multi) = drained else {
            unreachable!()
        };
        let mooncake = multi.mooncake.as_mut().unwrap();
        mooncake.operations.clear();
        for snapshot in &mut mooncake.pipeline {
            snapshot.queued = 0;
            snapshot.active = 0;
            snapshot.oldest_pending_seconds = 0.0;
            snapshot.queue_wait_seconds.clear();
            snapshot.execution_seconds.clear();
            snapshot.consumption_wait_seconds.clear();
        }
        super::record_scheduler_stats_with_handles(
            &handles,
            &SchedulerStats {
                kv_connector_stats: Some(drained),
                ..Default::default()
            },
        );
        let rendered = metrics.render().unwrap();
        assert!(rendered.contains(
            "vllm:mooncake_pipeline_queued{model_name=\"model\",engine=\"0\",stage=\"lookup\",rank=\"-1\"} 0"
        ));
        assert!(rendered.contains(
            "vllm:num_requests_deferred_by_reason{model_name=\"model\",engine=\"0\",reason=\"external_cache_lookup\"} 0"
        ));
        assert!(rendered.contains(
            "vllm:mooncake_pipeline_execution_seconds_count{model_name=\"model\",engine=\"0\",stage=\"lookup\",rank=\"-1\"} 1"
        ));
        assert!(rendered.contains(
            "vllm:mooncake_pipeline_consumption_wait_seconds_count{model_name=\"model\",engine=\"0\",stage=\"lookup\",rank=\"-1\"} 1"
        ));
    }

    fn nixl_stats() -> NixlStats {
        NixlStats {
            transfer_duration: vec![0.01, 0.02],
            post_duration: vec![0.001, 0.002],
            bytes_transferred: vec![4096, 8192],
            num_descriptors: vec![2, 4],
            num_failed_transfers: vec![],
            num_failed_notifications: vec![],
            num_kv_expired_reqs: vec![1],
        }
    }

    fn mooncake_stats() -> MooncakeStats {
        MooncakeStats {
            cache_layout: vec![],
            pipeline: vec![],
            operations: BTreeMap::from([(
                MooncakeOperation::LoadGet,
                vec![MooncakeRecord {
                    duration_seconds: 0.05,
                    num_keys: 3,
                    num_bytes: 1024,
                    status: MooncakeStatus::Ok,
                    num_failed_keys: 0,
                }],
            )]),
        }
    }

    /// Records one MultiConnector payload into Mooncake and NIXL metrics.
    #[test]
    fn kv_connector_stats_are_recorded_into_mooncake_and_nixl_metrics() {
        let metrics = Metrics::new();
        let handles = super::resolve_scheduler_stats_handles(&metrics.scheduler, "model", 0);

        let stats = SchedulerStats {
            kv_connector_stats: Some(KvConnectorStats::Multi(Box::new(MultiConnectorStats {
                nixl: Some(nixl_stats()),
                mooncake: Some(mooncake_stats()),
                ..Default::default()
            }))),
            ..Default::default()
        };

        super::record_scheduler_stats_with_handles(&handles, &stats);

        let rendered = metrics.render().unwrap();
        assert!(rendered.contains(
            "vllm:mooncake_store_operation_total{model_name=\"model\",engine=\"0\",\
             operation=\"load_get\",status=\"ok\"} 1"
        ));
        assert!(rendered.contains(
            "vllm:mooncake_store_operation_keys_total{model_name=\"model\",engine=\"0\",\
             operation=\"load_get\",status=\"ok\"} 3"
        ));
        assert!(rendered.contains(
            "vllm:mooncake_store_operation_bytes_total{model_name=\"model\",engine=\"0\",\
             operation=\"load_get\",status=\"ok\"} 1024"
        ));
        assert!(
            rendered.contains(
                "vllm:nixl_num_kv_expired_reqs_total{model_name=\"model\",engine=\"0\"} 1"
            )
        );
        assert!(
            rendered
                .contains("vllm:nixl_xfer_time_seconds_count{model_name=\"model\",engine=\"0\"} 2")
        );
    }

    #[test]
    fn checkpoint_stats_deserialize_and_export_without_changing_save_totals() {
        let payload = serde_json::json!({
            "save_put": [{"duration_seconds": 0.1, "num_keys": 2, "num_bytes": 768,
                "status": "ok", "num_failed_keys": 0}],
            "checkpoint_mamba_put": [{"duration_seconds": 0.1, "num_keys": 1,
                "num_bytes": 256, "status": "ok", "num_failed_keys": 0}],
            "checkpoint_attention_put": [{"duration_seconds": 0.1, "num_keys": 1,
                "num_bytes": 512, "status": "ok", "num_failed_keys": 0}],
            "checkpoint_attention_reuse": [{"duration_seconds": 0.01, "num_keys": 1,
                "num_bytes": 1024, "status": "ok", "num_failed_keys": 0}]
        });
        let decoded: KvConnectorStats = serde_json::from_value(payload).unwrap();
        assert!(matches!(&decoded, KvConnectorStats::Mooncake(_)));
        let metrics = Metrics::new();
        let handles = super::resolve_scheduler_stats_handles(&metrics.scheduler, "model", 0);
        super::record_scheduler_stats_with_handles(
            &handles,
            &SchedulerStats {
                kv_connector_stats: Some(decoded),
                ..Default::default()
            },
        );
        let rendered = metrics.render().unwrap();
        for (operation, bytes) in [
            ("save_put", 768),
            ("checkpoint_mamba_put", 256),
            ("checkpoint_attention_put", 512),
            ("checkpoint_attention_reuse", 1024),
        ] {
            assert!(rendered.contains(&format!(
                "vllm:mooncake_store_operation_bytes_total{{model_name=\"model\",engine=\"0\",operation=\"{operation}\",status=\"ok\"}} {bytes}"
            )));
        }
    }
    #[test]
    fn cache_inventory_and_expected_layout_export_and_clear() {
        let metrics = Metrics::new();
        let handles = super::resolve_scheduler_stats_handles(&metrics.scheduler, "model", 0);
        let connector: KvConnectorStats = serde_json::from_value(serde_json::json!({
            "_cache_layout": [{"scope": "model", "prefix": "model@tp_rank:0@group:1", "kind": "mamba"}]
        })).unwrap();
        assert!(matches!(&connector, KvConnectorStats::Mooncake(_)));
        super::record_scheduler_stats_with_handles(
            &handles,
            &SchedulerStats {
                cache_inventory: Some(std::collections::BTreeMap::from([(
                    "mamba.in_use_blocks".to_string(),
                    12.0,
                )])),
                kv_connector_stats: Some(connector),
                ..Default::default()
            },
        );
        let rendered = metrics.render().unwrap();
        assert!(rendered.contains("vllm:cache_inventory{model_name=\"model\",engine=\"0\",kind=\"mamba\",measure=\"in_use_blocks\"} 12"));
        assert!(rendered.contains("vllm:mooncake_cache_layout{model_name=\"model\",engine=\"0\",scope=\"model\",prefix=\"model@tp_rank:0@group:1\",kind=\"mamba\"} 1"));
        super::record_scheduler_stats_with_handles(
            &handles,
            &SchedulerStats {
                cache_inventory: Some(std::collections::BTreeMap::from([(
                    "mamba.in_use_blocks".to_string(),
                    0.0,
                )])),
                ..Default::default()
            },
        );
        assert!(metrics.render().unwrap().contains("vllm:cache_inventory{model_name=\"model\",engine=\"0\",kind=\"mamba\",measure=\"in_use_blocks\"} 0"));
    }
}
