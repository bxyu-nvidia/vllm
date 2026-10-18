# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-operation telemetry for MooncakeStoreConnector.

Records one row per Mooncake RPC (``save_exists``, ``save_put``, ``load_get``,
``lookup_exists``) with duration, key/byte counts, status, and failed-key
count. Exposed to the logger via ``KVConnectorLogging`` and to Prometheus
via ``MooncakeStorePromMetrics``. ``checkpoint_mamba_put`` and
``checkpoint_attention_put`` break out checkpoint traffic already included in
``save_put``; do not sum these subsets into total traffic. Their byte counts
are attempted bytes with success/failure status, like ``save_put``.
``checkpoint_attention_reuse`` counts avoided keys and bytes, not transfers.
"""

from dataclasses import dataclass
from statistics import fmean
from typing import Any

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
    KVConnectorPromMetrics,
    KVConnectorStats,
    PromMetric,
    PromMetricT,
)

from .pipeline import PIPELINE_BUCKETS, PIPELINE_KEY, PipelineSnapshot


def _nearest_rank_percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    rank = max(
        0, min(len(sorted_values) - 1, int(percentile * len(sorted_values) - 1e-12))
    )
    return sorted_values[rank]


@dataclass
class MooncakeStoreConnectorStats(KVConnectorStats):
    """Serializable Mooncake store communication telemetry."""

    def __post_init__(self):
        if not self.data:
            self.reset()

    def reset(self):
        self.data: dict[str, Any] = {}

    def is_empty(self) -> bool:
        return not self.data

    def aggregate(self, other: KVConnectorStats) -> KVConnectorStats:
        if other.is_empty():
            return self
        for operation, records in other.data.items():
            if operation == "_cache_layout":
                self.data[operation] = list(
                    {
                        s["prefix"]: s
                        for s in [*self.data.get(operation, []), *records]
                    }.values()
                )
                continue
            if operation == PIPELINE_KEY:
                snapshots = {
                    (s["stage"], s["rank"]): s for s in self.data.get(PIPELINE_KEY, [])
                }
                for record in records:
                    key = (record["stage"], record["rank"])
                    previous = snapshots.get(key)
                    latest = dict(record)
                    for field in (
                        "queue_wait_seconds",
                        "execution_seconds",
                        "consumption_wait_seconds",
                    ):
                        latest[field] = (
                            list(previous.get(field, [])) if previous else []
                        ) + list(record.get(field, []))
                    snapshots[key] = latest
                self.data[PIPELINE_KEY] = list(snapshots.values())
            else:
                self.data.setdefault(operation, []).extend(records)
        return self

    def reduce(self) -> dict[str, int | float]:
        reduced: dict[str, int | float] = {}
        for operation, records in sorted(self.data.items()):
            if operation == "_cache_layout":
                continue
            if operation == PIPELINE_KEY:
                for snapshot in records:
                    prefix = f"{snapshot['stage']}_rank{snapshot['rank']}"
                    for name in ("queued", "active", "oldest_pending_seconds"):
                        reduced[f"{prefix}_{name}"] = snapshot[name]
                    for name in (
                        "queue_wait_seconds",
                        "execution_seconds",
                        "consumption_wait_seconds",
                    ):
                        samples = snapshot.get(name, [])
                        if samples:
                            reduced[f"{prefix}_{name}_avg"] = fmean(samples)
                continue
            if not records:
                continue
            durations = [float(record["duration_seconds"]) for record in records]
            reduced[f"{operation}_count"] = len(records)
            reduced[f"{operation}_avg_ms"] = round(fmean(durations) * 1e3, 3)
            reduced[f"{operation}_p90_ms"] = round(
                _nearest_rank_percentile(durations, 0.9) * 1e3, 3
            )
            reduced[f"{operation}_total_keys"] = sum(
                int(record["num_keys"]) for record in records
            )
            reduced[f"{operation}_total_bytes"] = sum(
                int(record["num_bytes"]) for record in records
            )
            reduced[f"{operation}_failed_keys"] = sum(
                int(record["num_failed_keys"]) for record in records
            )
            reduced[f"{operation}_error_count"] = sum(
                1 for record in records if record["status"] == "error"
            )
        return reduced

    def record_operation(
        self,
        operation: str,
        duration_seconds: float,
        num_keys: int,
        *,
        num_bytes: int = 0,
        status: str = "ok",
        num_failed_keys: int = 0,
    ) -> None:
        self.data.setdefault(operation, []).append(
            {
                "duration_seconds": duration_seconds,
                "num_keys": num_keys,
                "num_bytes": num_bytes,
                "status": status,
                "num_failed_keys": num_failed_keys,
            }
        )


class MooncakeStorePromMetrics(KVConnectorPromMetrics):
    """Prometheus metrics for Mooncake store communication."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        metric_types: dict[type[PromMetric], type[PromMetricT]],
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
    ):
        super().__init__(vllm_config, metric_types, labelnames, per_engine_labelvalues)
        self._cache_layout = self._gauge_cls(
            name="vllm:mooncake_cache_layout",
            documentation="Store namespace layout for metadata inventory.",
            labelnames=labelnames + ["scope", "prefix", "kind"],
            multiprocess_mode="mostrecent",
        )
        metric_labelnames = labelnames + ["operation", "status"]
        self._metric_cache: dict[tuple[int, str, str], dict[str, PromMetric]] = {}
        pipeline_labels = labelnames + ["stage", "rank"]
        self._pipeline_metrics: dict[str, PromMetric] = {}
        for field, description in (
            ("queued", "Jobs waiting for a Mooncake pipeline worker."),
            ("active", "Jobs executing in a Mooncake pipeline worker."),
            ("oldest_pending_seconds", "Age of the oldest queued or active job."),
        ):
            self._pipeline_metrics[field] = self._gauge_cls(
                name=f"vllm:mooncake_pipeline_{field}",
                documentation=description,
                labelnames=pipeline_labels,
                multiprocess_mode="mostrecent",
            )
        for field in (
            "queue_wait_seconds",
            "execution_seconds",
            "consumption_wait_seconds",
        ):
            self._pipeline_metrics[field] = self._histogram_cls(
                name=f"vllm:mooncake_pipeline_{field}",
                documentation=f"Mooncake pipeline {field}; rank -1 is engine-wide.",
                labelnames=pipeline_labels,
                buckets=PIPELINE_BUCKETS,
            )

        self._histogram_operation_time = self._histogram_cls(
            name="vllm:mooncake_store_operation_time_seconds",
            documentation="Histogram of Mooncake store communication time.",
            buckets=[
                1e-3,
                5e-3,
                1e-2,
                5e-2,
                1e-1,
                2e-1,
                3e-1,
                4e-1,
                5e-1,
                7.5e-1,
                1.0,
                1.5,
                2.0,
                3.0,
                4.0,
            ],
            labelnames=metric_labelnames,
        )
        self._counter_operation_calls = self._counter_cls(
            name="vllm:mooncake_store_operation_total",
            documentation="Number of Mooncake store communication operations.",
            labelnames=metric_labelnames,
        )
        self._counter_operation_keys = self._counter_cls(
            name="vllm:mooncake_store_operation_keys_total",
            documentation="Number of Mooncake store keys touched by operations.",
            labelnames=metric_labelnames,
        )
        self._counter_operation_bytes = self._counter_cls(
            name="vllm:mooncake_store_operation_bytes_total",
            documentation="Number of bytes transferred by Mooncake store operations.",
            labelnames=metric_labelnames,
        )
        self._counter_failed_keys = self._counter_cls(
            name="vllm:mooncake_store_operation_failed_keys_total",
            documentation="Number of Mooncake store keys that failed in operations.",
            labelnames=metric_labelnames,
        )

    def _get_metrics(
        self,
        engine_idx: int,
        operation: str,
        status: str,
    ) -> dict[str, PromMetric]:
        cache_key = (engine_idx, operation, status)
        if cache_key not in self._metric_cache:
            label_values = self.per_engine_labelvalues[engine_idx] + [operation, status]
            self._metric_cache[cache_key] = {
                "time": self._histogram_operation_time.labels(*label_values),
                "calls": self._counter_operation_calls.labels(*label_values),
                "keys": self._counter_operation_keys.labels(*label_values),
                "bytes": self._counter_operation_bytes.labels(*label_values),
                "failed_keys": self._counter_failed_keys.labels(*label_values),
            }
        return self._metric_cache[cache_key]

    def observe(self, transfer_stats_data: dict[str, Any] | None, engine_idx: int = 0):
        if not transfer_stats_data:
            return
        for operation, records in transfer_stats_data.items():
            if operation == "_cache_layout":
                for row in records:
                    self._cache_layout.labels(
                        *self.per_engine_labelvalues[engine_idx],
                        row["scope"],
                        row["prefix"],
                        row["kind"],
                    ).set(1)
                continue
            if operation == PIPELINE_KEY:
                for snapshot in records:
                    self._observe_pipeline(snapshot, engine_idx)
                continue
            assert isinstance(records, list)
            for record in records:
                assert isinstance(record, dict)
                status = str(record["status"])
                metrics = self._get_metrics(engine_idx, operation, status)
                metrics["time"].observe(float(record["duration_seconds"]))
                metrics["calls"].inc()
                metrics["keys"].inc(int(record["num_keys"]))
                metrics["bytes"].inc(int(record["num_bytes"]))
                metrics["failed_keys"].inc(int(record["num_failed_keys"]))

    def _observe_pipeline(self, snapshot: PipelineSnapshot, engine_idx: int) -> None:
        labels = self.per_engine_labelvalues[engine_idx] + [
            snapshot["stage"],
            snapshot["rank"],
        ]
        for field in ("queued", "active", "oldest_pending_seconds"):
            self._pipeline_metrics[field].labels(*labels).set(snapshot[field])
        for field, samples in (
            ("queue_wait_seconds", snapshot["queue_wait_seconds"]),
            ("execution_seconds", snapshot["execution_seconds"]),
            ("consumption_wait_seconds", snapshot.get("consumption_wait_seconds", [])),
        ):
            for value in samples:
                self._pipeline_metrics[field].labels(*labels).observe(value)
