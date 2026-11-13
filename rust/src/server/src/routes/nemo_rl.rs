// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Opt-in control protocol for NeMo-RL's managed prefill/decode fleet.

use std::collections::BTreeSet;
use std::sync::Arc;

use axum::extract::{Request, State};
use axum::http::StatusCode;
use axum::middleware::{Next, from_fn_with_state};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use subtle::ConstantTimeEq as _;
use vllm_engine_core_client::EngineCoreClient;

use crate::error::{ApiError, server_error};
use crate::state::{ApiKeyHash, AppState, hash_api_key};
use crate::utils::utility_call_error;

pub(super) fn router(token: &str) -> Router<Arc<AppState>> {
    Router::new().route("/nemo_rl/capture/metadata", get(capture_metadata)).route(
        "/nemo_rl/refit",
        post(refit).route_layer(from_fn_with_state(hash_api_key(token), authenticate)),
    )
}

async fn authenticate(
    State(expected): State<ApiKeyHash>,
    request: Request,
    next: Next,
) -> Response {
    let authorized = request
        .headers()
        .get("authorization")
        .and_then(|header| header.to_str().ok())
        .and_then(|header| header.strip_prefix("Bearer "))
        .is_some_and(|token| bool::from(hash_api_key(token).ct_eq(&expected)));
    if !authorized {
        return (
            StatusCode::UNAUTHORIZED,
            Json(json!({"error": "Unauthorized PD control request"})),
        )
            .into_response();
    }
    next.run(request).await
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
struct CaptureMetadata {
    layer_ids: Vec<usize>,
    enabled: bool,
    max_logprobs: i64,
    logprobs_mode: String,
    #[serde(skip_serializing)]
    workers_per_engine: usize,
}

async fn metadata(client: &EngineCoreClient) -> Result<CaptureMetadata, ApiError> {
    let reports: Vec<Vec<CaptureMetadata>> = client
        .call_utility(
            "collective_rpc",
            (
                "pd_capture_metadata",
                None::<f64>,
                Vec::<Value>::new(),
                None::<Value>,
            ),
        )
        .await
        .map_err(|error| utility_call_error("pd_capture_metadata", error))?;
    merge_metadata(reports, client.engine_count())
}

fn merge_metadata(
    reports: Vec<Vec<CaptureMetadata>>,
    engines: usize,
) -> Result<CaptureMetadata, ApiError> {
    let mut combined = reports
        .first()
        .and_then(|workers| workers.first())
        .cloned()
        .ok_or_else(|| server_error!("PD workers returned no capture metadata"))?;
    if combined.workers_per_engine == 0
        || reports.len() != engines
        || reports.iter().any(|workers| workers.len() != combined.workers_per_engine)
    {
        return Err(server_error!(
            "PD capture acknowledgment count differs from its GPU topology"
        ));
    }
    let mut layers = BTreeSet::new();
    for report in reports.into_iter().flatten() {
        if report.enabled != combined.enabled
            || report.max_logprobs != combined.max_logprobs
            || report.logprobs_mode != combined.logprobs_mode
            || report.workers_per_engine != combined.workers_per_engine
        {
            return Err(server_error!(
                "PD workers disagree on capture configuration"
            ));
        }
        layers.extend(report.layer_ids);
    }
    combined.layer_ids = layers.into_iter().collect();
    Ok(combined)
}

async fn capture_metadata(State(state): State<Arc<AppState>>) -> Result<Json<Value>, ApiError> {
    let report = metadata(state.engine_core_client()).await?;
    Ok(Json(json!({
        "version": 1,
        "layer_ids": report.layer_ids,
        "enabled": report.enabled,
        "max_logprobs": report.max_logprobs,
        "logprobs_mode": report.logprobs_mode,
    })))
}

#[derive(Debug, Deserialize)]
#[serde(tag = "operation", rename_all = "snake_case", deny_unknown_fields)]
enum RefitRequest {
    Prepare {
        entries: Vec<(String, Vec<usize>, String)>,
        mtp_from_refit: bool,
    },
    Init {
        rank_prefix: usize,
        ip: String,
        port: u16,
        world_size: usize,
        train_world_size: usize,
    },
    Update {
        refit_timeout_s: f64,
    },
    Pause {},
    Resume {},
    Reset {},
}

async fn collective(
    client: &EngineCoreClient,
    workers: usize,
    method: &str,
    args: Value,
    require_true: bool,
) -> Result<(), ApiError> {
    // Preserve each DP engine's result list so a missing TP worker cannot be
    // hidden by a successful acknowledgment from another engine.
    let results: Vec<Vec<Value>> = client
        .call_utility("collective_rpc", (method, None::<f64>, args, None::<Value>))
        .await
        .map_err(|error| utility_call_error("collective_rpc", error))?;
    check_acknowledgments(&results, client.engine_count(), workers, require_true)
}

fn check_acknowledgments(
    results: &[Vec<Value>],
    engines: usize,
    workers: usize,
    require_true: bool,
) -> Result<(), ApiError> {
    if results.len() != engines || results.iter().any(|result| result.len() != workers) {
        return Err(server_error!(
            "PD refit acknowledgment count differs from its GPU topology"
        ));
    }
    if require_true && results.iter().flatten().any(|value| value != &Value::Bool(true)) {
        return Err(server_error!("PD refit failed on one or more GPU workers"));
    }
    Ok(())
}

async fn refit(
    State(state): State<Arc<AppState>>,
    Json(request): Json<RefitRequest>,
) -> Result<Json<Value>, ApiError> {
    let client = state.engine_core_client();
    let workers = metadata(client).await?.workers_per_engine;
    match request {
        RefitRequest::Prepare {
            entries,
            mtp_from_refit,
        } => {
            collective(
                client,
                workers,
                "pd_prepare_refit",
                json!([entries, mtp_from_refit]),
                true,
            )
            .await?;
        }
        RefitRequest::Init {
            rank_prefix,
            ip,
            port,
            world_size,
            train_world_size,
        } => {
            collective(
                client,
                workers,
                "init_collective",
                json!([rank_prefix, ip, port, world_size, train_world_size]),
                false,
            )
            .await?;
        }
        RefitRequest::Update { refit_timeout_s } => {
            if !refit_timeout_s.is_finite() || refit_timeout_s <= 0.0 {
                return Err(ApiError::invalid_request(
                    "refit_timeout_s must be positive and finite",
                    Some("refit_timeout_s"),
                ));
            }
            collective(
                client,
                workers,
                "update_weights_from_collective",
                json!([refit_timeout_s, false]),
                true,
            )
            .await?;
        }
        RefitRequest::Pause {} => {
            client
                .pause_scheduler(
                    vllm_engine_core_client::protocol::utility::PauseMode::Wait,
                    false,
                )
                .await
                .map_err(|error| utility_call_error("pause_scheduler", error))?;
        }
        RefitRequest::Resume {} => {
            client
                .resume_scheduler()
                .await
                .map_err(|error| utility_call_error("resume_scheduler", error))?;
        }
        RefitRequest::Reset {} => {
            if !client
                .reset_prefix_cache(false, true)
                .await
                .map_err(|error| utility_call_error("reset_prefix_cache", error))?
            {
                return Err(server_error!(
                    "PD cache invalidation failed; weights must not resume serving"
                ));
            }
            client
                .call_utility::<(), _>("reset_encoder_cache", ())
                .await
                .map_err(|error| utility_call_error("reset_encoder_cache", error))?;
        }
    }
    Ok(Json(json!({"ok": true})))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_dp_engine_and_tp_worker_must_acknowledge() {
        let success = vec![vec![json!(true), json!(true)]; 2];
        assert!(check_acknowledgments(&success, 2, 2, true).is_ok());
        assert!(check_acknowledgments(&success[..1], 2, 2, true).is_err());
        let mut missing = success.clone();
        missing[1].pop();
        assert!(check_acknowledgments(&missing, 2, 2, true).is_err());
        let mut failed = success;
        failed[1][1] = json!(false);
        assert!(check_acknowledgments(&failed, 2, 2, true).is_err());
    }

    #[test]
    fn control_protocol_rejects_arbitrary_methods_and_incomplete_updates() {
        assert!(
            serde_json::from_value::<RefitRequest>(
                json!({"operation": "collective_rpc", "method": "eval"})
            )
            .is_err()
        );
        assert!(serde_json::from_value::<RefitRequest>(json!({"operation": "update"})).is_err());
        assert!(
            serde_json::from_value::<RefitRequest>(
                json!({"operation": "reset", "reset_connector": false})
            )
            .is_err()
        );
    }
}
