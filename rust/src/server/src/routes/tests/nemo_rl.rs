// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use super::*;
use base64::Engine as _;
use base64::engine::general_purpose::STANDARD;
use vllm_engine_core_client::protocol::tensor::{WireArrayData, WireNdArray};

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn pd_completion_preserves_routes_across_frames_and_trims_stop_tail() {
    let (mut app, engine_task) = test_admin_app_with_engine_script(|dealer, push| {
        boxed_test_future(async move {
            let add = recv_engine_message(dealer).await;
            let request: EngineCoreRequest = rmp_serde::from_slice(&add[1]).unwrap();
            assert_eq!(
                request.sampling_params.as_ref().unwrap().routed_experts_prompt_start,
                1
            );
            let mut first = request_output(&request.request_id, bytes_to_token_ids(b"a"), None);
            first.routed_experts = Some(Box::new(
                WireNdArray::from_i32(vec![2, 1, 2], (0..4).collect()).unwrap(),
            ));
            let mut last = request_output(
                &request.request_id,
                bytes_to_token_ids(b"b!x"),
                Some(EngineCoreFinishReason::Length),
            );
            let mut routes = WireNdArray::from_i32(vec![3, 1, 2], (4..10).collect()).unwrap();
            let data = routes.data.as_raw_view().unwrap().clone();
            routes.data = WireArrayData::AuxIndex(1);
            last.routed_experts = Some(Box::new(routes));
            let outputs: EngineCoreOutputs = RequestBatchOutputs {
                outputs: vec![first, last],
                ..Default::default()
            }
            .into();
            push.send(
                ZmqMessage::try_from(vec![
                    Bytes::from(rmp_serde::to_vec_named(&outputs).unwrap()),
                    data,
                ])
                .unwrap(),
            )
            .await
            .unwrap();
        })
    })
    .await;
    let response = app
        .call(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "Qwen/Qwen1.5-0.5B-Chat", "prompt": [65, 66, 67],
                        "stream": false, "return_token_ids": true, "stop": "!",
                        "routed_experts_prompt_start": 1,
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    let status = response.status();
    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    assert_eq!(status, StatusCode::OK, "{}", String::from_utf8_lossy(&body));
    let response: serde_json::Value = serde_json::from_slice(&body).unwrap();
    let choice = &response["choices"][0];
    assert_eq!(choice["prompt_token_ids"], json!([65, 66, 67]));
    assert_eq!(choice["token_ids"], json!([97, 98, 33]));
    assert_eq!(choice["text"], "ab");
    let npy = STANDARD.decode(choice["routed_experts"].as_str().unwrap()).unwrap();
    let offset = 10 + usize::from(u16::from_le_bytes([npy[8], npy[9]]));
    assert!(std::str::from_utf8(&npy[10..offset]).unwrap().contains("'shape': (4, 1, 2)"));
    assert_eq!(
        &npy[offset..],
        (0i32..8).flat_map(i32::to_ne_bytes).collect::<Vec<_>>()
    );
    engine_task.finish().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn pd_refit_authenticates_and_resets_connector_before_acknowledging() {
    let (state, engine_task) = test_admin_state_with_ready_and_engine_script(default_ready_response(), |dealer, push| {
        boxed_test_future(async move {
            // Unauthorized requests must never reach the engine. The first RPC
            // belongs to the authenticated metadata request below.
            for (method, args, result) in [
                ("collective_rpc", json!(["pd_capture_metadata", null, [], null]), json!([
                    {"layer_ids": [3, 1], "enabled": true, "max_logprobs": 32, "logprobs_mode": "raw_logprobs", "workers_per_engine": 2},
                    {"layer_ids": [1, 3], "enabled": true, "max_logprobs": 32, "logprobs_mode": "raw_logprobs", "workers_per_engine": 2},
                ])),
                ("reset_prefix_cache", json!([false, true]), json!(true)),
                ("reset_encoder_cache", json!([]), json!(null)),
                ("collective_rpc", json!(["pd_clear_aux_cache", null, [], null]), json!([true, true])),
            ] {
                let msg = recv_engine_message(dealer).await;
                let payload = decode_value(&msg[1]).unwrap();
                let array = payload.as_array().unwrap();
                assert_eq!(array[2], Value::from(method));
                assert_eq!(array[3], rmpv::ext::to_value(args).unwrap());
                send_outputs(push, utility_outputs(array[1].as_u64().unwrap(), utility_result_value(result))).await;
            }
        })
    }).await;
    let mut app = super::super::nemo_rl::router("control-secret").with_state(state);
    for token in [None, Some("Bearer wrong"), Some("Bearer control-secret")] {
        let mut request = Request::builder()
            .method("POST")
            .uri("/nemo_rl/refit")
            .header("content-type", "application/json");
        if let Some(token) = token {
            request = request.header("authorization", token);
        }
        let response = app
            .call(request.body(Body::from(r#"{"operation":"reset"}"#)).unwrap())
            .await
            .unwrap();
        let expected = if token == Some("Bearer control-secret") {
            StatusCode::OK
        } else {
            StatusCode::UNAUTHORIZED
        };
        let status = response.status();
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        assert_eq!(status, expected, "{}", String::from_utf8_lossy(&body));
    }
    engine_task.finish().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn pd_refit_forwards_the_training_protocol_and_rejects_partial_updates() {
    let cases = vec![
        (
            json!({"operation": "prepare", "entries": [["weight", [8, 4], "bfloat16"]], "mtp_from_refit": false}),
            "collective_rpc",
            json!([
                "pd_prepare_refit",
                null,
                [[["weight", [8, 4], "bfloat16"]], false],
                null
            ]),
            json!([true, true]),
            StatusCode::OK,
        ),
        (
            json!({"operation": "init", "rank_prefix": 4, "ip": "127.0.0.1", "port": 31000, "world_size": 6, "train_world_size": 4}),
            "collective_rpc",
            json!(["init_collective", null, [4, "127.0.0.1", 31000, 6, 4], null]),
            json!([null, null]),
            StatusCode::OK,
        ),
        (
            json!({"operation": "update", "refit_timeout_s": 120.0}),
            "collective_rpc",
            json!(["update_weights_from_collective", null, [120.0, false], null]),
            json!([true, false]),
            StatusCode::INTERNAL_SERVER_ERROR,
        ),
        (
            json!({"operation": "drain_caches"}),
            "collective_rpc",
            json!(["pd_drain_caches", null, [], null]),
            json!([true, true]),
            StatusCode::OK,
        ),
        (
            json!({"operation": "pause"}),
            "drain_capture",
            json!([]),
            json!(false),
            StatusCode::INTERNAL_SERVER_ERROR,
        ),
        (
            json!({"operation": "resume"}),
            "resume_capture",
            json!([]),
            json!(false),
            StatusCode::INTERNAL_SERVER_ERROR,
        ),
        (
            json!({"operation": "pause"}),
            "pause_scheduler",
            json!(["wait", false]),
            json!(null),
            StatusCode::OK,
        ),
        (
            json!({"operation": "resume"}),
            "resume_scheduler",
            json!([]),
            json!(null),
            StatusCode::OK,
        ),
    ];
    let mut cases = cases;
    for (operation, args) in [
        ("check_transfers", json!([false, true])),
        ("pause_transfers", json!([true, false])),
        ("resume_transfers", json!([false, false])),
    ] {
        for (result, status) in [
            (true, StatusCode::OK),
            (false, StatusCode::INTERNAL_SERVER_ERROR),
        ] {
            cases.push((
                json!({"operation": operation}),
                "set_kv_transfer_paused",
                args.clone(),
                json!(result),
                status,
            ));
        }
    }
    cases.extend([
        (
            json!({"operation": "pause", "pause_mode": "keep"}),
            "suspend_capture",
            json!([]),
            json!(true),
            StatusCode::OK,
        ),
        (
            json!({"operation": "pause", "pause_mode": "keep"}),
            "pause_capture_admission",
            json!([]),
            json!(false),
            StatusCode::INTERNAL_SERVER_ERROR,
        ),
        (
            json!({"operation": "pause", "pause_mode": "keep"}),
            "suspend_capture",
            json!([]),
            json!(false),
            StatusCode::INTERNAL_SERVER_ERROR,
        ),
        (
            json!({"operation": "update", "pause_mode": "keep", "refit_timeout_s": 120.0}),
            "collective_rpc",
            json!(["update_weights_from_collective", null, [120.0, false], null]),
            json!([true, true]),
            StatusCode::OK,
        ),
        (
            json!({"operation": "update", "pause_mode": "keep", "refit_timeout_s": 120.0}),
            "suspend_capture",
            json!([]),
            json!(false),
            StatusCode::INTERNAL_SERVER_ERROR,
        ),
    ]);
    cases.push((
        json!({"operation": "drain_caches", "pause_mode": "keep"}),
        "collective_rpc",
        json!(["pd_drain_caches", null, [], null]),
        json!([true, true]),
        StatusCode::OK,
    ));
    let engine_cases = cases.clone();
    let (state, engine_task) = test_admin_state_with_ready_and_engine_script(default_ready_response(), move |dealer, push| {
        boxed_test_future(async move {
            for (body, method, args, result, _) in engine_cases {
                let metadata = json!([
                    {"layer_ids": [1], "enabled": true, "max_logprobs": 32, "logprobs_mode": "raw_logprobs", "workers_per_engine": 2},
                    {"layer_ids": [1], "enabled": true, "max_logprobs": 32, "logprobs_mode": "raw_logprobs", "workers_per_engine": 2},
                ]);
                let mut calls = vec![("collective_rpc", json!(["pd_capture_metadata", null, [], null]), metadata)];
                let keep = body["pause_mode"] == "keep";
                if body["operation"] == "pause" && keep && method == "suspend_capture" {
                    calls.push(("pause_capture_admission", json!([]), json!(true)));
                    calls.push(("pause_scheduler", json!(["keep", false]), json!(null)));
                }
                if method == "pause_scheduler" || (method == "collective_rpc" && args[0] == "update_weights_from_collective") {
                    calls.push((if keep { "suspend_capture" } else { "drain_capture" }, json!([]), json!(true)));
                }
                if method == "resume_scheduler" { calls.push(("resume_capture", json!([]), json!(true))); }
                if method == "collective_rpc" && args[0] == "pd_drain_caches" {
                    calls.push((if keep { "suspend_capture" } else { "drain_capture" }, if keep { json!([]) } else { json!([true]) }, json!(true)));
                }
                calls.push((method, args, result));
                for (method, args, result) in calls {
                    let msg = recv_engine_message(dealer).await;
                    let payload = decode_value(&msg[1]).unwrap();
                    let array = payload.as_array().unwrap();
                    assert_eq!(array[2], Value::from(method));
                    assert_eq!(array[3], rmpv::ext::to_value(args).unwrap());
                    send_outputs(push, utility_outputs(array[1].as_u64().unwrap(), utility_result_value(result))).await;
                }
            }
        })
    }).await;
    let mut app = super::super::nemo_rl::router("control-secret").with_state(state);
    for (body, _, _, _, expected) in cases {
        let response = app
            .call(
                Request::builder()
                    .method("POST")
                    .uri("/nemo_rl/refit")
                    .header("content-type", "application/json")
                    .header("authorization", "Bearer control-secret")
                    .body(Body::from(body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), expected);
        if expected == StatusCode::OK {
            let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
            let result: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
            assert_eq!(result["ok"], json!(true));
            let timings = result["timings_s"].as_object().unwrap();
            let phases: &[&str] = match body["operation"].as_str().unwrap() {
                "check_transfers" | "pause_transfers" | "resume_transfers" => {
                    &["pd_capture_metadata", "set_kv_transfer_paused"]
                }
                "prepare" => &["pd_capture_metadata", "pd_prepare_refit"],
                "init" => &["init_collective", "pd_capture_metadata"],
                "drain_caches" if body["pause_mode"] == "keep" => {
                    &["pd_capture_metadata", "pd_drain_caches", "suspend_capture"]
                }
                "drain_caches" => &[
                    "pd_capture_metadata",
                    "pd_drain_caches",
                    "release_capture_transients",
                ],
                "pause" if body["pause_mode"] == "keep" => &[
                    "pause_capture_admission",
                    "pause_scheduler",
                    "pd_capture_metadata",
                    "suspend_capture",
                ],
                "pause" => &["drain_capture", "pause_scheduler", "pd_capture_metadata"],
                "update" => &[
                    "pd_capture_metadata",
                    "suspend_capture",
                    "update_weights_from_collective",
                ],
                "resume" => &["pd_capture_metadata", "resume_capture", "resume_scheduler"],
                unexpected => panic!("unexpected operation {unexpected}"),
            };
            assert_eq!(
                timings.keys().map(String::as_str).collect::<Vec<_>>(),
                phases
            );
            assert!(
                timings.values().all(|v| v.as_f64().is_some_and(|t| t.is_finite() && t >= 0.0))
            );
        }
    }
    engine_task.finish().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn pd_chat_preserves_routes_in_json_and_streaming_with_stop_tail() {
    for streaming in [false, true] {
        let (mut app, engine_task) = test_admin_app_with_engine_script(|dealer, push| {
            boxed_test_future(async move {
                let add = recv_engine_message(dealer).await;
                let request: EngineCoreRequest = rmp_serde::from_slice(&add[1]).unwrap();
                assert_eq!(
                    request.sampling_params.as_ref().unwrap().routed_experts_prompt_start,
                    1
                );
                let prompt_count = request.prompt_token_ids.as_ref().unwrap().len();
                let rows = prompt_count - 1;
                let mut first = request_output(&request.request_id, bytes_to_token_ids(b"a"), None);
                first.routed_experts = Some(Box::new(
                    WireNdArray::from_i32(vec![rows, 1, 2], (0..(rows * 2) as i32).collect())
                        .unwrap(),
                ));
                let mut last = request_output(
                    &request.request_id,
                    bytes_to_token_ids(b"b!x"),
                    Some(EngineCoreFinishReason::Length),
                );
                last.routed_experts = Some(Box::new(
                    WireNdArray::from_i32(
                        vec![3, 1, 2],
                        ((rows * 2) as i32..((rows + 3) * 2) as i32).collect(),
                    )
                    .unwrap(),
                ));
                send_outputs(
                    push,
                    RequestBatchOutputs {
                        outputs: vec![first, last],
                        ..Default::default()
                    }
                    .into(),
                )
                .await;
            })
        })
        .await;
        let response = app
            .call(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        json!({"model": "Qwen/Qwen1.5-0.5B-Chat",
                            "messages": [{"role": "user", "content": "hi"}], "stream": streaming,
                            "return_token_ids": true, "stop": "!", "routed_experts_prompt_start": 1
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        let status = response.status();
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        assert_eq!(status, StatusCode::OK, "{}", String::from_utf8_lossy(&body));
        let responses: Vec<serde_json::Value> = if streaming {
            std::str::from_utf8(&body)
                .unwrap()
                .split("\n\n")
                .filter_map(|frame| frame.strip_prefix("data: "))
                .filter(|data| *data != "[DONE]")
                .map(|data| serde_json::from_str(data).unwrap())
                .collect()
        } else {
            vec![serde_json::from_slice(&body).unwrap()]
        };
        assert!(
            responses.iter().all(|r| r.get("error").is_none()),
            "{responses:?}"
        );
        let prompt_count =
            responses.iter().find_map(|r| r["prompt_token_ids"].as_array()).unwrap().len();
        let tokens: Vec<_> = responses
            .iter()
            .filter_map(|r| r["choices"][0]["token_ids"].as_array())
            .flatten()
            .map(|t| t.as_u64().unwrap())
            .collect();
        assert_eq!(tokens, vec![97, 98, 33]);
        let routes = responses
            .iter()
            .find_map(|r| r["choices"][0]["routed_experts"].as_str())
            .unwrap();
        let npy = STANDARD.decode(routes).unwrap();
        let offset = 10 + usize::from(u16::from_le_bytes([npy[8], npy[9]]));
        assert!(
            std::str::from_utf8(&npy[10..offset])
                .unwrap()
                .contains(&format!("'shape': ({}, 1, 2)", prompt_count + 1))
        );
        assert_eq!(
            &npy[offset..],
            (0i32..((prompt_count + 1) * 2) as i32)
                .flat_map(i32::to_ne_bytes)
                .collect::<Vec<_>>()
        );
        engine_task.finish().await;
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn capture_metrics_scrape_returns_engine_snapshot() {
    let text = "# TYPE nemo_rl_capture_executor_queued gauge\nnemo_rl_capture_executor_queued{engine=\"0\",operation=\"prepare\"} 17\n";
    let (state, engine_task) = test_admin_state_with_ready_and_engine_script(
        default_ready_response(),
        move |dealer, push| {
            boxed_test_future(async move {
                let message = recv_engine_message(dealer).await;
                let payload = decode_value(&message[1]).unwrap();
                let array = payload.as_array().unwrap();
                assert_eq!(array[2], Value::from("get_capture_metrics"));
                assert_eq!(array[3], Value::Array(vec![]));
                send_outputs(
                    push,
                    utility_outputs(
                        array[1].as_u64().unwrap(),
                        utility_result_value(json!(text)),
                    ),
                )
                .await;
            })
        },
    )
    .await;
    let mut app = super::super::nemo_rl::router("control-secret").with_state(state);
    let response = app
        .call(
            Request::builder()
                .uri("/nemo_rl/capture/metrics/0")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response.headers()["content-type"],
        "text/plain; version=0.0.4; charset=utf-8"
    );
    assert_eq!(
        to_bytes(response.into_body(), usize::MAX).await.unwrap(),
        text
    );
    engine_task.finish().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn pending_capture_result_and_release_target_the_owner() {
    let (state, engine_task) =
        test_admin_state_with_ready_and_engine_script(default_ready_response(), |dealer, push| {
            boxed_test_future(async move {
                for (method, result) in [
                    (
                        "resolve_capture",
                        json!({"disposition": "staged", "model_call_id": "call"}),
                    ),
                    ("release_capture", json!(true)),
                ] {
                    let msg = recv_engine_message(dealer).await;
                    let payload = decode_value(&msg[1]).unwrap();
                    let array = payload.as_array().unwrap();
                    assert_eq!(array[2], Value::from(method));
                    assert_eq!(
                        array[3],
                        rmpv::ext::to_value(json!(["opaque-ticket"])).unwrap()
                    );
                    send_outputs(
                        push,
                        utility_outputs(array[1].as_u64().unwrap(), utility_result_value(result)),
                    )
                    .await;
                }
            })
        })
        .await;
    let mut app = super::super::nemo_rl::router("control-secret").with_state(state);
    for method in ["GET", "DELETE"] {
        let response = app
            .call(
                Request::builder()
                    .method(method)
                    .uri("/nemo_rl/capture/result/0/opaque-ticket")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let body: serde_json::Value = serde_json::from_slice(&body).unwrap();
        if method == "GET" {
            assert_eq!(body["disposition"], "staged");
        } else {
            assert_eq!(body, json!({"ok":true}));
        }
    }
    engine_task.finish().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn capture_prompt_too_long_returns_normal_400_and_keeps_serving() {
    let (state, engine_task) =
        test_admin_state_with_ready_and_engine_script(default_ready_response(), |dealer, push| {
            boxed_test_future(async move {
                for (length_error, finish) in [
                    (Some((16, 16)), EngineCoreFinishReason::Error),
                    (Some((16, 17)), EngineCoreFinishReason::Error),
                    (None, EngineCoreFinishReason::Error),
                    (None, EngineCoreFinishReason::Stop),
                ] {
                    let add = recv_engine_message(dealer).await;
                    assert_eq!(add[0].as_ref(), &[0]);
                    let request: EngineCoreRequest = rmp_serde::from_slice(&add[1]).unwrap();
                    let mut output = request_output(&request.request_id, vec![], Some(finish));
                    output.prompt_length_error = length_error;
                    send_outputs(
                        push,
                        RequestBatchOutputs {
                            outputs: vec![output],
                            finished_requests: Some(BTreeSet::from([request.request_id])),
                            ..Default::default()
                        }
                        .into(),
                    )
                    .await;
                }
            })
        })
        .await;
    let mut state = Arc::try_unwrap(state).ok().unwrap();
    state.chat = state.chat.with_prefix_token_alignment(true);
    let mut app = build_router(Arc::new(state));
    for (index, expected_status) in [
        StatusCode::BAD_REQUEST,
        StatusCode::BAD_REQUEST,
        StatusCode::INTERNAL_SERVER_ERROR,
        StatusCode::OK,
    ]
    .into_iter()
    .enumerate()
    {
        let mut body = json!({
            "model": "Qwen/Qwen1.5-0.5B-Chat",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 1,
        });
        if index < 3 {
            body["nemo_rl_capture"] = json!({
                "version": 1, "admission": {"prev_len": 0},
                "weight_version": 0, "phase": "prefill",
                "top_logprobs": 0, "routed_experts": false,
            });
        }
        let response = app
            .call(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        let status = response.status();
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        assert_eq!(
            status,
            expected_status,
            "{}",
            String::from_utf8_lossy(&body)
        );
        if index < 2 {
            let expected = crate::error::chat_submit_error(
                "failed to submit chat request",
                vllm_chat::Error::Text(vllm_text::Error::PromptTooLong {
                    max_model_len: 16,
                    prompt_len: 16 + index as u32,
                }),
            )
            .to_error_response();
            assert_eq!(
                serde_json::from_slice::<serde_json::Value>(&body).unwrap(),
                serde_json::to_value(expected).unwrap()
            );
        }
    }
    engine_task.await.unwrap();
}
