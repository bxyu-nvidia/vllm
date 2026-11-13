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
    let engine_cases = cases.clone();
    let (state, engine_task) = test_admin_state_with_ready_and_engine_script(default_ready_response(), move |dealer, push| {
        boxed_test_future(async move {
            for (_, method, args, result, _) in engine_cases {
                let metadata = json!([
                    {"layer_ids": [1], "enabled": true, "max_logprobs": 32, "logprobs_mode": "raw_logprobs", "workers_per_engine": 2},
                    {"layer_ids": [1], "enabled": true, "max_logprobs": 32, "logprobs_mode": "raw_logprobs", "workers_per_engine": 2},
                ]);
                for (method, args, result) in [
                    ("collective_rpc", json!(["pd_capture_metadata", null, [], null]), metadata),
                    (method, args, result),
                ] {
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
    }
    engine_task.finish().await;
}
