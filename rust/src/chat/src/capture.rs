// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Coordination metadata for engine-local training capture.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::{Result, bail_invalid_prefix_alignment, invalid_prefix_alignment};
use crate::{ChatRequest, GenerationPromptMode};

pub const CAPTURE_CONTEXT_KEY: &str = "nemo_rl_capture";

/// The router assigns each leg of a logical model call its role.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CapturePhase {
    Prefill,
    Decode,
}

/// Boundary in the final `EngineCoreRequest.prompt_token_ids` array.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CaptureAlignment {
    /// First token after the completed assistant turn, excluding its terminator.
    pub suffix_start: usize,
    /// Template terminator to insert if the required parent ended without EOS.
    pub turn_end_token_id: u32,
    /// All valid EOS IDs, including alternatives emitted by the model.
    pub eos_token_ids: Vec<u32>,
}

/// Small request context; training arrays are carried separately by the engine.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CaptureContext {
    pub version: u32,
    /// Gym owns this schema. The capture adapter validates the full admission.
    pub admission: BTreeMap<String, Value>,
    pub weight_version: u64,
    pub phase: CapturePhase,
    pub top_logprobs: u32,
    pub routed_experts: bool,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub engine_capture: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prefill_key: Option<String>,
    /// Computed by the frontend, never accepted from the HTTP caller.
    #[serde(skip_deserializing, skip_serializing_if = "Option::is_none")]
    pub alignment: Option<CaptureAlignment>,
}

pub(crate) fn take_context(
    request: &mut ChatRequest,
    enabled: bool,
) -> Result<Option<CaptureContext>> {
    let Some(value) = request
        .sampling_params
        .vllm_xargs
        .as_mut()
        .and_then(|args| args.remove(CAPTURE_CONTEXT_KEY))
    else {
        return Ok(None);
    };
    let context: CaptureContext = serde_json::from_value(value)
        .map_err(|error| invalid_prefix_alignment!("invalid capture context: {error}"))?;
    if context.version != 1 {
        bail_invalid_prefix_alignment!("unsupported capture context version");
    }
    if !enabled {
        bail_invalid_prefix_alignment!("capture context requires --enable-prefix-token-alignment");
    }
    if request.has_multimodal()
        || request.prompt_truncation.is_some()
        || request.chat_options.generation_prompt_mode
            == GenerationPromptMode::ContinueFinalAssistant
    {
        bail_invalid_prefix_alignment!(
            "capture context requires untruncated text with a completed assistant turn"
        );
    }
    let prev_len = context.admission.get("prev_len").and_then(Value::as_u64).ok_or_else(|| {
        invalid_prefix_alignment!("capture admission requires nonnegative prev_len")
    })?;
    let prefix_len = request.required_prefix_token_ids.as_ref().map_or(0, Vec::len);
    if context.engine_capture && prefix_len != 0 {
        bail_invalid_prefix_alignment!(
            "engine capture resolves the parent; omit required_prefix_token_ids"
        );
    }
    if !context.engine_capture && prev_len != prefix_len as u64 {
        bail_invalid_prefix_alignment!(
            "capture parent must be resolved into required_prefix_token_ids before engine dispatch"
        );
    }
    Ok(Some(context))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn request(context: Value) -> ChatRequest {
        let mut request = ChatRequest::for_test();
        request.sampling_params.vllm_xargs = Some([(CAPTURE_CONTEXT_KEY.into(), context)].into());
        request
    }

    #[test]
    fn rejects_unsupported_or_unresolved_capture_before_engine_dispatch() {
        let valid = json!({"version":1,"admission":{"prev_len":0},"weight_version":7,"phase":"decode","top_logprobs":0,"routed_experts":false});
        assert!(take_context(&mut request(valid.clone()), false).is_err());
        for (key, value) in [
            ("version", json!(2)),
            ("weight_version", json!(-1)),
            ("weight_version", json!(true)),
            ("phase", json!("unknown")),
            ("admission", json!({"prev_len":1})),
            ("admission", json!({"prev_len":true})),
            ("alignment", json!({"suffix_start":0})),
        ] {
            let mut invalid = valid.clone();
            invalid[key] = value;
            assert!(take_context(&mut request(invalid), true).is_err(), "{key}");
        }
        let mut deferred = valid.clone();
        deferred["engine_capture"] = json!(true);
        deferred["admission"] = json!({"prev_len": 5});
        let mut child = request(deferred);
        assert!(take_context(&mut child, true).unwrap().unwrap().engine_capture);
        let context = take_context(&mut request(valid), true).unwrap().unwrap();
        assert_eq!(context.weight_version, 7);
        assert_eq!(context.alignment, None);
    }
}
