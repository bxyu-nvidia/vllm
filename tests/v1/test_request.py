# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import msgspec
import pytest

from vllm import SamplingParams
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.kv_hints import KvHintAction, KvHintsEnvelope
from vllm.v1.request import Request, RequestStatus


def test_request_status_fmt_str():
    """Test that the string representation of RequestStatus is correct."""
    assert f"{RequestStatus.WAITING}" == "WAITING"
    assert (
        f"{RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR}"
        == "WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR"
    )
    assert f"{RequestStatus.WAITING_FOR_REMOTE_KVS}" == "WAITING_FOR_REMOTE_KVS"
    assert f"{RequestStatus.WAITING_FOR_STREAMING_REQ}" == "WAITING_FOR_STREAMING_REQ"
    assert f"{RequestStatus.RUNNING}" == "RUNNING"
    assert f"{RequestStatus.PREEMPTED}" == "PREEMPTED"
    assert f"{RequestStatus.FINISHED_STOPPED}" == "FINISHED_STOPPED"
    assert f"{RequestStatus.FINISHED_LENGTH_CAPPED}" == "FINISHED_LENGTH_CAPPED"
    assert f"{RequestStatus.FINISHED_ABORTED}" == "FINISHED_ABORTED"
    assert f"{RequestStatus.FINISHED_IGNORED}" == "FINISHED_IGNORED"


def test_request_copies_session_id_from_engine_core_request():
    """Test that Request preserves session ID from EngineCoreRequest."""
    engine_request = EngineCoreRequest(
        request_id="request-1",
        prompt_token_ids=[1, 2, 3],
        mm_features=None,
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
        session_id="session-1",
    )

    request = Request.from_engine_core_request(engine_request, block_hasher=None)

    assert request.session_id == "session-1"


def test_request_copies_kv_hints_from_engine_core_request():
    """Test that Request preserves KV hints from EngineCoreRequest."""
    kv_hints = KvHintsEnvelope(
        protocol_version="0.1",
        message_id="msg-1",
        actions=[
            KvHintAction(
                action_id="action-1",
                action_type="example.action",
                action_version="1.0",
                payload={},
            )
        ],
    )
    engine_request = EngineCoreRequest(
        request_id="request-1",
        prompt_token_ids=[1, 2, 3],
        mm_features=None,
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
        kv_hints=kv_hints,
    )

    request = Request.from_engine_core_request(engine_request, block_hasher=None)

    assert request.kv_hints == kv_hints


@pytest.mark.parametrize("phase", ["prefill", "decode"])
@pytest.mark.parametrize("has_parent", [False, True])
def test_capture_context_survives_engine_wire_and_scheduler_conversion(
    phase, has_parent
):
    from vllm.v1.engine.capture import get_capture_context

    context = {
        "version": 1,
        "admission": {
            "model_call_id": "call-2",
            "prev_len": 2 if has_parent else 0,
            "staging_chain": ["parent-key"] if has_parent else [],
        },
        "weight_version": 7,
        "phase": phase,
        "top_logprobs": 8,
        "routed_experts": True,
    }
    if has_parent:
        context["alignment"] = {
            "suffix_start": 2,
            "turn_end_token_id": 2,
            "eos_token_ids": [2, 3],
        }
    request = EngineCoreRequest(
        request_id="request-1",
        prompt_token_ids=[10, 3, 40],
        mm_features=None,
        sampling_params=SamplingParams(
            max_tokens=1,
            extra_args={
                "nemo_rl_capture": context,
                "kv_transfer_params": {"remote_num_tokens": 2},
            },
        ),
        pooling_params=None,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )
    restored = msgspec.msgpack.decode(
        msgspec.msgpack.encode(request), type=EngineCoreRequest
    )
    capture = restored.capture_context
    assert capture is not None
    assert capture.weight_version == 7
    assert capture.phase == phase
    assert capture.admission == context["admission"]
    assert capture.top_logprobs == 8
    assert capture.routed_experts is True
    if has_parent:
        assert capture.alignment.suffix_start == 2
    else:
        assert capture.alignment is None
    scheduled = Request.from_engine_core_request(restored, block_hasher=None)
    assert (
        get_capture_context(
            scheduled.sampling_params.extra_args, scheduled.prompt_token_ids
        )
        == capture
    )
    assert scheduled.kv_transfer_params == {"remote_num_tokens": 2}


@pytest.mark.parametrize(
    "key,value",
    [
        ("version", 2),
        ("weight_version", -1),
        ("weight_version", True),
        ("phase", "unknown"),
        ("top_logprobs", -1),
        ("admission", {"prev_len": True}),
        ("admission", {"prev_len": 1}),
        (
            "alignment",
            {"suffix_start": 0, "turn_end_token_id": 2, "eos_token_ids": [2]},
        ),
    ],
)
def test_capture_context_rejects_invalid_metadata(key, value):
    from vllm.v1.engine.capture import get_capture_context

    context = {
        "version": 1,
        "admission": {"prev_len": 0},
        "weight_version": 7,
        "phase": "decode",
        "top_logprobs": 0,
        "routed_experts": False,
    }
    context[key] = value
    with pytest.raises((ValueError, msgspec.ValidationError)):
        get_capture_context({"nemo_rl_capture": context}, [10, 2, 40])


def test_requests_without_capture_have_no_context():
    from vllm.v1.engine.capture import get_capture_context

    assert get_capture_context(None, None) is None
    assert get_capture_context({"kv_transfer_params": {}}, [1]) is None


@pytest.mark.parametrize(
    "boundary",
    [
        {"suffix_start": 0, "turn_end_token_id": 2, "eos_token_ids": [2]},
        {"suffix_start": 4, "turn_end_token_id": 2, "eos_token_ids": [2]},
        {"suffix_start": 2, "turn_end_token_id": 2, "eos_token_ids": []},
        {"suffix_start": 2, "turn_end_token_id": 3, "eos_token_ids": [2]},
        {"suffix_start": 1, "turn_end_token_id": 2, "eos_token_ids": [2]},
    ],
)
def test_capture_rejects_boundaries_outside_prompt_or_without_terminator(boundary):
    from vllm.v1.engine.capture import get_capture_context

    context = {
        "version": 1,
        "admission": {"prev_len": 2},
        "weight_version": 7,
        "phase": "decode",
        "top_logprobs": 0,
        "routed_experts": False,
        "alignment": boundary,
    }
    with pytest.raises(ValueError, match="alignment boundary"):
        get_capture_context({"nemo_rl_capture": context}, [10, 2, 40])
