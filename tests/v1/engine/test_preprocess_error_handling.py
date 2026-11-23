# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import queue
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import msgspec
import pytest
import torch.cuda

from vllm import LLM, SamplingParams
from vllm.platforms import current_platform
from vllm.v1.engine import EngineCoreOutput, EngineCoreRequest, FinishReason
from vllm.v1.engine.capture_runtime import EngineCapture
from vllm.v1.engine.core import EngineCore, EngineCoreProc

MODEL_NAME = "hmellor/tiny-random-LlamaForCausalLM"


def test_preprocess_error_handling(monkeypatch: pytest.MonkeyPatch):
    """Test that preprocessing errors are handled gracefully."""
    if current_platform.is_rocm() or current_platform.is_xpu():
        pytest.skip(
            "Skipped on ROCm/XPU: this test only works with 'fork', "
            "but ROCm/XPU uses 'spawn'."
        )

    assert not torch.cuda.is_initialized(), (
        "fork needs to be used for the engine "
        "core process and this isn't possible if cuda is already initialized"
    )

    # Store original method to call for non-failing requests
    original_preprocess = EngineCore.preprocess_add_request

    # Monkeypatch to make preprocess_add_request raise an exception
    # only for requests with "FAIL" in the first token
    def conditional_failing_preprocess(self, request: EngineCoreRequest):
        # Fail if the first token id is 333
        if request.prompt_token_ids and request.prompt_token_ids[0] == 333:
            raise ValueError("Simulated preprocessing error!")
        return original_preprocess(self, request)

    monkeypatch.setattr(
        EngineCore, "preprocess_add_request", conditional_failing_preprocess
    )

    llm = LLM(model=MODEL_NAME)

    # Create a failing request by crafting a request with an invalid token
    # We need to use a direct approach since LLM.generate tokenizes for us
    from vllm.inputs import TokensPrompt

    # This should raise an exception due to the preprocessing failure
    # Special token id to trigger the failure
    failing_prompt = TokensPrompt(prompt_token_ids=[333])
    outputs = llm.generate(failing_prompt, SamplingParams(max_tokens=10))  # type: ignore
    assert len(outputs) == 1
    assert len(outputs[0].outputs[0].token_ids) == 0
    assert outputs[0].finished
    assert outputs[0].outputs[0].finish_reason == "error"

    # Verify the engine is still functional with a normal request
    outputs = llm.generate("Hello, my name is", SamplingParams(max_tokens=10))
    assert len(outputs) == 1
    assert len(outputs[0].outputs[0].token_ids) > 0
    assert outputs[0].outputs[0].finish_reason in ("stop", "length")


@pytest.mark.parametrize("aligned_len", [4, 5, None])
def test_capture_prompt_rejection_cleans_up_and_keeps_engine_usable(aligned_len):
    """Reject aligned overflow without scheduling it or poisoning capture."""
    runtime = EngineCapture()
    engine = EngineCoreProc.__new__(EngineCoreProc)
    engine.capture_runtime = runtime
    engine.engine_index = 2
    engine.output_queue = queue.Queue()
    engine.input_queue = queue.Queue()
    engine.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=4, logprobs_mode="raw_logprobs")
    )
    engine.preprocess_add_request = Mock(return_value=("prepared", 0))

    async def align(request):
        if request.request_id == "good":
            request.prompt_token_ids = [1, 2, 3]
        elif aligned_len is None:
            raise RuntimeError("storage unavailable")
        else:
            request.prompt_token_ids = [1] * aligned_len

    adapter = SimpleNamespace(
        prepare=AsyncMock(side_effect=align), discard=Mock(), close=AsyncMock()
    )
    runtime.adapter = adapter
    request = EngineCoreRequest(
        request_id="bad",
        prompt_token_ids=[1],
        mm_features=None,
        sampling_params=SamplingParams(
            max_tokens=3,
            extra_args={
                "nemo_rl_capture": {
                    "version": 1,
                    "admission": {"prev_len": 0},
                    "weight_version": 0,
                    "phase": "prefill",
                    "top_logprobs": 0,
                    "routed_experts": False,
                }
            },
        ),
        pooling_params=None,
        arrival_time=0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
        client_index=3,
    )
    try:
        engine._prepare_capture_request(request)
        client, batch = engine.output_queue.get(timeout=5)
        # Wait for the callback's finally block, without a timing-based sleep.
        asyncio.run_coroutine_threadsafe(asyncio.sleep(0), runtime.loop).result(5)
        assert client == 3
        assert set(batch.finished_requests) == {"bad"}
        output = batch.outputs[0]
        assert output.finish_reason == FinishReason.ERROR
        expected = None if aligned_len is None else (4, aligned_len)
        assert output.prompt_length_error == expected
        assert (
            msgspec.msgpack.decode(
                msgspec.msgpack.encode(output), type=EngineCoreOutput
            ).prompt_length_error
            == expected
        )
        engine.preprocess_add_request.assert_not_called()
        assert engine.input_queue.empty()
        adapter.discard.assert_called_once_with("bad")
        assert not runtime._active and not runtime._preparing
        assert not runtime._metadata and not runtime.pending
        assert runtime.failure is None

        valid = msgspec.structs.replace(request, request_id="good")
        engine._prepare_capture_request(valid)
        _, prepared = engine.input_queue.get(timeout=5)
        assert prepared == ("prepared", 0)
        assert valid.sampling_params.max_tokens == 1
    finally:
        runtime.close()
