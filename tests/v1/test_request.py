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


@pytest.mark.parametrize("prefix", [[10, 3], [10, 90]])
def test_engine_capture_aligns_parent_before_scheduler_hashing(prefix):
    from vllm.v1.engine.capture import align_capture_prompt, get_capture_context

    prompt = [100, 2, 40]
    raw = {
        "version": 1,
        "admission": {"prev_len": 2},
        "weight_version": 7,
        "phase": "prefill",
        "top_logprobs": 0,
        "routed_experts": False,
        "alignment": {
            "suffix_start": 2,
            "turn_end_token_id": 2,
            "eos_token_ids": [2, 3],
        },
    }
    context = get_capture_context({"nemo_rl_capture": raw}, prompt)
    assert context is not None
    expected = prefix + ([] if prefix[-1] == 3 else [2]) + [40]
    assert align_capture_prompt(context, prompt, prefix) == expected
    assert prompt == [100, 2, 40]
    with pytest.raises(ValueError, match="length"):
        align_capture_prompt(context, prompt, [10])


def test_engine_capture_lookup_does_not_block_other_requests_and_abort_cannot_add():
    import asyncio
    import threading
    from types import SimpleNamespace

    from vllm.v1.engine.capture_runtime import EngineCapture

    entered = threading.Event()
    added = threading.Event()
    discarded = threading.Event()
    failures = []

    class Adapter:
        async def prepare(self, request):
            if request.request_id == "slow":
                entered.set()
                await asyncio.Event().wait()

        def discard(self, request_id):
            assert request_id == "slow"
            discarded.set()

        def render_metrics(self):
            return "capture_in_flight 1\n"

        async def close(self):
            pass

    runtime = EngineCapture()
    runtime.adapter = Adapter()
    try:
        runtime.prepare(
            SimpleNamespace(
                request_id="slow",
                capture_context=SimpleNamespace(phase="decode", prefill_key="p"),
                prompt_token_ids=[1],
            ),
            lambda: failures.append("late ADD"),
            lambda: failures.append("error"),
        )
        assert entered.wait(5)
        # Metrics remain readable while admission is stalled, without draining it.
        assert runtime.render_metrics() == "capture_in_flight 1\n"
        runtime.prepare(
            SimpleNamespace(
                request_id="fast",
                capture_context=SimpleNamespace(phase="decode", prefill_key="p"),
                prompt_token_ids=[1],
            ),
            added.set,
            lambda: failures.append("error"),
        )
        assert added.wait(5)
        runtime.abort_pending(["slow"])
        assert discarded.wait(5)
    finally:
        runtime.close()
    assert not failures


def test_engine_capture_utility_waits_for_storage_ack():
    import asyncio
    import threading

    from vllm.v1.engine.capture_runtime import EngineCapture

    entered = threading.Event()

    class Adapter:
        release = None

        async def finalize(self, request_id, served_tokens, prefill_key):
            self.release = asyncio.Event()
            entered.set()
            await self.release.wait()
            return {"disposition": "staged", "served_tokens": served_tokens}

        async def drain(self, *, release_transients=False):
            pass

        async def close(self):
            pass

    adapter = Adapter()
    runtime = EngineCapture()
    runtime.adapter = adapter
    try:
        future = runtime.finalize("request", 2, None)
        assert entered.wait(5)
        assert not future.done()
        assert adapter.release is not None
        runtime.loop.call_soon_threadsafe(adapter.release.set)
        assert future.result(5) == {"disposition": "staged", "served_tokens": 2}
    finally:
        runtime.close()


def test_engine_capture_prepares_exact_prompt_before_block_hashing():
    import queue
    from types import SimpleNamespace

    from vllm.v1.engine import EngineCoreRequestType
    from vllm.v1.engine.capture_runtime import EngineCapture
    from vllm.v1.engine.core import EngineCoreProc

    class Adapter:
        async def prepare(self, request):
            request.prompt_token_ids = [10, 90, 2, 40]

        async def close(self):
            pass

    engine = EngineCoreProc.__new__(EngineCoreProc)
    engine.capture_runtime = EngineCapture()
    engine.capture_runtime.adapter = Adapter()
    engine.mm_receiver_cache = None
    engine.input_queue = queue.Queue()
    engine.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=8, logprobs_mode="raw_logprobs"),
        aux_output_config=SimpleNamespace(use_capture_handoff=False),
    )
    seen = []

    def hash_prompt(request):
        seen.append(list(request.prompt_token_ids))
        return []

    engine.request_block_hasher = hash_prompt
    request = EngineCoreRequest(
        request_id="capture",
        prompt_token_ids=[10, 2, 40],
        mm_features=None,
        sampling_params=SamplingParams(
            max_tokens=7,
            extra_args={
                "nemo_rl_capture": {
                    "version": 1,
                    "admission": {"prev_len": 0},
                    "weight_version": 0,
                    "phase": "decode",
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
    )
    try:
        engine._prepare_capture_request(request)
        kind, (scheduled, wave) = engine.input_queue.get(timeout=5)
        assert kind == EngineCoreRequestType.ADD
        assert scheduled.prompt_token_ids == [10, 90, 2, 40]
        assert scheduled.max_tokens == 4
        assert seen == [[10, 90, 2, 40]]
    finally:
        engine.capture_runtime.close()


def test_engine_capture_failure_still_allows_discard_and_shutdown():
    from vllm.v1.engine.capture_runtime import EngineCapture

    discarded = []

    class Adapter:
        def discard(self, request_id):
            discarded.append(request_id)

        async def close(self):
            raise OSError("storage unavailable")

    runtime = EngineCapture()
    runtime.adapter = Adapter()
    runtime.failure = ValueError("invalid capture output")
    try:
        with pytest.raises(RuntimeError, match="engine capture failed"):
            runtime.finalize("request", 1, None).result(5)
        runtime.discard("request").result(5)
        assert discarded == ["request"]
    finally:
        runtime.close()
    assert not runtime.thread.is_alive()


@pytest.mark.parametrize("cancel_caller", [False, True])
def test_engine_capture_drain_waits_for_write_and_blocks_new_admission(cancel_caller):
    import asyncio
    import threading
    from types import SimpleNamespace

    from vllm.v1.engine.capture_runtime import EngineCapture

    entered = threading.Event()
    release = None

    class Adapter:
        async def finalize(self, request_id, served_tokens, prefill_key):
            nonlocal release
            release = asyncio.Event()
            entered.set()
            await release.wait()
            return {"disposition": "staged"}

        async def drain(self, *, release_transients=False):
            pass

        async def close(self):
            pass

    runtime = EngineCapture()
    runtime.adapter = Adapter()
    runtime._active.add("request")
    try:
        write = runtime.finalize("request", 1, "prefill")
        assert entered.wait(5)
        if cancel_caller:
            assert write.cancel()
        drain = runtime.drain()
        with pytest.raises(RuntimeError, match="paused"):
            runtime.prepare(
                SimpleNamespace(request_id="child"), lambda: None, lambda: None
            )
        assert not drain.done()
        assert release is not None
        runtime.loop.call_soon_threadsafe(release.set)
        if not cancel_caller:
            assert write.result(5) == {"disposition": "staged"}
        assert drain.result(5) is True
        runtime.resume()
    finally:
        runtime.close()


def test_engine_capture_wire_output_omits_training_arrays_without_mutation():
    import numpy as np

    from vllm.v1.engine import EngineCoreOutput
    from vllm.v1.engine.capture_runtime import EngineCapture
    from vllm.v1.outputs import LogprobsLists

    runtime = EngineCapture()
    runtime._metadata["r"] = {"request_id": "r", "prompt_tokens": 100}
    output = EngineCoreOutput(
        "r",
        [3],
        new_logprobs=LogprobsLists(np.array([[3]]), np.array([[-0.5]]), np.array([1])),
        routed_experts=np.zeros((1, 1, 1)),
        kv_transfer_params={"remote_num_tokens": 80},
    )
    try:
        wire = runtime.wire_output(output, 2)
        assert wire.new_logprobs is None and wire.routed_experts is None
        assert output.new_logprobs is not None and output.routed_experts is not None
        assert "__nemo_rl_capture" not in output.kv_transfer_params
        assert wire.kv_transfer_params["__nemo_rl_capture"]["engine_index"] == 2
        assert wire.kv_transfer_params["remote_num_tokens"] == 80
    finally:
        runtime.close()


def test_engine_capture_finalization_follows_queued_outputs():
    import queue
    from concurrent.futures import Future
    from types import SimpleNamespace

    from vllm.v1.engine.core import EngineCoreProc

    engine = EngineCoreProc.__new__(EngineCoreProc)
    engine.output_queue = queue.Queue()
    events = []
    stored: Future[dict[str, str]] = Future()

    def finalize(request_id, served_tokens, prefill_key):
        events.append(("finalize", request_id, served_tokens, prefill_key))
        return stored

    engine.capture_runtime = SimpleNamespace(
        finish=lambda request_id: events.append(("finish", request_id)),
        finalize=finalize,
    )
    engine.abort_requests = lambda ids: events.append(("abort", ids))
    engine.output_queue.put(lambda: events.append(("output", "r")))
    result = engine.finalize_capture("r", 2, "prefill")
    assert events == [("abort", ["r"])]
    assert not result.done()
    engine.output_queue.get_nowait()()
    engine.output_queue.get_nowait()()
    assert events == [
        ("abort", ["r"]),
        ("output", "r"),
        ("finish", "r"),
        ("finalize", "r", 2, "prefill"),
    ]
    assert not result.done()
    stored.set_result({"disposition": "staged"})
    assert result.result(5) == {"disposition": "staged"}


@pytest.mark.parametrize("fail_output", [False, True])
def test_engine_capture_batch_preserves_outputs_before_finish_and_finalize(fail_output):
    from unittest.mock import patch

    from vllm.v1.engine import EngineCoreOutput
    from vllm.v1.engine.capture_runtime import EngineCapture

    events = []

    class Adapter:
        def observe(self, output):
            if fail_output:
                raise ValueError("bad training output")
            events.append((output.request_id, output.new_token_ids))

        def finish(self, request_id):
            events.append(("finish", request_id))

        async def finalize(self, request_id, served_tokens, prefill_key):
            events.append(("finalize", request_id))
            return {"disposition": "staged"}

        async def close(self):
            pass

    runtime = EngineCapture()
    runtime.adapter = Adapter()
    runtime._active.update(str(i) for i in range(2048))
    outputs = [EngineCoreOutput(str(i), [i]) for i in range(2048)]
    outputs.insert(1, EngineCoreOutput("uncaptured", [99]))
    try:
        with patch.object(
            runtime.loop,
            "call_soon_threadsafe",
            wraps=runtime.loop.call_soon_threadsafe,
        ) as schedule:
            runtime.observe_batch(outputs)
            assert schedule.call_count == 1
            runtime.observe_batch([EngineCoreOutput("uncaptured", [100])])
            assert schedule.call_count == 1
        # The caller may replace its list with wire copies immediately.
        outputs[:] = [EngineCoreOutput("wire", [])]
        runtime.finish("0")
        result = runtime.finalize("0", 1, None)
        if fail_output:
            with pytest.raises(RuntimeError, match="engine capture failed"):
                result.result(5)
            assert not events
        else:
            assert result.result(5) == {"disposition": "staged"}
            assert events == [(str(i), [i]) for i in range(2048)] + [
                ("finish", "0"),
                ("finalize", "0"),
            ]
    finally:
        runtime.close()


@pytest.mark.parametrize("fail_write", [False, True])
@pytest.mark.parametrize("suspend", [False, True])
def test_pending_capture_returns_before_cpu_work_and_preserves_completion_barriers(
    fail_write,
    suspend,
):
    import asyncio
    import threading

    from vllm.v1.engine.capture_runtime import EngineCapture

    entered = threading.Event()
    release = None

    class Adapter:
        async def finalize(self, request_id, served_tokens, prefill_key):
            nonlocal release
            release = asyncio.Event()
            entered.set()
            await release.wait()
            if fail_write:
                raise OSError("TQ rejected capture")
            return {"disposition": "staged", "served_tokens": served_tokens}

        async def suspend(self):
            pass

        async def drain(self, *, release_transients=False):
            pass

        async def close(self):
            pass

    runtime = EngineCapture()
    runtime.adapter = Adapter()
    runtime.result_url = "http://owner/nemo_rl/capture/result/3"
    runtime._active.add("request")
    runtime._metadata["request"] = {
        "phase": "decode",
        "rollout_id": "rollout",
        "model_call_id": "call",
    }
    try:
        receipt = runtime.finalize("request", 2, "prefill").result(5)
        assert receipt["disposition"] == "pending"
        assert receipt["rollout_id"] == "rollout"
        assert entered.wait(5)
        assert release is not None
        ticket = receipt["resolve_url"].rsplit("/", 1)[1]
        cancelled_reader = runtime.resolve_result(ticket)
        assert cancelled_reader.cancel()
        reader = runtime.resolve_result(ticket)
        drain = runtime.suspend() if suspend else runtime.drain()
        assert not reader.done() and not drain.done()
        with pytest.raises(ValueError, match="pending capture"):
            runtime.release_result(ticket).result(5)
        runtime.loop.call_soon_threadsafe(release.set)
        if fail_write:
            with pytest.raises(OSError, match="TQ rejected"):
                reader.result(5)
            with pytest.raises(OSError, match="TQ rejected"):
                drain.result(5)
            with pytest.raises(RuntimeError, match="failed capture"):
                runtime.resume()
        else:
            expected = {"disposition": "staged", "served_tokens": 2}
            assert reader.result(5) == expected
            assert runtime.resolve_result(ticket).result(5) == expected
            assert drain.result(5) is True
        assert runtime.release_result(ticket).result(5) is True
        assert runtime.release_result(ticket).result(5) is True
        with pytest.raises(KeyError):
            runtime.resolve_result(ticket).result(5)
    finally:
        runtime.close()


@pytest.mark.parametrize("phase", ["prefill", "decode"])
@pytest.mark.parametrize("remote", [0, 2, 4])
def test_capture_handoff_checks_prepared_route_boundary(phase, remote):
    from types import SimpleNamespace

    from vllm.v1.engine.capture import CaptureContext, validate_capture_handoff

    context = CaptureContext(
        version=1,
        admission={"prev_len": 2},
        weight_version=0,
        phase=phase,
        top_logprobs=0,
        routed_experts=True,
        prefill_key="prefill-key" if phase == "decode" else None,
    )
    transfer = {
        "do_remote_prefill": phase == "decode",
        "do_remote_decode": phase == "prefill",
        "remote_num_tokens": remote,
    }
    params = SamplingParams(
        routed_experts_prompt_start=max(1, min(remote, 3)) if phase == "decode" else 1,
        extra_args={
            "kv_transfer_params": transfer,
            "nemo_rl_capture": msgspec.to_builtins(context),
        },
    )
    request = SimpleNamespace(sampling_params=params, prompt_token_ids=[1, 2, 3, 4])
    validate_capture_handoff(request)
    params.routed_experts_prompt_start += 1
    with pytest.raises(ValueError, match="handoff boundary"):
        validate_capture_handoff(request)


@pytest.mark.parametrize(
    "missing",
    ["context", "routes", "prefill_key", "remote_num_tokens", "do_remote_prefill"],
)
def test_capture_handoff_rejects_incomplete_decode_metadata(missing):
    from types import SimpleNamespace

    from vllm.v1.engine.capture import CaptureContext, validate_capture_handoff

    context = CaptureContext(
        version=1,
        admission={"prev_len": 0},
        weight_version=0,
        phase="decode",
        top_logprobs=0,
        routed_experts=True,
        prefill_key="key",
    )
    transfer = {"do_remote_prefill": True, "remote_num_tokens": 2}
    if missing == "context":
        context = None
    elif missing == "routes":
        context.routed_experts = False
    elif missing == "prefill_key":
        context.prefill_key = None
    else:
        del transfer[missing]
    request = SimpleNamespace(
        prompt_token_ids=[1, 2, 3],
        sampling_params=SamplingParams(
            routed_experts_prompt_start=2,
            extra_args={
                "kv_transfer_params": transfer,
                "nemo_rl_capture": msgspec.to_builtins(context),
            },
        ),
    )
    with pytest.raises(ValueError, match="requires"):
        validate_capture_handoff(request)


def test_engine_rejects_handoff_without_configured_adapter_before_scheduling():
    from types import SimpleNamespace

    from vllm.v1.engine.core import EngineCore

    engine = object.__new__(EngineCore)
    engine.vllm_config = SimpleNamespace(
        aux_output_config=SimpleNamespace(use_capture_handoff=True)
    )
    engine.capture_runtime = None
    with pytest.raises(RuntimeError, match="configured capture adapter"):
        engine.preprocess_add_request(None)
    engine.capture_runtime = object()
    with pytest.raises(ValueError, match="managed routed-experts"):
        engine.preprocess_add_request(
            SimpleNamespace(
                capture_context=None, sampling_params=None, prompt_token_ids=[1]
            )
        )


@pytest.mark.parametrize("backoff", [1, 4])
def test_nixl_prefill_truncation_preserves_capture_prompt(backoff):
    from types import SimpleNamespace

    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_scheduler import (
        NixlBaseConnectorScheduler,
    )

    prompt = list(range(11))
    core_request = EngineCoreRequest(
        request_id="prefill",
        prompt_token_ids=prompt,
        mm_features=None,
        sampling_params=SamplingParams(
            max_tokens=1, extra_args={"kv_transfer_params": {"do_remote_decode": True}}
        ),
        pooling_params=None,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )
    scheduled = Request.from_engine_core_request(core_request, block_hasher=None)
    scheduler = SimpleNamespace(_prefill_backoff=lambda: backoff)
    NixlBaseConnectorScheduler._truncate_request_for_prefill(scheduler, scheduled)
    assert scheduled.prompt_token_ids == list(range(11 - backoff))
    assert scheduled.num_prompt_tokens == 11 - backoff
    assert list(scheduled.all_token_ids) == scheduled.prompt_token_ids
    assert core_request.prompt_token_ids == list(range(11))
    assert prompt == list(range(11))
    # Re-admission must not truncate it twice.
    NixlBaseConnectorScheduler._truncate_request_for_prefill(scheduler, scheduled)
    assert scheduled.num_prompt_tokens == 11 - backoff


@pytest.mark.parametrize("fail_cleanup", [False, True])
def test_capture_drain_awaits_actor_cleanup_and_preserves_pending_receipts(
    fail_cleanup,
):
    import asyncio
    import threading

    from vllm.v1.engine.capture_runtime import EngineCapture

    entered, release = threading.Event(), threading.Event()

    class Adapter:
        async def finalize(self, *args):
            return {"disposition": "staged", "staging_key": "record"}

        async def drain(self, *, release_transients=False):
            entered.set()
            await asyncio.to_thread(release.wait)
            if fail_cleanup:
                raise OSError("actor cleanup failed")

        async def close(self):
            pass

    runtime = EngineCapture()
    runtime.adapter = Adapter()
    runtime.result_url = "http://engine/result"
    runtime._active.add("call")
    runtime._metadata["call"] = {
        "phase": "decode",
        "rollout_id": "r",
        "model_call_id": "c",
    }
    try:
        receipt = runtime.finalize("call", 1, "prefill").result(5)
        ticket = receipt["resolve_url"].rsplit("/", 1)[1]
        drain = runtime.drain(release_transients=True)
        assert entered.wait(5)
        assert not drain.done()
        with pytest.raises(RuntimeError, match="undrained"):
            runtime.resume()
        release.set()
        if fail_cleanup:
            with pytest.raises(OSError, match="actor cleanup failed"):
                drain.result(5)
            with pytest.raises(RuntimeError, match="failed capture"):
                runtime.resume()
        else:
            assert drain.result(5) is True
            runtime.resume()
        assert runtime.resolve_result(ticket).result(5) == {
            "disposition": "staged",
            "staging_key": "record",
        }
        assert runtime.release_result(ticket).result(5)
        assert not runtime.results
    finally:
        release.set()
        runtime.close()


@pytest.mark.parametrize("barrier_failure", [None, "storage", "cancel"])
def test_capture_suspension_fences_outputs_and_retains_unfinished_calls(
    barrier_failure,
):
    import asyncio
    import queue
    import threading
    from types import SimpleNamespace

    from vllm.v1.core.sched.interface import PauseState
    from vllm.v1.engine import EngineCoreOutput
    from vllm.v1.engine.capture_runtime import EngineCapture
    from vllm.v1.engine.core import EngineCoreProc

    preparing, added, entered = (threading.Event() for _ in range(3))
    release_prepare, release_storage = threading.Event(), threading.Event()
    tokens = []
    events = []

    class Adapter:
        async def prepare(self, request):
            preparing.set()
            await asyncio.to_thread(release_prepare.wait)

        def observe(self, output):
            tokens.extend(output.new_token_ids)

        async def suspend(self):
            entered.set()
            await asyncio.to_thread(release_storage.wait)
            if barrier_failure == "storage":
                raise OSError("storage flush failed")

        def finish(self, request_id):
            events.append("finish")

        async def finalize(self, request_id, served_tokens, prefill_key):
            events.append("finalize")
            return {"tokens": list(tokens)}

        async def close(self):
            pass

    runtime = EngineCapture()
    runtime.adapter = Adapter()
    engine = EngineCoreProc.__new__(EngineCoreProc)
    engine.capture_runtime = runtime
    engine.output_queue = queue.Queue()
    engine.scheduler = SimpleNamespace(pause_state=PauseState.UNPAUSED)
    engine.batch_queue = []
    request = SimpleNamespace(
        request_id="r",
        prompt_token_ids=[1],
        capture_context=SimpleNamespace(phase="decode", prefill_key="handoff"),
    )
    try:
        runtime.prepare(request, added.set, lambda: pytest.fail("prepare failed"))
        assert preparing.wait(5)
        with pytest.raises(RuntimeError, match="keep"):
            engine.suspend_capture()
        engine.scheduler.pause_state = PauseState.PAUSED_ALL
        engine.batch_queue = [object()]
        with pytest.raises(RuntimeError, match="keep"):
            engine.suspend_capture()
        engine.batch_queue.clear()
        suspended = engine.suspend_capture()
        assert not suspended.done()
        # Simulate the output thread reaching the fence.
        engine.output_queue.get_nowait()()
        with pytest.raises(RuntimeError, match="paused"):
            runtime.prepare(request, added.set, lambda: None)
        with pytest.raises(RuntimeError, match="undrained"):
            runtime.resume()
        assert not entered.is_set()  # Prepare must settle before actor flushing.
        release_prepare.set()
        assert added.wait(5) and entered.wait(5)
        if barrier_failure == "cancel":
            # The utility future cannot cancel an already-running storage fence.
            assert not suspended.cancel()
        release_storage.set()
        if barrier_failure == "storage":
            with pytest.raises(OSError, match="storage flush failed"):
                suspended.result(5)
            with pytest.raises(RuntimeError, match="failed capture"):
                runtime.resume()
            assert not events
            return
        assert suspended.result(5) is True
        assert runtime._metadata["r"]["prefill_key"] == "handoff"
        runtime.resume()

        for token in (2, 3):
            engine.output_queue.put_nowait(
                lambda token=token: runtime.observe(EngineCoreOutput("r", [token]))
            )
            suspended = engine.suspend_capture()
            assert not suspended.done()
            engine.output_queue.get_nowait()()  # Output before the fence.
            engine.output_queue.get_nowait()()  # Capture suspension fence.
            assert suspended.result(5) is True
            assert tokens == list(range(2, token + 1))
            assert not events  # Suspension must never finish/finalize the call.
            runtime.resume()
        runtime.finish("r")
        assert runtime.finalize("r", 2, "handoff").result(5) == {"tokens": [2, 3]}
        assert events == ["finish", "finalize"]
    finally:
        release_prepare.set()
        release_storage.set()
        runtime.close()


@pytest.mark.parametrize("operation", ["drain", "cancel", "close"])
def test_incomplete_capture_barrier_never_authorizes_resume(operation):
    import asyncio
    import threading

    from vllm.v1.engine.capture_runtime import EngineCapture

    entered, cancelled = threading.Event(), threading.Event()

    class Adapter:
        async def suspend(self):
            if operation == "drain":
                return
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        async def close(self):
            pass

    runtime = EngineCapture()
    runtime.adapter = Adapter()
    runtime._active.add("unfinished")
    try:
        if operation != "drain":
            suspended = runtime.suspend()
            assert entered.wait(5)
            with pytest.raises(RuntimeError, match="barrier is already pending"):
                runtime.drain()
            if operation == "close":
                runtime.close()
                assert not runtime.thread.is_alive()
            else:
                assert suspended.cancel()
            assert cancelled.wait(5)
        else:
            # A full drain must remain strict even after a successful suspension.
            assert runtime.suspend().result(5) is True
            with pytest.raises(RuntimeError, match="unfinalized"):
                runtime.drain().result(5)
        with pytest.raises(RuntimeError, match="failed capture"):
            runtime.resume()
    finally:
        if runtime.thread.is_alive():
            runtime.close()
