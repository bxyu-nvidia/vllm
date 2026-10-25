# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import vllm.v1.worker.gpu.model_runner as model_runner_module
from vllm.config import CacheConfig, ParallelConfig
from vllm.model_executor.warmup.jit_warmup import JitWarmupRegistry
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.worker.gpu.async_utils import async_copy_to_np
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.model_runner import ExecuteModelState, GPUModelRunner
from vllm.v1.worker.gpu.spec_decode.standalone_ar import speculator as ar_module
from vllm.v1.worker.gpu.spec_decode.utils import get_drafter_hidden_states


def test_non_last_pp_rank_uses_global_batch_for_sample_feedback():
    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.is_last_pp_rank = False
    local_batch = object()
    global_batch = SimpleNamespace(idx_mapping=object())
    runner.pcp_manager = SimpleNamespace(
        global_batch=global_batch,
        restore_for_sampling=Mock(),
    )
    runner.pp_handler = SimpleNamespace(receive=Mock(return_value=False))
    runner.postprocess_num_computed_tokens = Mock()
    runner.model_state = SimpleNamespace(postprocess_state=Mock())
    runner.kv_connector = SimpleNamespace(post_forward=Mock(return_value=None))
    runner.eplb = SimpleNamespace(step=Mock())
    runner.execute_model_state = ExecuteModelState(
        input_batch=local_batch,
        attn_metadata=None,
        slot_mappings_by_layer=None,
        hidden_states=None,
        aux_hidden_states=None,
        dp_sync_state=None,
        finished_req_ids=set(),
        ec_connector_output=None,
        cudagraph_stats=None,
        num_spec_tokens_to_schedule=0,
    )

    runner.sample_tokens(None)

    runner.pp_handler.receive.assert_called_once_with(global_batch)
    runner.postprocess_num_computed_tokens.assert_called_once_with(global_batch)
    runner.model_state.postprocess_state.assert_called_once_with(
        global_batch.idx_mapping, 0
    )
    runner.pcp_manager.restore_for_sampling.assert_not_called()


@pytest.mark.parametrize(
    ("replayssm", "has_connector"), [(True, True), (True, False), (False, True)]
)
@pytest.mark.parametrize("speculative_steps", [0, 5])
def test_replayssm_pd_output_waits_for_canonical_state(
    monkeypatch, replayssm, has_connector, speculative_steps
):
    """PD cannot observe completion using only the early token-copy event."""
    events = []
    runner = Mock()
    runner.vllm_config = SimpleNamespace(use_flashinfer_replayssm=replayssm)
    runner.is_last_pp_rank = True
    runner.pp_handler = runner.pcp_manager = None
    runner.speculator = runner.aux_output_connector = runner.adaptive_verification = (
        None
    )
    runner.model = Mock(spec=["compute_logits"])
    runner.num_speculative_steps = speculative_steps
    runner.execute_model_state = Mock(
        input_batch=Mock(req_ids=["request"], idx_mapping=torch.tensor([0])),
        finished_req_ids=set(),
        num_spec_tokens_to_schedule=speculative_steps,
    )
    if speculative_steps:
        runner.req_states.draft_tokens = torch.zeros((1, speculative_steps))
        runner.speculator = Mock(supports_mm_inputs=False)

        def draft(*_args, **_kwargs):
            events.append("draft")
            return torch.ones((1, speculative_steps))

        runner.speculator.propose.side_effect = draft
        monkeypatch.setattr(
            model_runner_module, "use_workspace_lane", contextlib.nullcontext
        )
    runner.sample.return_value = (Mock(), torch.tensor([1]), torch.tensor([0]))
    runner.postprocess_sampled.side_effect = lambda *_: events.append("materialize")
    runner.output_copy_stream.wait_stream.side_effect = lambda _: events.append("wait")
    output = Mock()
    output.copy_event.record.side_effect = lambda _: events.append("ready")

    def start_output_copy(**_kwargs):
        events.append("copy")
        return output

    monkeypatch.setattr(model_runner_module, "AsyncOutput", start_output_copy)
    if has_connector:
        runner.kv_connector.post_forward.side_effect = lambda _: events.append("save")
    else:
        runner.kv_connector = model_runner_module.NO_OP_KV_CONNECTOR

    assert GPUModelRunner.sample_tokens(runner, None) is output

    expected = ["copy", "materialize"]
    if speculative_steps:
        expected += ["draft"]
    if has_connector:
        expected += ["save"]
    if replayssm and has_connector:
        expected += ["wait", "ready"]
        runner.output_copy_stream.wait_stream.assert_called_once_with(
            runner.main_stream
        )
        output.copy_event.record.assert_called_once_with(runner.output_copy_stream)
    else:
        output.copy_event.record.assert_not_called()
    assert events == expected


def test_qsa_circular_group_uses_custom_slot_mapping(monkeypatch):
    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.max_model_len = 262144
    runner.is_encoder_decoder = False
    runner.dcp_size = 1
    runner.dcp_rank = 0
    runner.cp_interleave = 1
    runner.cache_config = SimpleNamespace(enable_prefix_caching=True)
    parallel_config = SimpleNamespace(
        decode_context_parallel_size=1,
        cp_kv_cache_interleave_size=1,
    )
    runner.parallel_config = parallel_config
    runner.vllm_config = SimpleNamespace(
        parallel_config=parallel_config,
        cache_config=SimpleNamespace(mamba_cache_mode="none"),
    )
    runner.jit_warmup_registry = JitWarmupRegistry(runner.vllm_config)
    runner.model_state = SimpleNamespace(
        get_additional_cg_support=lambda: (),
        num_new_sampled_tokens_per_step=1,
    )
    runner.speculator = None
    runner.req_states = []
    runner.input_buffers = SimpleNamespace(query_start_loc=None)
    runner.vocab_size = 1
    runner.max_num_reqs = 1
    runner.max_num_tokens = 2
    runner.device = torch.device("cuda")

    raw_spec = CircularBufferSpec(
        block_size=8,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
    )
    compressed_spec = FullAttentionSpec(
        block_size=262144,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=1,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                layer_names=["raw"],
                kv_cache_spec=UniformTypeKVCacheSpecs(
                    block_size=8,
                    kv_cache_specs={"raw": raw_spec},
                ),
            ),
            KVCacheGroupSpec(layer_names=["compressed"], kv_cache_spec=compressed_spec),
        ],
    )

    class FakeAttnCGSupport:
        def narrow(self, *args):
            return self

    attn_cg_support = FakeAttnCGSupport()
    monkeypatch.setattr(
        model_runner_module,
        "init_attn_backend",
        lambda *args, **kwargs: ([], attn_cg_support, [8, 262144]),
    )
    monkeypatch.setattr(
        model_runner_module,
        "maybe_create_adaptive_verification_manager",
        lambda **kwargs: None,
    )

    captured = {}

    class BlockTablesCaptured(Exception):
        pass

    def capture_block_tables(**kwargs):
        captured.update(kwargs)
        raise BlockTablesCaptured

    monkeypatch.setattr(model_runner_module, "BlockTables", capture_block_tables)

    with pytest.raises(BlockTablesCaptured):
        runner.initialize_kv_cache(kv_cache_config)

    assert captured["max_num_blocks_per_group"] == [1, 1]
    assert captured["slot_mapping_enabled"] == [False, True]


@pytest.mark.parametrize(
    ("mamba_cache_mode", "num_speculative_blocks", "expected"),
    [
        pytest.param("align", 0, 65_536, id="align-prefix-cache"),
        pytest.param("none", 7, 8, id="no-prefix-cache-with-speculation"),
    ],
)
def test_initialize_kv_cache_does_not_dcp_shard_mamba_block_table(
    monkeypatch,
    mamba_cache_mode: str,
    num_speculative_blocks: int,
    expected: int,
):
    """Mamba/GDN block-table rows index global positions, unlike DCP KV."""
    max_model_len = 1_048_576
    attention_block_size = 1_536
    mamba_block_size = 16
    dcp_size = 8
    full_attention_spec = FullAttentionSpec(
        block_size=attention_block_size,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.bfloat16,
    )
    mamba_spec = MambaSpec(
        shapes=((1,),),
        dtypes=(torch.bfloat16,),
        block_size=mamba_block_size,
        mamba_cache_mode=mamba_cache_mode,
        num_speculative_blocks=num_speculative_blocks,
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=1,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(["attention"], full_attention_spec),
            KVCacheGroupSpec(["kda"], mamba_spec),
        ],
    )
    parallel_config = SimpleNamespace(
        decode_context_parallel_size=dcp_size,
        cp_kv_cache_interleave_size=1,
    )
    vllm_config = SimpleNamespace(
        parallel_config=parallel_config,
        cache_config=SimpleNamespace(mamba_cache_mode=mamba_cache_mode),
    )
    runner = SimpleNamespace(
        max_model_len=max_model_len,
        is_encoder_decoder=False,
        vllm_config=vllm_config,
        parallel_config=parallel_config,
    )

    class _CapturedWidths(Exception):
        pass

    captured: list[int] = []

    def capture_width(max_num_blocks: int, *_args, **_kwargs) -> int:
        captured.append(max_num_blocks)
        if len(captured) == 2:
            raise _CapturedWidths
        return max_num_blocks

    monkeypatch.setattr(model_runner_module, "get_block_table_width", capture_width)

    with pytest.raises(_CapturedWidths):
        GPUModelRunner.initialize_kv_cache(runner, kv_cache_config)

    # Attention KV is local to one of eight DCP ranks; KDA state is replicated
    # and therefore needs one table entry for every global 16-token page.
    assert captured == [86, expected]


def test_append_block_ids_rejects_write_past_row_capacity():
    """Reject an oversized staged write before it can corrupt the next row."""

    class _BlockTable:
        gpu = torch.empty((2, 4), dtype=torch.int32)

        def stage_write(self, *_args):
            pytest.fail("an oversized write must not be staged")

    block_tables = BlockTables.__new__(BlockTables)
    block_tables.num_kv_cache_groups = 1
    block_tables.block_tables = [_BlockTable()]
    block_tables.num_blocks = SimpleNamespace(
        np=torch.tensor([[0, 3]], dtype=torch.int32)
    )

    with pytest.raises(
        RuntimeError,
        match=r"request 1, group 0 exceeds row capacity \(5 > 4\)",
    ):
        block_tables.append_block_ids(
            req_index=1,
            new_block_ids=([4, 5],),
            overwrite=False,
        )

    assert block_tables.num_blocks.np[0, 1] == 3


def _make_capture_runner(captured: bool) -> GPUModelRunner:
    """Minimal V2 runner for capture_model: fakes everything except the
    cudagraph_manager's needs_capture decision."""
    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.model_state = SimpleNamespace(
        supports_mm_inputs=False, capture_inner_cudagraphs=lambda *args: None
    )
    runner.cudagraph_manager = SimpleNamespace(
        needs_capture=lambda: captured,
        capture=lambda *args, **kwargs: None,
    )
    runner.lora_config = None
    runner.maybe_setup_dummy_loras = lambda _cfg: contextlib.nullcontext()
    runner.speculator = None
    runner.adaptive_verification = None
    runner.model = None
    runner.input_buffers = None
    runner.pcp_manager = None
    runner.intermediate_tensors = None
    runner.block_tables = None
    runner.attn_groups = None
    runner.kv_cache_config = None
    runner.use_aux_hidden_state_outputs = False
    runner.kv_connector = model_runner_module.NO_OP_KV_CONNECTOR
    return runner


def test_capture_model_locks_workspace_after_capture(monkeypatch):
    """A workspace resize after capture frees the buffer the captured graphs
    baked in, so capture_model must lock the workspace before returning
    (https://github.com/vllm-project/vllm/issues/55336)."""
    runner = _make_capture_runner(captured=True)
    monkeypatch.setattr(
        model_runner_module, "freeze_gc_for_cudagraph_capture", contextlib.nullcontext
    )
    monkeypatch.setattr(torch.accelerator, "empty_cache", lambda: None)
    monkeypatch.setattr(
        torch.accelerator, "get_memory_info", lambda: (1 << 30, 1 << 30)
    )
    lock_calls = []
    monkeypatch.setattr(
        model_runner_module, "lock_workspace", lambda: lock_calls.append("lock")
    )

    runner.capture_model()

    assert lock_calls == ["lock"]


def test_capture_model_skips_lock_when_nothing_captured(monkeypatch):
    """With no graphs to capture (e.g. enforce_eager) there is nothing baked
    into the workspace, so the early return must not lock it."""
    runner = _make_capture_runner(captured=False)
    lock_calls = []
    monkeypatch.setattr(
        model_runner_module, "lock_workspace", lambda: lock_calls.append("lock")
    )

    assert runner.capture_model() == 0
    assert lock_calls == []


def test_capture_model_profile_only_skips_lock(monkeypatch):
    """The memory-profiling capture pass runs before kernel warmup and the
    real capture; locking there would stop the warmup from growing the
    workspace to its scheduler-realistic size."""
    runner = _make_capture_runner(captured=True)
    monkeypatch.setattr(
        model_runner_module, "freeze_gc_for_cudagraph_capture", contextlib.nullcontext
    )
    monkeypatch.setattr(torch.accelerator, "empty_cache", lambda: None)
    monkeypatch.setattr(
        torch.accelerator, "get_memory_info", lambda: (1 << 30, 1 << 30)
    )
    lock_calls = []
    monkeypatch.setattr(
        model_runner_module, "lock_workspace", lambda: lock_calls.append("lock")
    )

    runner.capture_model(profile_only=True)

    assert lock_calls == []


@pytest.mark.parametrize("target_buffer", ["absent", "none", "tensor"])
def test_get_drafter_hidden_states_tolerates_missing_target_buffer(target_buffer):
    """Targets allocate the MTP hidden buffer only for hidden-state drafters."""
    hidden_states = torch.zeros(4, 8)
    buffer = torch.arange(16 * 8, dtype=torch.float32).view(16, 8)
    if target_buffer == "absent":
        model = SimpleNamespace()
    else:
        returned = buffer if target_buffer == "tensor" else None
        model = SimpleNamespace(get_mtp_target_hidden_states=lambda: returned)

    out = get_drafter_hidden_states(model, hidden_states)

    if target_buffer == "tensor":
        assert torch.equal(out, buffer[:4])
    else:
        assert out is hidden_states


def test_async_copy_to_np_does_not_alias_reused_buffer():
    buffer = torch.zeros(4, dtype=torch.int64)

    snapshot = async_copy_to_np(buffer)
    buffer.fill_(1)

    assert snapshot.tolist() == [0, 0, 0, 0]


@pytest.mark.parametrize(
    ("draft_dtype", "expected_dtype"),
    [("auto", "auto"), ("bfloat16", "bfloat16"), (None, "fp8_e4m3")],
)
def test_standalone_draft_kv_dtype_override(monkeypatch, draft_dtype, expected_dtype):
    @dataclass
    class Config:
        cache_config: CacheConfig
        parallel_config: ParallelConfig
        model_config: object
        quant_config: object | None = None

    target = Config(CacheConfig(cache_dtype="fp8_e4m3"), ParallelConfig(), object())
    speculator = object.__new__(ar_module.StandaloneARSpeculator)
    speculator.vllm_config = target
    speculator.draft_model_config = object()
    speculator.speculative_config = SimpleNamespace(
        kv_cache_dtype=draft_dtype, draft_parallel_config=ParallelConfig()
    )
    get_model = Mock(return_value=object())
    monkeypatch.setattr(ar_module, "get_model", get_model)
    monkeypatch.setattr(ar_module, "set_model_tag", lambda _: contextlib.nullcontext())

    speculator.load_draft_model(object(), set())

    draft_config = get_model.call_args.kwargs["vllm_config"]
    assert draft_config.cache_config.cache_dtype == expected_dtype
    assert speculator.attn_vllm_config.cache_config.cache_dtype == expected_dtype
    assert target.cache_config.cache_dtype == "fp8_e4m3"


@pytest.mark.parametrize(
    "num_reqs,num_tokens,padded_tokens,mode,padded_reqs",
    [
        (1, 6, 8, "PIECEWISE", None),
        (2, 12, 16, "PIECEWISE", None),
        (1, 6, 128, "PIECEWISE", None),  # padding to another DP rank's size
        (1, 6, 6, "NONE", 1),
        (1, 6, 12, "FULL", 2),
    ],
)
def test_dummy_graph_padding_preserves_dp_sync(
    monkeypatch, num_reqs, num_tokens, padded_tokens, mode, padded_reqs
):
    """The MTP drafter must reuse the target's DP sync after dummy graph padding."""
    from vllm.config.compilation import CUDAGraphMode
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.worker.gpu import dp_utils
    from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
    from vllm.v1.worker.gpu.input_batch import InputBuffers
    from vllm.v1.worker.utils import get_uniform_decode_token_count

    mode = CUDAGraphMode[mode]
    desc = BatchExecutionDescriptor(
        cg_mode=mode, num_tokens=padded_tokens, num_reqs=padded_reqs
    )
    sync = dp_utils.DPSyncState(
        num_tokens_across_dp=torch.full((4,), padded_tokens, dtype=torch.int32),
        uniform_token_count=6,
        eager=mode == CUDAGraphMode.NONE,
        num_reqs=num_reqs,
    )
    runner = Mock()
    runner.lora_config = runner.ubatch_runner = runner.pcp_manager = None
    runner.is_encoder_decoder = False
    runner.decode_query_len = 6
    runner.dp_size, runner.dp_rank = 4, 3
    runner.input_buffers = InputBuffers(4, padded_tokens, torch.device("cpu"))
    runner.gather_batch_req_state = lambda output, dummy: (
        model_runner_module.GPUModelRunner.gather_batch_req_state(runner, output, dummy)
    )
    dispatch = Mock(return_value=(desc, sync))
    monkeypatch.setattr(model_runner_module, "dispatch_cg_and_sync_dp", dispatch)

    class BatchPrepared(Exception):
        pass

    runner.prepare_dummy_attn.side_effect = BatchPrepared
    output = SchedulerOutput.make_empty()
    output.num_scheduled_tokens = {str(i): 6 for i in range(num_reqs)}
    output.total_num_scheduled_tokens = num_tokens
    with pytest.raises(BatchPrepared):
        GPUModelRunner.execute_model(runner, output, dummy_run=True)

    # Inspect the batch actually passed to attention, before any GPU kernels run.
    batch = runner.prepare_dummy_attn.call_args.args[0]
    assert dispatch.call_args.args[3] == 6
    assert batch.num_tokens == (
        padded_tokens if mode == CUDAGraphMode.FULL else num_tokens
    )
    assert batch.num_tokens_after_padding == padded_tokens
    assert batch.num_scheduled_tokens.tolist() == [6] * (padded_reqs or num_reqs)
    assert batch.query_start_loc_np[-1] == batch.num_tokens
    assert batch.logits_indices.tolist() == list(range(5, batch.num_tokens, 6))
    assert batch.input_ids.shape == batch.positions.shape == (padded_tokens,)
    GPUModelRunner.prepare_dummy_attn(runner, batch)
    runner.block_tables.get_dummy_slot_mappings.assert_called_once_with(padded_tokens)

    uniform = get_uniform_decode_token_count(
        batch.num_reqs,
        batch.num_tokens,
        int(batch.num_scheduled_tokens.max()),
        batch.decode_graph_eligible,
    )
    manager = Mock()
    manager.dispatch.return_value = desc
    # Reusing the agreement must not issue another collective on only idle ranks.
    monkeypatch.setattr(
        dp_utils, "get_dp_group", Mock(side_effect=AssertionError("new collective"))
    )
    _, reused = dp_utils.dispatch_cg_and_sync_dp(
        manager,
        batch.num_reqs,
        batch.num_tokens_after_padding,
        uniform,
        dp_size=4,
        dp_rank=3,
        dp_sync_state=sync,
    )
    assert reused is sync


@pytest.mark.parametrize("num_reqs", [1, 2])
@pytest.mark.parametrize("collect_metrics", [False, True])
@pytest.mark.parametrize("dummy_run", [False, True])
@pytest.mark.parametrize("microbatched", [False, True])
@pytest.mark.parametrize("drafting", [False, True])
def test_mtp_metadata_is_full_batch_and_rebuilt_after_dbo_join(
    monkeypatch, microbatched, drafting, dummy_run, collect_metrics, num_reqs
):
    """MTP must not see a microbatch's offsets or overwrite live builder buffers."""
    from vllm.config.compilation import CUDAGraphMode
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
    from vllm.v1.worker.gpu.dp_utils import DPSyncState
    from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
    from vllm.v1.worker.gpu.ubatch_utils import create_whole_request_ubatch_slices

    events = []
    runner = Mock()
    runner.lora_config = runner.pcp_manager = None
    runner.is_encoder_decoder = runner.uses_inputs_embeds = False
    runner.use_aux_hidden_state_outputs = False
    runner.is_first_pp_rank = runner.is_last_pp_rank = True
    runner.speculator = Mock() if drafting else None
    runner.decode_query_len = 6
    runner.dp_size, runner.dp_rank, runner.dcp_size = 4, 0, 1
    runner.parallel_config.num_ubatches = 2
    runner.model_state.requires_whole_request_ubatches = True
    runner.input_buffers = InputBuffers(4, 24, torch.device("cpu"))
    runner.observability_config.enable_dbo_metrics = collect_metrics
    runner.observability_config.cudagraph_metrics = False
    runner.aux_output_connector = None
    runner.parallel_config.use_ubatching = microbatched
    runner.gather_batch_req_state.return_value = (
        SimpleNamespace(num_tokens=18, has_prefill=True),
        6,
    )
    runner.prepare_inputs.return_value = InputBatch.make_dummy(
        num_reqs, 18, runner.input_buffers, num_tokens_after_padding=24
    )
    runner.model_state.prepare_inputs.return_value = {}
    blocks = (torch.tensor([[11], [22]]),)
    slots = torch.arange(18).reshape(1, -1)
    runner.prepare_dummy_attn.return_value = (blocks, slots)
    runner.prepare_attn.return_value = (blocks, slots)
    full_metadata = {"attention": object()}
    full_slots = {"attention": slots[0]}

    def prepare_metadata(batch, mode, block_tables, slot_mappings, *args, **kwargs):
        events.append("full_metadata")
        assert batch.num_reqs == num_reqs
        assert batch.query_start_loc_np.tolist() == (
            [0, 18] if num_reqs == 1 else [0, 9, 18]
        )
        assert block_tables is blocks and slot_mappings is slots
        return full_metadata

    runner.model_state.prepare_attn.side_effect = prepare_metadata

    def target_forward(*args, **kwargs):
        events.append("target_joined")
        return torch.arange(18).reshape(18, 1)

    if microbatched:
        runner.ubatch_runner.prepare.side_effect = (
            lambda batch, *args, **kwargs: SimpleNamespace(
                slices=create_whole_request_ubatch_slices(batch)
            )
        )
        runner.ubatch_runner.run.side_effect = target_forward
    else:
        runner.ubatch_runner = None
        runner.model.side_effect = target_forward
    desc = BatchExecutionDescriptor(
        cg_mode=CUDAGraphMode.NONE,
        num_tokens=18,
        num_reqs=num_reqs,
        num_ubatches=2 if microbatched else 1,
    )
    sync = DPSyncState(torch.full((4,), 18), None, True, 2)

    def dispatch(*args, **kwargs):
        assert kwargs["allow_ubatching"] == microbatched
        if kwargs["dbo_stats"] is not None and microbatched:
            kwargs["dbo_stats"].outcome = "executed"
        return desc, sync

    monkeypatch.setattr(model_runner_module, "dispatch_cg_and_sync_dp", dispatch)
    monkeypatch.setattr(
        model_runner_module,
        "build_slot_mappings_by_layer",
        lambda slot_mappings, config: full_slots,
    )
    monkeypatch.setattr(
        model_runner_module,
        "set_forward_context",
        lambda *a, **kw: contextlib.nullcontext(),
    )
    output = SchedulerOutput.make_empty()
    output.num_scheduled_tokens = {str(i): 6 for i in range(num_reqs)}
    output.total_num_scheduled_tokens = 6 * num_reqs
    GPUModelRunner.execute_model(runner, output, dummy_run=dummy_run)

    state = runner.execute_model_state
    if collect_metrics and not dummy_run:
        assert state.dbo_stats is not None
        assert state.dbo_stats.outcome == ("executed" if microbatched else "disabled")
        assert state.dbo_stats.microbatch_tokens == (
            ([18, 0] if num_reqs == 1 else [9, 9]) if microbatched else []
        )
        assert state.dbo_stats.microbatch_requests == (
            ([1, 0] if num_reqs == 1 else [1, 1]) if microbatched else []
        )
        assert state.dbo_stats.microbatch_padding_tokens == (
            ([6, 1] if num_reqs == 1 else [0, 6]) if microbatched else []
        )
    else:
        assert state.dbo_stats is None
    if microbatched and not drafting:
        assert events == ["target_joined"]
        assert state.attn_metadata is None
    else:
        assert events == (
            ["target_joined", "full_metadata"]
            if microbatched
            else ["full_metadata", "target_joined"]
        )
        assert state.attn_metadata is full_metadata
        assert state.slot_mappings_by_layer is full_slots
    assert state.dp_sync_state is sync
    torch.testing.assert_close(state.hidden_states, torch.arange(18).reshape(18, 1))
