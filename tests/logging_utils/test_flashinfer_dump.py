# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import sys
from types import ModuleType

import pytest
import torch

from vllm.logging_utils import flashinfer_dump as dump

pytestmark = pytest.mark.cpu_test


@pytest.fixture
def capture(tmp_path, monkeypatch):
    monkeypatch.setenv("VLLM_FLASHINFER_DUMP_DIR", str(tmp_path))
    monkeypatch.setattr(dump, "_session", None)
    monkeypatch.setattr(dump, "_active_tokens", None)
    flashinfer = ModuleType("flashinfer")
    api = ModuleType("flashinfer.api_logging")
    api.__dict__["_vllm_capture_installed"] = True
    flashinfer.__dict__["api_logging"] = api
    monkeypatch.setitem(sys.modules, "flashinfer", flashinfer)
    yield tmp_path


def plus_one(x):
    return x + 1


def twice(x):
    return x * 2


def mutate(x, *, out):
    out.copy_(x + 4)


def test_ten_per_api_not_per_forward_and_no_warmup(capture):
    path = capture
    x = torch.arange(4)
    for tokens, dummy in [(20_000, True), (16_384, False), (12, False)]:
        with dump.flashinfer_dump_forward(tokens, dummy=dummy):
            dump.capture_call(plus_one, (x,), {})
    assert dump._session is None
    assert not list(path.rglob("inputs.pt"))
    # One large real forward can contain all ten samples from multiple layers.
    with dump.flashinfer_dump_forward(16_385):
        for _ in range(12):
            assert torch.equal(dump.capture_call(plus_one, (x,), {}), x + 1)
    with dump.flashinfer_dump_forward(20_000):
        for _ in range(12):
            dump.capture_call(plus_one, (x,), {})
            assert torch.equal(dump.capture_call(twice, (x,), {}), x * 2)
    records = [json.loads(p.read_text()) for p in path.rglob("metadata.json")]
    assert len(records) == 20
    for function in ("plus_one", "twice"):
        assert sorted(
            r["call_index"] for r in records if r["function"] == function
        ) == list(range(1, 11))
    assert dump._active_tokens is None


def test_snapshot_preserves_views_aliasing_and_inplace_outputs(capture):
    path = capture
    base = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    x = base[:, ::2]
    out = base[:, 1::2]
    original = x.clone()
    with dump.flashinfer_dump_forward(20_000):
        assert dump.capture_call(mutate, (x,), {"out": out}) is None
    directory = next(path.rglob("inputs.pt")).parent
    positional, keyword = torch.load(directory / "inputs.pt", weights_only=True)
    assert positional[0].stride() == x.stride()
    assert (
        positional[0].untyped_storage().data_ptr()
        == keyword["out"].untyped_storage().data_ptr()
    )
    assert torch.equal(keyword["out"], original + 1)
    # Loading the saved arguments and invoking the API reproduces its mutation.
    mutate(*positional, **keyword)
    after = torch.load(directory / "arguments_after.pt", weights_only=True)
    torch.testing.assert_close((positional, keyword), after)
    assert torch.equal(after[1]["out"], original + 4)
    assert torch.load(directory / "outputs.pt", weights_only=True) is None
    base.zero_()
    assert torch.equal(
        torch.load(directory / "arguments_after.pt", weights_only=True)[1]["out"],
        original + 4,
    )


def test_opaque_inputs_are_recorded_but_not_claimed_replayable(capture):
    path = capture

    def operation(plan, x):
        return x + 1

    with dump.flashinfer_dump_forward(20_000):
        dump.capture_call(operation, (object(), torch.ones(3)), {})
    record = json.loads(next(path.rglob("metadata.json")).read_text())
    assert record["unsupported_types"] == ["builtins.object"]
    assert not record["standalone_replay"]
    assert record["status"] == "complete"


def test_operation_failure_propagates_without_retry(capture):
    path = capture
    calls = []

    def operation(x):
        calls.append(x)
        raise ValueError("operation failed")

    with (
        pytest.raises(ValueError, match="operation failed"),
        dump.flashinfer_dump_forward(20_000),
    ):
        dump.capture_call(operation, (7,), {})
    assert calls == [7]
    assert dump._active_tokens is None
    record = json.loads(next(path.rglob("metadata.json")).read_text())
    assert record["status"] == "inputs_saved"
    assert not list(path.rglob("outputs.pt"))


def test_io_failure_disables_capture_without_repeating_operation(capture, monkeypatch):
    calls = []

    def operation(x):
        calls.append(x)
        return x + 1

    def fail(*args):
        raise OSError("disk full")

    monkeypatch.setattr(dump, "_save", fail)
    with dump.flashinfer_dump_forward(20_000):
        assert dump.capture_call(operation, (3,), {}) == 4
        assert dump.capture_call(operation, (4,), {}) == 5
    assert calls == [3, 4]
    assert dump._session.failed


def test_replay_restores_inputs_each_iteration_and_compares_parameters(capture):
    import importlib.util
    from pathlib import Path

    source = Path(__file__).parents[2] / "benchmarks/kernels/replay_flashinfer_dump.py"
    spec = importlib.util.spec_from_file_location("replay_flashinfer_dump", source)
    assert spec is not None and spec.loader is not None
    replay = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(replay)
    path = capture
    x = torch.arange(4, dtype=torch.float32)
    with dump.flashinfer_dump_forward(20_000):
        dump.capture_call(mutate, (x,), {"out": x})
    directory = next(path.rglob("inputs.pt")).parent
    expected = torch.load(directory / "arguments_after.pt", weights_only=True)
    for _ in range(3):
        positional, keyword = replay.load_snapshot(
            directory / "inputs.pt", torch.device("cpu")
        )
        mutate(*positional, **keyword)
        replay.compare_snapshot((positional, keyword), expected, rtol=0, atol=0)
    parameters = {"dtype": torch.float32, "device": torch.device("cpu"), "flag": True}
    replay.compare_snapshot(parameters, parameters, rtol=0, atol=0)
    with pytest.raises(AssertionError):
        replay.compare_snapshot(
            {"dtype": torch.float32}, {"dtype": torch.float16}, rtol=0, atol=0
        )


def test_import_hook_covers_aliases_and_late_imports_without_flashinfer_changes(
    tmp_path,
):
    import os
    import subprocess
    import sys
    import textwrap

    package = tmp_path / "flashinfer"
    package.mkdir()
    (package / "__init__.py").write_text("from .ops import operation\n")
    (package / "api_logging.py").write_text(
        textwrap.dedent("""
        def flashinfer_api(function=None, *, trace=None):
            def decorate(f):
                f.fi_trace = trace
                return f
            return decorate if function is None else decorate(function)
    """)
    )
    (package / "ops.py").write_text(
        textwrap.dedent("""
        from .api_logging import flashinfer_api
        @flashinfer_api(trace="preserved")
        def operation(x):
            x.add_(1)
            return x
    """)
    )
    (package / "lazy_op.py").write_text(
        textwrap.dedent("""
        from .api_logging import flashinfer_api
        @flashinfer_api
        def operation(x):
            return x * 3
    """)
    )
    before = {p.name: p.read_bytes() for p in package.glob("*.py")}
    environment = os.environ.copy()
    environment["VLLM_FLASHINFER_DUMP_DIR"] = str(tmp_path / "captures")
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent("""
            import sys
            sys.path.insert(0, sys.argv[1])
            import vllm  # installs the hook before importing the dependency
            from flashinfer import operation as alias
            from flashinfer.api_logging import _vllm_capture_installed
            assert _vllm_capture_installed
            assert alias.fi_trace == "preserved"
            from vllm.logging_utils.flashinfer_dump import flashinfer_dump_forward
            from vllm.logging_utils.flashinfer_capture_hook import _CaptureFinder
            assert not any(isinstance(f, _CaptureFinder) for f in sys.meta_path)
            import torch
            x = torch.zeros(3)
            alias(x)  # warmup: no active capture
            with flashinfer_dump_forward(20000, dummy=True):
                alias(x)
            with flashinfer_dump_forward(20000):
                alias(x)
                from flashinfer.lazy_op import operation
                operation(x)
        """),
            str(tmp_path),
        ],
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    assert {p.name: p.read_bytes() for p in package.glob("*.py")} == before
    records = [
        json.loads(p.read_text())
        for p in (tmp_path / "captures").rglob("metadata.json")
    ]
    assert len(records) == 2
    assert {r["module"] for r in records} == {"flashinfer.ops", "flashinfer.lazy_op"}
    assert all(r["call_index"] == 1 and r["status"] == "complete" for r in records)
