# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Capture ten calls per FlashInfer API in real forwards above 16,384 tokens."""

import json
import os
import socket
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from threading import Lock
from typing import TypeVar
from uuid import uuid4

import torch

import vllm.envs as envs
from vllm.logger import init_logger

logger = init_logger(__name__)
_T = TypeVar("_T")
_lock = Lock()
_active_tokens: int | None = None


@dataclass
class _Session:
    path: Path
    counts: dict[str, int] = field(default_factory=dict)
    failed: bool = False


_session: _Session | None = None


def _pack(value: object, unsupported: set[str]) -> object:
    if isinstance(value, torch.Tensor):
        # Saving before/after the call preserves views, strides and aliasing.
        # No clone of the cache/weights is allocated on the GPU.
        return value.detach()
    if value is None or isinstance(
        value, (bool, int, float, str, Enum, torch.dtype, torch.device)
    ):
        return value
    if isinstance(value, tuple):
        return tuple(_pack(item, unsupported) for item in value)
    if isinstance(value, list):
        return [_pack(item, unsupported) for item in value]
    if isinstance(value, dict):
        return {key: _pack(item, unsupported) for key, item in value.items()}
    name = f"{type(value).__module__}.{type(value).__qualname__}"
    unsupported.add(name)
    # Native plans, process groups and wrapper objects cannot be reconstructed
    # from a tensor dump. Keep the call for inspection, but reject its replay.
    return {"unreplayable_type": name}


def _save(path: Path, value: object) -> None:
    pending = path.with_suffix(".tmp")
    torch.save(value, pending)
    pending.replace(path)


def capture_call(function: Callable[..., _T], args: tuple, kwargs: dict) -> _T:
    """Snapshot inputs before execution and returns/buffers after execution."""
    session = _session
    tokens = _active_tokens
    if session is None or session.failed or tokens is None:
        return function(*args, **kwargs)
    name = f"{function.__module__}.{function.__qualname__}"
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        return function(*args, **kwargs)
    with _lock:
        index = session.counts.get(name, 0) + 1
        if index > 10:
            selected = False
        else:
            selected = True
            session.counts[name] = index
    if not selected:
        return function(*args, **kwargs)
    path = session.path / name / f"call-{index:02d}"
    unsupported: set[str] = set()
    metadata = {
        "module": function.__module__,
        "function": function.__qualname__,
        "call_index": index,
        "forward_tokens": tokens,
        "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
        "status": "inputs_saved",
    }
    try:
        if torch.cuda.is_available():
            torch.accelerator.synchronize()
        path.mkdir(parents=True, exist_ok=False)
        _save(path / "inputs.pt", _pack((args, kwargs), unsupported))
        metadata["unsupported_types"] = sorted(unsupported)
        (path / "metadata.json").write_text(json.dumps(metadata, indent=2))
    except Exception:
        session.failed = True
        logger.exception("FlashInfer capture failed; disabling further dumps")
        return function(*args, **kwargs)

    # The operation is executed exactly once. Its exceptions propagate normally.
    result = function(*args, **kwargs)
    try:
        if torch.cuda.is_available():
            torch.accelerator.synchronize()
        _save(path / "outputs.pt", _pack(result, unsupported))
        # Many APIs return None and mutate arguments (norm, Mamba, KV writes).
        _save(path / "arguments_after.pt", _pack((args, kwargs), unsupported))
        metadata["unsupported_types"] = sorted(unsupported)
        metadata["standalone_replay"] = (
            not unsupported and "<locals>" not in function.__qualname__
        ) and not any(
            part in function.__module__.split(".") for part in ("comm", "moe_ep")
        )
        metadata["status"] = "complete"
        (path / "metadata.json").write_text(json.dumps(metadata, indent=2))
        logger.info("Saved FlashInfer call: %s (%d forward tokens)", path, tokens)
    except Exception:
        session.failed = True
        logger.exception("FlashInfer output capture failed; disabling further dumps")
    return result


@contextmanager
def flashinfer_dump_forward(num_tokens: int, *, dummy: bool = False) -> Iterator[None]:
    """Enable operation capture only for qualifying real model forwards."""
    global _session, _active_tokens
    directory = envs.VLLM_FLASHINFER_DUMP_DIR
    if not directory or dummy or num_tokens <= 16_384:
        yield
        return
    if _session is None:
        from flashinfer import api_logging

        if not getattr(api_logging, "_vllm_capture_installed", False):
            raise RuntimeError(
                "Set VLLM_FLASHINFER_DUMP_DIR before importing vLLM, "
                "and import vLLM before FlashInfer for API capture."
            )
        name = f"{socket.gethostname()}-{os.getpid()}-{uuid4().hex[:8]}"
        _session = _Session(Path(directory) / name)
    previous = _active_tokens
    _active_tokens = num_tokens
    try:
        yield
    finally:
        _active_tokens = previous
