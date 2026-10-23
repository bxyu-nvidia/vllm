# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in interception of FlashInfer's existing API decorator at import time.

This module must remain torch/FlashInfer-free: vLLM imports it before either
library so aliases and lazily loaded FlashInfer submodules are also intercepted.
The installed FlashInfer package and its compiled kernels are not modified.
"""

import functools
import importlib.util
import sys
from collections.abc import Callable, Sequence
from importlib.abc import MetaPathFinder
from importlib.machinery import ModuleSpec
from types import ModuleType


def _patch(module: ModuleType) -> None:
    original = module.__dict__["flashinfer_api"]

    def decorate(function: Callable | None = None, **options: object) -> Callable:
        def wrap(function: Callable) -> Callable:
            decorated = original(function, **options)

            @functools.wraps(decorated)
            def captured(*args: object, **kwargs: object) -> object:
                from vllm.logging_utils.flashinfer_dump import capture_call

                return capture_call(decorated, args, kwargs)

            return captured

        return wrap if function is None else wrap(function)

    module.__dict__["flashinfer_api"] = decorate
    module.__dict__["_vllm_capture_installed"] = True


class _CaptureFinder(MetaPathFinder):
    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None = None,
        target: ModuleType | None = None,
    ) -> ModuleSpec | None:
        if fullname != "flashinfer.api_logging":
            return None
        sys.meta_path.remove(self)
        spec = importlib.util.find_spec(fullname)
        if spec is None or spec.loader is None:
            raise ImportError("FlashInfer capture requires flashinfer.api_logging")
        original_exec = spec.loader.exec_module

        def exec_then_patch(module: ModuleType) -> None:
            original_exec(module)
            _patch(module)

        spec.loader.exec_module = exec_then_patch  # type: ignore[method-assign]
        return spec


def install() -> None:
    """Intercept API decoration before FlashInfer imports any kernel modules."""
    loaded = sys.modules.get("flashinfer.api_logging")
    if loaded is not None:
        if getattr(loaded, "_vllm_capture_installed", False):
            return
        raise RuntimeError(
            "For FlashInfer capture, set VLLM_FLASHINFER_DUMP_DIR before "
            "importing vLLM, and import vLLM before FlashInfer."
        )
    if not any(isinstance(finder, _CaptureFinder) for finder in sys.meta_path):
        sys.meta_path.insert(0, _CaptureFinder())
