# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only keep-refit transfer barriers; no model or device initialization."""

from unittest.mock import MagicMock

import pytest

from vllm.v1.engine.core import EngineCore

pytestmark = pytest.mark.cpu_test


def _core() -> EngineCore:
    core = object.__new__(EngineCore)
    core.scheduler = MagicMock()
    core.model_executor = MagicMock()
    return core


def test_transfer_pause_requires_every_worker_before_scheduler_ack():
    core = _core()
    connector = core.scheduler.get_kv_connector.return_value
    connector.supports_transfer_pause = True
    core.model_executor.collective_rpc.return_value = [True, False]
    with pytest.raises(RuntimeError, match="worker failed"):
        core.set_kv_transfer_paused(True)
    connector.set_transfer_paused.assert_not_called()
    core.model_executor.collective_rpc.return_value = [True, True]
    assert core.set_kv_transfer_paused(False, True)
    connector.set_transfer_paused.assert_not_called()
    assert core.set_kv_transfer_paused(True)
    connector.set_transfer_paused.assert_called_once_with(True)


def test_transfer_pause_rejects_unsupported_connector_before_worker_mutation():
    core = _core()
    core.scheduler.get_kv_connector.return_value.supports_transfer_pause = False
    with pytest.raises(NotImplementedError, match="scheduler KV connector"):
        core.set_kv_transfer_paused(True)
    core.model_executor.collective_rpc.assert_not_called()
