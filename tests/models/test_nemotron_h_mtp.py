# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

import vllm.model_executor.models.nemotron_h_mtp as nemotron_h_mtp

pytestmark = pytest.mark.cpu_test


def test_mtp_attention_marks_draft_kv(monkeypatch):
    def init_base_layer(self, **kwargs):
        self.mixer = SimpleNamespace(attn=SimpleNamespace(is_eagle_draft=False))

    monkeypatch.setattr(
        nemotron_h_mtp.NemotronHAttentionDecoderLayer,
        "__init__",
        init_base_layer,
    )

    layer = nemotron_h_mtp.NemotronHMTPAttentionDecoderLayer(
        config=SimpleNamespace(), layer_idx=0
    )

    assert layer.mixer.attn.is_eagle_draft
