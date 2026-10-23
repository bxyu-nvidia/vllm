# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from torch import nn

from vllm.distributed.eplb.eplb_state import EplbLayerState
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
from vllm.model_executor.models import nemotron_h_mtp as mtp
from vllm.model_executor.models.interfaces import get_mixture_of_experts_model


@pytest.fixture
def draft(request):
    pattern, redundant = request.param
    predictor = nn.Module()
    predictor.layers = nn.ModuleDict()
    predictor.make_empty_intermediate_tensors = Mock()
    for i, char in enumerate(pattern):
        if char == "*":
            layer = nn.Identity()
        else:
            layer = mtp.NemotronHMTPMoEDecoderLayer.__new__(
                mtp.NemotronHMTPMoEDecoderLayer
            )
            nn.Module.__init__(layer)
            experts = MoERunner.__new__(MoERunner)
            nn.Module.__init__(experts)
            experts.router = SimpleNamespace(eplb_state=EplbLayerState())
            experts.routed_experts = Mock()
            experts.routed_experts.get_expert_weights.return_value = [torch.zeros(2)]
            layer.mixer = SimpleNamespace(
                experts=experts,
                n_logical_experts=8,
                n_physical_experts=8 + redundant,
                n_local_physical_experts=(8 + redundant) // 4,
                n_routed_experts=8,
                n_shared_experts=1,
                n_redundant_experts=redundant,
            )
        predictor.layers[str(i)] = layer

    config = Mock()
    draft_hf_config = config.speculative_config.draft_model_config.hf_config
    draft_hf_config.get_text_config.return_value = SimpleNamespace(
        num_hidden_layers=20, vocab_size=128, hidden_size=64, n_group=1
    )
    config.parallel_config.eplb_config.num_redundant_experts = redundant
    with (
        patch.object(mtp.NemotronHMTP, "_find_quant_config", return_value=None),
        patch.object(mtp, "NemotronHMultiTokenPredictor", return_value=predictor),
        patch.object(mtp, "ParallelLMHead"),
        patch.object(mtp, "LogitsProcessor"),
    ):
        return mtp.NemotronHMTP(vllm_config=config)


@pytest.mark.parametrize("draft", [("*E", 0), ("*E*E", 4)], indirect=True)
def test_mtp_eplb_discovers_experts_and_attaches_live_state(draft):
    """Draft registration must populate every MoE router before its first forward."""
    assert get_mixture_of_experts_model(draft) is draft
    mixers = [
        layer.mixer for layer in draft.model.layers.values() if hasattr(layer, "mixer")
    ]
    assert draft.num_moe_layers == len(mixers)
    assert draft.num_logical_experts == draft.num_routed_experts == 8
    assert draft.num_physical_experts == 8 + draft.num_redundant_experts
    assert draft.num_local_physical_experts == draft.num_physical_experts // 4
    assert draft.num_shared_experts == draft.num_expert_groups == 1

    loads = torch.zeros(
        (draft.num_moe_layers, draft.num_physical_experts), dtype=torch.int32
    )
    mapping = torch.arange(8).repeat(draft.num_moe_layers, 1).unsqueeze(-1)
    replicas = torch.ones((draft.num_moe_layers, 8), dtype=torch.int32)
    draft.set_eplb_state(loads, mapping, replicas)

    for i, mixer in enumerate(mixers):
        assert draft.moe_layers[i] is mixer.experts
        assert draft.expert_weights[i][0] is mixer.experts.get_expert_weights()[0]
        state = mixer.experts.eplb_state
        # State must alias the balancer's tensors so updates survive graph capture.
        state.expert_load_view.add_(i + 1)
        assert torch.equal(loads[i], torch.full_like(loads[i], i + 1))
        mapping[i, 0, 0] = 7
        replicas[i, 0] = 2
        assert state.logical_to_physical_map[0, 0] == 7
        assert state.logical_replica_count[0] == 2


@pytest.mark.parametrize("draft", [("*E*E", 4)], indirect=True)
def test_mtp_eplb_updates_physical_expert_metadata(draft):
    draft.update_physical_experts_metadata(16, 3)
    assert draft.num_physical_experts == 16
    assert draft.num_local_physical_experts == 3
    assert draft.num_redundant_experts == 8
    for layer in draft.model.layers.values():
        if hasattr(layer, "mixer"):
            assert layer.mixer.n_physical_experts == 16
            assert layer.mixer.n_local_physical_experts == 3
            assert layer.mixer.n_redundant_experts == 8
            layer.mixer.experts.routed_experts.update_expert_map.assert_called_once()


@pytest.mark.parametrize("draft", [("*", 0)], indirect=True)
def test_attention_only_mtp_is_not_registered_with_eplb(draft):
    assert draft.num_moe_layers == 0
    assert get_mixture_of_experts_model(draft) is None
