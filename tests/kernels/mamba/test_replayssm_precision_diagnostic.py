# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks for the diagnostic's algebra and rounding attribution.

The diagnostic maps a state plus token-major inputs/history to counterfactual
outputs. Compare to a sequential oracle and isolated cancellation examples so
incorrect algebra cannot masquerade as an improvement in kernel precision.
"""

import pytest
import torch

from benchmarks.replayssm.check_fp16_cache import (
    ablate_saved_call,
    recurrence_scan,
    replay_history,
    replay_precision_variants,
)


@pytest.mark.parametrize("checkpoint", [False, True])
@pytest.mark.parametrize("count", [0, 5, 16])
@pytest.mark.parametrize("tokens", [1, 6])
@pytest.mark.parametrize("state_dtype", [torch.float16, torch.float32])
def test_unrounded_factorization_matches_recurrence(
    checkpoint, count, tokens, state_dtype
):
    rng = torch.Generator().manual_seed(17)

    def rand(*shape):
        return torch.randn(*shape, generator=rng).bfloat16()

    state = rand(4, 3, 8).to(state_dtype)
    x, b, c = rand(tokens, 4, 3), rand(tokens, 2, 8), rand(tokens, 2, 8)
    a, d = -torch.arange(1, 5).float(), rand(4, 1)
    dt = torch.sigmoid(rand(tokens, 4).float()) * 0.1
    history = (
        rand(count, 4, 3),
        torch.sigmoid(rand(count, 4).float()) * 0.1,
        rand(count, 2, 8),
    )
    initial = state.double()
    if count:
        _, states = recurrence_scan(
            initial,
            *history[:2],
            a,
            history[2],
            torch.zeros_like(history[2]),
            dtype=torch.float64,
        )
        initial = states[-1]
    expected, _ = recurrence_scan(initial, x, dt, a, b, c, d, dtype=torch.float64)
    variants = {
        name: (out, reconstructed)
        for name, out, reconstructed in replay_precision_variants(
            state, history, x, dt, a, b, c, d, checkpoint=checkpoint
        )
    }
    assert len(variants) == 8
    actual, reconstructed = variants["without_readout_cb_reconstruction"]
    torch.testing.assert_close(actual.double(), expected, rtol=5e-5, atol=5e-6)
    torch.testing.assert_close(reconstructed.double(), initial, rtol=5e-5, atol=5e-6)


@pytest.mark.parametrize("factor", ["readout", "cb", "reconstruction"])
def test_ablation_attributes_isolated_rounding_and_interaction(factor):
    state = torch.zeros(1, 1, 2)
    x, dt = torch.zeros(1, 1, 1), torch.zeros(1, 1)
    a, d = torch.zeros(1), torch.zeros(1, 1)
    b, c = torch.zeros(1, 1, 2), torch.tensor([[[1.0, -1.0]]])
    history = (x[:0], dt[:0], b[:0])
    epsilon = 1 / 1024
    if factor == "readout":
        state[0, 0] = torch.tensor([1 + epsilon, 1])
    elif factor == "cb":
        x.fill_(1)
        dt.fill_(1 + epsilon)
        b[0, 0, 0] = 1
    else:
        history = (
            torch.ones_like(x),
            torch.full_like(dt, 1 + epsilon),
            torch.tensor([[[1.0, 0.0]]]),
        )
    variants = {
        name: (out.item(), reconstructed)
        for name, out, reconstructed in replay_precision_variants(
            state,
            history,
            x,
            dt,
            a,
            b,
            c,
            d,
            checkpoint=factor == "reconstruction",
        )
    }
    expected = epsilon if factor == "readout" else 1 + epsilon
    assert variants["native"][0] != expected
    assert variants["without_readout_cb_reconstruction"][0] == expected
    if factor != "reconstruction":
        assert variants[f"without_{factor}"][0] == expected
    else:
        # Removing only reconstruction rounding is hidden by BF16 readout.
        assert variants["without_reconstruction"][0] == variants["native"][0]
        assert variants["without_readout_reconstruction"][0] == expected
        assert variants["without_reconstruction"][1][0, 0, 0] == expected


def test_history_wrap_excludes_uncommitted_tokens():
    snapshot = {
        "x_cache": torch.full((2, 4, 9, 3), float("nan")),
        "dt_cache": torch.full((2, 4, 9), float("nan")),
        "b_cache": torch.full((2, 2, 9, 8), float("nan")),
    }
    for tensor in snapshot.values():
        for value, index in enumerate([7, 8, 0, 1], start=1):
            tensor[0, :, index] = value
    for result in replay_history(snapshot, start=7, count=4):
        assert result.isfinite().all()
        torch.testing.assert_close(
            result.reshape(4, -1),
            torch.arange(1, 5).float()[:, None].expand(4, result[0].numel()),
        )


def test_masked_future_decay_does_not_overflow():
    state = torch.zeros(1, 1, 1)
    x = b = c = torch.ones(6, 1, 1)
    dt, a, d = torch.ones(6, 1), torch.tensor([-1000.0]), torch.zeros(1, 1)
    for _, out, _ in replay_precision_variants(
        state, (x[:0], dt[:0], b[:0]), x, dt, a, b, c, d, checkpoint=False
    ):
        torch.testing.assert_close(out, x)


def test_saved_ablation_excludes_rejected_outputs(tmp_path):
    tensors = {
        "replay_state": torch.ones(2, 1, 1, 1),
        "x_cache": torch.zeros(2, 1, 8, 1),
        "b_cache": torch.zeros(2, 1, 8, 1),
        "dt_cache": torch.zeros(2, 1, 8),
        "x": torch.zeros(1, 2, 1, 1),
        "dt": torch.zeros(1, 2, 1, 1),
        "dt_bias": torch.zeros(1, 1),
        "a": torch.zeros(1),
        "b": torch.zeros(1, 2, 1, 1),
        "c": torch.ones(1, 2, 1, 1),
        "d": torch.zeros(1, 1),
        "replay_out": torch.tensor([1.0, float("nan")]).reshape(1, 2, 1, 1),
    }
    path = tmp_path / "case.pt"
    torch.save(
        {
            "format_version": 1,
            "case": {},
            "tensors": tensors,
            "step": 0,
            "accepted": 1,
            "ring_start": 0,
            "committed": 0,
            "checkpoint": False,
        },
        path,
    )
    metrics = ablate_saved_call(path)["metrics"]["empty_history"]
    for metric in metrics.values():
        assert metric["finite"]
        assert metric["elements"] == 1
        assert metric["relative_rms"] == 0
