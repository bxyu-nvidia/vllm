# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Request metadata for an engine-local training capture adapter.

This module transports and validates coordination data only. It does not join a
Ray cluster, resolve parent tokens, or acknowledge TQ writes.
"""

from typing import Annotated, Any, Literal

import msgspec

CAPTURE_CONTEXT_KEY = "nemo_rl_capture"
NonNegativeInt = Annotated[int, msgspec.Meta(ge=0)]
TokenId = Annotated[int, msgspec.Meta(ge=0, le=2**32 - 1)]


class CaptureAlignment(msgspec.Struct, forbid_unknown_fields=True):  # type: ignore[call-arg]
    """Boundary in EngineCoreRequest.prompt_token_ids, after the turn terminator."""

    suffix_start: NonNegativeInt
    turn_end_token_id: TokenId
    eos_token_ids: list[TokenId]


class CaptureContext(msgspec.Struct, forbid_unknown_fields=True):  # type: ignore[call-arg]
    """Router identity and frontend-computed alignment for one P/D leg."""

    version: Literal[1]
    admission: dict[str, Any]
    weight_version: NonNegativeInt
    phase: Literal["prefill", "decode"]
    top_logprobs: NonNegativeInt
    routed_experts: bool
    alignment: CaptureAlignment | None = None


def get_capture_context(
    extra_args: dict[str, Any] | None,
    prompt_token_ids: list[int] | None,
) -> CaptureContext | None:
    """Read capture metadata without loading any RL or TQ dependency."""
    if extra_args is None or CAPTURE_CONTEXT_KEY not in extra_args:
        return None
    context = msgspec.convert(extra_args[CAPTURE_CONTEXT_KEY], CaptureContext)
    prev_len = context.admission.get("prev_len")
    if type(prev_len) is not int or prev_len < 0:
        raise ValueError("capture admission requires nonnegative prev_len")
    if prompt_token_ids is None:
        raise ValueError("capture context requires prompt_token_ids")
    alignment = context.alignment
    if prev_len == 0:
        if alignment is not None:
            raise ValueError("parentless capture must not replace a prompt prefix")
    elif alignment is None:
        raise ValueError("capture with a parent requires an alignment boundary")
    elif (
        not 0 < alignment.suffix_start <= len(prompt_token_ids)
        or not alignment.eos_token_ids
        or alignment.turn_end_token_id not in alignment.eos_token_ids
        or prompt_token_ids[alignment.suffix_start - 1] not in alignment.eos_token_ids
    ):
        raise ValueError("invalid capture alignment boundary")
    return context
