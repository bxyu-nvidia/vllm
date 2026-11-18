# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Request metadata for an engine-local training capture adapter.

The optional engine adapter resolves parents and stages native outputs. Ordinary
serving does not import any RL or TransferQueue dependency.
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
    engine_capture: bool = False


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


def align_capture_prompt(
    context: CaptureContext, prompt: list[int], prefix: list[int]
) -> list[int]:
    """Splice the exact stored parent before scheduler hashing or KV lookup.

    In engine capture mode the frontend boundary addresses the rendered prompt;
    otherwise it addresses a prompt that the frontend already aligned.
    """
    if len(prefix) != context.admission["prev_len"]:
        raise ValueError("stored parent length differs from capture admission")
    if not prefix:
        return prompt
    alignment = context.alignment
    if alignment is None:
        raise ValueError("capture parent requires an alignment boundary")
    if not context.engine_capture:
        if prompt[: len(prefix)] != prefix:
            raise ValueError("prompt differs from the committed capture parent")
        return prompt
    aligned = list(prefix)
    if prefix[-1] not in alignment.eos_token_ids:
        aligned.append(alignment.turn_end_token_id)
    aligned.extend(prompt[alignment.suffix_start :])
    return aligned
