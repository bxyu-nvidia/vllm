# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Request metadata for an engine-local training capture adapter.

The optional engine adapter resolves parents and stages native outputs. Ordinary
serving does not import any RL or TransferQueue dependency.
"""

from typing import TYPE_CHECKING, Annotated, Any, Literal

import msgspec

if TYPE_CHECKING:
    from vllm.v1.engine import EngineCoreRequest

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
    prefill_key: str | None = None


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
    """Splice the stored parent into the rendered prompt before scheduler hashing."""
    if len(prefix) != context.admission["prev_len"]:
        raise ValueError("stored parent length differs from capture admission")
    if not prefix:
        return prompt
    alignment = context.alignment
    if alignment is None:
        raise ValueError("capture parent requires an alignment boundary")
    aligned = list(prefix)
    if prefix[-1] not in alignment.eos_token_ids:
        aligned.append(alignment.turn_end_token_id)
    aligned.extend(prompt[alignment.suffix_start :])
    return aligned


def validate_capture_handoff(request: "EngineCoreRequest") -> None:
    """Validate the adapter's route boundary before NIXL scheduler admission."""
    params = request.sampling_params
    raw_context = (params.extra_args or {}).get(CAPTURE_CONTEXT_KEY) if params else None
    # The adapter has replaced the rendered prefix with exact parent tokens.
    # Its original alignment boundary no longer indexes the prepared prompt.
    context = msgspec.convert(raw_context, CaptureContext) if raw_context else None
    prompt = request.prompt_token_ids
    if context is None or not context.routed_experts or params is None or not prompt:
        raise ValueError("use_capture_handoff requires managed routed-experts capture")
    previous = context.admission.get("prev_len")
    if type(previous) is not int or previous < 0:
        raise ValueError("capture admission requires nonnegative prev_len")
    if previous > len(prompt):
        raise ValueError("capture parent exceeds the aligned prompt")
    start = max(previous - 1, 0)
    transfer = (params.extra_args or {}).get("kv_transfer_params") or {}
    if context.phase == "decode":
        remote = transfer.get("remote_num_tokens")
        if (
            not context.prefill_key
            or not transfer.get("do_remote_prefill")
            or type(remote) is not int
            or not 0 <= remote <= len(prompt)
        ):
            raise ValueError("decode capture requires an acknowledged prefill handoff")
        start = max(start, min(remote, len(prompt) - 1))
    elif not transfer.get("do_remote_decode") or transfer.get("do_remote_prefill"):
        raise ValueError("prefill capture requires a NIXL producer request")
    if params.routed_experts_prompt_start != start:
        raise ValueError(
            "capture route start differs from the prepared handoff boundary"
        )
