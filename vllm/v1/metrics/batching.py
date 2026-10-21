# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler-visible prefill batching, excluding cache-only and draft steps.

A forward is one nonempty main-model schedule, not one layer/kernel or pipeline
microbatch. Decode-only KV consumers are excluded to avoid counting P/D requests
twice. Batch totals include decode rows in mixed batches. Per-request rows
cover initial prefill, including recomputation before its first output. Completed
request counts are emitted at first output; aborted unfinished prefills are absent.
"""

from collections import Counter
from dataclasses import dataclass, field

# Keep Rust scheduler metric fields/reasons in sync.
BATCH_COUNTER_FIELDS = (
    "completed_requests",
    "request_forwards",
    "request_tokens",
    "request_forward_observations",
    "batch_tokens",
    "batch_requests",
    "batches",
    "unused_token_slots",
    "underfilled_batches",
)
UNDERFILL_REASONS = (
    "mamba_alignment",
    "long_prefill_limit",
    "prefill_lookahead",
    "kv_capacity",
    "kv_lookup_pending",
    "kv_load_pending",
    "blocked_request",
    "request_limit",
    "dp_prefill_throttle",
    "lora_limit",
    "chunked_prefill_disabled",
    "input_budget_reserved",
    "scheduling_token_limit",
    "encoder_constraint",
    "in_flight_output",
    "preemption",
    "request_tail",
    "no_other_constraint_observed",
)


@dataclass
class PrefillBatchingStats:
    """Delta observations drained with scheduler stats; no request-ID labels."""

    completed_requests: int = 0
    request_forwards: int = 0
    request_tokens: int = 0
    request_forward_observations: int = 0
    batch_tokens: int = 0
    batch_requests: int = 0
    batches: int = 0
    unused_token_slots: int = 0
    underfilled_batches: int = 0
    underfilled_reasons: dict[str, int] = field(default_factory=dict)

    def observe_batch(
        self,
        request_tokens: list[int],
        *,
        total_tokens: int,
        requests: int,
        token_budget: int,
        reasons: list[str],
    ) -> None:
        """Observe a prefill-containing batch, counting each constraint once."""
        if not request_tokens:
            return
        self.request_tokens += sum(request_tokens)
        self.request_forward_observations += len(request_tokens)
        self.batch_tokens += total_tokens
        self.batch_requests += requests
        self.batches += 1
        unused = max(0, token_budget - total_tokens)
        self.unused_token_slots += unused
        if unused:
            self.underfilled_batches += 1
            counts = Counter(self.underfilled_reasons)
            counts.update(set(reasons) or {"no_other_constraint_observed"})
            self.underfilled_reasons = dict(counts)
