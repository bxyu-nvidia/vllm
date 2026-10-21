# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler-visible prefill batching, lookup latency, and request delays.

A forward is one nonempty main-model schedule, not one layer/kernel or pipeline
microbatch. Decode-only KV consumers are excluded to avoid counting P/D requests
twice. Batch totals include decode rows in mixed batches. Per-request rows
cover initial prefill, including recomputation before its first output. Completed
request counts are emitted at first output; aborted unfinished prefills are absent.
Batch/forward counters exclude cache-only and draft steps. Lookup/delay counters
also cover schedules without a forward. Delay counts are request-step observations,
not unique requests or a snapshot of the entire queue: unvisited requests are
unknown. The denominator includes any step that examines an initial prefill,
even if another constraint prevents it from running.
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
    "delay_steps",
    "kv_lookup_completed",
    "kv_lookup_duration_ns",
    "kv_lookup_pending_completed",
    "kv_lookup_pending_duration_ns",
    "kv_lookup_cancelled",
)
DELAY_REASONS = ("mamba_alignment", "kv_load_pending", "kv_lookup_pending")
DELAY_COUNTER_FIELDS = (
    "delayed_requests",
    "blocked_requests",
)
UNDERFILL_REASONS = (
    "mamba_alignment",
    "mamba_alignment_chunk_boundary",
    "mamba_alignment_budget_too_small",
    "mamba_alignment_resume_boundary",
    "mamba_alignment_last_cache_boundary",
    "mamba_alignment_prompt_checkpoint",
    "mamba_alignment_shared_prefix",
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
    delay_steps: int = 0
    kv_lookup_completed: int = 0
    kv_lookup_duration_ns: int = 0
    kv_lookup_pending_completed: int = 0
    kv_lookup_pending_duration_ns: int = 0
    kv_lookup_cancelled: int = 0
    delayed_requests: dict[str, int] = field(default_factory=dict)
    blocked_requests: dict[str, int] = field(default_factory=dict)

    def observe_delays(
        self,
        candidates: dict[str, str],
        *,
        scheduled_tokens: dict[str, int],
        has_prefill: bool,
    ) -> None:
        """Count request-step observations, including steps without a forward.

        Candidates have a sole observed blocker, not proven counterfactual
        readiness: later admission checks and pending cache hit lengths may be
        unknown.
        """
        if not has_prefill and not candidates:
            return
        self.delay_steps += 1
        for request_id, reason in candidates.items():
            self.delayed_requests[reason] = self.delayed_requests.get(reason, 0) + 1
            if not scheduled_tokens.get(request_id):
                self.blocked_requests[reason] = self.blocked_requests.get(reason, 0) + 1

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
