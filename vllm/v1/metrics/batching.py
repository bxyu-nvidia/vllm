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
Readiness is observed after lookup/load, alignment, and encoder/lookahead gates,
but before KV allocation. Ready counts include scheduled requests; ready-unscheduled
counts include allocation failures or subsequent preemption. Requests not examined
in this step are reported separately, never assumed ready. Readiness observations
are engine-step counts, including steps without a forward, not queue snapshots.
Outcome accounting partitions all active initial prefills once per step, including
unexamined requests. It records the observed stopping condition, not every possible
blocker. A scheduled request wins over earlier constraints; partial work is scheduled.
The underfilled subset uses the same partition and only nonempty prefill forwards.
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
    "kv_lookup_timed_completed",
    "kv_lookup_timed_failed",
    "kv_lookup_queue_duration_ns",
    "kv_lookup_execution_duration_ns",
    "kv_lookup_observation_duration_ns",
    "readiness_steps",
    "ready_requests",
    "ready_unscheduled_requests",
    "unexamined_requests",
    "outcome_steps",
    "underfilled_outcome_steps",
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


REQUEST_OUTCOMES = (
    "scheduled",
    "kv_lookup_pending",
    "kv_load_pending",
    "in_flight_output",
    "mamba_alignment",
    "prefill_lookahead",
    "dp_prefill_throttle",
    "kv_capacity",
    "encoder_pending",
    "lora_limit",
    "grammar_pending",
    "streaming_input_pending",
    "chunked_prefill_disabled",
    "spec_decode_padding",
    "model_length_limit",
    "not_examined",
    "other",
)
QUEUE_STOP_REASONS = (
    "token_budget",
    "input_budget_reserved",
    "request_limit",
    "kv_capacity",
    "mamba_alignment",
    "prefill_lookahead",
    "dp_prefill_throttle",
    "chunked_prefill_disabled",
    "spec_decode_padding",
    "encoder_pending",
    "scheduler_paused",
    "preemption",
    "other",
)
# Field -> Prometheus label and its bounded values. No request IDs are exported.
OUTCOME_COUNTER_FIELDS = {
    "request_outcomes": ("outcome", REQUEST_OUTCOMES),
    "unexamined_by_stop_reason": ("stop_reason", QUEUE_STOP_REASONS),
    "underfilled_request_outcomes": ("outcome", REQUEST_OUTCOMES),
    "underfilled_unexamined_by_stop_reason": ("stop_reason", QUEUE_STOP_REASONS),
}


@dataclass
class PrefillLookupStats:
    """Completed instrumented lookup attempts, including observed failures.

    All three durations share this cohort. Execution includes the client RPC
    round trip, worker processing and response decoding; it is not GPU time or
    just the store existence check. Discarded attempts are excluded.
    """

    completed: int = 0
    failed: int = 0
    queue_duration_ns: int = 0
    execution_duration_ns: int = 0
    observation_duration_ns: int = 0

    def aggregate(self, other: "PrefillLookupStats") -> None:
        self.completed += other.completed
        self.failed += other.failed
        self.queue_duration_ns += other.queue_duration_ns
        self.execution_duration_ns += other.execution_duration_ns
        self.observation_duration_ns += other.observation_duration_ns


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
    kv_lookup_timed_completed: int = 0
    kv_lookup_timed_failed: int = 0
    kv_lookup_queue_duration_ns: int = 0
    kv_lookup_execution_duration_ns: int = 0
    kv_lookup_observation_duration_ns: int = 0
    readiness_steps: int = 0
    ready_requests: int = 0
    ready_unscheduled_requests: int = 0
    unexamined_requests: int = 0
    delayed_requests: dict[str, int] = field(default_factory=dict)
    blocked_requests: dict[str, int] = field(default_factory=dict)

    outcome_steps: int = 0
    underfilled_outcome_steps: int = 0
    request_outcomes: dict[str, int] = field(default_factory=dict)
    unexamined_by_stop_reason: dict[str, int] = field(default_factory=dict)
    underfilled_request_outcomes: dict[str, int] = field(default_factory=dict)
    underfilled_unexamined_by_stop_reason: dict[str, int] = field(default_factory=dict)

    def observe_outcomes(
        self,
        outcomes: dict[str, int],
        unexamined: dict[str, int],
        *,
        underfilled: bool,
    ) -> None:
        """Partition active initial prefills once per step by observed outcome.

        Unexamined requests inherit the queue's stop reason, not a claim that
        they themselves are blocked by it. Underfilled observations include only
        nonempty prefill-containing forwards below max_num_batched_tokens.
        """
        if not outcomes:
            return
        self.outcome_steps += 1
        if underfilled:
            self.underfilled_outcome_steps += 1
        for prefix in ("", "underfilled_") if underfilled else ("",):
            for field_name, counts in (
                ("request_outcomes", outcomes),
                ("unexamined_by_stop_reason", unexamined),
            ):
                target = getattr(self, prefix + field_name)
                for reason, count in counts.items():
                    target[reason] = target.get(reason, 0) + count

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
