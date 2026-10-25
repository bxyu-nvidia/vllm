# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass, field


@dataclass
class DBOTimingStats:
    """CUDA-event stream windows for one eligible eager AG/RS target forward.

    Compute/communication are unions of windows, overlap is their intersection.
    Windows exclude explicit handoff waits but can include launch gaps and NCCL
    waiting for peers; they are not a measure of SM utilization or speedup.
    """

    compute_seconds: float
    comm_seconds: float
    overlap_seconds: float
    wall_seconds: float


DBO_TIMING_FIELDS = ("compute", "comm", "overlap", "wall")
DBO_TIMING_BUCKETS = (0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 30)


@dataclass
class DBOStats:
    """One successful non-dummy target forward on one DP engine.

    Outcomes are executed, disabled, dp_unavailable, insufficient_requests,
    rank_ineligible, or below_threshold. Fallback reasons describe the first
    failed group-wide eligibility check, not every possible blocker.
    Microbatch lists are empty for unsplit steps. Tokens exclude DP padding;
    requests count the real requests touched by each slice. For whole-request
    AG/RS, insufficient_requests means no rank can split; one-request peers
    participate with a padding-only microbatch, counted in padding tokens.
    """

    outcome: str
    microbatch_tokens: list[int] = field(default_factory=list)
    microbatch_requests: list[int] = field(default_factory=list)
    microbatch_padding_tokens: list[int] = field(default_factory=list)
    timing: DBOTimingStats | None = None


DBO_TOKEN_BUCKETS = (0, 32, 128, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536)
DBO_REQUEST_BUCKETS = (0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)
