# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass, field


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


DBO_TOKEN_BUCKETS = (0, 32, 128, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536)
DBO_REQUEST_BUCKETS = (0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)
