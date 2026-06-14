# SPDX-License-Identifier: CC-BY-NC-4.0
"""Query-budget heuristics shared across ARTS domains.

The orchestrator's per-query segment length and its number of tool-call rounds
both scale with the recording length so that short recordings stay cheap while
long ones get enough refinement budget.
"""

from __future__ import annotations

MIN_QUERY_SEC = 0.1    # Floor: allow sub-second queries for short recordings
MAX_QUERY_SEC = 120.0  # Ceiling: cap at 120s to avoid context bloat from chunked results
QUERY_FRACTION = 0.15  # Fraction-based limit within the floor/ceiling

DEFAULT_TOOL_ROUNDS = 15


def get_max_query_seconds(recording_duration_sec: float) -> float:
    """Adaptive per-query limit: fraction-based, clamped to [MIN, MAX]."""
    return max(MIN_QUERY_SEC, min(MAX_QUERY_SEC, QUERY_FRACTION * recording_duration_sec))


def get_tool_rounds(recording_duration_sec: float, cli_override: int | None = None) -> int:
    """Scale tool rounds with recording length. CLI override takes precedence."""
    if cli_override is not None:
        return cli_override
    if recording_duration_sec <= 300:
        return DEFAULT_TOOL_ROUNDS
    elif recording_duration_sec <= 900:
        return 20
    elif recording_duration_sec <= 3600:
        return 25
    else:
        return 40
