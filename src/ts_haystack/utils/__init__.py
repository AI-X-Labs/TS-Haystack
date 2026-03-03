# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors
# SPDX-License-Identifier: MIT

"""
Utilities for TS-Haystack benchmark.

- Timestamp conversion utilities
- Position sampling utilities
- Context length utilities
"""

from ts_haystack.utils.context_utils import (
    format_context_dir,
    parse_context_dir,
)
from ts_haystack.utils.timestamp_utils import (
    compute_duration_string,
    format_time_range,
    format_timestamp,
    ms_to_timestamp,
    parse_time_string,
    samples_to_timestamp,
    samples_to_timestamp_from_background,
)
from ts_haystack.utils.position_utils import (
    check_position_conflicts,
    compute_gaps,
    find_non_overlapping_position,
    find_sequential_positions,
    get_activity_region_at_position,
    sample_distinct_durations,
    sample_position_with_mode,
)
from ts_haystack.utils.answer_evaluation import (
    compute_time_range_iou,
    evaluate_answer,
    extract_final_answer,
    normalize_boolean,
    normalize_integer,
    parse_time_range,
)
from ts_haystack.utils.oracle_utils import (
    format_oracle_timeline,
)

__all__ = [
    # Context utilities
    "format_context_dir",
    "parse_context_dir",
    # Timestamp utilities
    "parse_time_string",
    "format_timestamp",
    "samples_to_timestamp",
    "samples_to_timestamp_from_background",
    "format_time_range",
    "compute_duration_string",
    "ms_to_timestamp",
    # Position utilities
    "sample_position_with_mode",
    "find_non_overlapping_position",
    "find_sequential_positions",
    "compute_gaps",
    "sample_distinct_durations",
    "check_position_conflicts",
    "get_activity_region_at_position",
    # Answer evaluation utilities
    "extract_final_answer",
    "parse_time_range",
    "compute_time_range_iou",
    "normalize_boolean",
    "normalize_integer",
    "evaluate_answer",
    # Oracle utilities
    "format_oracle_timeline",
]
