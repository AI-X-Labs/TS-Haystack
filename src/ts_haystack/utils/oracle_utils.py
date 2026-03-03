# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors
# SPDX-License-Identifier: MIT

"""
Oracle mode utilities for TS-Haystack.

Provides functions to format ground-truth activity segmentation as text
for oracle training mode. The oracle receives the same inputs as regular
training but includes activity labellings/segmentation metadata.
"""

import json
from typing import Dict, List, Tuple, Union

from ts_haystack.utils.timestamp_utils import (
    format_timestamp,
    parse_time_string,
)
from datetime import timedelta


def format_oracle_timeline(
    needles: Union[str, List[Dict]],
    difficulty_config: Union[str, Dict],
    recording_time_start: str,
    recording_time_end: str,
    context_length_samples: int,
) -> str:
    """
    Format ground-truth activity segmentation as a text timeline.

    This function creates a human-readable timeline of activities that can be
    prepended to prompts in oracle training mode. It shows the exact segmentation
    of activities with timestamps, marking inserted needles with [inserted] tags.

    Args:
        needles: JSON string or list of needle dictionaries. Each needle has:
            - "activity": activity label
            - "timestamp_start": start time string
            - "timestamp_end": end time string
            - "insert_position_frac": fractional position (0.0 to 1.0)
            - "duration_samples": duration in samples
        difficulty_config: JSON string or dict with:
            - "background_activities": list of background activity names
            - "global_timeline": (optional, state_query only) list of
              [start_frac, end_frac, activity] tuples
        recording_time_start: Recording start time string (e.g., "6:00:00.000 AM")
        recording_time_end: Recording end time string (e.g., "7:40:00.000 AM")
        context_length_samples: Total number of samples in the context window

    Returns:
        Formatted timeline string like:
            Activity Timeline (Ground Truth):
            Recording: 6:00:00.000 AM to 7:40:00.000 AM

            6:00:00.000 AM - 6:22:12.000 AM: sitting
            6:22:12.000 AM - 6:28:30.500 AM: bicycling  [inserted]
            6:28:30.500 AM - 7:40:00.000 AM: sitting
    """
    # Parse inputs if they're JSON strings
    if isinstance(needles, str):
        try:
            needles = json.loads(needles)
        except json.JSONDecodeError:
            needles = []

    if isinstance(difficulty_config, str):
        try:
            difficulty_config = json.loads(difficulty_config)
        except json.JSONDecodeError:
            difficulty_config = {}

    # Ensure needles is a list
    if needles is None:
        needles = []

    # Ensure difficulty_config is a dict
    if difficulty_config is None:
        difficulty_config = {}

    # Get background activities (default to "background" if not specified)
    background_activities = difficulty_config.get("background_activities", ["background"])
    if not background_activities:
        background_activities = ["background"]
    primary_background = background_activities[0]

    # Build segments: (start_frac, end_frac, activity, is_inserted)
    segments: List[Tuple[float, float, str, bool]] = []

    # Check if we have global_timeline (state_query task)
    global_timeline = difficulty_config.get("global_timeline")

    if global_timeline:
        # For state_query: use global_timeline as background
        for entry in global_timeline:
            if len(entry) >= 3:
                start_frac, end_frac, activity = entry[0], entry[1], entry[2]
                segments.append((start_frac, end_frac, activity, False))

    # Add needles as inserted segments
    for needle in needles:
        if not isinstance(needle, dict):
            continue

        activity = needle.get("activity", "unknown")
        insert_frac = needle.get("insert_position_frac", 0.0)
        duration_samples = needle.get("duration_samples", 0)

        # Calculate end fraction
        if context_length_samples > 0 and duration_samples > 0:
            duration_frac = duration_samples / context_length_samples
            end_frac = min(insert_frac + duration_frac, 1.0)
        else:
            # Fallback: use a small duration
            end_frac = min(insert_frac + 0.01, 1.0)

        segments.append((insert_frac, end_frac, activity, True))

    # If no global_timeline (non-state_query tasks), fill gaps with background
    if not global_timeline and segments:
        # Get all needle segments sorted
        needle_segments = sorted([s for s in segments if s[3]], key=lambda x: x[0])

        # Build background segments to fill gaps
        background_segments = []
        current_pos = 0.0

        for start_frac, end_frac, _, _ in needle_segments:
            # Add background before this needle if there's a gap
            if start_frac > current_pos:
                background_segments.append((current_pos, start_frac, primary_background, False))
            current_pos = max(current_pos, end_frac)

        # Add trailing background if needed
        if current_pos < 1.0:
            background_segments.append((current_pos, 1.0, primary_background, False))

        # Combine background and needle segments
        segments = background_segments + needle_segments
    elif not global_timeline and not segments:
        # No needles and no timeline - entire recording is background
        segments = [(0.0, 1.0, primary_background, False)]

    # Sort segments by start time
    segments.sort(key=lambda x: x[0])

    # Convert fractions to timestamps
    start_dt = parse_time_string(recording_time_start)
    end_dt = parse_time_string(recording_time_end)

    # Handle day wraparound
    if end_dt < start_dt:
        end_dt += timedelta(days=1)

    total_duration = (end_dt - start_dt).total_seconds()

    def frac_to_timestamp(frac: float) -> str:
        """Convert fractional position to timestamp string."""
        frac = max(0.0, min(1.0, frac))
        result_dt = start_dt + timedelta(seconds=total_duration * frac)
        return format_timestamp(result_dt)

    # Format timeline
    lines = [
        "Activity Timeline (Ground Truth):",
        f"Recording: {recording_time_start} to {recording_time_end}",
        "",
    ]

    for start_frac, end_frac, activity, is_inserted in segments:
        start_ts = frac_to_timestamp(start_frac)
        end_ts = frac_to_timestamp(end_frac)

        if is_inserted:
            lines.append(f"{start_ts} - {end_ts}: {activity}  [inserted]")
        else:
            lines.append(f"{start_ts} - {end_ts}: {activity}")

    return "\n".join(lines)


if __name__ == "__main__":
    # Test the function with sample data
    print("=" * 60)
    print("Oracle Timeline Formatter Test")
    print("=" * 60)

    # Test case 1: Simple needles without global_timeline
    needles_1 = [
        {
            "activity": "walking",
            "insert_position_frac": 0.3,
            "duration_samples": 500,
            "timestamp_start": "6:18:00.000 AM",
            "timestamp_end": "6:23:00.000 AM",
        },
        {
            "activity": "running",
            "insert_position_frac": 0.7,
            "duration_samples": 300,
            "timestamp_start": "6:42:00.000 AM",
            "timestamp_end": "6:45:00.000 AM",
        },
    ]
    difficulty_1 = {"background_activities": ["sitting"]}

    print("\nTest 1: Simple needles with background filling")
    print("-" * 40)
    result_1 = format_oracle_timeline(
        needles=needles_1,
        difficulty_config=difficulty_1,
        recording_time_start="6:00:00.000 AM",
        recording_time_end="7:00:00.000 AM",
        context_length_samples=10000,
    )
    print(result_1)

    # Test case 2: State query with global_timeline
    needles_2 = [
        {
            "activity": "bicycling",
            "insert_position_frac": 0.35,
            "duration_samples": 200,
        },
    ]
    difficulty_2 = {
        "background_activities": ["sitting", "standing"],
        "global_timeline": [
            [0.0, 0.3, "sitting"],
            [0.3, 0.7, "walking"],
            [0.7, 1.0, "sitting"],
        ],
    }

    print("\n" + "=" * 60)
    print("\nTest 2: State query with global_timeline")
    print("-" * 40)
    result_2 = format_oracle_timeline(
        needles=json.dumps(needles_2),  # Test JSON string input
        difficulty_config=json.dumps(difficulty_2),
        recording_time_start="2:00:00.000 PM",
        recording_time_end="4:00:00.000 PM",
        context_length_samples=10000,
    )
    print(result_2)

    # Test case 3: No needles (all background)
    print("\n" + "=" * 60)
    print("\nTest 3: No needles (all background)")
    print("-" * 40)
    result_3 = format_oracle_timeline(
        needles=[],
        difficulty_config={"background_activities": ["standing"]},
        recording_time_start="9:00:00.000 AM",
        recording_time_end="10:00:00.000 AM",
        context_length_samples=10000,
    )
    print(result_3)

    print("\n" + "=" * 60)
    print("Oracle Timeline Formatter Test Complete!")
    print("=" * 60)
