# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors
# SPDX-License-Identifier: MIT

"""
Position sampling utilities for TS-Haystack.

Provides functions for needle position sampling with different modes,
finding non-overlapping positions, and computing gaps between regions.
"""

from typing import List, Optional, Tuple

import numpy as np


def sample_position_with_mode(
    context_length: int,
    needle_length: int,
    mode: str,
    margin: int,
    rng: np.random.Generator,
) -> Optional[int]:
    """
    Sample needle insertion position based on mode.

    Args:
        context_length: Total context window length in samples
        needle_length: Length of needle to insert in samples
        mode: Position mode - "beginning", "middle", "end", or "random"
        margin: Minimum margin from window edges in samples
        rng: Random number generator

    Returns:
        Position index, or None if needle doesn't fit

    Modes:
        - "beginning": First 25% of window (after margin)
        - "middle": Central 50% of window
        - "end": Last 25% of window (before margin)
        - "random": Anywhere with margin constraints
    """
    # Maximum valid start position
    max_pos = context_length - needle_length - margin

    if max_pos <= margin:
        return None  # Needle doesn't fit

    if mode == "beginning":
        # First 25% of valid range
        range_end = margin + int((max_pos - margin) * 0.25)
        range_end = max(margin + 1, range_end)
        return int(rng.integers(margin, range_end))

    elif mode == "middle":
        # Central 50% of valid range
        range_size = max_pos - margin
        quarter = int(range_size * 0.25)
        range_start = margin + quarter
        range_end = max_pos - quarter
        if range_end <= range_start:
            return int(rng.integers(margin, max_pos))
        return int(rng.integers(range_start, range_end))

    elif mode == "end":
        # Last 25% of valid range
        range_start = max_pos - int((max_pos - margin) * 0.25)
        range_start = min(max_pos - 1, range_start)
        return int(rng.integers(range_start, max_pos))

    else:  # "random"
        return int(rng.integers(margin, max_pos))


def find_non_overlapping_position(
    context_length: int,
    needle_length: int,
    occupied: List[Tuple[int, int]],
    min_gap: int,
    rng: np.random.Generator,
    max_attempts: int = 100,
    margin: int = 0,
) -> Optional[int]:
    """
    Find a position that doesn't conflict with occupied ranges.

    Args:
        context_length: Total context window length in samples
        needle_length: Length of needle to insert in samples
        occupied: List of (start, end) tuples of occupied ranges
        min_gap: Minimum gap required between needles in samples
        rng: Random number generator
        max_attempts: Maximum sampling attempts before giving up
        margin: Minimum margin from window edges

    Returns:
        Valid position index, or None if no valid position found
    """
    max_start = context_length - needle_length - margin

    if max_start <= margin:
        return None

    for _ in range(max_attempts):
        # Sample candidate position
        candidate = int(rng.integers(margin, max_start))
        candidate_end = candidate + needle_length

        # Check for conflicts with all occupied ranges
        valid = True
        for occ_start, occ_end in occupied:
            # Check overlap with gap consideration
            # No overlap if: candidate_end + min_gap <= occ_start
            #            or: candidate >= occ_end + min_gap
            if not (candidate_end + min_gap <= occ_start or candidate >= occ_end + min_gap):
                valid = False
                break

        if valid:
            return candidate

    return None


def find_sequential_positions(
    context_length: int,
    needle_lengths: List[int],
    min_gap: int,
    margin: int,
    rng: np.random.Generator,
    max_gap: Optional[int] = None,
) -> Optional[List[int]]:
    """
    Find sequential non-overlapping positions for multiple needles.

    Ensures needles are placed in order with minimum gaps between them.

    Args:
        context_length: Total context window length in samples
        needle_lengths: List of needle lengths in samples
        min_gap: Minimum gap between consecutive needles
        margin: Minimum margin from window edges
        rng: Random number generator
        max_gap: Maximum gap between consecutive needles (optional).
                 If provided, gaps are constrained to [min_gap, max_gap].
                 Useful for tasks requiring tight adjacency (e.g., antecedent).

    Returns:
        List of positions, or None if needles don't fit
    """
    n_needles = len(needle_lengths)
    total_needle_length = sum(needle_lengths)
    total_gaps = (n_needles - 1) * min_gap
    total_required = total_needle_length + total_gaps + 2 * margin

    if total_required > context_length:
        return None

    # Available slack to distribute
    slack = context_length - total_required

    # Sample random extra gaps
    if n_needles > 1:
        if max_gap is not None:
            # Constrained mode: limit extra gap per needle pair
            max_extra_per_gap = max_gap - min_gap
            # Distribute slack with cap on inter-needle gaps
            # Extra gaps: [before_first, between_1_2, ..., between_n-1_n, after_last]
            extra_gaps = []
            # Before first needle: can use more slack
            extra_gaps.append(int(rng.integers(0, slack // (n_needles + 1) + 1)))
            # Between needles: constrained by max_gap
            for _ in range(n_needles - 1):
                extra_gaps.append(int(rng.integers(0, max_extra_per_gap + 1)))
            # After last needle: remaining slack goes here (not constrained)
            remaining_slack = slack - sum(extra_gaps)
            extra_gaps.append(max(0, remaining_slack))
        else:
            # Original behavior: distribute slack freely
            extra_gaps = rng.integers(0, slack // (n_needles + 1) + 1, size=n_needles + 1)
    else:
        extra_gaps = [rng.integers(0, slack + 1)]

    # Compute positions
    positions = []
    current_pos = margin + int(extra_gaps[0])

    for i, length in enumerate(needle_lengths):
        positions.append(current_pos)
        current_pos += length + min_gap
        if i + 1 < len(extra_gaps):
            current_pos += int(extra_gaps[i + 1])

    return positions


def compute_gaps(
    occupied: List[Tuple[int, int]],
    context_length: int,
) -> List[Tuple[int, int, int]]:
    """
    Compute gap ranges between occupied regions.

    Args:
        occupied: List of (start, end) tuples of occupied ranges
        context_length: Total context window length

    Returns:
        List of (start, end, length) tuples for each gap
    """
    if not occupied:
        return [(0, context_length, context_length)]

    # Sort by start position
    sorted_occupied = sorted(occupied, key=lambda x: x[0])

    gaps = []

    # Gap before first needle
    if sorted_occupied[0][0] > 0:
        gap_end = sorted_occupied[0][0]
        gaps.append((0, gap_end, gap_end))

    # Gaps between needles
    for i in range(len(sorted_occupied) - 1):
        gap_start = sorted_occupied[i][1]
        gap_end = sorted_occupied[i + 1][0]
        if gap_end > gap_start:
            gaps.append((gap_start, gap_end, gap_end - gap_start))

    # Gap after last needle
    last_end = sorted_occupied[-1][1]
    if last_end < context_length:
        gaps.append((last_end, context_length, context_length - last_end))

    return gaps


def sample_distinct_durations(
    n: int,
    min_duration: int,
    max_duration: int,
    min_diff: int,
    rng: np.random.Generator,
) -> Optional[List[int]]:
    """
    Sample n distinct durations with minimum difference between them.

    Ensures no two durations are closer than min_diff to avoid ties
    in comparison tasks.

    Args:
        n: Number of durations to sample
        min_duration: Minimum duration value
        max_duration: Maximum duration value
        min_diff: Minimum difference between any two durations
        rng: Random number generator

    Returns:
        List of n distinct durations, or None if not feasible
    """
    # Check feasibility
    range_needed = (n - 1) * min_diff
    if range_needed > (max_duration - min_duration):
        return None

    # Sample base durations and space them out
    available_range = max_duration - min_duration - range_needed
    base = int(rng.integers(min_duration, min_duration + available_range + 1))

    durations = []
    for i in range(n):
        # Add some random variation within the spacing
        variation = int(rng.integers(0, min_diff // 2 + 1)) if i > 0 else 0
        durations.append(base + i * min_diff + variation)

    # Shuffle to randomize order
    rng.shuffle(durations)
    return durations


def check_position_conflicts(
    position: int,
    length: int,
    occupied: List[Tuple[int, int]],
    min_gap: int = 0,
) -> bool:
    """
    Check if a position conflicts with occupied ranges.

    Args:
        position: Proposed start position
        length: Length of region to place
        occupied: List of (start, end) tuples of occupied ranges
        min_gap: Minimum gap required

    Returns:
        True if there's a conflict, False if position is valid
    """
    end = position + length

    for occ_start, occ_end in occupied:
        # Conflict if ranges overlap (with gap consideration)
        if not (end + min_gap <= occ_start or position >= occ_end + min_gap):
            return True

    return False


def get_activity_region_at_position(
    position: int,
    activity_timeline: List[Tuple[float, float, str]],
    context_length: int,
) -> Optional[Tuple[int, int, str]]:
    """
    Get the activity region (start, end, activity) containing a position.

    Args:
        position: Sample position
        activity_timeline: List of (start_frac, end_frac, activity) tuples
        context_length: Total context length in samples

    Returns:
        (start_samples, end_samples, activity) or None if not found
    """
    position_frac = position / context_length

    for start_frac, end_frac, activity in activity_timeline:
        if start_frac <= position_frac < end_frac:
            return (
                int(start_frac * context_length),
                int(end_frac * context_length),
                activity,
            )

    return None
