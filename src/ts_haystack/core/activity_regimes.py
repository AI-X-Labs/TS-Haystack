# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors
# SPDX-License-Identifier: MIT

"""
Activity regime definitions for WillettsSpecific2018 label scheme.

Activities within the same regime have similar statistical properties
(variance, frequency content) in accelerometer signals, making them
appropriate as distractors for each other.

This module groups activities into two broad regimes:
1. Sedentary: Low movement intensity, low signal variance
2. Active: Significant movement, higher signal variance

Using two regimes (instead of more granular groupings) ensures sufficient
activities per group for meaningful distractor sampling.
"""

from typing import Dict, List, Set


# =============================================================================
# Regime Definitions
# =============================================================================

# Two-regime grouping for WillettsSpecific2018 labels
# Grouped by signal variance / movement intensity
WILLETTS_ACTIVITY_REGIMES: Dict[str, List[str]] = {
    # Sedentary regime: Low movement intensity, low signal variance
    # Includes sleep, sitting, standing
    "sedentary": [
        "sleep",
        "sitting",
        "standing",
        "vehicle",
    ],

    # Active regime: Significant movement, higher signal variance
    # Includes walking, exercise, manual labor, household chores, vehicle
    "active": [
        "walking",
        "mixed-activity",
        "bicycling",
        "manual-work",
        "sports",
        "household-chores",
    ],
}

# Reverse mapping: activity -> regime
ACTIVITY_TO_REGIME: Dict[str, str] = {
    activity: regime
    for regime, activities in WILLETTS_ACTIVITY_REGIMES.items()
    for activity in activities
}

# All known activities
ALL_ACTIVITIES: Set[str] = set(ACTIVITY_TO_REGIME.keys())


# =============================================================================
# Utility Functions
# =============================================================================


def get_regime(activity: str) -> str:
    """
    Get the regime for a given activity.

    Args:
        activity: Activity label (WillettsSpecific2018)

    Returns:
        Regime name ("sedentary" or "active")

    Raises:
        ValueError: If activity is not in the mapping
    """
    if activity not in ACTIVITY_TO_REGIME:
        raise ValueError(
            f"Unknown activity '{activity}'. "
            f"Known activities: {sorted(ALL_ACTIVITIES)}"
        )
    return ACTIVITY_TO_REGIME[activity]


def get_regime_activities(regime: str) -> Set[str]:
    """
    Get all activities in a given regime.

    Args:
        regime: Regime name ("sedentary" or "active")

    Returns:
        Set of activity labels in that regime

    Raises:
        ValueError: If regime is not known
    """
    if regime not in WILLETTS_ACTIVITY_REGIMES:
        raise ValueError(
            f"Unknown regime '{regime}'. "
            f"Known regimes: {list(WILLETTS_ACTIVITY_REGIMES.keys())}"
        )
    return set(WILLETTS_ACTIVITY_REGIMES[regime])


def get_same_regime_activities(activity: str) -> Set[str]:
    """
    Get all activities in the same regime as the given activity.

    Args:
        activity: Activity label

    Returns:
        Set of all activities in the same regime (including the input activity)
    """
    regime = ACTIVITY_TO_REGIME.get(activity)
    if regime is None:
        return set()
    return set(WILLETTS_ACTIVITY_REGIMES[regime])


def get_distractor_candidates(target_activity: str) -> Set[str]:
    """
    Get activities that can serve as distractors for the target.

    Distractors are activities in the SAME regime but different from target.
    They have similar statistical properties, forcing the model to distinguish
    activity patterns rather than just detecting variance changes.

    Args:
        target_activity: The activity being asked about

    Returns:
        Set of activity labels that can serve as distractors
    """
    same_regime = get_same_regime_activities(target_activity)
    return same_regime - {target_activity}


def get_other_regime_activities(activity: str) -> Set[str]:
    """
    Get all activities in the OTHER regime from the given activity.

    Useful for selecting negative targets that are clearly distinguishable
    from inserted needles (different signal characteristics).

    Args:
        activity: Activity label

    Returns:
        Set of activities from the other regime
    """
    regime = ACTIVITY_TO_REGIME.get(activity)
    if regime is None:
        return ALL_ACTIVITIES.copy()

    other_regime = "active" if regime == "sedentary" else "sedentary"
    return set(WILLETTS_ACTIVITY_REGIMES[other_regime])


def get_regime_for_activities(activities: Set[str]) -> str:
    """
    Get the regime that contains all given activities.

    Args:
        activities: Set of activity labels

    Returns:
        Regime name if all activities are in the same regime

    Raises:
        ValueError: If activities span multiple regimes or are unknown
    """
    if not activities:
        raise ValueError("Cannot determine regime for empty activity set")

    regimes = {ACTIVITY_TO_REGIME.get(a) for a in activities}
    regimes.discard(None)

    if len(regimes) != 1:
        raise ValueError(
            f"Activities span multiple regimes or contain unknowns: {activities}"
        )

    return regimes.pop()


def filter_activities_by_regime(
    activities: Set[str],
    regime: str,
) -> Set[str]:
    """
    Filter a set of activities to only those in the specified regime.

    Args:
        activities: Set of activity labels to filter
        regime: Target regime ("sedentary" or "active")

    Returns:
        Subset of activities that belong to the specified regime
    """
    regime_activities = get_regime_activities(regime)
    return activities & regime_activities
