# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors
# SPDX-License-Identifier: MIT

"""
Activity transition matrix for TS-Haystack.

Computes P(activity_next | activity_current) from timeline data, enabling
realistic needle-background pairings based on natural activity sequences.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
from tqdm import tqdm

from ts_haystack.core.data_structures import (
    ParticipantTimeline,
)
from ts_haystack.core.timeline_builder import (
    TS_HAYSTACK_DIR,
    TimelineBuilder,
)


TRANSITION_MATRIX_PATH = os.path.join(TS_HAYSTACK_DIR, "transition_matrix.json")


def get_transition_matrix_path() -> Path:
    """Get path to transition matrix file."""
    return Path(TRANSITION_MATRIX_PATH)


class TransitionMatrix:
    """
    Computes and stores aggregate activity transition probabilities.

    The transition matrix captures P(activity_next | activity_current) from
    observed activity sequences across all participants.
    """

    def __init__(self):
        """Initialize empty transition matrix."""
        self.activities: List[str] = []
        self.activity_to_idx: Dict[str, int] = {}
        self.matrix: Optional[np.ndarray] = None
        self._counts: Optional[np.ndarray] = None

    def build_from_timelines(
        self,
        timelines: Dict[str, ParticipantTimeline],
    ) -> None:
        """
        Build transition matrix from participant timelines.

        For each consecutive (bout_i, bout_{i+1}) pair across all participants:
            counts[activity_i][activity_{i+1}] += 1

        Then row-normalize to get probabilities.
        """
        # Collect all activities
        all_activities: Set[str] = set()
        for timeline in timelines.values():
            all_activities.update(timeline.activities_present)

        self.activities = sorted(all_activities)
        self.activity_to_idx = {act: i for i, act in enumerate(self.activities)}
        n = len(self.activities)

        # Count transitions
        self._counts = np.zeros((n, n), dtype=np.int64)

        for timeline in tqdm(timelines.values(), desc="Computing transitions"):
            bouts = timeline.timeline
            for i in range(len(bouts) - 1):
                from_idx = self.activity_to_idx[bouts[i].activity]
                to_idx = self.activity_to_idx[bouts[i + 1].activity]
                self._counts[from_idx, to_idx] += 1

        # Row-normalize to get probabilities
        row_sums = self._counts.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        self.matrix = self._counts / row_sums

    def get_transition_prob(self, from_activity: str, to_activity: str) -> float:
        """Get P(to_activity | from_activity)."""
        if from_activity not in self.activity_to_idx:
            return 0.0
        if to_activity not in self.activity_to_idx:
            return 0.0

        from_idx = self.activity_to_idx[from_activity]
        to_idx = self.activity_to_idx[to_activity]
        return float(self.matrix[from_idx, to_idx])

    def sample_successor(
        self,
        activity: str,
        exclude: Optional[Set[str]] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> Optional[str]:
        """Sample a plausible successor activity based on transition probabilities."""
        if rng is None:
            rng = np.random.default_rng()

        if activity not in self.activity_to_idx:
            return None

        from_idx = self.activity_to_idx[activity]
        probs = self.matrix[from_idx].copy()

        if exclude:
            for exc in exclude:
                if exc in self.activity_to_idx:
                    probs[self.activity_to_idx[exc]] = 0.0

        total = probs.sum()
        if total == 0:
            return None
        probs = probs / total

        idx = rng.choice(len(self.activities), p=probs)
        return self.activities[idx]

    def sample_predecessor(
        self,
        activity: str,
        exclude: Optional[Set[str]] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> Optional[str]:
        """Sample a plausible predecessor activity."""
        if rng is None:
            rng = np.random.default_rng()

        if activity not in self.activity_to_idx:
            return None

        to_idx = self.activity_to_idx[activity]
        counts = self._counts[:, to_idx].astype(float)

        if exclude:
            for exc in exclude:
                if exc in self.activity_to_idx:
                    counts[self.activity_to_idx[exc]] = 0.0

        total = counts.sum()
        if total == 0:
            return None
        probs = counts / total

        idx = rng.choice(len(self.activities), p=probs)
        return self.activities[idx]

    def save(self, path: Optional[Path] = None, overwrite: bool = False) -> None:
        """Save transition matrix to JSON."""
        if path is None:
            path = get_transition_matrix_path()

        os.makedirs(path.parent, exist_ok=True)

        if not overwrite and path.exists():
            print(f"Transition matrix already exists at {path}. Use --overwrite to rebuild.")
            return

        data = {
            "activities": self.activities,
            "matrix": self.matrix.tolist(),
            "counts": self._counts.tolist(),
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        print(f"Saved transition matrix to {path}")

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "TransitionMatrix":
        """Load transition matrix from JSON."""
        if path is None:
            path = get_transition_matrix_path()

        if not path.exists():
            raise FileNotFoundError(f"Transition matrix not found at {path}")

        with open(path) as f:
            data = json.load(f)

        tm = cls()
        tm.activities = data["activities"]
        tm.activity_to_idx = {act: i for i, act in enumerate(tm.activities)}
        tm.matrix = np.array(data["matrix"])
        tm._counts = np.array(data["counts"])

        return tm

    def print_summary(self) -> None:
        """Print transition matrix summary and the full matrix."""
        print(f"\nTransition Matrix Summary")
        print("=" * 60)
        print(f"Activities: {len(self.activities)}")
        print(f"Total transitions: {int(self._counts.sum()):,}")

        if self.matrix is not None:
            self._print_full_matrix()

    def _print_full_matrix(self) -> None:
        """Print the full transition matrix as a formatted table."""
        n = len(self.activities)
        if n == 0:
            return

        # Abbreviate activity names for column headers (max 8 chars)
        abbrev = [act[:8] for act in self.activities]

        # Calculate column width
        col_width = max(10, max(len(a) for a in abbrev) + 1)
        row_label_width = max(len(act) for act in self.activities) + 2

        print(f"\nTransition Probabilities P(column | row):")
        print("-" * (row_label_width + col_width * n + 4))

        # Header row
        header = " " * row_label_width + "\u2502"
        for a in abbrev:
            header += f"{a:^{col_width}}"
        print(header)
        print("-" * row_label_width + "\u253c" + "-" * (col_width * n))

        # Data rows
        for i, act in enumerate(self.activities):
            row = f"{act:<{row_label_width}}\u2502"
            for j in range(n):
                prob = self.matrix[i, j]
                if prob == 0:
                    row += f"{'\u00b7':^{col_width}}"
                elif prob >= 0.1:
                    row += f"{prob:^{col_width}.2f}"
                else:
                    row += f"{prob:^{col_width}.3f}"
            print(row)

        print("-" * (row_label_width + col_width * n + 4))

        # Also print raw counts
        print(f"\nRaw Transition Counts:")
        print("-" * (row_label_width + col_width * n + 4))

        # Header row
        header = " " * row_label_width + "\u2502"
        for a in abbrev:
            header += f"{a:^{col_width}}"
        print(header)
        print("-" * row_label_width + "\u253c" + "-" * (col_width * n))

        # Data rows
        for i, act in enumerate(self.activities):
            row = f"{act:<{row_label_width}}\u2502"
            for j in range(n):
                count = int(self._counts[i, j])
                if count == 0:
                    row += f"{'\u00b7':^{col_width}}"
                else:
                    row += f"{count:^{col_width}}"
            print(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build activity transition matrix for TS-Haystack"
    )
    parser.add_argument(
        "--max-participants", "-n",
        type=int,
        default=None,
        help="Limit number of participants (for testing)"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Force rebuild even if file exists"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("TS-Haystack Transition Matrix")
    print("=" * 60)

    matrix_path = get_transition_matrix_path()

    # Load existing matrix if available (unless --overwrite)
    if matrix_path.exists() and not args.overwrite:
        print(f"Loading existing transition matrix from {matrix_path}")
        matrix = TransitionMatrix.load(matrix_path)
    else:
        # Build from timelines
        available_pids = TimelineBuilder.get_available_participants()
        if not available_pids:
            print("Error: No timelines found. Run timeline_builder.py first.")
            exit(1)

        print(f"Found {len(available_pids)} participant timelines")
        print("Building transition matrix...")

        timelines = TimelineBuilder.load_all_timelines(max_participants=args.max_participants)

        matrix = TransitionMatrix()
        matrix.build_from_timelines(timelines)
        matrix.save(overwrite=args.overwrite)

    matrix.print_summary()

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
