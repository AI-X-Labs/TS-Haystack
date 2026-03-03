# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors
# SPDX-License-Identifier: MIT

"""
Cross-participant bout indexer for TS-Haystack.

This module aggregates bouts from all participant timelines into a single
queryable index, enabling efficient sampling of "needles" for task generation.
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
import polars as pl
import pyarrow.parquet as pq
from tqdm import tqdm

from ts_haystack.core.data_structures import (
    ActivityStats,
    BoutIndex,
    BoutRef,
    ParticipantTimeline,
)
from ts_haystack.core.timeline_builder import (
    TS_HAYSTACK_DIR,
    TimelineBuilder,
)


# Output paths
BOUT_INDEX_PARQUET = os.path.join(TS_HAYSTACK_DIR, "bout_index.parquet")
BOUT_INDEX_JSON = os.path.join(TS_HAYSTACK_DIR, "bout_index.json")


def get_bout_index_path(fmt: str = "parquet") -> Path:
    """Get path to bout index file."""
    if fmt == "parquet":
        return Path(BOUT_INDEX_PARQUET)
    return Path(BOUT_INDEX_JSON)


class BoutIndexer:
    """
    Aggregates bouts across all participants into a queryable index.

    The bout index enables efficient needle sampling by:
    - Grouping bouts by activity for O(1) activity lookup
    - Pre-computing statistics for duration-based filtering
    - Supporting participant exclusion for cross-participant sampling
    """

    def __init__(self, min_bout_duration_ms: int = 100):
        """
        Initialize the bout indexer.

        Args:
            min_bout_duration_ms: Minimum bout duration to include in index
        """
        self.min_bout_duration_ms = min_bout_duration_ms

    def build_index(self, timelines: Dict[str, ParticipantTimeline]) -> BoutIndex:
        """
        Build cross-participant bout index from timelines.

        Args:
            timelines: Dictionary mapping participant ID to timeline

        Returns:
            BoutIndex with all bouts grouped by activity
        """
        by_activity: Dict[str, List[BoutRef]] = defaultdict(list)
        participants_by_activity: Dict[str, Set[str]] = defaultdict(set)

        for pid, timeline in tqdm(timelines.items(), desc="Building bout index"):
            for bout in timeline.timeline:
                if bout.duration_ms >= self.min_bout_duration_ms:
                    ref = BoutRef(
                        pid=pid,
                        start_ms=bout.start_ms,
                        end_ms=bout.end_ms,
                        duration_ms=bout.duration_ms,
                        activity=bout.activity,
                    )
                    by_activity[bout.activity].append(ref)
                    participants_by_activity[bout.activity].add(pid)

        # Compute statistics per activity
        activity_stats = {}
        for activity, bouts in by_activity.items():
            durations = np.array([b.duration_ms for b in bouts])

            activity_stats[activity] = ActivityStats(
                activity=activity,
                count=len(bouts),
                total_duration_ms=int(np.sum(durations)),
                mean_duration_ms=float(np.mean(durations)),
                min_duration_ms=int(np.min(durations)),
                max_duration_ms=int(np.max(durations)),
                participant_count=len(participants_by_activity[activity]),
            )

        return BoutIndex(by_activity=dict(by_activity), activity_stats=activity_stats)

    def save_index(
        self,
        index: BoutIndex,
        overwrite: bool = False,
        save_json: bool = True,
    ) -> None:
        """
        Save bout index to parquet and optionally JSON.

        Args:
            index: BoutIndex to save
            overwrite: Force overwrite if files exist
            save_json: Also save as JSON (default: True)
        """
        os.makedirs(TS_HAYSTACK_DIR, exist_ok=True)

        parquet_path = get_bout_index_path("parquet")

        if not overwrite and parquet_path.exists():
            print(f"Bout index already exists at {parquet_path}. Use --overwrite to rebuild.")
            return

        # Save as parquet
        rows = []
        for activity, bouts in index.by_activity.items():
            for bout in bouts:
                rows.append({
                    "activity": activity,
                    "pid": bout.pid,
                    "start_ms": bout.start_ms,
                    "end_ms": bout.end_ms,
                    "duration_ms": bout.duration_ms,
                })

        df = pl.DataFrame(rows)
        table = df.to_arrow()
        pq.write_table(table, parquet_path, compression="snappy")
        print(f"Saved bout index to {parquet_path}")

        # Save as JSON if requested
        if save_json:
            self._save_json(index)

    def _save_json(self, index: BoutIndex) -> None:
        """Save bout index as JSON (includes full bout data)."""
        json_path = get_bout_index_path("json")

        data = {
            "total_bouts": index.total_bouts,
            "activities": index.activities,
            "activity_stats": {
                act: stats.to_dict() for act, stats in index.activity_stats.items()
            },
            "by_activity": {
                act: [bout.to_dict() for bout in bouts]
                for act, bouts in index.by_activity.items()
            },
        }

        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Saved bout index JSON to {json_path}")

    @staticmethod
    def load_index(fmt: str = "parquet") -> BoutIndex:
        """
        Load pre-computed bout index.

        Args:
            fmt: Format to load from ("parquet" or "json")

        Returns:
            BoutIndex
        """
        if fmt == "json":
            return BoutIndexer._load_from_json()
        return BoutIndexer._load_from_parquet()

    @staticmethod
    def _load_from_parquet() -> BoutIndex:
        """Load bout index from parquet file."""
        parquet_path = get_bout_index_path("parquet")
        json_path = get_bout_index_path("json")

        if not parquet_path.exists():
            raise FileNotFoundError(f"Bout index not found at {parquet_path}")

        df = pl.read_parquet(parquet_path)

        by_activity: Dict[str, List[BoutRef]] = defaultdict(list)
        for row in df.iter_rows(named=True):
            ref = BoutRef(
                pid=row["pid"],
                start_ms=row["start_ms"],
                end_ms=row["end_ms"],
                duration_ms=row["duration_ms"],
                activity=row["activity"],
            )
            by_activity[row["activity"]].append(ref)

        # Load statistics from JSON if available
        activity_stats = {}
        if json_path.exists():
            with open(json_path) as f:
                data = json.load(f)
            for act, stats_data in data.get("activity_stats", {}).items():
                activity_stats[act] = ActivityStats.from_dict(stats_data)

        return BoutIndex(by_activity=dict(by_activity), activity_stats=activity_stats)

    @staticmethod
    def _load_from_json() -> BoutIndex:
        """Load bout index from JSON file."""
        json_path = get_bout_index_path("json")

        if not json_path.exists():
            raise FileNotFoundError(f"Bout index JSON not found at {json_path}")

        with open(json_path) as f:
            data = json.load(f)

        by_activity = {
            act: [BoutRef.from_dict(b) for b in bouts]
            for act, bouts in data["by_activity"].items()
        }

        activity_stats = {
            act: ActivityStats.from_dict(stats)
            for act, stats in data["activity_stats"].items()
        }

        return BoutIndex(by_activity=by_activity, activity_stats=activity_stats)

    @staticmethod
    def sample_bout(
        index: BoutIndex,
        activity: str,
        min_duration_ms: int = 0,
        max_duration_ms: Optional[int] = None,
        exclude_pids: Optional[Set[str]] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> Optional[BoutRef]:
        """
        Sample a bout matching the given criteria.

        Args:
            index: BoutIndex to sample from
            activity: Target activity type
            min_duration_ms: Minimum bout duration
            max_duration_ms: Maximum bout duration (optional)
            exclude_pids: Participant IDs to exclude
            rng: Random generator for reproducibility

        Returns:
            BoutRef if found, None if no matching bout exists
        """
        if rng is None:
            rng = np.random.default_rng()

        candidates = index.get_bouts_for_activity(
            activity,
            min_duration_ms=min_duration_ms,
            max_duration_ms=max_duration_ms,
            exclude_pids=exclude_pids,
        )

        if not candidates:
            return None

        idx = rng.integers(0, len(candidates))
        return candidates[idx]

    @staticmethod
    def print_summary(index: BoutIndex) -> None:
        """Print a summary of the bout index."""
        print(f"\nBout Index Summary")
        print("=" * 70)
        print(f"Total bouts: {index.total_bouts:,}")
        print(f"Activities: {len(index.activities)}")
        print()

        print(f"{'Activity':<25} {'Count':>10} {'Mean (s)':>10} {'Min (s)':>10} {'Max (s)':>10} {'PIDs':>6}")
        print("-" * 75)

        for activity in sorted(index.activities):
            stats = index.activity_stats[activity]
            print(
                f"{activity:<25} "
                f"{stats.count:>10,} "
                f"{stats.mean_duration_ms / 1000:>10.1f} "
                f"{stats.min_duration_ms / 1000:>10.1f} "
                f"{stats.max_duration_ms / 1000:>10.1f} "
                f"{stats.participant_count:>6}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build cross-participant bout index for TS-Haystack"
    )
    parser.add_argument(
        "--min-bout-duration-ms", "-m",
        type=int,
        default=100,
        help="Minimum bout duration in ms (default: 100)"
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
        help="Force rebuild even if files exist"
    )
    parser.add_argument(
        "--no-json",
        action="store_true",
        help="Skip saving JSON file"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("TS-Haystack Bout Indexer")
    print("=" * 60)

    available_pids = TimelineBuilder.get_available_participants()
    if not available_pids:
        print("Error: No timelines found. Run timeline_builder.py first.")
        exit(1)

    print(f"Found {len(available_pids)} participant timelines")

    timelines = TimelineBuilder.load_all_timelines(max_participants=args.max_participants)

    indexer = BoutIndexer(min_bout_duration_ms=args.min_bout_duration_ms)
    index = indexer.build_index(timelines)

    indexer.save_index(index, overwrite=args.overwrite, save_json=not args.no_json)

    BoutIndexer.print_summary(index)

    print("\n" + "=" * 60)
    print("Bout indexing complete!")
    print("=" * 60)
