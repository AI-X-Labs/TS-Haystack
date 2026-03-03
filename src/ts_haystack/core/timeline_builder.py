# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors
# SPDX-License-Identifier: MIT

"""
Timeline builder for extracting activity bouts from Capture-24 sensor data.

This module provides functionality to:
- Load raw sensor data and annotations per participant
- Map annotations to activity labels using a specified label scheme
- Merge consecutive same-activity samples into contiguous bouts
- Save timelines as parquet files for efficient loading
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from joblib import Parallel, delayed
from tqdm import tqdm

from ts_haystack.capture24.loader import (
    CAPTURE24_DATA_DIR,
    get_sensor_data_dir,
    load_participant_sensor_data,
)
from ts_haystack.capture24.classification import (
    load_label_mapping,
)
from ts_haystack.core.data_structures import (
    BoutRecord,
    ParticipantTimeline,
)


# Output directory for timelines
TS_HAYSTACK_DIR = os.path.join(CAPTURE24_DATA_DIR, "ts_haystack")
TIMELINES_DIR = os.path.join(TS_HAYSTACK_DIR, "timelines")


def get_timelines_dir() -> Path:
    """Get the path to the timelines directory."""
    return Path(TIMELINES_DIR)


def get_timeline_path(pid: str, fmt: str = "parquet") -> Path:
    """
    Get the path for a participant's timeline file.

    Args:
        pid: Participant ID (e.g., "P001")
        fmt: File format ("parquet" or "json")

    Returns:
        Path to the timeline file
    """
    return get_timelines_dir() / f"{pid}.{fmt}"


class TimelineBuilder:
    """
    Builds activity timelines by merging consecutive same-activity annotations
    into contiguous bouts.

    The timeline extraction process:
    1. Load participant sensor data with annotations
    2. Map annotations to labels using the specified label scheme
    3. Identify activity transitions (where label changes)
    4. Group consecutive same-label samples into bouts
    5. Filter by minimum bout duration
    6. Save as parquet for efficient loading

    Example:
        >>> builder = TimelineBuilder(label_scheme="WillettsSpecific2018")
        >>> timeline = builder.build_participant_timeline("P001")
        >>> print(f"Found {timeline.num_bouts} bouts")
        >>> print(f"Activities: {timeline.activities_present}")
    """

    def __init__(
        self,
        label_scheme: str = "WillettsSpecific2018",
        source_hz: int = 100,
        min_bout_duration_ms: int = 100,  # 0.1 seconds = 10 samples at 100Hz
    ):
        """
        Initialize the timeline builder.

        Args:
            label_scheme: Label scheme for mapping annotations to activities
            source_hz: Source data sampling frequency in Hz
            min_bout_duration_ms: Minimum bout duration to keep (default: 100ms)
        """
        self.label_scheme = label_scheme
        self.source_hz = source_hz
        self.min_bout_duration_ms = min_bout_duration_ms

        # Load label mapping
        self.label_mapping = load_label_mapping(label_scheme)

    def build_participant_timeline(self, pid: str) -> ParticipantTimeline:
        """
        Build timeline for a single participant.

        Args:
            pid: Participant ID (e.g., "P001")

        Returns:
            ParticipantTimeline with all bouts for this participant
        """
        # Load sensor data with annotations
        df = load_participant_sensor_data(pid, downsample_hz=self.source_hz)

        # Extract bouts from annotations
        bouts = self._merge_annotations_to_bouts(df)

        # Build bouts_by_activity index
        bouts_by_activity: Dict[str, List[BoutRecord]] = defaultdict(list)
        for bout in bouts:
            bouts_by_activity[bout.activity].append(bout)

        # Compute recording metadata
        recording_start_ms = int(df["timestamp_ms"].min())
        recording_end_ms = int(df["timestamp_ms"].max())
        total_duration_ms = recording_end_ms - recording_start_ms

        return ParticipantTimeline(
            participant_id=pid,
            total_duration_ms=total_duration_ms,
            recording_start_ms=recording_start_ms,
            recording_end_ms=recording_end_ms,
            timeline=bouts,
            bouts_by_activity=dict(bouts_by_activity),
        )

    def _merge_annotations_to_bouts(self, df: pl.DataFrame) -> List[BoutRecord]:
        """
        Merge consecutive same-activity samples into bouts.

        Algorithm:
        1. Map annotations to labels using label_scheme
        2. Identify activity transitions (label[i] != label[i-1])
        3. Assign bout IDs via cumulative sum of transitions
        4. Group by bout_id and aggregate start/end timestamps
        5. Filter by min_bout_duration_ms

        Args:
            df: Polars DataFrame with columns: timestamp_ms, annotation

        Returns:
            List of BoutRecord in chronological order
        """
        # Map annotations to labels
        # Create a mapping expression
        df = df.with_columns(
            pl.col("annotation")
            .replace(self.label_mapping, default=None)
            .alias("label")
        )

        # Filter out rows with unmapped annotations
        df = df.filter(pl.col("label").is_not_null())

        if len(df) == 0:
            return []

        # Identify transitions: where label differs from previous row
        df = df.with_columns(
            (pl.col("label") != pl.col("label").shift(1))
            .fill_null(True)
            .alias("is_transition")
        )

        # Assign bout IDs via cumulative sum
        df = df.with_columns(
            pl.col("is_transition").cum_sum().alias("bout_id")
        )

        # Group by bout_id and aggregate
        bouts_df = df.group_by("bout_id").agg(
            pl.col("timestamp_ms").min().alias("start_ms"),
            pl.col("timestamp_ms").max().alias("end_ms"),
            pl.col("label").first().alias("activity"),
        )

        # Compute duration and filter
        # Note: end_ms is the timestamp of the last sample, so we add one sample duration
        sample_duration_ms = 1000 // self.source_hz  # e.g., 10ms at 100Hz
        bouts_df = bouts_df.with_columns(
            (pl.col("end_ms") - pl.col("start_ms") + sample_duration_ms).alias("duration_ms")
        )

        # Filter by minimum duration
        bouts_df = bouts_df.filter(pl.col("duration_ms") >= self.min_bout_duration_ms)

        # Sort by start time
        bouts_df = bouts_df.sort("start_ms")

        # Convert to BoutRecord objects
        bouts = []
        for row in bouts_df.iter_rows(named=True):
            bouts.append(
                BoutRecord(
                    start_ms=row["start_ms"],
                    end_ms=row["end_ms"],
                    activity=row["activity"],
                    duration_ms=row["duration_ms"],
                )
            )

        return bouts

    def build_all_timelines(
        self,
        n_jobs: int = 1,
        max_participants: Optional[int] = None,
        overwrite: bool = False,
        save_json: bool = False,
    ) -> None:
        """
        Build and save timelines for all participants.

        Args:
            n_jobs: Number of parallel jobs
            max_participants: Limit number of participants (for testing)
            overwrite: Force rebuild even if files exist
            save_json: Also save human-readable JSON files
        """
        # Create output directory
        os.makedirs(TIMELINES_DIR, exist_ok=True)

        # Get list of participants from sensor data directory
        sensor_dir = Path(get_sensor_data_dir(self.source_hz))
        participant_dirs = sorted(sensor_dir.glob("pid=P*"))
        pids = [d.name.replace("pid=", "") for d in participant_dirs]

        if max_participants is not None:
            pids = pids[:max_participants]

        print(f"Building timelines for {len(pids)} participants...")
        print(f"  Label scheme: {self.label_scheme}")
        print(f"  Min bout duration: {self.min_bout_duration_ms}ms")
        print(f"  Output: {TIMELINES_DIR}")

        # Filter out already processed if not overwriting
        if not overwrite:
            pids_to_process = [
                pid for pid in pids if not get_timeline_path(pid).exists()
            ]
            if len(pids_to_process) < len(pids):
                print(f"  Skipping {len(pids) - len(pids_to_process)} already processed")
            pids = pids_to_process

        if len(pids) == 0:
            print("  All timelines already exist. Use --overwrite to rebuild.")
            return

        # Process participants
        def process_one(pid: str) -> Dict:
            """Process a single participant and return stats."""
            try:
                timeline = self.build_participant_timeline(pid)
                self._save_timeline(timeline, save_json=save_json)
                return {
                    "pid": pid,
                    "success": True,
                    "num_bouts": timeline.num_bouts,
                    "activities": len(timeline.activities_present),
                    "duration_hours": timeline.total_duration_ms / (1000 * 60 * 60),
                }
            except Exception as e:
                return {"pid": pid, "success": False, "error": str(e)}

        if n_jobs == 1:
            results = [process_one(pid) for pid in tqdm(pids, desc="Building timelines")]
        else:
            results = Parallel(n_jobs=n_jobs)(
                delayed(process_one)(pid) for pid in tqdm(pids, desc="Building timelines")
            )

        # Print summary
        successes = [r for r in results if r["success"]]
        failures = [r for r in results if not r["success"]]

        print(f"\nCompleted: {len(successes)} participants")
        if failures:
            print(f"Failed: {len(failures)} participants")
            for f in failures[:5]:
                print(f"  {f['pid']}: {f['error']}")

        if successes:
            total_bouts = sum(r["num_bouts"] for r in successes)
            total_hours = sum(r["duration_hours"] for r in successes)
            print(f"Total bouts: {total_bouts:,}")
            print(f"Total recording time: {total_hours:.1f} hours")

    def _save_timeline(self, timeline: ParticipantTimeline, save_json: bool = False) -> None:
        """
        Save a timeline to parquet (and optionally JSON).

        Args:
            timeline: Timeline to save
            save_json: Also save human-readable JSON
        """
        pid = timeline.participant_id

        # Save as parquet
        parquet_path = get_timeline_path(pid, "parquet")
        self._save_timeline_parquet(timeline, parquet_path)

        # Optionally save as JSON for debugging
        if save_json:
            json_path = get_timeline_path(pid, "json")
            with open(json_path, "w") as f:
                json.dump(timeline.to_dict(), f, indent=2)

    def _save_timeline_parquet(self, timeline: ParticipantTimeline, path: Path) -> None:
        """Save timeline to parquet format."""
        # Convert to flat structure for parquet
        data = {
            "participant_id": timeline.participant_id,
            "recording_start_ms": timeline.recording_start_ms,
            "recording_end_ms": timeline.recording_end_ms,
            "total_duration_ms": timeline.total_duration_ms,
            "bout_start_ms": [b.start_ms for b in timeline.timeline],
            "bout_end_ms": [b.end_ms for b in timeline.timeline],
            "bout_activity": [b.activity for b in timeline.timeline],
            "bout_duration_ms": [b.duration_ms for b in timeline.timeline],
        }

        df = pl.DataFrame(data)
        table = df.to_arrow()
        pq.write_table(table, path, compression="snappy")

    @staticmethod
    def load_timeline(pid: str) -> ParticipantTimeline:
        """
        Load a pre-computed timeline for a participant.

        Args:
            pid: Participant ID

        Returns:
            ParticipantTimeline
        """
        path = get_timeline_path(pid, "parquet")
        if not path.exists():
            raise FileNotFoundError(f"Timeline not found for {pid} at {path}")

        df = pl.read_parquet(path)

        # Parquet format: one row per bout, with metadata duplicated
        # Get metadata from first row
        first_row = df.row(0, named=True)

        # Reconstruct bouts from all rows
        bouts = []
        for row in df.iter_rows(named=True):
            bouts.append(BoutRecord(
                start_ms=row["bout_start_ms"],
                end_ms=row["bout_end_ms"],
                activity=row["bout_activity"],
                duration_ms=row["bout_duration_ms"],
            ))

        # Build bouts_by_activity
        bouts_by_activity: Dict[str, List[BoutRecord]] = defaultdict(list)
        for bout in bouts:
            bouts_by_activity[bout.activity].append(bout)

        return ParticipantTimeline(
            participant_id=first_row["participant_id"],
            total_duration_ms=first_row["total_duration_ms"],
            recording_start_ms=first_row["recording_start_ms"],
            recording_end_ms=first_row["recording_end_ms"],
            timeline=bouts,
            bouts_by_activity=dict(bouts_by_activity),
        )

    @staticmethod
    def load_all_timelines(max_participants: Optional[int] = None) -> Dict[str, ParticipantTimeline]:
        """
        Load all pre-computed timelines.

        Args:
            max_participants: Limit number of participants (for testing)

        Returns:
            Dictionary mapping participant ID to timeline
        """
        timelines_dir = get_timelines_dir()
        if not timelines_dir.exists():
            raise FileNotFoundError(f"Timelines directory not found at {timelines_dir}")

        timeline_files = sorted(timelines_dir.glob("P*.parquet"))

        if max_participants is not None:
            timeline_files = timeline_files[:max_participants]

        timelines = {}
        for path in tqdm(timeline_files, desc="Loading timelines"):
            pid = path.stem
            timelines[pid] = TimelineBuilder.load_timeline(pid)

        return timelines

    @staticmethod
    def get_available_participants() -> List[str]:
        """Get list of participants with available timelines."""
        timelines_dir = get_timelines_dir()
        if not timelines_dir.exists():
            return []
        return sorted([p.stem for p in timelines_dir.glob("P*.parquet")])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build activity timelines from Capture-24 data"
    )
    parser.add_argument(
        "--label-scheme", "-l",
        type=str,
        default="WillettsSpecific2018",
        help="Label scheme for activity mapping (default: WillettsSpecific2018)"
    )
    parser.add_argument(
        "--min-bout-duration-ms", "-m",
        type=int,
        default=100,
        help="Minimum bout duration in ms (default: 100)"
    )
    parser.add_argument(
        "--n-jobs", "-j",
        type=int,
        default=1,
        help="Number of parallel jobs (default: 1)"
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
        "--save-json",
        action="store_true",
        help="Also save human-readable JSON files"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("TS-Haystack Timeline Builder")
    print("=" * 60)

    builder = TimelineBuilder(
        label_scheme=args.label_scheme,
        min_bout_duration_ms=args.min_bout_duration_ms,
    )

    builder.build_all_timelines(
        n_jobs=args.n_jobs,
        max_participants=args.max_participants,
        overwrite=args.overwrite,
        save_json=args.save_json,
    )

    print("\n" + "=" * 60)
    print("Timeline building complete!")
    print("=" * 60)
