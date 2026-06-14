# SPDX-FileCopyrightText: 2026 Anonymous Authors
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Cross-participant bout index for LTAF-Haystack."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl

from src.datasets.ltaf_haystack.core.data_structures import (
    LTAFActivityStats,
    LTAFBoutIndex,
    LTAFBoutRef,
    LTAFParticipantTimeline,
)


LTAF_HAYSTACK_DIR = Path("data") / "ltafdb" / "ltaf_haystack"
BOUT_INDEX_PARQUET = LTAF_HAYSTACK_DIR / "bout_index.parquet"
BOUT_INDEX_JSON = LTAF_HAYSTACK_DIR / "bout_index.json"


def get_ltaf_bout_index_path(fmt: str = "parquet", output_root: Path | None = None) -> Path:
    if output_root is not None:
        base = Path(output_root)
        if fmt == "json":
            return base / "bout_index.json"
        return base / "bout_index.parquet"

    if fmt == "json":
        return BOUT_INDEX_JSON
    return BOUT_INDEX_PARQUET


class LTAFBoutIndexer:
    """Aggregates participant timelines into a cross-participant bout index."""

    def __init__(self, min_bout_duration_samples: int = 16):
        self.min_bout_duration_samples = int(min_bout_duration_samples)

    def build_index(self, timelines: dict[str, LTAFParticipantTimeline]) -> LTAFBoutIndex:
        by_activity: dict[str, list[LTAFBoutRef]] = defaultdict(list)
        participants_by_activity: dict[str, set[str]] = defaultdict(set)

        for participant_id, timeline in timelines.items():
            for bout in timeline.timeline:
                if bout.duration_samples < self.min_bout_duration_samples:
                    continue
                by_activity[bout.activity].append(
                    LTAFBoutRef(
                        participant_id=participant_id,
                        record_id=timeline.record_id,
                        start_sample=bout.start_sample,
                        end_sample=bout.end_sample,
                        duration_samples=bout.duration_samples,
                        activity=bout.activity,
                    )
                )
                participants_by_activity[bout.activity].add(participant_id)

        stats: dict[str, LTAFActivityStats] = {}
        for activity, refs in by_activity.items():
            durations = np.array([r.duration_samples for r in refs], dtype=np.int64)
            stats[activity] = LTAFActivityStats(
                activity=activity,
                count=len(refs),
                mean_duration_samples=float(np.mean(durations)) if len(durations) else 0.0,
                min_duration_samples=int(np.min(durations)) if len(durations) else 0,
                max_duration_samples=int(np.max(durations)) if len(durations) else 0,
            )

        return LTAFBoutIndex(by_activity=dict(by_activity), activity_stats=stats)

    def save_index(
        self,
        index: LTAFBoutIndex,
        overwrite: bool = False,
        save_json: bool = True,
        output_root: Path | None = None,
    ) -> None:
        parquet_path = get_ltaf_bout_index_path("parquet", output_root=output_root)
        parquet_path.parent.mkdir(parents=True, exist_ok=True)

        if parquet_path.exists() and not overwrite:
            print(f"LTAF bout index already exists at {parquet_path}. Use --overwrite to rebuild.")
            return

        rows: list[dict[str, object]] = []
        for activity, refs in index.by_activity.items():
            for ref in refs:
                rows.append(
                    {
                        "activity": activity,
                        "participant_id": ref.participant_id,
                        "record_id": ref.record_id,
                        "start_sample": ref.start_sample,
                        "end_sample": ref.end_sample,
                        "duration_samples": ref.duration_samples,
                    }
                )

        if rows:
            df = pl.DataFrame(rows)
        else:
            df = pl.DataFrame(
                {
                    "activity": [],
                    "participant_id": [],
                    "record_id": [],
                    "start_sample": [],
                    "end_sample": [],
                    "duration_samples": [],
                }
            )
        df.write_parquet(parquet_path, compression="snappy")
        print(f"Saved LTAF bout index to {parquet_path}")

        if save_json:
            payload = {
                "total_bouts": index.total_bouts,
                "activities": index.activities,
                "activity_stats": {
                    act: stats.to_dict() for act, stats in index.activity_stats.items()
                },
            }
            json_path = get_ltaf_bout_index_path("json", output_root=output_root)
            with json_path.open("w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, sort_keys=True)
            print(f"Saved LTAF bout index metadata to {json_path}")

    @staticmethod
    def load_index(fmt: str = "parquet", output_root: Path | None = None) -> LTAFBoutIndex:
        if fmt == "json":
            json_path = get_ltaf_bout_index_path("json", output_root=output_root)
            if not json_path.exists():
                raise FileNotFoundError(f"LTAF bout index JSON not found at {json_path}")
            with json_path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            stats = {
                activity: LTAFActivityStats.from_dict(raw)
                for activity, raw in payload.get("activity_stats", {}).items()
            }
            return LTAFBoutIndex(by_activity={}, activity_stats=stats)

        parquet_path = get_ltaf_bout_index_path("parquet", output_root=output_root)
        if not parquet_path.exists():
            raise FileNotFoundError(f"LTAF bout index not found at {parquet_path}")

        df = pl.read_parquet(parquet_path)
        by_activity: dict[str, list[LTAFBoutRef]] = defaultdict(list)
        for row in df.iter_rows(named=True):
            ref = LTAFBoutRef(
                participant_id=str(row["participant_id"]),
                record_id=str(row["record_id"]),
                start_sample=int(row["start_sample"]),
                end_sample=int(row["end_sample"]),
                duration_samples=int(row["duration_samples"]),
                activity=str(row["activity"]),
            )
            by_activity[ref.activity].append(ref)

        stats: dict[str, LTAFActivityStats] = {}
        json_path = get_ltaf_bout_index_path("json", output_root=output_root)
        if json_path.exists():
            with json_path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            for activity, raw in payload.get("activity_stats", {}).items():
                stats[activity] = LTAFActivityStats.from_dict(raw)
        else:
            for activity, refs in by_activity.items():
                durations = np.array([r.duration_samples for r in refs], dtype=np.int64)
                stats[activity] = LTAFActivityStats(
                    activity=activity,
                    count=len(refs),
                    mean_duration_samples=float(np.mean(durations)) if len(durations) else 0.0,
                    min_duration_samples=int(np.min(durations)) if len(durations) else 0,
                    max_duration_samples=int(np.max(durations)) if len(durations) else 0,
                )

        return LTAFBoutIndex(by_activity=dict(by_activity), activity_stats=stats)

    @staticmethod
    def print_summary(index: LTAFBoutIndex, max_rows: Optional[int] = None) -> None:
        print("\nLTAF Bout Index Summary")
        print("=" * 64)
        print(f"Total bouts: {index.total_bouts:,}")
        print(f"Activities: {len(index.activities)}")

        activities = sorted(index.activities)
        if max_rows is not None:
            activities = activities[:max_rows]

        print(f"{'Activity':<18} {'Count':>10} {'Mean (smp)':>12} {'Min':>8} {'Max':>8}")
        print("-" * 64)
        for activity in activities:
            stats = index.activity_stats.get(activity)
            if stats is None:
                continue
            print(
                f"{activity:<18} "
                f"{stats.count:>10,} "
                f"{stats.mean_duration_samples:>12.1f} "
                f"{stats.min_duration_samples:>8} "
                f"{stats.max_duration_samples:>8}"
            )
