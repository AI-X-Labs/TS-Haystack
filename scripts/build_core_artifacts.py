#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors
# SPDX-License-Identifier: MIT

"""
Build all Phase 1 artifacts for TS-Haystack benchmark.

This script orchestrates the generation of:
1. Per-participant activity timelines
2. Cross-participant bout index
3. Activity transition matrix

Usage:
    python scripts/build_core_artifacts.py

    # With options
    python scripts/build_core_artifacts.py \
        --n-jobs 8 \
        --label-scheme WillettsSpecific2018 \
        --min-bout-duration-ms 100
"""

import argparse
import time

from ts_haystack.core.bout_indexer import BoutIndexer
from ts_haystack.core.timeline_builder import TimelineBuilder
from ts_haystack.core.transition_matrix import TransitionMatrix


def main():
    parser = argparse.ArgumentParser(
        description="Build all Phase 1 artifacts for TS-Haystack"
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
        help="Also save human-readable JSON files for timelines"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("TS-Haystack Phase 1: Building Core Artifacts")
    print("=" * 70)
    print(f"Label scheme: {args.label_scheme}")
    print(f"Min bout duration: {args.min_bout_duration_ms}ms")
    print(f"Parallel jobs: {args.n_jobs}")
    if args.max_participants:
        print(f"Max participants: {args.max_participants}")
    print()

    total_start = time.time()

    # Step 1: Build timelines
    print("\n" + "=" * 70)
    print("Step 1/3: Building participant timelines")
    print("=" * 70)

    step_start = time.time()
    timeline_builder = TimelineBuilder(
        label_scheme=args.label_scheme,
        min_bout_duration_ms=args.min_bout_duration_ms,
    )
    timeline_builder.build_all_timelines(
        n_jobs=args.n_jobs,
        max_participants=args.max_participants,
        overwrite=args.overwrite,
        save_json=args.save_json,
    )
    print(f"Step 1 completed in {time.time() - step_start:.1f}s")

    # Step 2: Build bout index
    print("\n" + "=" * 70)
    print("Step 2/3: Building bout index")
    print("=" * 70)

    step_start = time.time()
    timelines = TimelineBuilder.load_all_timelines(max_participants=args.max_participants)
    print(f"Loaded {len(timelines)} timelines")

    bout_indexer = BoutIndexer(min_bout_duration_ms=args.min_bout_duration_ms)
    bout_index = bout_indexer.build_index(timelines)
    bout_indexer.save_index(bout_index, overwrite=args.overwrite, save_json=args.save_json)
    BoutIndexer.print_summary(bout_index)
    print(f"Step 2 completed in {time.time() - step_start:.1f}s")

    # Step 3: Build transition matrix
    print("\n" + "=" * 70)
    print("Step 3/3: Building transition matrix")
    print("=" * 70)

    step_start = time.time()
    transition_matrix = TransitionMatrix()
    transition_matrix.build_from_timelines(timelines)
    transition_matrix.save(overwrite=args.overwrite)
    transition_matrix.print_summary()
    print(f"Step 3 completed in {time.time() - step_start:.1f}s")

    # Summary
    total_time = time.time() - total_start
    print("\n" + "=" * 70)
    print("Phase 1 Complete!")
    print("=" * 70)
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"Participants: {len(timelines)}")
    print(f"Total bouts: {bout_index.total_bouts:,}")
    print(f"Activities: {len(bout_index.activities)}")


if __name__ == "__main__":
    main()
