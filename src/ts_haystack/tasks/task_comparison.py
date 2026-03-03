# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors
# SPDX-License-Identifier: MIT

"""
Comparison Task Generator for TS-Haystack benchmark.

Task 7: "What was the {longest/shortest} period {with/without} {activity}?"

This task tests comparison and negation reasoning: the model must identify
the extremum (longest or shortest) period either containing or lacking
a specific activity.
"""

from typing import List, Optional, Tuple

import numpy as np

from ts_haystack.core import (
    DifficultyConfig,
    GeneratedSample,
    InsertedNeedle,
)
from ts_haystack.tasks.base_task import BaseTaskGenerator
from ts_haystack.utils import (
    compute_gaps,
    sample_distinct_durations,
)


class ComparisonTaskGenerator(BaseTaskGenerator):
    """
    Task 7: Comparison - "What was the {longest/shortest} period {with/without} {activity}?"

    This task tests:
    - Comparison reasoning: finding extrema among multiple periods
    - Negation understanding: "with" vs "without" an activity
    - Duration estimation: comparing lengths of different periods

    Algorithm:
    1. Sample question type: extremum (longest/shortest) x polarity (with/without)
    2. Sample background
    3. Sample target activity NOT in background
    4. Insert N bouts (>=2) with DISTINCT durations (no ties)
    5. For "with" polarity: answer is extremum needle bout
    6. For "without" polarity: compute gaps and answer is extremum gap
    7. Q: "What was the {longest/shortest} period {with/without} {activity}?"
    8. A: Time range of the extremum period

    Question Variants (2x2 = 4 types):
    | Extremum  | Polarity | Meaning                          |
    |-----------|----------|----------------------------------|
    | longest   | with     | Longest bout OF the activity     |
    | shortest  | with     | Shortest bout OF the activity    |
    | longest   | without  | Longest gap BETWEEN bouts        |
    | shortest  | without  | Shortest gap BETWEEN bouts       |

    Difficulty Knobs:
    - min_bouts / max_bouts: More periods = harder
    - min_duration_diff_ms: Smaller diff = closer to ties = harder
    - extremum: "longest" vs "shortest"
    - polarity: "with" vs "without"

    Answer Type: time_range
    """

    @property
    def task_name(self) -> str:
        return "comparison"

    @property
    def answer_type(self) -> str:
        return "time_range"

    def generate_sample(
        self,
        difficulty: DifficultyConfig,
        rng: np.random.Generator,
    ) -> GeneratedSample:
        """
        Generate a single comparison task sample.

        Args:
            difficulty: Difficulty configuration
            rng: Random number generator (from SeedManager)

        Returns:
            GeneratedSample with comparison question and time_range answer
        """
        context_length = difficulty.context_length_samples
        min_bouts = difficulty.task_specific.get("min_bouts", 2)
        max_bouts = difficulty.task_specific.get("max_bouts", 4)
        min_duration_diff_ms = difficulty.get_effective_min_duration_diff_ms(self.source_hz)
        min_gap_samples = difficulty.get_effective_min_gap_samples()
        margin_samples = difficulty.get_effective_margin_samples()

        # =====================================================================
        # Step 1: Sample question type (extremum x polarity)
        # =====================================================================
        extremum = rng.choice(["longest", "shortest"])
        polarity = rng.choice(["with", "without"])

        # =====================================================================
        # Step 2: Sample background
        # =====================================================================
        background = self.background_sampler.sample_background(
            context_length_samples=context_length,
            purity=difficulty.background_purity,
            rng=rng,
        )

        if background is None:
            return self._create_invalid_sample(
                "Failed to sample background",
                difficulty,
            )

        # Validate annotation coverage
        is_valid, reason = self._validate_background_coverage(background, difficulty)
        if not is_valid:
            return self._create_invalid_sample(reason, difficulty)

        # =====================================================================
        # Step 3: Sample target activity (NOT in background)
        # =====================================================================
        all_activities = set(self.needle_sampler.get_available_activities())
        candidate_activities = all_activities - background.activities_present

        if not candidate_activities:
            return self._create_invalid_sample(
                "No candidate target activities available",
                difficulty,
            )

        target_activity = rng.choice(list(candidate_activities))

        # =====================================================================
        # Step 4: Sample number of bouts and their durations
        # =====================================================================
        n_bouts = int(rng.integers(min_bouts, max_bouts + 1))

        min_duration_ms, max_duration_ms = difficulty.get_needle_length_range_ms(
            self.source_hz
        )

        # Sample distinct durations (ensures no ties)
        durations_ms = sample_distinct_durations(
            n=n_bouts,
            min_duration=min_duration_ms,
            max_duration=max_duration_ms,
            min_diff=min_duration_diff_ms,
            rng=rng,
        )

        if durations_ms is None:
            return self._create_invalid_sample(
                f"Could not sample {n_bouts} distinct durations in range "
                f"[{min_duration_ms}, {max_duration_ms}] with min_diff={min_duration_diff_ms}",
                difficulty,
            )

        # =====================================================================
        # Step 5: Sample and insert bouts with non-overlapping positions
        # =====================================================================
        # Initialize signal
        current_signal = (
            background.x.copy(),
            background.y.copy(),
            background.z.copy(),
        )
        occupied_ranges: List[Tuple[int, int]] = []
        inserted_needles: List[InsertedNeedle] = []
        actual_durations_ms: List[int] = []

        for duration_ms in durations_ms:
            # Sample needle from bout index
            needle = self.needle_sampler.sample_needle(
                activity=target_activity,
                min_duration_ms=duration_ms,
                rng=rng,
            )

            if needle is None:
                continue  # Skip if no suitable needle found

            # Trim to exact duration
            target_samples = int(duration_ms * self.source_hz / 1000)
            target_samples = min(target_samples, needle.n_samples)
            trimmed_needle = self._trim_needle(needle, target_samples)

            # Find valid non-overlapping position
            position = self._find_valid_position(
                context_length=context_length,
                needle_length=trimmed_needle.n_samples,
                occupied_ranges=occupied_ranges,
                min_gap=min_gap_samples,
                rng=rng,
                margin=margin_samples,
            )

            if position is None:
                continue  # No valid position found

            # Record occupied range
            occupied_ranges.append((position, position + trimmed_needle.n_samples))

            # Apply style transfer and insert
            local_stats = self.style_transfer.compute_local_statistics(
                background=current_signal,
                position=position,
            )
            transferred = self.style_transfer.transfer(trimmed_needle, local_stats)

            current_signal = self.style_transfer.insert_with_blending(
                background=current_signal,
                needle=(transferred.x, transferred.y, transferred.z),
                position=position,
            )

            # Create needle metadata
            inserted_needle = InsertedNeedle(
                activity=target_activity,
                source_pid=trimmed_needle.source_pid,
                source_start_ms=trimmed_needle.start_ms,
                source_end_ms=trimmed_needle.end_ms,
                insert_position_samples=position,
                insert_position_frac=position / context_length,
                duration_samples=trimmed_needle.n_samples,
                duration_ms=duration_ms,  # Use requested duration for consistency
                timestamp_start=self._samples_to_timestamp(position, background),
                timestamp_end=self._samples_to_timestamp(
                    position + trimmed_needle.n_samples, background
                ),
            )
            inserted_needles.append(inserted_needle)
            actual_durations_ms.append(duration_ms)

        # Verify we have at least 2 bouts for comparison
        if len(inserted_needles) < 2:
            return self._create_invalid_sample(
                f"Could only insert {len(inserted_needles)} bouts, need at least 2 for comparison",
                difficulty,
            )

        # Sort needles by position for consistent ordering
        inserted_needles.sort(key=lambda n: n.insert_position_samples)

        # =====================================================================
        # Step 6: Determine answer based on extremum and polarity
        # =====================================================================
        if polarity == "with":
            # Find longest/shortest period WITH the activity (i.e., needle bouts)
            periods = [
                (nm.timestamp_start, nm.timestamp_end, nm.duration_samples)
                for nm in inserted_needles
            ]
        else:
            # Find longest/shortest period WITHOUT the activity (i.e., gaps)
            occupied_for_gaps = [
                (nm.insert_position_samples, nm.insert_position_samples + nm.duration_samples)
                for nm in inserted_needles
            ]
            gap_ranges = compute_gaps(occupied_for_gaps, context_length)

            # Convert gaps to (timestamp_start, timestamp_end, duration_samples)
            periods = [
                (
                    self._samples_to_timestamp(start, background),
                    self._samples_to_timestamp(end, background),
                    length,
                )
                for start, end, length in gap_ranges
            ]

        if not periods:
            return self._create_invalid_sample(
                f"No valid periods found for polarity='{polarity}'",
                difficulty,
            )

        # Sort by duration to find extremum
        # longest = descending (largest first), shortest = ascending (smallest first)
        periods_sorted = sorted(
            periods,
            key=lambda p: p[2],
            reverse=(extremum == "longest"),
        )

        answer_period = periods_sorted[0]
        runner_up = periods_sorted[1] if len(periods_sorted) > 1 else None

        # =====================================================================
        # Step 7: Generate Q/A using template bank
        # =====================================================================
        # Convert duration to ms for template
        duration_ms_answer = int(answer_period[2] * 1000 / self.source_hz)

        # Select appropriate template set based on polarity:
        # - "comparison_with": templates for finding longest/shortest activity BOUTS
        # - "comparison_without": templates for finding longest/shortest GAPS between bouts
        template_task = f"comparison_{polarity}"

        question, answer = self.template_bank.sample(
            task=template_task,
            rng=rng,
            extremum=extremum,
            activity=target_activity,
            start=answer_period[0],
            end=answer_period[1],
            duration_ms=duration_ms_answer,
        )

        # Compute difficulty indicator: how close is runner-up?
        duration_diff_to_runner_up = None
        if runner_up:
            duration_diff_to_runner_up = abs(answer_period[2] - runner_up[2])

        # Build difficulty config with task-specific info
        full_difficulty_config = {
            **difficulty.to_dict(),
            "target_activity": target_activity,
            "extremum": extremum,
            "polarity": polarity,
            "n_bouts_inserted": len(inserted_needles),
            "all_periods": [
                {"start": p[0], "end": p[1], "duration_samples": p[2]}
                for p in periods
            ],
            "answer_period": {
                "start": answer_period[0],
                "end": answer_period[1],
                "duration_samples": answer_period[2],
            },
            "runner_up_period": {
                "start": runner_up[0],
                "end": runner_up[1],
                "duration_samples": runner_up[2],
            } if runner_up else None,
            "duration_diff_to_runner_up_samples": duration_diff_to_runner_up,
            "background_activities": list(background.activities_present),
        }

        final_x, final_y, final_z = current_signal

        return GeneratedSample(
            x=final_x,
            y=final_y,
            z=final_z,
            task_type=self.task_name,
            context_length_samples=context_length,
            background_pid=background.pid,
            recording_time_range=background.recording_time_context,
            question=question,
            answer=answer,
            answer_type=self.answer_type,
            needles=inserted_needles,
            difficulty_config=full_difficulty_config,
            is_valid=True,
        )


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate comparison task dataset for TS-Haystack"
    )
    parser.add_argument(
        "--context-lengths",
        type=int,
        nargs="+",
        default=[10000, 50000],
        help="Context lengths in samples",
    )
    parser.add_argument(
        "--samples-per-split",
        type=int,
        nargs=3,
        default=[1000, 100, 100],
        help="Samples for train, val, test",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Master seed",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of parallel jobs",
    )
    parser.add_argument(
        "--min-bouts",
        type=int,
        default=2,
        help="Minimum bouts to insert",
    )
    parser.add_argument(
        "--max-bouts",
        type=int,
        default=4,
        help="Maximum bouts to insert",
    )
    parser.add_argument(
        "--min-duration-diff",
        type=int,
        default=2000,
        help="Minimum duration difference between bouts in ms",
    )
    parser.add_argument(
        "--needle-ratio-min",
        type=float,
        default=0.02,
        help="Minimum needle length as fraction of context (default: 0.02 = 2%%)",
    )
    parser.add_argument(
        "--needle-ratio-max",
        type=float,
        default=0.08,
        help="Maximum needle length as fraction of context (default: 0.08 = 8%%)",
    )
    parser.add_argument(
        "--needle-position",
        type=str,
        choices=["random", "beginning", "middle", "end"],
        default="random",
        help="Needle position mode",
    )
    parser.add_argument(
        "--background-purity",
        type=str,
        choices=["pure", "mixed", "any"],
        default="pure",
        help="Background purity mode ('any' samples random window, adapts to context)",
    )
    parser.add_argument(
        "--min-gap-ratio",
        type=float,
        default=0.02,
        help="Min gap as fraction of context (default: 0.02 = 2%%)",
    )
    parser.add_argument(
        "--min-gap-max-samples",
        type=int,
        default=100,
        help="Maximum min_gap in samples (default: 100)",
    )
    parser.add_argument(
        "--margin-ratio",
        type=float,
        default=0.02,
        help="Margin as fraction of context (default: 0.02 = 2%%)",
    )
    parser.add_argument(
        "--margin-max-samples",
        type=int,
        default=100,
        help="Maximum margin in samples (default: 100)",
    )

    args = parser.parse_args()

    print("Creating ComparisonTaskGenerator...")
    generator = ComparisonTaskGenerator.create_with_artifacts(seed=args.seed)

    for context_length in args.context_lengths:
        print(f"\nGenerating for context_length={context_length}...")

        difficulty = DifficultyConfig(
            context_length_samples=context_length,
            needle_position=args.needle_position,
            needle_length_ratio_range=(args.needle_ratio_min, args.needle_ratio_max),
            background_purity=args.background_purity,
            task_specific={
                "min_bouts": args.min_bouts,
                "max_bouts": args.max_bouts,
                "min_duration_diff_ratio": 0.02,
                "min_duration_diff_max_ms": args.min_duration_diff,
                "min_gap_ratio": args.min_gap_ratio,
                "min_gap_max_samples": args.min_gap_max_samples,
                "margin_ratio": args.margin_ratio,
                "margin_max_samples": args.margin_max_samples,
            },
        )

        for split, n_samples in zip(
            ["train", "val", "test"], args.samples_per_split
        ):
            print(f"  Generating {n_samples} {split} samples...")
            samples = generator.generate_dataset(
                n_samples=n_samples,
                difficulty=difficulty,
                split=split,
                n_jobs=args.n_jobs,
            )

            output_path = generator.save_dataset(
                samples=samples,
                split=split,
                context_length=context_length,
            )
            print(f"  Saved to: {output_path}")

    print("\nDone!")
