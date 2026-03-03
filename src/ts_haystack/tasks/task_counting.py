# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors
# SPDX-License-Identifier: MIT

"""
Counting Task Generator for TS-Haystack benchmark.

Task 3: "How many {activity} bouts occurred in this recording?"

This task requires the model to count the number of occurrences
of a specific activity within the time series window.
"""

from typing import List, Tuple

import numpy as np

from ts_haystack.core import (
    DifficultyConfig,
    GeneratedSample,
    InsertedNeedle,
)
from ts_haystack.tasks.base_task import BaseTaskGenerator


class CountingTaskGenerator(BaseTaskGenerator):
    """
    Task 3: Counting - "How many {activity} bouts occurred?"

    Algorithm:
    1. Sample background window
    2. Determine target activity (NOT in background)
    3. Sample N (number of bouts to insert) from configured range
    4. For each bout:
       a. Sample needle from bout index
       b. Find non-overlapping position with min_gap constraint
       c. Apply style transfer and insert
    5. Generate Q/A with integer count answer

    Key Constraint: Minimum gap between bouts to ensure distinguishability.

    Difficulty Knobs:
    - min_bouts / max_bouts: Higher counts are harder
    - min_gap_samples: Smaller gaps make bouts harder to distinguish
    - needle_length_ratio_range: Shorter bouts (smaller ratio) are harder to detect
    - background_purity: Mixed backgrounds add confusion

    Answer Type: integer
    """

    @property
    def task_name(self) -> str:
        return "counting"

    @property
    def answer_type(self) -> str:
        return "integer"

    def generate_sample(
        self,
        difficulty: DifficultyConfig,
        rng: np.random.Generator,
    ) -> GeneratedSample:
        """
        Generate a single counting task sample.

        Args:
            difficulty: Difficulty configuration
            rng: Random number generator (from SeedManager)

        Returns:
            GeneratedSample with counting question and integer answer
        """
        context_length = difficulty.context_length_samples

        # Step 1: Sample background window
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

        # Step 2: Determine target activity (NOT in background)
        all_activities = set(self.needle_sampler.get_available_activities())
        candidate_activities = all_activities - background.activities_present

        if not candidate_activities:
            return self._create_invalid_sample(
                "No candidate activities available",
                difficulty,
            )

        target_activity = rng.choice(list(candidate_activities))

        # Step 3: Sample N (number of bouts to insert)
        min_bouts = difficulty.task_specific.get("min_bouts", 1)
        max_bouts = difficulty.task_specific.get("max_bouts", 5)

        # Check how many bouts are available for this activity
        # No PID exclusion needed - we're selecting a different activity
        min_duration_ms, max_duration_ms = difficulty.get_needle_length_range_ms(
            self.source_hz
        )
        available_bouts = self.needle_sampler.count_available_bouts(
            activity=target_activity,
            min_duration_ms=min_duration_ms,
        )

        if available_bouts == 0:
            return self._create_invalid_sample(
                f"No bouts available for {target_activity}",
                difficulty,
            )

        # Limit max_bouts by available bouts
        max_bouts = min(max_bouts, available_bouts)
        if max_bouts < min_bouts:
            min_bouts = max_bouts

        n_bouts = int(rng.integers(min_bouts, max_bouts + 1))

        # Step 4: Insert N bouts
        min_gap_samples = difficulty.get_effective_min_gap_samples()
        margin_samples = difficulty.get_effective_margin_samples()

        # Initialize signal
        current_signal = (
            background.x.copy(),
            background.y.copy(),
            background.z.copy(),
        )
        occupied_ranges: List[Tuple[int, int]] = []
        inserted_needles: List[InsertedNeedle] = []

        for i in range(n_bouts):
            # Sample needle from bout index
            # No PID exclusion needed - target_activity is not in background
            needle = self.needle_sampler.sample_needle(
                activity=target_activity,
                min_duration_ms=min_duration_ms,
                rng=rng,
            )

            if needle is None:
                print(f"Warning: No suitable needle found, skipping counting task")
                continue  # Skip if no suitable needle found

            # Determine needle length (cap by actual needle duration)
            actual_max_duration_ms = min(max_duration_ms, needle.duration_ms)
            target_duration_ms = int(rng.integers(min_duration_ms, actual_max_duration_ms + 1))
            target_samples = int(target_duration_ms * self.source_hz / 1000)
            target_samples = min(target_samples, needle.n_samples)

            # Trim needle
            trimmed_needle = self._trim_needle(needle, target_samples)

            # Find non-overlapping position
            position = self._find_valid_position(
                context_length=context_length,
                needle_length=trimmed_needle.n_samples,
                occupied_ranges=occupied_ranges,
                min_gap=min_gap_samples,
                rng=rng,
                margin=margin_samples,
            )

            if position is None:
                print(f"Warning: No suitable position found, skipping counting task")
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
            inserted_needle = self._create_inserted_needle(
                needle=trimmed_needle,
                position=position,
                context_length=context_length,
                background=background,
            )
            inserted_needles.append(inserted_needle)

        # Check if we inserted at least one bout
        actual_count = len(inserted_needles)
        if actual_count == 0:
            return self._create_invalid_sample(
                "Could not insert any bouts",
                difficulty,
            )

        # Step 5: Generate Q/A using template bank
        question, answer = self.template_bank.sample(
            task="counting",
            rng=rng,
            activity=target_activity,
            count=actual_count,
        )

        # Sort needles by position for consistent ordering
        inserted_needles.sort(key=lambda n: n.insert_position_samples)

        # Build difficulty config with task-specific info
        full_difficulty_config = {
            **difficulty.to_dict(),
            "target_activity": target_activity,
            "requested_bouts": n_bouts,
            "actual_bouts": actual_count,
            "min_gap_samples": min_gap_samples,
            "background_activities": list(background.activities_present),
            "bout_positions": [n.insert_position_samples for n in inserted_needles],
            "bout_durations_samples": [n.duration_samples for n in inserted_needles],
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
            answer=str(actual_count),
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
        description="Generate counting task dataset for TS-Haystack"
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
        default=1,
        help="Minimum bouts to insert",
    )
    parser.add_argument(
        "--max-bouts",
        type=int,
        default=5,
        help="Maximum bouts to insert",
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

    print("Creating CountingTaskGenerator...")
    generator = CountingTaskGenerator.create_with_artifacts(seed=args.seed)

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
