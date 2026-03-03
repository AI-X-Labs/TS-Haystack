# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors
# SPDX-License-Identifier: MIT

"""
Ordering Task Generator for TS-Haystack benchmark.

Task 4: "Did {activity_a} occur before {activity_b}?"
        "Which occurred first, {activity_a} or {activity_b}?"

This task requires the model to determine the temporal order
of two different activities within the time series window.
"""

import numpy as np

from ts_haystack.core import (
    DifficultyConfig,
    GeneratedSample,
    InsertedNeedle,
    NeedleSample,
)
from ts_haystack.tasks.base_task import BaseTaskGenerator
from ts_haystack.utils import find_sequential_positions


class OrderingTaskGenerator(BaseTaskGenerator):
    """
    Task 4: Temporal Ordering - "Did {activity_a} occur before {activity_b}?"

    Algorithm:
    1. Sample two distinct activities (A and B)
    2. Sample background EXCLUDING both A and B (prevents ambiguity)
    3. Sample needles for both activities
    4. Randomly decide temporal order (50/50 for balanced labels)
    5. Compute sequential positions using find_sequential_positions
    6. Insert both needles in temporal order with style transfer
    7. Generate Q/A based on the true order

    Why Needle Insertion (not natural bout selection):
    - Natural selection would allow solving via activity transition matrix
    - Insertion gives experimental control and prevents data leakage
    - Ensures balanced positive/negative labels

    Difficulty Knobs:
    - needle_length_ratio_range: Shorter needles (smaller ratio) are harder to detect
    - min_gap_samples: Smaller gap makes ordering harder
    - context_length_samples: Longer windows require more scanning
    - question_format: "boolean" vs "category"

    Answer Type: boolean (default) or category
    """

    @property
    def task_name(self) -> str:
        return "ordering"

    @property
    def answer_type(self) -> str:
        return "boolean"

    def generate_sample(
        self,
        difficulty: DifficultyConfig,
        rng: np.random.Generator,
    ) -> GeneratedSample:
        """
        Generate a single ordering task sample.

        Args:
            difficulty: Difficulty configuration
            rng: Random number generator (from SeedManager)

        Returns:
            GeneratedSample with ordering question and boolean/category answer
        """
        context_length = difficulty.context_length_samples
        min_gap = difficulty.get_effective_min_gap_samples()
        margin = difficulty.get_effective_margin_samples()

        # =====================================================================
        # Step 1: Sample two distinct activities
        # =====================================================================
        all_activities = list(self.needle_sampler.get_available_activities())

        if len(all_activities) < 2:
            return self._create_invalid_sample(
                "Need at least 2 activities for ordering",
                difficulty,
            )

        # Sample two different activities
        sampled = rng.choice(all_activities, size=2, replace=False)
        activity_a, activity_b = sampled[0], sampled[1]

        # =====================================================================
        # Step 2: Sample background EXCLUDING both activities
        # =====================================================================
        # This ensures the only occurrences of activity_a and activity_b
        # in the final signal are our inserted needles
        background = self.background_sampler.sample_background(
            context_length_samples=context_length,
            purity=difficulty.background_purity,
            excluded_activities={activity_a, activity_b},
            rng=rng,
        )

        if background is None:
            return self._create_invalid_sample(
                "Failed to sample background excluding both activities",
                difficulty,
            )

        # Validate annotation coverage
        is_valid, reason = self._validate_background_coverage(background, difficulty)
        if not is_valid:
            return self._create_invalid_sample(reason, difficulty)

        # Verify exclusion worked
        if activity_a in background.activities_present or activity_b in background.activities_present:
            return self._create_invalid_sample(
                "Background contains one of the target activities",
                difficulty,
            )

        # =====================================================================
        # Step 3: Sample needles for both activities
        # =====================================================================
        # Note: No PID exclusion needed - we're selecting activities not in
        # the background, so even if from same participant, data won't overlap
        min_duration_ms, max_duration_ms = difficulty.get_needle_length_range_ms(
            self.source_hz
        )

        needle_a = self.needle_sampler.sample_needle(
            activity=activity_a,
            min_duration_ms=min_duration_ms,
            rng=rng,
        )

        needle_b = self.needle_sampler.sample_needle(
            activity=activity_b,
            min_duration_ms=min_duration_ms,
            rng=rng,
        )

        if needle_a is None:
            return self._create_invalid_sample(
                f"Could not sample needle for {activity_a}",
                difficulty,
            )

        if needle_b is None:
            return self._create_invalid_sample(
                f"Could not sample needle for {activity_b}",
                difficulty,
            )

        # Trim needles to target lengths
        trimmed_a = self._trim_needle_to_range(needle_a, min_duration_ms, max_duration_ms, rng)
        trimmed_b = self._trim_needle_to_range(needle_b, min_duration_ms, max_duration_ms, rng)

        # =====================================================================
        # Step 4: Randomly decide temporal order (50/50 balance)
        # =====================================================================
        a_first = rng.random() < 0.5

        if a_first:
            # A comes before B
            first_needle = trimmed_a
            second_needle = trimmed_b
            first_activity = activity_a
            second_activity = activity_b
        else:
            # B comes before A
            first_needle = trimmed_b
            second_needle = trimmed_a
            first_activity = activity_b
            second_activity = activity_a

        # =====================================================================
        # Step 5: Compute sequential positions
        # =====================================================================
        needle_lengths = [first_needle.n_samples, second_needle.n_samples]

        positions = find_sequential_positions(
            context_length=context_length,
            needle_lengths=needle_lengths,
            min_gap=min_gap,
            margin=margin,
            rng=rng,
        )

        if positions is None:
            return self._create_invalid_sample(
                "Needles too long for context window",
                difficulty,
            )

        pos_first, pos_second = positions

        # =====================================================================
        # Step 6: Insert both needles in temporal order
        # =====================================================================
        # Start with background signal
        current_signal = (
            background.x.copy(),
            background.y.copy(),
            background.z.copy(),
        )

        # Insert first needle
        local_stats_first = self.style_transfer.compute_local_statistics(
            background=current_signal,
            position=pos_first,
        )
        transferred_first = self.style_transfer.transfer(first_needle, local_stats_first)
        current_signal = self.style_transfer.insert_with_blending(
            background=current_signal,
            needle=(transferred_first.x, transferred_first.y, transferred_first.z),
            position=pos_first,
        )

        # Insert second needle
        local_stats_second = self.style_transfer.compute_local_statistics(
            background=current_signal,
            position=pos_second,
        )
        transferred_second = self.style_transfer.transfer(second_needle, local_stats_second)
        current_signal = self.style_transfer.insert_with_blending(
            background=current_signal,
            needle=(transferred_second.x, transferred_second.y, transferred_second.z),
            position=pos_second,
        )

        # Create needle metadata
        needle_metadata = [
            self._create_inserted_needle(first_needle, pos_first, context_length, background),
            self._create_inserted_needle(second_needle, pos_second, context_length, background),
        ]
        # Update activity names (since _create_inserted_needle uses needle.activity)
        needle_metadata[0] = InsertedNeedle(
            activity=first_activity,
            source_pid=first_needle.source_pid,
            source_start_ms=first_needle.start_ms,
            source_end_ms=first_needle.end_ms,
            insert_position_samples=pos_first,
            insert_position_frac=pos_first / context_length,
            duration_samples=first_needle.n_samples,
            duration_ms=first_needle.duration_ms,
            timestamp_start=self._samples_to_timestamp(pos_first, background),
            timestamp_end=self._samples_to_timestamp(pos_first + first_needle.n_samples, background),
        )
        needle_metadata[1] = InsertedNeedle(
            activity=second_activity,
            source_pid=second_needle.source_pid,
            source_start_ms=second_needle.start_ms,
            source_end_ms=second_needle.end_ms,
            insert_position_samples=pos_second,
            insert_position_frac=pos_second / context_length,
            duration_samples=second_needle.n_samples,
            duration_ms=second_needle.duration_ms,
            timestamp_start=self._samples_to_timestamp(pos_second, background),
            timestamp_end=self._samples_to_timestamp(pos_second + second_needle.n_samples, background),
        )

        # =====================================================================
        # Step 7: Generate Q/A
        # =====================================================================
        question_format = difficulty.task_specific.get("question_format", "boolean")

        # Generate question and answer using template bank
        # Question always asks about activity_a and activity_b (original order)
        question, answer = self.template_bank.sample(
            task="ordering",
            rng=rng,
            activity_a=activity_a,
            activity_b=activity_b,
            a_before_b=a_first,
            first_activity=first_activity,
            second_activity=second_activity,
        )

        # Determine final answer type
        if question_format == "category":
            # For category format, answer is the activity name
            answer = first_activity
            final_answer_type = "category"
        else:
            final_answer_type = "boolean"

        # Calculate gap between needles
        gap_samples = pos_second - (pos_first + first_needle.n_samples)

        # Build difficulty config
        full_difficulty_config = {
            **difficulty.to_dict(),
            "activity_a": activity_a,
            "activity_b": activity_b,
            "a_first": a_first,
            "first_activity": first_activity,
            "second_activity": second_activity,
            "pos_first": pos_first,
            "pos_second": pos_second,
            "gap_samples": gap_samples,
            "question_format": question_format,
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
            answer_type=final_answer_type,
            needles=needle_metadata,
            difficulty_config=full_difficulty_config,
            is_valid=True,
        )

    def _trim_needle_to_range(
        self,
        needle: NeedleSample,
        min_duration_ms: int,
        max_duration_ms: int,
        rng: np.random.Generator,
    ) -> NeedleSample:
        """
        Trim needle to a random duration within the specified range.

        Args:
            needle: Original needle sample
            min_duration_ms: Minimum duration in milliseconds
            max_duration_ms: Maximum duration in milliseconds
            rng: Random number generator

        Returns:
            Trimmed NeedleSample
        """
        # Cap max duration by actual needle duration
        actual_max_ms = min(max_duration_ms, needle.duration_ms)

        # Sample target duration
        target_duration_ms = int(rng.integers(min_duration_ms, actual_max_ms + 1))

        # Convert to samples
        target_samples = int(target_duration_ms * self.source_hz / 1000)
        target_samples = min(target_samples, needle.n_samples)

        return self._trim_needle(needle, target_samples)


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate ordering task dataset for TS-Haystack"
    )
    parser.add_argument(
        "--context-lengths",
        type=int,
        nargs="+",
        default=[10000, 20000],
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
        "--question-format",
        choices=["boolean", "category"],
        default="boolean",
        help="Question format: boolean or category",
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
        default=0.10,
        help="Maximum needle length as fraction of context (default: 0.10 = 10%%)",
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

    print("Creating OrderingTaskGenerator...")
    generator = OrderingTaskGenerator.create_with_artifacts(seed=args.seed)

    for context_length in args.context_lengths:
        print(f"\nGenerating for context_length={context_length}...")

        difficulty = DifficultyConfig(
            context_length_samples=context_length,
            needle_position=args.needle_position,
            needle_length_ratio_range=(args.needle_ratio_min, args.needle_ratio_max),
            background_purity=args.background_purity,
            task_specific={
                "min_gap_ratio": args.min_gap_ratio,
                "min_gap_max_samples": args.min_gap_max_samples,
                "margin_ratio": args.margin_ratio,
                "margin_max_samples": args.margin_max_samples,
                "question_format": args.question_format,
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
