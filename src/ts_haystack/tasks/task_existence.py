# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors
# SPDX-License-Identifier: MIT

"""
Existence Task Generator for TS-Haystack benchmark.

Task 1: "Is there {activity} in this recording?"

This is the simplest task - a binary classification asking whether
a specific activity is present in the time series window.

Updated with distractor insertion to prevent variance-based detection shortcuts.
When the background is homogeneous, inserting multiple needles from the same
activity regime forces the model to distinguish between similar activities
rather than just detecting variance changes.
"""

from typing import List, Optional, Set, Tuple

import numpy as np

from ts_haystack.core import (
    DifficultyConfig,
    GeneratedSample,
    InsertedNeedle,
    NeedleSample,
    WILLETTS_ACTIVITY_REGIMES,
    get_regime_activities,
)
from ts_haystack.tasks.base_task import BaseTaskGenerator


class ExistenceTaskGenerator(BaseTaskGenerator):
    """
    Task 1: Existence - "Is there {activity} in this recording?"

    Updated Algorithm (with distractor insertion):
    1. Sample background window
    2. Decide positive (50%) or negative (50%) formulation
    3. Randomly select a regime (sedentary or active)
    4. Compute insertable_activities = regime_activities - background_activities
    5. For negative case: ensure at least 1 activity reserved for target
    6. Sample N needles from insertable_activities
    7. Insert all needles at non-overlapping positions
    8. Select target:
       - Positive: random activity from inserted_activities
       - Negative: random activity from (insertable_activities - inserted_activities)
    9. Generate Q/A pair using template bank

    This prevents variance-based detection shortcuts by inserting multiple
    similar activities, forcing the model to learn activity-specific patterns.

    Difficulty Knobs:
    - context_length_samples: Longer windows are harder to scan
    - background_purity: "mixed" backgrounds have more activity variety
    - needle_length_ratio_range: Shorter needles (smaller ratio) are harder to detect
    - needle_position: Position affects difficulty (edges vs middle)
    - min_distractors / max_distractors: More distractors increases difficulty
    - min_gap_samples: Gap between inserted needles

    Answer Type: boolean (Yes/No)
    """

    @property
    def task_name(self) -> str:
        return "existence"

    @property
    def answer_type(self) -> str:
        return "boolean"

    def generate_sample(
        self,
        difficulty: DifficultyConfig,
        rng: np.random.Generator,
    ) -> GeneratedSample:
        """
        Generate a single existence task sample with distractor insertion.

        Args:
            difficulty: Difficulty configuration
            rng: Random number generator (from SeedManager)

        Returns:
            GeneratedSample with existence question and boolean answer
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
                "Failed to sample background", difficulty
            )

        # Validate annotation coverage
        is_valid, reason = self._validate_background_coverage(background, difficulty)
        if not is_valid:
            return self._create_invalid_sample(reason, difficulty)

        # Step 2: Decide positive or negative sample (50/50 balance)
        is_positive = rng.random() < 0.5

        # Step 3: Randomly select a regime
        regimes = list(WILLETTS_ACTIVITY_REGIMES.keys())
        selected_regime = regimes[rng.integers(0, len(regimes))]
        regime_activities = get_regime_activities(selected_regime)

        # Step 4: Compute insertable activities (regime - background)
        insertable_activities = regime_activities - background.activities_present

        if not insertable_activities:
            return self._create_invalid_sample(
                f"No insertable activities for regime '{selected_regime}' "
                f"(background has: {background.activities_present})",
                difficulty,
            )

        # Step 5: For negative case, we need at least 2 insertable activities:
        # - At least 1 to insert as distractor
        # - At least 1 to ask about (that's NOT inserted)
        # For positive case, we need at least 1 insertable activity
        if not is_positive and len(insertable_activities) < 2:
            # Fall back to positive case
            is_positive = True

        # Step 6: Determine how many needles to insert
        min_distractors = difficulty.task_specific.get("min_distractors", 1)
        max_distractors = difficulty.task_specific.get("max_distractors", 3)

        # Cap by available activities (reserve 1 for negative target if needed)
        max_insertable = len(insertable_activities) if is_positive else len(insertable_activities) - 1
        max_insertable = max(1, max_insertable)  # At least 1

        n_needles = int(rng.integers(
            min(min_distractors, max_insertable),
            min(max_distractors, max_insertable) + 1
        ))

        # Step 7: Sample needles from regime
        min_duration_ms, max_duration_ms = difficulty.get_needle_length_range_ms(
            self.source_hz
        )
        needles = self.needle_sampler.sample_needles_for_regime(
            regime_activities=insertable_activities,
            n_needles=n_needles,
            min_duration_ms=min_duration_ms,
            rng=rng,
        )

        if not needles:
            return self._create_invalid_sample(
                f"Failed to sample needles for regime '{selected_regime}'",
                difficulty,
            )

        # Get the set of activities we successfully sampled
        inserted_activities = {n.activity for n in needles}

        # Step 8: Insert all needles at non-overlapping positions
        min_gap = difficulty.get_effective_min_gap_samples()
        margin = difficulty.get_effective_margin_samples()

        final_x = background.x.copy()
        final_y = background.y.copy()
        final_z = background.z.copy()
        occupied_ranges: List[Tuple[int, int]] = []
        inserted_needle_metadata: List[InsertedNeedle] = []

        for needle in needles:
            # Determine target length for this needle
            capped_max_ms = min(max_duration_ms, needle.duration_ms)
            if capped_max_ms < min_duration_ms:
                continue  # Skip if needle is too short

            target_duration_ms = int(rng.integers(min_duration_ms, capped_max_ms + 1))
            target_samples = int(target_duration_ms * self.source_hz / 1000)
            target_samples = min(target_samples, needle.n_samples)

            # Trim needle
            trimmed_needle = self._trim_needle(needle, target_samples)

            # Find non-overlapping position
            position = self._find_valid_position(
                context_length=context_length,
                needle_length=trimmed_needle.n_samples,
                occupied_ranges=occupied_ranges,
                min_gap=min_gap,
                rng=rng,
                margin=margin,
            )

            if position is None:
                continue  # Skip if no valid position found

            # Insert needle
            final_x, final_y, final_z = self._insert_needle(
                background=background,
                needle=trimmed_needle,
                position=position,
                current_signal=(final_x, final_y, final_z),
            )

            # Record occupied range
            occupied_ranges.append((position, position + trimmed_needle.n_samples))

            # Create metadata
            inserted_needle_metadata.append(
                self._create_inserted_needle(
                    needle=trimmed_needle,
                    position=position,
                    context_length=context_length,
                    background=background,
                )
            )

        if not inserted_needle_metadata:
            return self._create_invalid_sample(
                "Failed to insert any needles",
                difficulty,
            )

        # Update inserted_activities based on what we actually inserted
        actually_inserted = {m.activity for m in inserted_needle_metadata}

        # Step 9: Select target activity
        if is_positive:
            # Ask about an activity that WAS inserted
            target_activity = list(actually_inserted)[
                rng.integers(0, len(actually_inserted))
            ]
        else:
            # Ask about an activity in the same regime that was NOT inserted
            available_negatives = insertable_activities - actually_inserted
            if not available_negatives:
                # Fall back to positive if no negatives available
                is_positive = True
                target_activity = list(actually_inserted)[
                    rng.integers(0, len(actually_inserted))
                ]
            else:
                target_activity = list(available_negatives)[
                    rng.integers(0, len(available_negatives))
                ]

        # Step 10: Generate Q/A using template bank
        question, answer = self.template_bank.sample(
            task="existence",
            rng=rng,
            activity=target_activity,
            exists=is_positive,
        )

        # Build difficulty config with task-specific info
        full_difficulty_config = {
            **difficulty.to_dict(),
            "target_activity": target_activity,
            "is_positive": is_positive,
            "selected_regime": selected_regime,
            "inserted_activities": list(actually_inserted),
            "n_needles_inserted": len(inserted_needle_metadata),
            "background_activities": list(background.activities_present),
        }

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
            needles=inserted_needle_metadata,
            difficulty_config=full_difficulty_config,
            is_valid=True,
        )


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="Generate existence task dataset for TS-Haystack"
    )
    parser.add_argument(
        "--context-lengths",
        type=int,
        nargs="+",
        default=[1000, 10000],
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
        "--min-distractors",
        type=int,
        default=1,
        help="Minimum number of distractor needles to insert",
    )
    parser.add_argument(
        "--max-distractors",
        type=int,
        default=3,
        help="Maximum number of distractor needles to insert",
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

    print("Creating ExistenceTaskGenerator...")
    generator = ExistenceTaskGenerator.create_with_artifacts(seed=args.seed)

    for context_length in args.context_lengths:
        print(f"\nGenerating for context_length={context_length}...")

        difficulty = DifficultyConfig(
            context_length_samples=context_length,
            needle_position=args.needle_position,
            needle_length_ratio_range=(args.needle_ratio_min, args.needle_ratio_max),
            background_purity=args.background_purity,
            task_specific={
                "min_distractors": args.min_distractors,
                "max_distractors": args.max_distractors,
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
