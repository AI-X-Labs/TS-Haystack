# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors
# SPDX-License-Identifier: MIT

"""
Localization Task Generator for TS-Haystack benchmark.

Task 2: "When did the {activity} bout occur?"

This task requires the model to identify the temporal location of a
specific activity within the time series window.

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


class LocalizationTaskGenerator(BaseTaskGenerator):
    """
    Task 2: Localization - "When did the {activity} bout occur?"

    Updated Algorithm (with distractor insertion):
    1. Sample background window
    2. Randomly select a regime (sedentary or active)
    3. Compute insertable_activities = regime_activities - background_activities
    4. Sample N needles from insertable_activities
    5. Insert all needles at non-overlapping positions
    6. Randomly select ONE inserted needle as the target
    7. Ask: "When did the {target_activity} bout occur?"
    8. Answer: timestamp range of the target needle

    This prevents variance-based detection shortcuts by inserting multiple
    similar activities. The model must identify the specific activity pattern
    among distractors with similar statistical properties.

    Difficulty Knobs:
    - context_length_samples: Longer windows are harder to scan
    - needle_position: "beginning", "middle", "end", "random"
    - needle_length_ratio_range: Shorter needles (smaller ratio) are harder to localize
    - background_purity: "mixed" backgrounds add confusion
    - min_distractors / max_distractors: More distractors increases difficulty
    - min_gap_samples: Gap between inserted needles

    Answer Type: timestamp (time range)
    """

    @property
    def task_name(self) -> str:
        return "localization"

    @property
    def answer_type(self) -> str:
        return "timestamp"

    def generate_sample(
        self,
        difficulty: DifficultyConfig,
        rng: np.random.Generator,
    ) -> GeneratedSample:
        """
        Generate a single localization task sample with distractor insertion.

        Args:
            difficulty: Difficulty configuration
            rng: Random number generator (from SeedManager)

        Returns:
            GeneratedSample with localization question and timestamp answer
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

        # Step 2: Randomly select a regime
        regimes = list(WILLETTS_ACTIVITY_REGIMES.keys())
        selected_regime = regimes[rng.integers(0, len(regimes))]
        regime_activities = get_regime_activities(selected_regime)

        # Step 3: Compute insertable activities (regime - background)
        insertable_activities = regime_activities - background.activities_present

        if not insertable_activities:
            return self._create_invalid_sample(
                f"No insertable activities for regime '{selected_regime}' "
                f"(background has: {background.activities_present})",
                difficulty,
            )

        # Step 4: Determine how many needles to insert
        # For localization, we want at least 2 needles to have distractors
        min_distractors = difficulty.task_specific.get("min_distractors", 2)
        max_distractors = difficulty.task_specific.get("max_distractors", 4)

        # Cap by available activities
        max_insertable = len(insertable_activities)
        n_needles = int(rng.integers(
            min(min_distractors, max_insertable),
            min(max_distractors, max_insertable) + 1
        ))

        # Step 5: Sample needles from regime
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

        # Step 6: Insert all needles at non-overlapping positions
        min_gap = difficulty.get_effective_min_gap_samples()
        margin = difficulty.get_effective_margin_samples()

        final_x = background.x.copy()
        final_y = background.y.copy()
        final_z = background.z.copy()
        occupied_ranges: List[Tuple[int, int]] = []
        inserted_needle_metadata: List[InsertedNeedle] = []
        needle_to_metadata_map: List[Tuple[NeedleSample, InsertedNeedle]] = []

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
            metadata = self._create_inserted_needle(
                needle=trimmed_needle,
                position=position,
                context_length=context_length,
                background=background,
            )
            inserted_needle_metadata.append(metadata)
            needle_to_metadata_map.append((trimmed_needle, metadata))

        if not inserted_needle_metadata:
            return self._create_invalid_sample(
                "Failed to insert any needles",
                difficulty,
            )

        # Step 7: Randomly select ONE needle as the target
        target_idx = int(rng.integers(0, len(inserted_needle_metadata)))
        target_metadata = inserted_needle_metadata[target_idx]
        target_activity = target_metadata.activity

        # Step 8: Generate Q/A using template bank
        question, answer = self.template_bank.sample(
            task="localization",
            rng=rng,
            activity=target_activity,
            start=target_metadata.timestamp_start,
            end=target_metadata.timestamp_end,
        )

        # Get list of actually inserted activities
        actually_inserted = [m.activity for m in inserted_needle_metadata]

        # Build difficulty config with task-specific info
        full_difficulty_config = {
            **difficulty.to_dict(),
            "target_activity": target_activity,
            "target_needle_index": target_idx,
            "target_position_samples": target_metadata.insert_position_samples,
            "target_position_frac": target_metadata.insert_position_frac,
            "target_duration_samples": target_metadata.duration_samples,
            "target_duration_ms": target_metadata.duration_ms,
            "selected_regime": selected_regime,
            "inserted_activities": actually_inserted,
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

    parser = argparse.ArgumentParser(
        description="Generate localization task dataset for TS-Haystack"
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
        default=2,
        help="Minimum number of distractor needles to insert (including target)",
    )
    parser.add_argument(
        "--max-distractors",
        type=int,
        default=4,
        help="Maximum number of distractor needles to insert (including target)",
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

    print("Creating LocalizationTaskGenerator...")
    generator = LocalizationTaskGenerator.create_with_artifacts(seed=args.seed)

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
