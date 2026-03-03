# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors
# SPDX-License-Identifier: MIT

"""
Anomaly Localization Task Generator for TS-Haystack benchmark.

Task 10: "Is there an anomaly in this recording, and if so, when did it occur?"

This task combines anomaly detection with temporal localization: the model must
not only detect cross-regime violations but also specify WHEN they occur.

Key difference from Anomaly Detection:
- Anomaly Detection: "Is there an anomaly?" -> Yes/No + what
- Anomaly Localization: "Is there an anomaly, and when?" -> Yes/No + what + when

Anomaly Definition:
- Cross-regime insertion: active activity in sedentary background (or vice versa)
- Same-regime activities are NOT anomalous

Sample Design (with mandatory distractors to prevent detection shortcuts):
- Positive: 1 cross-regime insertion (anomaly) + N same-regime distractors - must report time range
- Negative: N same-regime distractors only (no anomaly)

Both cases have needle insertions, so the model cannot use "insertion detected"
as a shortcut. It must identify the regime mismatch AND localize it.

Answer Format:
- Positive: "Yes, there is anomalous {activity} activity from {start} to {end}."
- Negative: "No, the recording shows consistent {regime} activity."
"""

from typing import List, Optional, Set, Tuple

import numpy as np

from ts_haystack.core import (
    DifficultyConfig,
    GeneratedSample,
    get_regime_activities,
)
from ts_haystack.tasks.task_anomaly_detection import (
    AnomalyDetectionTaskGenerator,
)


class AnomalyLocalizationTaskGenerator(AnomalyDetectionTaskGenerator):
    """
    Task 10: Anomaly Localization - "Is there an anomaly, and when did it occur?"

    Extends AnomalyDetectionTaskGenerator to include temporal localization.
    The answer for positive samples includes the time range of the anomaly.

    Algorithm (inherits from AnomalyDetectionTaskGenerator with mandatory distractors):
    1. Sample background from a single regime (pure)
    2. Determine background regime
    3. Decide positive (50%) or negative (50%)
    4. Positive: insert 1 OPPOSITE regime needle (anomaly) + n_distractors SAME regime needles
    5. Negative: insert n_distractors SAME regime needles only (no anomaly)
    6. For positive samples, answer includes temporal location of the anomaly

    Difficulty Knobs:
    - Same as AnomalyDetection (min_distractors, max_distractors are mandatory)
    - More distractors make localization harder (more candidates to rule out)

    Answer Type: time_range (timestamp for positive, boolean for negative)
    Evaluation:
    - Detection: Accuracy (exact match on Yes/No)
    - Localization: IoU (Intersection over Union) of predicted vs ground truth time range
    """

    @property
    def task_name(self) -> str:
        return "anomaly_localization"

    @property
    def answer_type(self) -> str:
        return "time_range"  # Includes temporal information (like Task 2: Localization)

    def _generate_positive_sample(
        self,
        background,
        background_regime: str,
        difficulty: DifficultyConfig,
        rng: np.random.Generator,
    ) -> GeneratedSample:
        """Generate a positive sample with cross-regime anomaly + same-regime distractors."""
        context_length = difficulty.context_length_samples

        # Get activities from the OPPOSITE regime (these are anomalous)
        opposite_regime = "active" if background_regime == "sedentary" else "sedentary"
        anomaly_candidates = get_regime_activities(opposite_regime)

        # Get insertable same-regime activities for distractors
        same_regime_activities = get_regime_activities(background_regime) - background.activities_present

        # Sample and insert anomaly (cross-regime)
        result = self._sample_and_insert_needle(
            background, anomaly_candidates, difficulty, rng
        )

        if result is None:
            return self._create_invalid_sample(
                f"Failed to sample anomaly needle from {opposite_regime} regime",
                difficulty,
            )

        final_signal, anomaly_needle = result
        final_x, final_y, final_z = final_signal

        # Add MANDATORY same-regime distractors (prevents detection shortcuts)
        min_distractors = difficulty.task_specific.get("min_distractors", 1)
        max_distractors = difficulty.task_specific.get("max_distractors", 3)
        n_distractors = int(rng.integers(min_distractors, max_distractors + 1))

        distractor_needles = []
        occupied_ranges = [(anomaly_needle.insert_position_samples,
                           anomaly_needle.insert_position_samples + anomaly_needle.duration_samples)]

        for _ in range(n_distractors):
            if not same_regime_activities:
                break
            result = self._sample_and_insert_needle(
                background, same_regime_activities, difficulty, rng,
                current_signal=(final_x, final_y, final_z),
                occupied_ranges=occupied_ranges,
            )
            if result:
                (final_x, final_y, final_z), distractor = result
                distractor_needles.append(distractor)
                occupied_ranges.append((distractor.insert_position_samples,
                                       distractor.insert_position_samples + distractor.duration_samples))

        # Generate Q/A with time range (use positive template - requires anomaly_activity, start, end)
        question, answer = self.template_bank.sample(
            task="anomaly_localization_positive",
            rng=rng,
            anomaly_activity=anomaly_needle.activity,
            start=anomaly_needle.timestamp_start,
            end=anomaly_needle.timestamp_end,
        )

        # Build difficulty config
        full_difficulty_config = {
            **difficulty.to_dict(),
            "is_positive": True,
            "background_regime": background_regime,
            "anomaly_regime": opposite_regime,
            "anomaly_activity": anomaly_needle.activity,
            "anomaly_start": anomaly_needle.timestamp_start,
            "anomaly_end": anomaly_needle.timestamp_end,
            "anomaly_position_samples": anomaly_needle.insert_position_samples,
            "anomaly_duration_samples": anomaly_needle.duration_samples,
            "background_activities": list(background.activities_present),
            "n_distractors": len(distractor_needles),
        }

        all_needles = [anomaly_needle] + distractor_needles

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
            needles=all_needles,
            difficulty_config=full_difficulty_config,
            is_valid=True,
        )

    def _generate_negative_sample(
        self,
        background,
        background_regime: str,
        difficulty: DifficultyConfig,
        rng: np.random.Generator,
    ) -> GeneratedSample:
        """Generate a negative sample with same-regime distractors only (no anomaly)."""
        context_length = difficulty.context_length_samples

        # Get activities from SAME regime (these are NOT anomalous)
        same_regime_activities = get_regime_activities(background_regime) - background.activities_present

        if not same_regime_activities:
            return self._create_invalid_sample(
                f"No insertable same-regime activities for {background_regime}",
                difficulty,
            )

        # Determine number of distractors (same as positive case for consistency)
        min_distractors = difficulty.task_specific.get("min_distractors", 1)
        max_distractors = difficulty.task_specific.get("max_distractors", 3)
        n_distractors = int(rng.integers(min_distractors, max_distractors + 1))

        # Insert multiple same-regime distractors (NO cross-regime = NO anomaly)
        final_x = background.x.copy()
        final_y = background.y.copy()
        final_z = background.z.copy()
        distractor_needles = []
        occupied_ranges = []

        for _ in range(n_distractors):
            if not same_regime_activities:
                break
            result = self._sample_and_insert_needle(
                background, same_regime_activities, difficulty, rng,
                current_signal=(final_x, final_y, final_z),
                occupied_ranges=occupied_ranges if occupied_ranges else None,
            )
            if result:
                (final_x, final_y, final_z), distractor = result
                distractor_needles.append(distractor)
                occupied_ranges.append((distractor.insert_position_samples,
                                       distractor.insert_position_samples + distractor.duration_samples))

        if not distractor_needles:
            return self._create_invalid_sample(
                f"Failed to insert any same-regime distractors for {background_regime}",
                difficulty,
            )

        # Generate Q/A (use negative template - only needs background_regime)
        question, answer = self.template_bank.sample(
            task="anomaly_localization_negative",
            rng=rng,
            background_regime=background_regime,
        )

        full_difficulty_config = {
            **difficulty.to_dict(),
            "is_positive": False,
            "background_regime": background_regime,
            "n_distractors": len(distractor_needles),
            "distractor_activities": [n.activity for n in distractor_needles],
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
            needles=distractor_needles,
            difficulty_config=full_difficulty_config,
            is_valid=True,
        )


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate anomaly localization task dataset for TS-Haystack"
    )
    parser.add_argument(
        "--context-lengths",
        type=int,
        nargs="+",
        default=[10000],
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
        default=0.03,
        help="Minimum needle length as fraction of context",
    )
    parser.add_argument(
        "--needle-ratio-max",
        type=float,
        default=0.15,
        help="Maximum needle length as fraction of context",
    )
    parser.add_argument(
        "--min-distractors",
        type=int,
        default=1,
        help="Minimum number of same-regime distractor needles",
    )
    parser.add_argument(
        "--max-distractors",
        type=int,
        default=3,
        help="Maximum number of same-regime distractor needles",
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

    print("Creating AnomalyLocalizationTaskGenerator...")
    generator = AnomalyLocalizationTaskGenerator.create_with_artifacts(seed=args.seed)

    for context_length in args.context_lengths:
        print(f"\nGenerating for context_length={context_length}...")

        difficulty = DifficultyConfig(
            context_length_samples=context_length,
            needle_position="random",
            needle_length_ratio_range=(args.needle_ratio_min, args.needle_ratio_max),
            background_purity="pure",
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
