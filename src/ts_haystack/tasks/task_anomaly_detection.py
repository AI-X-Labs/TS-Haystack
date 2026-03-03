# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors
# SPDX-License-Identifier: MIT

"""
Anomaly Detection Task Generator for TS-Haystack benchmark.

Task 9: "Is there an anomaly in this recording?"

This task tests contextual reasoning: detecting when an activity is anomalous
relative to the background regime, without being told what to look for.

Key difference from Existence:
- Existence: "Is there walking?" (target given)
- Anomaly Detection: "Is there an anomaly?" (must identify the anomaly type)

Anomaly Definition:
- Cross-regime insertion: active activity in sedentary background (or vice versa)
- Same-regime activities are NOT anomalous

Sample Design (with mandatory distractors to prevent detection shortcuts):
- Positive: 1 cross-regime insertion (anomaly) + N same-regime distractors
- Negative: N same-regime distractors only (no anomaly)

Both cases have needle insertions, so the model cannot use "insertion detected"
as a shortcut. It must identify the regime mismatch (cross-regime = anomaly).

Answer Format:
- Positive: "Yes, there is anomalous {activity} activity in the {regime} background."
- Negative: "No, the recording shows consistent {regime} activity."
"""

from typing import List, Optional, Set, Tuple

import numpy as np

from ts_haystack.core import (
    DifficultyConfig,
    GeneratedSample,
    InsertedNeedle,
    NeedleSample,
    WILLETTS_ACTIVITY_REGIMES,
    ACTIVITY_TO_REGIME,
    get_regime,
    get_regime_activities,
)
from ts_haystack.tasks.base_task import BaseTaskGenerator


class AnomalyDetectionTaskGenerator(BaseTaskGenerator):
    """
    Task 9: Anomaly Detection - "Is there an anomaly in this recording?"

    Algorithm (with mandatory distractors):
    1. Sample background from a single regime (pure)
    2. Determine background regime
    3. Compute insertable same-regime activities
    4. Decide positive (50%) or negative (50%)
    5. Determine n_distractors from config (min_distractors to max_distractors)
    6. Positive: insert 1 OPPOSITE regime needle (anomaly) + n_distractors SAME regime needles
    7. Negative: insert n_distractors SAME regime needles only (no anomaly)
    8. Generate Q/A with explanation

    Difficulty Knobs:
    - needle_length_ratio_range: Smaller anomalies are harder to detect
    - min_distractors / max_distractors: Number of same-regime distractors (mandatory)

    Answer Type: boolean (Yes/No)
    Evaluation: Accuracy (exact match on Yes/No + activity identification for positive)
    """

    @property
    def task_name(self) -> str:
        return "anomaly_detection"

    @property
    def answer_type(self) -> str:
        return "boolean"  # Yes/No

    def generate_sample(
        self,
        difficulty: DifficultyConfig,
        rng: np.random.Generator,
    ) -> GeneratedSample:
        """Generate a single anomaly detection sample."""
        context_length = difficulty.context_length_samples

        # Step 1: Sample PURE background (single regime)
        background = self.background_sampler.sample_background(
            context_length_samples=context_length,
            purity="pure",  # Force pure for clear regime identification
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

        # Step 2: Determine background regime
        background_regime = self._get_dominant_regime(background.activities_present)
        if background_regime is None:
            return self._create_invalid_sample(
                f"Could not determine regime for activities: {background.activities_present}",
                difficulty,
            )

        # Step 3: Decide positive or negative (50/50)
        is_positive = rng.random() < 0.5

        # Step 4/5: Generate sample based on positive/negative
        if is_positive:
            return self._generate_positive_sample(
                background, background_regime, difficulty, rng
            )
        else:
            return self._generate_negative_sample(
                background, background_regime, difficulty, rng
            )

    def _get_dominant_regime(self, activities: Set[str]) -> Optional[str]:
        """Determine the dominant regime from a set of activities."""
        if not activities:
            return None

        regimes = {}
        for activity in activities:
            regime = ACTIVITY_TO_REGIME.get(activity)
            if regime:
                regimes[regime] = regimes.get(regime, 0) + 1

        if not regimes:
            return None

        return max(regimes, key=regimes.get)

    def _sample_and_insert_needle(
        self,
        background,
        target_activities: Set[str],
        difficulty: DifficultyConfig,
        rng: np.random.Generator,
        current_signal: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
        occupied_ranges: Optional[List[Tuple[int, int]]] = None,
    ) -> Optional[Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], InsertedNeedle]]:
        """Helper to sample and insert a needle from target activities."""
        context_length = difficulty.context_length_samples
        min_duration_ms, max_duration_ms = difficulty.get_needle_length_range_ms(
            self.source_hz
        )

        # Sample a needle from the target activities
        needle = self.needle_sampler.sample_needle_for_context(
            context_activities=set(),  # Don't exclude any activities
            min_duration_ms=min_duration_ms,
            use_transition_probs=False,  # Use uniform sampling from target set
            rng=rng,
        )

        # If that didn't work, try sampling directly from one of the target activities
        if needle is None or needle.activity not in target_activities:
            # Sample from a random activity in target_activities
            target_list = list(target_activities)
            if not target_list:
                return None
            chosen_activity = target_list[rng.integers(0, len(target_list))]
            needle = self.needle_sampler.sample_needle(
                activity=chosen_activity,
                min_duration_ms=min_duration_ms,
                rng=rng,
            )

        if needle is None:
            return None

        # Verify the needle activity is in our target set
        if needle.activity not in target_activities:
            return None

        # Trim needle to target length
        capped_max_ms = min(max_duration_ms, needle.duration_ms)
        if capped_max_ms < min_duration_ms:
            return None

        target_duration_ms = int(rng.integers(min_duration_ms, capped_max_ms + 1))
        target_samples = int(target_duration_ms * self.source_hz / 1000)
        target_samples = min(target_samples, needle.n_samples)
        trimmed_needle = self._trim_needle(needle, target_samples)

        # Find insertion position
        margin = difficulty.get_effective_margin_samples()
        min_gap = difficulty.get_effective_min_gap_samples()

        if occupied_ranges:
            position = self._find_valid_position(
                context_length=context_length,
                needle_length=trimmed_needle.n_samples,
                occupied_ranges=occupied_ranges,
                min_gap=min_gap,
                rng=rng,
                margin=margin,
            )
        else:
            position = self._sample_position(
                context_length=context_length,
                needle_length=trimmed_needle.n_samples,
                position_mode=difficulty.needle_position,
                rng=rng,
                margin_samples=margin,
            )

        if position is None:
            return None

        # Insert needle
        if current_signal is None:
            current_signal = (background.x.copy(), background.y.copy(), background.z.copy())

        final_signal = self._insert_needle(
            background=background,
            needle=trimmed_needle,
            position=position,
            current_signal=current_signal,
        )

        # Create needle metadata
        inserted_needle = self._create_inserted_needle(
            needle=trimmed_needle,
            position=position,
            context_length=context_length,
            background=background,
        )

        return final_signal, inserted_needle

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

        # Generate Q/A (use positive template - requires anomaly_activity)
        question, answer = self.template_bank.sample(
            task="anomaly_detection_positive",
            rng=rng,
            anomaly_activity=anomaly_needle.activity,
            background_regime=background_regime,
        )

        # Build difficulty config
        full_difficulty_config = {
            **difficulty.to_dict(),
            "is_positive": True,
            "background_regime": background_regime,
            "anomaly_regime": opposite_regime,
            "anomaly_activity": anomaly_needle.activity,
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
            task="anomaly_detection_negative",
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
        description="Generate anomaly detection task dataset for TS-Haystack"
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

    print("Creating AnomalyDetectionTaskGenerator...")
    generator = AnomalyDetectionTaskGenerator.create_with_artifacts(seed=args.seed)

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
