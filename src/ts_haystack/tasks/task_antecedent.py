# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors
# SPDX-License-Identifier: MIT

"""
Antecedent Task Generator for TS-Haystack benchmark.

Task 6: "What activity occurred immediately before {target_activity}?"

This task tests temporal relationship understanding: the model must identify
what activity came immediately before a specified target activity.
"""

import numpy as np

from ts_haystack.core import (
    DifficultyConfig,
    GeneratedSample,
    InsertedNeedle,
)
from ts_haystack.tasks.base_task import BaseTaskGenerator
from ts_haystack.utils import find_sequential_positions


class AntecedentTaskGenerator(BaseTaskGenerator):
    """
    Task 6: Antecedent - "What activity occurred immediately before {target_activity}?"

    This task tests the model's ability to:
    1. Locate the target activity bout
    2. Identify what came immediately before it
    3. Report the antecedent activity (not the target)

    Algorithm:
    1. Sample background (prefer low-activity for clear signal)
    2. Sample antecedent activity A (NOT in background)
    3. Sample target activity T (NOT in background, different from A)
    4. Sample needles for both A and T
    5. Insert A first, then T immediately adjacent with small gap
    6. Q: "What activity occurred immediately before {target_activity}?"
    7. A: The antecedent activity

    Design Choice - Two-Needle Insertion:
    - Control over antecedent-target pairing
    - Prevents learning from natural transition statistics
    - Clear adjacency relationship

    Example:
        Background:  [...sleep............................]
        + Antecedent: [...sleep...][SEDENTARY][...sleep...]
        + Target:     [...sleep...][SEDENTARY][WALKING][...sleep...]
        Q: "What occurred immediately before walking?"
        A: "Sedentary"

    Difficulty Knobs:
    - adjacency_gap_samples: Smaller gap = clearer adjacency (easier)
    - background_mode: "low_activity" vs "mixed"
    - use_transition_probs: Use transition matrix for realistic pairs

    Answer Type: category
    """

    @property
    def task_name(self) -> str:
        return "antecedent"

    @property
    def answer_type(self) -> str:
        return "category"

    def generate_sample(
        self,
        difficulty: DifficultyConfig,
        rng: np.random.Generator,
    ) -> GeneratedSample:
        """
        Generate a single antecedent task sample.

        Args:
            difficulty: Difficulty configuration
            rng: Random number generator (from SeedManager)

        Returns:
            GeneratedSample with antecedent question and category answer
        """
        context_length = difficulty.context_length_samples
        adjacency_gap_samples = difficulty.task_specific.get("adjacency_gap_samples", 10)
        margin_samples = difficulty.get_effective_margin_samples()
        background_mode = difficulty.task_specific.get("background_mode", "low_activity")
        use_transition_probs = difficulty.task_specific.get("use_transition_probs", False)

        # =====================================================================
        # Step 1: Sample background (prefer low-activity for clean signal)
        # =====================================================================
        if background_mode == "low_activity":
            # Select from sleep or standing/sitting periods for cleaner signal
            background = self.background_sampler.sample_background(
                context_length_samples=context_length,
                allowed_activities={"sleep", "standing", "sitting"},
                purity="pure",
                rng=rng,
            )
            # Fallback to any pure background if low-activity not available
            if background is None:
                print(f"Low activity background not available, fallback to any pure activity background type")
                background = self.background_sampler.sample_background(
                    context_length_samples=context_length,
                    purity="pure",
                    rng=rng,
                )
        else:
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
        # Step 2: Sample antecedent activity A (NOT in background)
        # =====================================================================
        all_activities = set(self.needle_sampler.get_available_activities())
        candidate_activities = all_activities - background.activities_present

        if len(candidate_activities) < 2:
            return self._create_invalid_sample(
                "Need at least 2 candidate activities (one for antecedent, one for target)",
                difficulty,
            )

        antecedent_activity = rng.choice(list(candidate_activities))

        # =====================================================================
        # Step 3: Sample target activity T (different from A, NOT in background)
        # =====================================================================
        candidate_target = candidate_activities - {antecedent_activity}

        if not candidate_target:
            return self._create_invalid_sample(
                "No candidate target activities after selecting antecedent",
                difficulty,
            )

        if use_transition_probs:
            # Sample target based on transition probability P(T | A)
            target_activity = self.needle_sampler.transition_matrix.sample_successor(
                antecedent_activity,
                exclude={antecedent_activity},
                rng=rng,
            )
            # Verify target is in candidates, otherwise fall back to random
            if target_activity not in candidate_target:
                target_activity = rng.choice(list(candidate_target))
        else:
            target_activity = rng.choice(list(candidate_target))

        # =====================================================================
        # Step 4: Sample needles for both activities
        # =====================================================================
        min_duration_ms, max_duration_ms = difficulty.get_needle_length_range_ms(
            self.source_hz
        )

        antecedent_needle = self.needle_sampler.sample_needle(
            activity=antecedent_activity,
            min_duration_ms=min_duration_ms,
            rng=rng,
        )

        target_needle = self.needle_sampler.sample_needle(
            activity=target_activity,
            min_duration_ms=min_duration_ms,
            rng=rng,
        )

        if antecedent_needle is None:
            return self._create_invalid_sample(
                f"Could not sample antecedent needle for {antecedent_activity}",
                difficulty,
            )

        if target_needle is None:
            return self._create_invalid_sample(
                f"Could not sample target needle for {target_activity}",
                difficulty,
            )

        # Trim needles to target durations
        antecedent_trimmed = self._trim_needle_to_range(
            antecedent_needle, min_duration_ms, max_duration_ms, rng
        )
        target_trimmed = self._trim_needle_to_range(
            target_needle, min_duration_ms, max_duration_ms, rng
        )

        # =====================================================================
        # Step 5: Compute sequential positions (antecedent then target)
        # =====================================================================
        needle_lengths = [antecedent_trimmed.n_samples, target_trimmed.n_samples]

        # For antecedent task, we need TIGHT adjacency between needles.
        # Use max_gap to constrain the spacing to be close to adjacency_gap_samples.
        # Allow some flexibility (up to 2x the configured gap) but keep it tight.
        max_adjacency_gap = difficulty.task_specific.get(
            "max_adjacency_gap_samples", adjacency_gap_samples * 2
        )

        positions = find_sequential_positions(
            context_length=context_length,
            needle_lengths=needle_lengths,
            min_gap=adjacency_gap_samples,
            margin=margin_samples,
            rng=rng,
            max_gap=max_adjacency_gap,
        )

        if positions is None:
            return self._create_invalid_sample(
                "Needles too long for context window with adjacency constraint",
                difficulty,
            )

        antecedent_pos, target_pos = positions

        # =====================================================================
        # Step 6: Insert both needles in order (antecedent first, then target)
        # =====================================================================
        # Start with background signal
        current_signal = (
            background.x.copy(),
            background.y.copy(),
            background.z.copy(),
        )

        # Insert antecedent needle first
        local_stats_antecedent = self.style_transfer.compute_local_statistics(
            background=current_signal,
            position=antecedent_pos,
        )
        transferred_antecedent = self.style_transfer.transfer(
            antecedent_trimmed, local_stats_antecedent
        )
        current_signal = self.style_transfer.insert_with_blending(
            background=current_signal,
            needle=(transferred_antecedent.x, transferred_antecedent.y, transferred_antecedent.z),
            position=antecedent_pos,
        )

        # Insert target needle second (immediately after antecedent)
        local_stats_target = self.style_transfer.compute_local_statistics(
            background=current_signal,
            position=target_pos,
        )
        transferred_target = self.style_transfer.transfer(
            target_trimmed, local_stats_target
        )
        current_signal = self.style_transfer.insert_with_blending(
            background=current_signal,
            needle=(transferred_target.x, transferred_target.y, transferred_target.z),
            position=target_pos,
        )

        # =====================================================================
        # Step 7: Create needle metadata
        # =====================================================================
        needle_metadata = [
            InsertedNeedle(
                activity=antecedent_activity,
                source_pid=antecedent_trimmed.source_pid,
                source_start_ms=antecedent_trimmed.start_ms,
                source_end_ms=antecedent_trimmed.end_ms,
                insert_position_samples=antecedent_pos,
                insert_position_frac=antecedent_pos / context_length,
                duration_samples=antecedent_trimmed.n_samples,
                duration_ms=antecedent_trimmed.duration_ms,
                timestamp_start=self._samples_to_timestamp(antecedent_pos, background),
                timestamp_end=self._samples_to_timestamp(
                    antecedent_pos + antecedent_trimmed.n_samples, background
                ),
            ),
            InsertedNeedle(
                activity=target_activity,
                source_pid=target_trimmed.source_pid,
                source_start_ms=target_trimmed.start_ms,
                source_end_ms=target_trimmed.end_ms,
                insert_position_samples=target_pos,
                insert_position_frac=target_pos / context_length,
                duration_samples=target_trimmed.n_samples,
                duration_ms=target_trimmed.duration_ms,
                timestamp_start=self._samples_to_timestamp(target_pos, background),
                timestamp_end=self._samples_to_timestamp(
                    target_pos + target_trimmed.n_samples, background
                ),
            ),
        ]

        # =====================================================================
        # Step 8: Generate Q/A using template bank
        # =====================================================================
        question, answer = self.template_bank.sample(
            task="antecedent",
            rng=rng,
            target_activity=target_activity,
            antecedent_activity=antecedent_activity,
        )

        # Compute the actual gap between needles
        actual_gap = target_pos - (antecedent_pos + antecedent_trimmed.n_samples)

        # Build difficulty config with task-specific info
        full_difficulty_config = {
            **difficulty.to_dict(),
            "antecedent_activity": antecedent_activity,
            "target_activity": target_activity,
            "adjacency_gap_samples": actual_gap,
            "antecedent_pos": antecedent_pos,
            "target_pos": target_pos,
            "antecedent_duration_samples": antecedent_trimmed.n_samples,
            "target_duration_samples": target_trimmed.n_samples,
            "background_mode": background_mode,
            "use_transition_probs": use_transition_probs,
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
            answer=antecedent_activity,
            answer_type=self.answer_type,
            needles=needle_metadata,
            difficulty_config=full_difficulty_config,
            is_valid=True,
        )

    def _trim_needle_to_range(
        self,
        needle,
        min_duration_ms: int,
        max_duration_ms: int,
        rng: np.random.Generator,
    ):
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
        description="Generate antecedent task dataset for TS-Haystack"
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
        "--adjacency-gap",
        type=int,
        default=10,
        help="Gap in samples between antecedent and target needles",
    )
    parser.add_argument(
        "--background-mode",
        choices=["low_activity", "mixed"],
        default="low_activity",
        help="Background sampling mode",
    )
    parser.add_argument(
        "--use-transition-probs",
        action="store_true",
        help="Use transition matrix for activity pairing",
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

    print("Creating AntecedentTaskGenerator...")
    generator = AntecedentTaskGenerator.create_with_artifacts(seed=args.seed)

    for context_length in args.context_lengths:
        print(f"\nGenerating for context_length={context_length}...")

        difficulty = DifficultyConfig(
            context_length_samples=context_length,
            needle_position=args.needle_position,
            needle_length_ratio_range=(args.needle_ratio_min, args.needle_ratio_max),
            background_purity=args.background_purity,
            task_specific={
                "adjacency_gap_samples": args.adjacency_gap,
                "margin_ratio": args.margin_ratio,
                "margin_max_samples": args.margin_max_samples,
                "background_mode": args.background_mode,
                "use_transition_probs": args.use_transition_probs,
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
