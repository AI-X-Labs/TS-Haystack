# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors
# SPDX-License-Identifier: MIT

"""
State Query Task Generator for TS-Haystack benchmark.

Task 5: "What was the overall activity level when the {needle_activity} occurred?"

This task tests cross-scale integration: the model must simultaneously attend to
a local event (needle) and identify the global activity regime at that moment.
"""

from typing import List, Optional, Tuple

import numpy as np

from ts_haystack.core import (
    DifficultyConfig,
    GeneratedSample,
)
from ts_haystack.tasks.base_task import BaseTaskGenerator
from ts_haystack.utils import get_activity_region_at_position


class StateQueryTaskGenerator(BaseTaskGenerator):
    """
    Task 5: State Query - "What was the activity level when {needle_activity} occurred?"

    This task tests cross-scale integration:
    - Local scale: Detect the inserted needle event (seconds)
    - Global scale: Identify the surrounding activity regime (minutes)

    Algorithm:
    1. Sample a MIXED background (must have multiple activity states)
    2. Extract activity regions from the background timeline
    3. Select a target global state (region) large enough for needle insertion
    4. Sample needle activity NOT in background
    5. Insert needle WITHIN the selected global state
    6. Model must report the global state, not the needle activity

    Example:
        Background: [...sleep...][...sedentary...][...sleep...]
        Needle: [walking] inserted within sedentary region
        Q: "What was the overall activity level when the walking spike occurred?"
        A: "sedentary"

    Difficulty Knobs:
    - min_global_states / max_global_states: More states = harder
    - position_mode: "center" (easy) vs "near_boundary" (hard)
    - min_state_duration_samples: Shorter states = harder
    - needle_length_ratio_range: Shorter needles (smaller ratio) = harder to detect

    Answer Type: category
    """

    @property
    def task_name(self) -> str:
        return "state_query"

    @property
    def answer_type(self) -> str:
        return "category"

    def generate_sample(
        self,
        difficulty: DifficultyConfig,
        rng: np.random.Generator,
    ) -> GeneratedSample:
        """
        Generate a single state query task sample.

        Args:
            difficulty: Difficulty configuration
            rng: Random number generator (from SeedManager)

        Returns:
            GeneratedSample with state query question and category answer
        """
        context_length = difficulty.context_length_samples
        min_global_states = difficulty.task_specific.get("min_global_states", 2)
        max_global_states = difficulty.task_specific.get("max_global_states", 5)
        min_state_duration_samples = difficulty.get_effective_min_state_duration_samples()
        position_mode = difficulty.task_specific.get("position_mode", "random")
        boundary_margin_frac = difficulty.task_specific.get("boundary_margin_frac", 0.1)

        # =====================================================================
        # Step 1: Sample a MIXED background (must have multiple activities)
        # For short contexts where mixed is impossible, fallback to "any"
        # =====================================================================
        used_fallback = False
        try:
            background = self.background_sampler.sample_background(
                context_length_samples=context_length,
                purity="mixed",  # Must have multiple activities
                min_activity_count=min_global_states,
                max_activity_count=max_global_states,
                rng=rng,
            )
        except ValueError:
            # Fallback to "any" for short contexts where mixed is impossible
            # This allows pure backgrounds for short contexts
            used_fallback = True
            background = self.background_sampler.sample_background(
                context_length_samples=context_length,
                purity="any",  # Accept any window (including pure)
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
        # Step 2: Extract and validate activity regions
        # =====================================================================
        # activity_timeline is List[(start_frac, end_frac, activity)]
        activity_timeline = background.activity_timeline

        # For fallback mode (short contexts), accept any number of states >= 1
        # For normal mode, require min_global_states
        required_states = 1 if used_fallback else min_global_states
        if len(activity_timeline) < required_states:
            return self._create_invalid_sample(
                f"Background has {len(activity_timeline)} states, need >= {required_states}",
                difficulty,
            )

        # =====================================================================
        # Step 3: Find valid states (large enough to contain needle)
        # =====================================================================
        min_needle_samples, max_needle_samples = difficulty.get_needle_length_range_samples()
        # Need state large enough for needle + margins
        required_state_samples = min_needle_samples + 2 * int(
            min_needle_samples * boundary_margin_frac
        )

        valid_states: List[Tuple[int, int, str, int]] = []  # (start, end, activity, duration)
        for start_frac, end_frac, activity in activity_timeline:
            state_start = int(start_frac * context_length)
            state_end = int(end_frac * context_length)
            state_duration = state_end - state_start

            if state_duration >= max(min_state_duration_samples, required_state_samples):
                valid_states.append((state_start, state_end, activity, state_duration))

        if not valid_states:
            return self._create_invalid_sample(
                "No valid states large enough for needle insertion",
                difficulty,
            )

        # =====================================================================
        # Step 4: Select target global state
        # =====================================================================
        state_idx = int(rng.integers(0, len(valid_states)))
        state_start, state_end, target_global_activity, state_duration = valid_states[state_idx]

        # =====================================================================
        # Step 5: Sample needle activity (NOT in background)
        # =====================================================================
        all_activities = set(self.needle_sampler.get_available_activities())
        candidate_needles = all_activities - background.activities_present

        if not candidate_needles:
            return self._create_invalid_sample(
                "No candidate needle activities available",
                difficulty,
            )

        needle_activity = rng.choice(list(candidate_needles))

        # =====================================================================
        # Step 6: Sample needle from bout index
        # =====================================================================
        min_duration_ms, max_duration_ms = difficulty.get_needle_length_range_ms(
            self.source_hz
        )

        needle = self.needle_sampler.sample_needle(
            activity=needle_activity,
            min_duration_ms=min_duration_ms,
            rng=rng,
        )

        if needle is None:
            return self._create_invalid_sample(
                f"Could not sample needle for {needle_activity}",
                difficulty,
            )

        # Trim needle to target duration
        actual_max_ms = min(max_duration_ms, needle.duration_ms)
        target_duration_ms = int(rng.integers(min_duration_ms, actual_max_ms + 1))
        target_samples = int(target_duration_ms * self.source_hz / 1000)
        target_samples = min(target_samples, needle.n_samples)
        trimmed_needle = self._trim_needle(needle, target_samples)

        # =====================================================================
        # Step 7: Sample insertion position WITHIN the target global state
        # =====================================================================
        state_margin = int(state_duration * boundary_margin_frac)
        safe_start = state_start + state_margin
        safe_end = state_end - state_margin - trimmed_needle.n_samples

        if safe_end <= safe_start:
            return self._create_invalid_sample(
                "State too small for needle with margins",
                difficulty,
            )

        if position_mode == "center":
            # Insert at center of state
            position = (safe_start + safe_end) // 2
        elif position_mode == "near_boundary":
            # Insert near one of the state boundaries (harder)
            boundary_zone = state_margin // 2
            if rng.random() < 0.5:
                # Near start boundary
                position = safe_start + int(rng.integers(0, max(1, boundary_zone)))
            else:
                # Near end boundary
                position = safe_end - int(rng.integers(0, max(1, boundary_zone)))
        else:  # "random"
            position = int(rng.integers(safe_start, safe_end))

        # =====================================================================
        # Step 8: Apply style transfer and insert needle
        # =====================================================================
        final_x, final_y, final_z = self._insert_needle(
            background=background,
            needle=trimmed_needle,
            position=position,
        )

        # Create needle metadata
        inserted_needle = self._create_inserted_needle(
            needle=trimmed_needle,
            position=position,
            context_length=context_length,
            background=background,
        )

        # =====================================================================
        # Step 9: Generate Q/A using template bank
        # =====================================================================
        question, answer = self.template_bank.sample(
            task="state_query",
            rng=rng,
            needle_activity=needle_activity,
            global_state=target_global_activity,
        )

        # Compute distance to nearest state boundary (difficulty indicator)
        dist_to_start = position - state_start
        dist_to_end = state_end - (position + trimmed_needle.n_samples)
        dist_to_boundary = min(dist_to_start, dist_to_end)

        # Build difficulty config with task-specific info
        full_difficulty_config = {
            **difficulty.to_dict(),
            "needle_activity": needle_activity,
            "global_activity": target_global_activity,
            "global_timeline": list(activity_timeline),  # Keep as list of tuples
            "position_mode": position_mode,
            "dist_to_boundary_samples": dist_to_boundary,
            "state_start_samples": state_start,
            "state_end_samples": state_end,
            "state_duration_samples": state_duration,
            "n_global_states": len(activity_timeline),
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
            answer=target_global_activity,
            answer_type=self.answer_type,
            needles=[inserted_needle],
            difficulty_config=full_difficulty_config,
            is_valid=True,
        )


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate state query task dataset for TS-Haystack"
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
        "--min-global-states",
        type=int,
        default=2,
        help="Minimum number of global activity states in background",
    )
    parser.add_argument(
        "--max-global-states",
        type=int,
        default=5,
        help="Maximum number of global activity states in background",
    )
    parser.add_argument(
        "--position-mode",
        choices=["center", "near_boundary", "random"],
        default="random",
        help="Position mode for needle insertion within state",
    )
    parser.add_argument(
        "--needle-ratio-min",
        type=float,
        default=0.01,
        help="Minimum needle length as fraction of context (default: 0.01 = 1%%)",
    )
    parser.add_argument(
        "--needle-ratio-max",
        type=float,
        default=0.02,
        help="Maximum needle length as fraction of context (default: 0.02 = 2%%)",
    )

    args = parser.parse_args()

    print("Creating StateQueryTaskGenerator...")
    generator = StateQueryTaskGenerator.create_with_artifacts(seed=args.seed)

    for context_length in args.context_lengths:
        print(f"\nGenerating for context_length={context_length}...")

        difficulty = DifficultyConfig(
            context_length_samples=context_length,
            needle_position="random",
            needle_length_ratio_range=(args.needle_ratio_min, args.needle_ratio_max),
            background_purity="mixed",  # Required for this task
            task_specific={
                "min_global_states": args.min_global_states,
                "max_global_states": args.max_global_states,
                "position_mode": args.position_mode,
                "min_state_duration_ratio": 0.20,
                "min_state_duration_max_samples": 500,
                "boundary_margin_frac": 0.1,
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
