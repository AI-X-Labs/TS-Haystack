# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors
# SPDX-License-Identifier: MIT

"""
Multi-Hop Localization Task Generator for TS-Haystack benchmark.

Task 8: "When did the {ordinal} {target_activity} bout occur {direction} the {anchor_activity}?"

This task tests multi-step reasoning: the model must first locate an anchor activity,
then find and count target activities relative to that anchor, and finally report
the K-th occurrence.
"""

from typing import List, Optional, Tuple

import numpy as np

from ts_haystack.core import (
    DifficultyConfig,
    GeneratedSample,
    InsertedNeedle,
    NeedleSample,
)
from ts_haystack.tasks.base_task import BaseTaskGenerator


class MultiHopTaskGenerator(BaseTaskGenerator):
    """
    Task 8: Multi-Hop Localization - "When did the Kth {target} occur {direction} {anchor}?"

    This task tests multi-step reasoning capabilities:
    1. Locate the anchor activity in the signal
    2. Identify target activity bouts relative to the anchor
    3. Count to the K-th occurrence in the specified direction
    4. Report the time range of that specific bout

    Algorithm:
    1. Sample background EXCLUDING both anchor and target activities
    2. Sample anchor activity (A) and target activity (T), both NOT in background
    3. Sample K value (1, 2, or 3) and direction ("before" or "after")
    4. Position anchor to leave room for K targets in the specified direction
    5. Insert K target needles in temporal order relative to anchor
    6. Optionally add distractor targets on the opposite side (tests direction understanding)
    7. Generate Q/A asking for the K-th target relative to anchor

    Example:
        Background: [...sleep...................................]
        + Anchor:   [...sleep...][BICYCLING][...sleep...........]
        + Targets:  [...sleep...][BICYCLING][...sleep][WALKING_1][...][WALKING_2][...]
        Q: "When did the 2nd walking bout occur after the bicycling?"
        A: "The 2nd walking bout after bicycling occurred from 7:45 AM to 8:02 AM."

    Difficulty Knobs:
    - k_distribution: Probability for K values [P(K=1), P(K=2), P(K=3)]
    - direction_mode: "random", "after_only", "before_only"
    - n_distractors_opposite: Target bouts on opposite side of anchor (confounders)
    - min_gap_samples: Minimum gap between bouts
    - needle_length_ratio_range: Duration range for needles as fraction of context

    Answer Type: time_range
    """

    @property
    def task_name(self) -> str:
        return "multi_hop"

    @property
    def answer_type(self) -> str:
        return "time_range"

    def generate_sample(
        self,
        difficulty: DifficultyConfig,
        rng: np.random.Generator,
    ) -> GeneratedSample:
        """
        Generate a single multi-hop localization task sample.

        Args:
            difficulty: Difficulty configuration
            rng: Random number generator (from SeedManager)

        Returns:
            GeneratedSample with multi-hop question and time_range answer
        """
        context_length = difficulty.context_length_samples
        min_gap = difficulty.get_effective_min_gap_samples()
        margin = difficulty.get_effective_margin_samples()

        # K distribution: probability for K=1, K=2, K=3
        k_distribution = difficulty.task_specific.get("k_distribution", [0.4, 0.4, 0.2])
        direction_mode = difficulty.task_specific.get("direction_mode", "random")
        n_distractors_opposite = difficulty.task_specific.get("n_distractors_opposite", 0)

        # =====================================================================
        # Step 1: Sample K and direction
        # =====================================================================
        K = int(rng.choice([1, 2, 3], p=k_distribution))

        if direction_mode == "after_only":
            direction = "after"
        elif direction_mode == "before_only":
            direction = "before"
        else:
            direction = rng.choice(["before", "after"])

        # =====================================================================
        # Step 2: Sample anchor and target activities
        # =====================================================================
        all_activities = list(self.needle_sampler.get_available_activities())

        if len(all_activities) < 2:
            return self._create_invalid_sample(
                "Need at least 2 activities for multi-hop",
                difficulty,
            )

        # Sample two different activities for anchor and target
        sampled = rng.choice(all_activities, size=2, replace=False)
        anchor_activity, target_activity = sampled[0], sampled[1]

        # =====================================================================
        # Step 3: Sample background EXCLUDING both activities
        # =====================================================================
        background = self.background_sampler.sample_background(
            context_length_samples=context_length,
            purity=difficulty.background_purity,
            excluded_activities={anchor_activity, target_activity},
            rng=rng,
        )

        if background is None:
            return self._create_invalid_sample(
                "Failed to sample background excluding anchor and target",
                difficulty,
            )

        # Validate annotation coverage
        is_valid, reason = self._validate_background_coverage(background, difficulty)
        if not is_valid:
            return self._create_invalid_sample(reason, difficulty)

        # Verify exclusion
        if anchor_activity in background.activities_present or target_activity in background.activities_present:
            return self._create_invalid_sample(
                "Background contains anchor or target activity",
                difficulty,
            )

        # =====================================================================
        # Step 4: Sample anchor needle
        # =====================================================================
        min_duration_ms, max_duration_ms = difficulty.get_needle_length_range_ms(
            self.source_hz
        )

        anchor_needle = self.needle_sampler.sample_needle(
            activity=anchor_activity,
            min_duration_ms=min_duration_ms,
            rng=rng,
        )

        if anchor_needle is None:
            return self._create_invalid_sample(
                f"Could not sample anchor needle for {anchor_activity}",
                difficulty,
            )

        # Trim anchor to target duration
        anchor_trimmed = self._trim_needle_to_range(
            anchor_needle, min_duration_ms, max_duration_ms, rng
        )

        # =====================================================================
        # Step 5: Calculate space requirements and sample anchor position
        # =====================================================================
        # Estimate space needed for K targets + distractors
        target_needle_samples = int(min_duration_ms * self.source_hz / 1000)
        space_for_targets = K * (target_needle_samples + min_gap)
        space_for_distractors = n_distractors_opposite * (target_needle_samples + min_gap)

        total_min_space = (
            margin +
            space_for_distractors +  # Distractors on one side
            anchor_trimmed.n_samples +
            min_gap +
            space_for_targets +  # Targets on the other side
            margin
        )

        if total_min_space > context_length:
            return self._create_invalid_sample(
                f"Not enough space for anchor + {K} targets + {n_distractors_opposite} distractors",
                difficulty,
            )

        # Position anchor based on direction
        if direction == "after":
            # Targets come after anchor, so anchor should be early enough
            # Layout: [margin][distractors_before][anchor][gap][targets...][margin]
            min_anchor_pos = margin + space_for_distractors
            max_anchor_pos = context_length - anchor_trimmed.n_samples - min_gap - space_for_targets - margin
        else:
            # Targets come before anchor, so anchor should be late enough
            # Layout: [margin][targets...][gap][anchor][distractors_after][margin]
            min_anchor_pos = margin + space_for_targets + min_gap
            max_anchor_pos = context_length - anchor_trimmed.n_samples - space_for_distractors - margin

        if max_anchor_pos <= min_anchor_pos:
            return self._create_invalid_sample(
                "Invalid anchor position range",
                difficulty,
            )

        anchor_pos = int(rng.integers(min_anchor_pos, max_anchor_pos + 1))

        # =====================================================================
        # Step 6: Sample and position K target needles
        # =====================================================================
        target_needles: List[NeedleSample] = []
        target_positions: List[int] = []

        if direction == "after":
            # Targets come after anchor
            current_pos = anchor_pos + anchor_trimmed.n_samples + min_gap

            for i in range(K):
                needle = self.needle_sampler.sample_needle(
                    activity=target_activity,
                    min_duration_ms=min_duration_ms,
                    rng=rng,
                )

                if needle is None:
                    return self._create_invalid_sample(
                        f"Could not sample target needle {i+1} for {target_activity}",
                        difficulty,
                    )

                trimmed = self._trim_needle_to_range(needle, min_duration_ms, max_duration_ms, rng)

                # Add random extra gap for variation
                extra_gap = int(rng.integers(0, min_gap + 1))
                pos = current_pos + extra_gap

                # Verify position is valid
                if pos + trimmed.n_samples > context_length - margin:
                    return self._create_invalid_sample(
                        f"Target {i+1} position exceeds context length",
                        difficulty,
                    )

                target_positions.append(pos)
                target_needles.append(trimmed)
                current_pos = pos + trimmed.n_samples + min_gap

        else:
            # Targets come before anchor (sample in reverse, then flip)
            positions_reverse: List[Tuple[int, NeedleSample]] = []
            current_pos = anchor_pos - min_gap

            for i in range(K):
                needle = self.needle_sampler.sample_needle(
                    activity=target_activity,
                    min_duration_ms=min_duration_ms,
                    rng=rng,
                )

                if needle is None:
                    return self._create_invalid_sample(
                        f"Could not sample target needle {i+1} for {target_activity}",
                        difficulty,
                    )

                trimmed = self._trim_needle_to_range(needle, min_duration_ms, max_duration_ms, rng)

                # Position needle ending before current_pos
                extra_gap = int(rng.integers(0, min_gap + 1))
                pos = current_pos - trimmed.n_samples - extra_gap

                # Verify position is valid
                if pos < margin:
                    return self._create_invalid_sample(
                        f"Target {i+1} position is before margin",
                        difficulty,
                    )

                positions_reverse.append((pos, trimmed))
                current_pos = pos - min_gap

            # Reverse to get chronological order (earliest first)
            for pos, needle in reversed(positions_reverse):
                target_positions.append(pos)
                target_needles.append(needle)

        # =====================================================================
        # Step 7: Optionally add distractor targets on opposite side
        # =====================================================================
        distractor_needles: List[NeedleSample] = []
        distractor_positions: List[int] = []

        if n_distractors_opposite > 0:
            if direction == "after":
                # Add distractors BEFORE anchor
                current_pos = anchor_pos - min_gap

                for _ in range(n_distractors_opposite):
                    d_needle = self.needle_sampler.sample_needle(
                        activity=target_activity,
                        min_duration_ms=min_duration_ms,
                        rng=rng,
                    )

                    if d_needle is None:
                        continue  # Skip if can't sample

                    d_trimmed = self._trim_needle_to_range(d_needle, min_duration_ms, max_duration_ms, rng)
                    pos = current_pos - d_trimmed.n_samples

                    if pos >= margin:
                        distractor_positions.append(pos)
                        distractor_needles.append(d_trimmed)
                        current_pos = pos - min_gap

            else:
                # Add distractors AFTER anchor
                current_pos = anchor_pos + anchor_trimmed.n_samples + min_gap

                for _ in range(n_distractors_opposite):
                    d_needle = self.needle_sampler.sample_needle(
                        activity=target_activity,
                        min_duration_ms=min_duration_ms,
                        rng=rng,
                    )

                    if d_needle is None:
                        continue

                    d_trimmed = self._trim_needle_to_range(d_needle, min_duration_ms, max_duration_ms, rng)
                    pos = current_pos

                    if pos + d_trimmed.n_samples <= context_length - margin:
                        distractor_positions.append(pos)
                        distractor_needles.append(d_trimmed)
                        current_pos = pos + d_trimmed.n_samples + min_gap

        # =====================================================================
        # Step 8: Insert all needles with style transfer
        # =====================================================================
        # Collect all insertions: (needle, position, activity, label, ordinal)
        all_insertions: List[Tuple[NeedleSample, int, str, str, int]] = []

        # Anchor
        all_insertions.append((anchor_trimmed, anchor_pos, anchor_activity, "anchor", 0))

        # Target needles (K of them in specified direction)
        for i, (needle, pos) in enumerate(zip(target_needles, target_positions)):
            # Ordinal is based on temporal position relative to anchor
            if direction == "after":
                ordinal = i + 1  # 1st, 2nd, 3rd after
            else:
                ordinal = i + 1  # 1st, 2nd, 3rd before (in chronological order)
            all_insertions.append((needle, pos, target_activity, f"target_{direction}", ordinal))

        # Distractor needles
        for needle, pos in zip(distractor_needles, distractor_positions):
            opposite_dir = "before" if direction == "after" else "after"
            all_insertions.append((needle, pos, target_activity, f"distractor_{opposite_dir}", 0))

        # Sort by position for sequential insertion
        all_insertions.sort(key=lambda x: x[1])

        # Insert needles
        current_signal = (
            background.x.copy(),
            background.y.copy(),
            background.z.copy(),
        )
        needle_metadata: List[InsertedNeedle] = []

        for needle, pos, activity, label, ordinal in all_insertions:
            # Apply style transfer
            local_stats = self.style_transfer.compute_local_statistics(
                background=current_signal,
                position=pos,
            )
            transferred = self.style_transfer.transfer(needle, local_stats)

            # Insert with blending
            current_signal = self.style_transfer.insert_with_blending(
                background=current_signal,
                needle=(transferred.x, transferred.y, transferred.z),
                position=pos,
            )

            # Create metadata
            inserted = InsertedNeedle(
                activity=activity,
                source_pid=needle.source_pid,
                source_start_ms=needle.start_ms,
                source_end_ms=needle.end_ms,
                insert_position_samples=pos,
                insert_position_frac=pos / context_length,
                duration_samples=needle.n_samples,
                duration_ms=needle.duration_ms,
                timestamp_start=self._samples_to_timestamp(pos, background),
                timestamp_end=self._samples_to_timestamp(pos + needle.n_samples, background),
            )
            needle_metadata.append(inserted)

        # =====================================================================
        # Step 9: Determine the answer (K-th target in specified direction)
        # =====================================================================
        # Find the K-th target needle (ordinal == K)
        answer_needle: Optional[InsertedNeedle] = None

        for (needle, pos, activity, label, ordinal), metadata in zip(all_insertions, needle_metadata):
            if label == f"target_{direction}" and ordinal == K:
                answer_needle = metadata
                break

        if answer_needle is None:
            return self._create_invalid_sample(
                f"Could not find {K}-th target in direction '{direction}'",
                difficulty,
            )

        # =====================================================================
        # Step 10: Generate Q/A using template bank
        # =====================================================================
        question, answer = self.template_bank.sample(
            task="multi_hop",
            rng=rng,
            target_activity=target_activity,
            anchor_activity=anchor_activity,
            k=K,  # Will be converted to ordinal in template
            direction=direction,
            start=answer_needle.timestamp_start,
            end=answer_needle.timestamp_end,
        )

        # Build difficulty config with task-specific info
        full_difficulty_config = {
            **difficulty.to_dict(),
            "anchor_activity": anchor_activity,
            "target_activity": target_activity,
            "K": K,
            "direction": direction,
            "anchor_pos": anchor_pos,
            "target_positions": target_positions,
            "n_distractors_inserted": len(distractor_needles),
            "n_targets_inserted": len(target_needles),
            "answer_target_index": K - 1,  # 0-indexed
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
        description="Generate multi-hop localization task dataset for TS-Haystack"
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
        "--direction-mode",
        choices=["random", "after_only", "before_only"],
        default="random",
        help="Direction mode for target placement",
    )
    parser.add_argument(
        "--n-distractors",
        type=int,
        default=0,
        help="Number of distractor targets on opposite side of anchor",
    )
    parser.add_argument(
        "--k-max",
        type=int,
        choices=[1, 2, 3],
        default=3,
        help="Maximum K value (ordinal)",
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
        default=0.06,
        help="Maximum needle length as fraction of context (default: 0.06 = 6%%)",
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

    # Build k_distribution based on k_max
    if args.k_max == 1:
        k_distribution = [1.0, 0.0, 0.0]
    elif args.k_max == 2:
        k_distribution = [0.5, 0.5, 0.0]
    else:
        k_distribution = [0.4, 0.4, 0.2]

    print("Creating MultiHopTaskGenerator...")
    generator = MultiHopTaskGenerator.create_with_artifacts(seed=args.seed)

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
                "k_distribution": k_distribution,
                "direction_mode": args.direction_mode,
                "n_distractors_opposite": args.n_distractors,
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
