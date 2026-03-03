# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors
# SPDX-License-Identifier: MIT

"""
Base task generator for TS-Haystack benchmark.

Provides abstract base class that all task generators inherit from.
Implements shared functionality for sample generation, batch processing,
and output serialization.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import json
import numpy as np
import polars as pl

from ts_haystack.core import (
    BackgroundSample,
    BackgroundSampler,
    BoutIndexer,
    DifficultyConfig,
    GeneratedSample,
    InsertedNeedle,
    NeedleSampler,
    NeedleSample,
    PromptTemplateBank,
    SeedManager,
    SignalStatistics,
    StyleTransfer,
    TaskConfig,
    TimelineBuilder,
    TransitionMatrix,
)
from ts_haystack.utils import (
    format_context_dir,
    samples_to_timestamp,
    sample_position_with_mode,
    find_non_overlapping_position,
)
from ts_haystack.capture24.loader import DATA_DIR


# Default output directory
TS_HAYSTACK_DATA_DIR = Path(DATA_DIR) / "ts_haystack"


class BaseTaskGenerator(ABC):
    """
    Abstract base class for all task generators.

    Design Principles:
    1. Dependency injection: All Phase 2 components passed via constructor
    2. Single RNG per sample: Each generate_sample() receives its own RNG
    3. Separation of concerns: This class orchestrates, components do the work
    4. Parallel-safe: Stateless generate_sample() enables parallel execution

    Usage:
        # Initialize with Phase 2 components
        generator = ConcreteTaskGenerator(
            background_sampler=bg_sampler,
            needle_sampler=needle_sampler,
            style_transfer=style_transfer,
            template_bank=template_bank,
            seed_manager=seed_manager,
        )

        # Generate single sample
        rng = seed_manager.get_sample_rng("task", 10000, "train", 0)
        sample = generator.generate_sample(difficulty, rng)

        # Generate batch
        samples = generator.generate_dataset(
            n_samples=1000,
            difficulty=difficulty,
            split="train",
        )
    """

    def __init__(
        self,
        background_sampler: BackgroundSampler,
        needle_sampler: NeedleSampler,
        style_transfer: StyleTransfer,
        template_bank: PromptTemplateBank,
        seed_manager: SeedManager,
        source_hz: int = 100,
    ):
        """
        Initialize task generator with Phase 2 components.

        Args:
            background_sampler: Sampler for background windows
            needle_sampler: Sampler for needle bouts
            style_transfer: Style transfer and blending
            template_bank: NL template bank for Q/A diversity
            seed_manager: Seed manager for reproducibility
            source_hz: Source data sampling frequency
        """
        self.background_sampler = background_sampler
        self.needle_sampler = needle_sampler
        self.style_transfer = style_transfer
        self.template_bank = template_bank
        self.seed_manager = seed_manager
        self.source_hz = source_hz

    # =========================================================================
    # Abstract Properties & Methods (Must be implemented by subclasses)
    # =========================================================================

    @property
    @abstractmethod
    def task_name(self) -> str:
        """
        Unique task identifier.

        Examples: 'existence', 'localization', 'counting', etc.
        """
        ...

    @property
    @abstractmethod
    def answer_type(self) -> str:
        """
        Type of answer produced by this task.

        One of: 'boolean', 'timestamp', 'integer', 'category', 'time_range'
        """
        ...

    @abstractmethod
    def generate_sample(
        self,
        difficulty: DifficultyConfig,
        rng: np.random.Generator,
    ) -> GeneratedSample:
        """
        Generate a single sample using the provided RNG.

        This method MUST be:
        - Deterministic given the same RNG state
        - Stateless (no side effects on self)
        - Self-contained (all randomness from rng parameter)

        Args:
            difficulty: Difficulty configuration for this sample
            rng: Random number generator (from SeedManager)

        Returns:
            GeneratedSample (may have is_valid=False if generation failed)
        """
        ...

    # =========================================================================
    # Shared Helper Methods
    # =========================================================================

    def _samples_to_timestamp(
        self,
        sample_idx: int,
        background: BackgroundSample,
    ) -> str:
        """
        Convert sample index to human-readable timestamp.

        Args:
            sample_idx: Sample index within the background window
            background: BackgroundSample with time context

        Returns:
            Human-readable timestamp string (e.g., "6:45 AM")
        """
        return samples_to_timestamp(
            sample_idx=sample_idx,
            total_samples=background.n_samples,
            start_time_str=background.recording_time_context[0],
            end_time_str=background.recording_time_context[1],
        )

    def _validate_background_coverage(
        self,
        background: BackgroundSample,
        difficulty: DifficultyConfig,
    ) -> Tuple[bool, str]:
        """
        Validate that background has sufficient annotation coverage.

        Backgrounds with large unlabeled gaps (where annotations don't map to the
        label scheme) can create ambiguous samples. This validation ensures that
        at least `min_annotation_coverage` of the background window has activity
        annotations.

        Args:
            background: BackgroundSample to validate
            difficulty: DifficultyConfig with min_annotation_coverage threshold

        Returns:
            Tuple of (is_valid, reason_string)

        Example:
            If background.annotation_coverage = 0.45 and min_annotation_coverage = 0.6,
            returns (False, "Low annotation coverage: 45.0% < 60.0% required")
        """
        coverage = background.annotation_coverage
        min_coverage = difficulty.min_annotation_coverage

        if coverage < min_coverage:
            return (
                False,
                f"Low annotation coverage: {coverage:.1%} < {min_coverage:.1%} required",
            )
        return True, ""

    def _sample_position(
        self,
        context_length: int,
        needle_length: int,
        position_mode: str,
        rng: np.random.Generator,
        margin_samples: int = 100,
    ) -> Optional[int]:
        """
        Sample needle insertion position based on mode.

        Args:
            context_length: Total context window length in samples
            needle_length: Length of needle to insert in samples
            position_mode: "beginning", "middle", "end", or "random"
            rng: Random number generator
            margin_samples: Minimum margin from window edges

        Returns:
            Position index, or None if needle doesn't fit
        """
        return sample_position_with_mode(
            context_length=context_length,
            needle_length=needle_length,
            mode=position_mode,
            margin=margin_samples,
            rng=rng,
        )

    def _find_valid_position(
        self,
        context_length: int,
        needle_length: int,
        occupied_ranges: List[Tuple[int, int]],
        min_gap: int,
        rng: np.random.Generator,
        max_attempts: int = 100,
        margin: int = 0,
    ) -> Optional[int]:
        """
        Find non-overlapping position for needle insertion.

        Args:
            context_length: Total context window length in samples
            needle_length: Length of needle to insert in samples
            occupied_ranges: List of (start, end) tuples of occupied regions
            min_gap: Minimum gap between needles in samples
            rng: Random number generator
            max_attempts: Maximum sampling attempts
            margin: Minimum margin from window edges

        Returns:
            Valid position index, or None if no valid position found
        """
        return find_non_overlapping_position(
            context_length=context_length,
            needle_length=needle_length,
            occupied=occupied_ranges,
            min_gap=min_gap,
            rng=rng,
            max_attempts=max_attempts,
            margin=margin,
        )

    def _compute_local_stats(
        self,
        background: BackgroundSample,
        position: int,
        window_samples: int = 500,
    ) -> SignalStatistics:
        """
        Compute local statistics at insertion position for style transfer.

        Args:
            background: BackgroundSample with sensor data
            position: Insertion position in samples
            window_samples: Window size for computing statistics

        Returns:
            SignalStatistics for the local region
        """
        return self.style_transfer.compute_local_statistics(
            background=(background.x, background.y, background.z),
            position=position,
            window_samples=window_samples,
        )

    def _trim_needle(
        self,
        needle: NeedleSample,
        target_samples: int,
    ) -> NeedleSample:
        """
        Trim needle to target number of samples.

        Uses NeedleSample.trim() which trims from center to preserve
        characteristic signal.

        Args:
            needle: Original needle sample
            target_samples: Target number of samples

        Returns:
            Trimmed NeedleSample
        """
        return needle.trim(target_samples)

    def _create_inserted_needle(
        self,
        needle: NeedleSample,
        position: int,
        context_length: int,
        background: BackgroundSample,
    ) -> InsertedNeedle:
        """
        Create InsertedNeedle metadata for a placed needle.

        Args:
            needle: The needle sample
            position: Insertion position in samples
            context_length: Total context length in samples
            background: Background sample for timestamp computation

        Returns:
            InsertedNeedle with all metadata
        """
        return InsertedNeedle(
            activity=needle.activity,
            source_pid=needle.source_pid,
            source_start_ms=needle.start_ms,
            source_end_ms=needle.end_ms,
            insert_position_samples=position,
            insert_position_frac=position / context_length,
            duration_samples=needle.n_samples,
            duration_ms=needle.duration_ms,
            timestamp_start=self._samples_to_timestamp(position, background),
            timestamp_end=self._samples_to_timestamp(position + needle.n_samples, background),
        )

    def _create_invalid_sample(
        self,
        reason: str,
        difficulty: DifficultyConfig,
    ) -> GeneratedSample:
        """
        Create a marked-invalid sample for retry handling.

        Args:
            reason: Reason for invalidity
            difficulty: Difficulty configuration used

        Returns:
            GeneratedSample with is_valid=False
        """
        return GeneratedSample(
            x=np.array([], dtype=np.float32),
            y=np.array([], dtype=np.float32),
            z=np.array([], dtype=np.float32),
            task_type=self.task_name,
            context_length_samples=difficulty.context_length_samples,
            background_pid="",
            recording_time_range=("", ""),
            question="",
            answer="",
            answer_type=self.answer_type,
            needles=[],
            difficulty_config=difficulty.to_dict(),
            is_valid=False,
            validation_notes=reason,
        )

    def _insert_needle(
        self,
        background: BackgroundSample,
        needle: NeedleSample,
        position: int,
        current_signal: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply style transfer and insert needle into signal.

        Args:
            background: BackgroundSample for statistics computation
            needle: Needle to insert
            position: Insertion position
            current_signal: Optional existing signal to modify
                           (if None, uses background data)

        Returns:
            Tuple of (x, y, z) arrays with needle inserted
        """
        if current_signal is None:
            current_signal = (background.x.copy(), background.y.copy(), background.z.copy())

        # Compute local statistics for style transfer
        local_stats = self.style_transfer.compute_local_statistics(
            background=current_signal,
            position=position,
        )

        # Apply style transfer
        transferred = self.style_transfer.transfer(needle, local_stats)

        # Insert with blending
        return self.style_transfer.insert_with_blending(
            background=current_signal,
            needle=(transferred.x, transferred.y, transferred.z),
            position=position,
        )

    # =========================================================================
    # Batch Generation
    # =========================================================================

    def generate_dataset(
        self,
        n_samples: int,
        difficulty: DifficultyConfig,
        split: str,
        n_jobs: int = 1,
        retry_factor: float = 2.0,
        verbose: bool = True,
    ) -> List[GeneratedSample]:
        """
        Generate n_samples for a split with automatic retry handling.

        Uses pre-computed seeds from SeedManager for reproducibility.

        Args:
            n_samples: Number of valid samples to generate
            difficulty: Difficulty configuration
            split: Data split ("train", "val", "test")
            n_jobs: Number of parallel jobs (1 = sequential)
            retry_factor: Factor to multiply n_samples for retry seeds
            verbose: Whether to print progress

        Returns:
            List of n_samples valid GeneratedSample objects
        """
        max_seeds = int(n_samples * retry_factor)
        sample_seeds = self.seed_manager.get_sample_seeds(
            task=self.task_name,
            context_length=difficulty.context_length_samples,
            split=split,
            n_samples=max_seeds,
        )

        if n_jobs == 1:
            return self._generate_sequential(
                sample_seeds, difficulty, n_samples, verbose
            )
        else:
            return self._generate_parallel(
                sample_seeds, difficulty, n_samples, n_jobs, verbose
            )

    def _generate_sequential(
        self,
        sample_seeds: List[int],
        difficulty: DifficultyConfig,
        n_samples: int,
        verbose: bool = True,
    ) -> List[GeneratedSample]:
        """
        Generate samples sequentially with retry support.

        Args:
            sample_seeds: Pre-computed seeds for each attempt
            difficulty: Difficulty configuration
            n_samples: Number of valid samples to generate
            verbose: Whether to print progress

        Returns:
            List of valid GeneratedSample objects
        """
        samples = []
        attempts = 0
        failures = 0

        for seed in sample_seeds:
            if len(samples) >= n_samples:
                break

            rng = np.random.default_rng(seed)
            sample = self.generate_sample(difficulty, rng)
            attempts += 1

            if sample.is_valid:
                samples.append(sample)
                if verbose and len(samples) % 100 == 0:
                    print(f"  Generated {len(samples)}/{n_samples} samples "
                          f"({failures} failures)")
            else:
                failures += 1

        if len(samples) < n_samples:
            print(f"WARNING: Only generated {len(samples)}/{n_samples} samples "
                  f"after {attempts} attempts ({failures} failures)")

        return samples

    def _generate_parallel(
        self,
        sample_seeds: List[int],
        difficulty: DifficultyConfig,
        n_samples: int,
        n_jobs: int,
        verbose: bool = True,
    ) -> List[GeneratedSample]:
        """
        Generate samples in parallel.

        Args:
            sample_seeds: Pre-computed seeds for each attempt
            difficulty: Difficulty configuration
            n_samples: Number of valid samples to generate
            n_jobs: Number of parallel jobs
            verbose: Whether to print progress

        Returns:
            List of valid GeneratedSample objects
        """
        from joblib import Parallel, delayed

        def generate_one(seed: int) -> GeneratedSample:
            rng = np.random.default_rng(seed)
            return self.generate_sample(difficulty, rng)

        if verbose:
            print(f"Generating {n_samples} samples using {n_jobs} jobs...")

        # Generate all samples in parallel
        all_samples = Parallel(n_jobs=n_jobs)(
            delayed(generate_one)(seed)
            for seed in sample_seeds[:int(n_samples * 1.5)]  # Initial batch
        )

        # Filter valid samples
        valid_samples = [s for s in all_samples if s.is_valid]

        # If we don't have enough, generate more
        seed_idx = int(n_samples * 1.5)
        while len(valid_samples) < n_samples and seed_idx < len(sample_seeds):
            batch_size = min(n_samples - len(valid_samples), n_jobs * 10)
            batch_seeds = sample_seeds[seed_idx:seed_idx + batch_size]
            seed_idx += batch_size

            batch_samples = Parallel(n_jobs=n_jobs)(
                delayed(generate_one)(seed) for seed in batch_seeds
            )
            valid_samples.extend([s for s in batch_samples if s.is_valid])

        if len(valid_samples) < n_samples:
            print(f"WARNING: Only generated {len(valid_samples)}/{n_samples} "
                  f"valid samples")

        return valid_samples[:n_samples]

    # =========================================================================
    # Output Serialization
    # =========================================================================

    def save_dataset(
        self,
        samples: List[GeneratedSample],
        split: str,
        context_length: int,
        output_dir: Optional[Path] = None,
    ) -> Path:
        """
        Save generated samples to parquet file.

        Directory structure: {output_dir}/{context_s}s/{task}/{split}/data.parquet
        E.g., tasks/100s/existence/train/data.parquet

        Args:
            samples: List of GeneratedSample objects
            split: Data split ("train", "val", "test")
            context_length: Context length in samples
            output_dir: Output directory (default: data/capture24/ts_haystack/tasks/)

        Returns:
            Path to saved parquet file
        """
        if output_dir is None:
            output_dir = TS_HAYSTACK_DATA_DIR / "tasks"

        # Convert samples to seconds for human-readable directory name
        context_s = context_length / self.source_hz
        task_dir = output_dir / format_context_dir(context_s) / self.task_name / split
        task_dir.mkdir(parents=True, exist_ok=True)

        # Convert samples to dictionaries
        records = []
        for sample in samples:
            record = sample.to_dict()
            # Serialize nested objects as JSON strings
            record["needles"] = json.dumps(record["needles"])
            record["difficulty_config"] = json.dumps(record["difficulty_config"])
            records.append(record)

        # Create DataFrame and save
        df = pl.DataFrame(records)
        output_path = task_dir / "data.parquet"
        df.write_parquet(output_path)

        return output_path

    def save_metadata(
        self,
        config: TaskConfig,
        generation_stats: Dict[str, Any],
        context_length: int,
        output_dir: Optional[Path] = None,
    ) -> Path:
        """
        Save task metadata alongside generated data.

        Directory structure: {output_dir}/{context_s}s/{task}/metadata.json
        E.g., tasks/100s/existence/metadata.json

        Args:
            config: Task configuration used
            generation_stats: Statistics about generation
            context_length: Context length in samples
            output_dir: Output directory

        Returns:
            Path to saved metadata file
        """
        if output_dir is None:
            output_dir = TS_HAYSTACK_DATA_DIR / "tasks"

        # Convert samples to seconds for human-readable directory name
        context_s = context_length / self.source_hz
        task_dir = output_dir / format_context_dir(context_s) / self.task_name
        task_dir.mkdir(parents=True, exist_ok=True)

        metadata = {
            "task_name": self.task_name,
            "answer_type": self.answer_type,
            "context_length_samples": context_length,
            "context_length_seconds": context_s,
            "source_hz": self.source_hz,
            "config": config.to_dict(),
            "seed_config": self.seed_manager.get_metadata(),
            "generation_stats": generation_stats,
        }

        output_path = task_dir / "metadata.json"
        with open(output_path, "w") as f:
            json.dump(metadata, f, indent=2)

        return output_path

    # =========================================================================
    # Factory Method
    # =========================================================================

    @classmethod
    def create_with_artifacts(
        cls,
        seed: int = 42,
        source_hz: int = 100,
        transfer_mode: str = "mean_only",
        blend_mode: str = "cosine",
    ) -> "BaseTaskGenerator":
        """
        Factory method to create generator with loaded Phase 1 artifacts.

        Loads timelines, bout index, and transition matrix from disk,
        then initializes all Phase 2 components.

        Args:
            seed: Master seed for reproducibility
            source_hz: Source data sampling frequency
            transfer_mode: Style transfer mode ("mean_only" or "full")
            blend_mode: Blending mode ("cosine" or "linear")

        Returns:
            Initialized task generator
        """
        # Load Phase 1 artifacts
        timelines = TimelineBuilder.load_all_timelines()
        bout_index = BoutIndexer.load_index()
        transition_matrix = TransitionMatrix.load()

        # Initialize Phase 2 components
        seed_manager = SeedManager(master_seed=seed)
        background_sampler = BackgroundSampler(timelines, bout_index, source_hz)
        needle_sampler = NeedleSampler(bout_index, transition_matrix, source_hz)
        style_transfer = StyleTransfer(
            transfer_mode=transfer_mode,
            blend_mode=blend_mode,
        )
        template_bank = PromptTemplateBank()

        return cls(
            background_sampler=background_sampler,
            needle_sampler=needle_sampler,
            style_transfer=style_transfer,
            template_bank=template_bank,
            seed_manager=seed_manager,
            source_hz=source_hz,
        )
