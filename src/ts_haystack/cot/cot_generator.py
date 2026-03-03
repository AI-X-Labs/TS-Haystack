# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

"""
TS-Haystack Chain-of-Thought (CoT) Generator.

This module generates LLM-based chain-of-thought rationales for TS-Haystack
benchmark samples. It uses rich metadata (needle positions, activities, timestamps)
to create grounded, natural reasoning.

Key Features:
- Parallel processing with ThreadPoolExecutor
- Incremental saving for resume capability
- Answer validation to ensure rationale consistency
- Plot generation for visual LLM input

Usage:
    from ts_haystack.cot import (
        TSHaystackCoTGenerator,
        OpenAICoTClient,
    )

    client = OpenAICoTClient()
    generator = TSHaystackCoTGenerator(client)
    generator.process_dataset(input_parquet, output_parquet)
"""

import base64
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import polars as pl
from PIL import Image
from tqdm import tqdm

from ts_haystack.cot.llm_client import (
    OpenAICoTClient,
    OpenAIConfig,
)
from ts_haystack.cot.plot_generator import (
    create_accelerometer_plot_from_sample,
)
from ts_haystack.cot.prompt_builder import (
    create_cot_prompt,
)


@dataclass
class GenerationStats:
    """Statistics for tracking generation progress."""
    total_processed: int = 0
    successful: int = 0
    failed: int = 0
    api_errors: int = 0
    validation_errors: int = 0
    task_counts: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "total_processed": self.total_processed,
            "successful": self.successful,
            "failed": self.failed,
            "api_errors": self.api_errors,
            "validation_errors": self.validation_errors,
            "task_counts": self.task_counts,
        }


class TSHaystackCoTGenerator:
    """
    Generate LLM-based CoT rationales for TS-Haystack samples.

    This generator:
    1. Reads parquet files with sample data and metadata
    2. Creates accelerometer plots for visual input
    3. Builds prompts with rich ground truth context
    4. Calls LLM to generate natural reasoning
    5. Validates answer consistency
    6. Saves results incrementally

    Args:
        llm_client: OpenAI API client for generation
        include_plot: If True, include accelerometer plot in LLM input
        annotate_needles: If True, annotate needle regions in plots
        max_workers: Number of parallel workers for API calls
        validate_answers: If True, validate generated answers match expected
        save_interval: Save partial results every N samples
        debug_mode: If True, save debug outputs (JSON + plots) for each sample
        debug_output_dir: Directory to save debug outputs (required if debug_mode=True)

    Example:
        >>> client = OpenAICoTClient()
        >>> generator = TSHaystackCoTGenerator(client, max_workers=4)
        >>> generator.process_dataset(
        ...     Path("data/capture24/ts_haystack/tasks/100s/existence/train/data.parquet"),
        ...     Path("data/capture24/ts_haystack/cot/100s/existence/train/data.parquet"),
        ... )

    Debug Mode Example:
        >>> generator = TSHaystackCoTGenerator(
        ...     client,
        ...     debug_mode=True,
        ...     debug_output_dir=Path("debug_output/cot"),
        ... )
        # Creates JSON files with sample data, prompt, rationale, and plot images
    """

    def __init__(
        self,
        llm_client: OpenAICoTClient,
        include_plot: bool = True,
        annotate_needles: bool = True,
        max_workers: int = 4,
        validate_answers: bool = True,
        save_interval: int = 100,
        debug_mode: bool = False,
        debug_output_dir: Optional[Path] = None,
    ):
        self.llm_client = llm_client
        self.include_plot = include_plot
        self.annotate_needles = annotate_needles
        self.max_workers = max_workers
        self.validate_answers = validate_answers
        self.save_interval = save_interval
        self.debug_mode = debug_mode
        self.debug_output_dir = debug_output_dir

        # Thread-safe locks
        self._save_lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self._debug_lock = threading.Lock()

    def generate_rationale(
        self, sample: Dict, return_debug_info: bool = False
    ) -> Optional[str] | Tuple[Optional[str], Optional[Image.Image], str]:
        """
        Generate rationale for a single sample.

        Args:
            sample: Dict containing sample data (x_axis, y_axis, z_axis,
                   question, answer, needles, difficulty_config, etc.)
            return_debug_info: If True, also return plot image and prompt for debugging

        Returns:
            If return_debug_info=False: Full rationale string ending with "Answer: ...", or None if failed
            If return_debug_info=True: Tuple of (rationale, plot_image, prompt)
        """
        # Create plot if requested
        image = None
        if self.include_plot:
            try:
                image = create_accelerometer_plot_from_sample(
                    sample,
                    annotate_needles=self.annotate_needles,
                )
            except Exception as e:
                print(f"  [WARNING] Plot generation failed: {e}")
                # Continue without plot

        # Build prompt
        prompt = create_cot_prompt(sample)

        # Generate response
        response = self.llm_client.generate(prompt, image=image)

        if response is None:
            if return_debug_info:
                return None, image, prompt
            return None

        # Extract rationale and answer
        rationale = response.get("rationale", "")
        answer = response.get("answer", "")

        # Combine into full response (avoid duplicate "Answer:" if LLM already included it)
        if "Answer:" in rationale:
            # LLM already included answer, use as-is
            full_rationale = rationale
        else:
            full_rationale = f"{rationale} Answer: {answer}"

        if return_debug_info:
            return full_rationale, image, prompt
        return full_rationale

    def _validate_answer(self, generated: str, expected: str) -> bool:
        """
        Validate that the generated answer matches the expected answer.

        Args:
            generated: Full rationale string ending with "Answer: ..."
            expected: Expected answer string

        Returns:
            True if answers match, False otherwise
        """
        # Extract answer from generated rationale
        if "Answer:" in generated:
            gen_answer = generated.split("Answer:")[-1].strip()
        else:
            gen_answer = generated.strip()

        # Normalize for comparison
        gen_normalized = gen_answer.lower().strip().rstrip(".")
        exp_normalized = expected.lower().strip().rstrip(".")

        # Direct match
        if gen_normalized == exp_normalized:
            return True

        # Boolean equivalents
        bool_map = {"yes": "yes", "no": "no", "true": "yes", "false": "no"}
        if gen_normalized in bool_map and exp_normalized in bool_map:
            return bool_map[gen_normalized] == bool_map[exp_normalized]

        return False

    def _save_debug_output(
        self,
        sample: Dict,
        idx: int,
        rationale: str,
        image: Optional[Image.Image],
        prompt: str,
        task_type: str,
    ) -> None:
        """
        Save debug output for a single sample.

        Creates a JSON file with sample data, rationale, and prompt,
        plus a PNG file with the plot image if available.

        Args:
            sample: Original sample data dict
            idx: Sample index
            rationale: Generated rationale
            image: Plot image (PIL Image) or None
            prompt: The prompt sent to the LLM
            task_type: Task type for directory organization
        """
        if not self.debug_output_dir:
            return

        with self._debug_lock:
            # Create task-specific debug directory
            debug_dir = self.debug_output_dir / task_type
            debug_dir.mkdir(parents=True, exist_ok=True)

            # Prepare sample data for JSON (filter out large arrays)
            sample_json = {}
            for key, value in sample.items():
                # Skip large arrays (accelerometer data)
                if key in ("x_axis", "y_axis", "z_axis"):
                    # Include shape info but not full data
                    if hasattr(value, "__len__"):
                        sample_json[f"{key}_length"] = len(value)
                    continue
                # Handle non-JSON-serializable types
                try:
                    json.dumps(value)
                    sample_json[key] = value
                except (TypeError, ValueError):
                    sample_json[key] = str(value)

            # Build debug output
            debug_data = {
                "sample_idx": idx,
                "task_type": task_type,
                "sample_data": sample_json,
                "prompt": prompt,
                "rationale": rationale,
                "timestamp": datetime.now().isoformat(),
            }

            # Add base64-encoded plot if available
            if image is not None:
                buffer = BytesIO()
                image.save(buffer, format="PNG")
                buffer.seek(0)
                debug_data["plot_base64"] = base64.b64encode(buffer.read()).decode("utf-8")

            # Save JSON
            json_path = debug_dir / f"sample_{idx:06d}.json"
            with open(json_path, "w") as f:
                json.dump(debug_data, f, indent=2)

            # Also save plot as separate PNG for easy viewing
            if image is not None:
                png_path = debug_dir / f"sample_{idx:06d}_plot.png"
                image.save(png_path)

    def _process_sample(
        self,
        sample: Dict,
        idx: int,
        stats: GenerationStats,
    ) -> Optional[Dict]:
        """
        Process a single sample and return result dict.

        Args:
            sample: Sample data dict
            idx: Sample index
            stats: Stats object for tracking (thread-safe updates)

        Returns:
            Dict with sample data + rationale, or None if failed
        """
        try:
            task_type = sample.get("task_type", "unknown")

            # Generate with debug info if debug mode is enabled
            if self.debug_mode:
                result = self.generate_rationale(sample, return_debug_info=True)
                rationale, image, prompt = result
            else:
                rationale = self.generate_rationale(sample)
                image, prompt = None, ""

            if rationale is None:
                with self._stats_lock:
                    stats.api_errors += 1
                return None

            # Validate answer if requested
            if self.validate_answers:
                expected_answer = sample.get("answer", "")
                if not self._validate_answer(rationale, expected_answer):
                    with self._stats_lock:
                        stats.validation_errors += 1
                    # Still return the rationale, but log the mismatch
                    print(f"  [VALIDATION] Sample {idx}: answer mismatch")

            # Save debug output if debug mode is enabled
            if self.debug_mode and self.debug_output_dir:
                self._save_debug_output(
                    sample=sample,
                    idx=idx,
                    rationale=rationale,
                    image=image,
                    prompt=prompt,
                    task_type=task_type,
                )

            # Update stats
            with self._stats_lock:
                stats.successful += 1
                stats.task_counts[task_type] = stats.task_counts.get(task_type, 0) + 1

            return {
                "idx": idx,
                "rationale": rationale,
            }

        except Exception as e:
            print(f"  [ERROR] Sample {idx} failed: {e}")
            with self._stats_lock:
                stats.failed += 1
            return None

    def process_dataset(
        self,
        input_parquet: Path,
        output_parquet: Path,
        resume: bool = True,
        max_samples: Optional[int] = None,
    ) -> GenerationStats:
        """
        Process entire parquet file, adding rationale column.

        Args:
            input_parquet: Path to input parquet file
            output_parquet: Path to output parquet file
            resume: If True, resume from existing output file
            max_samples: If set, process only this many samples (for testing)

        Returns:
            GenerationStats with processing statistics
        """
        print(f"\n{'='*60}")
        print(f"Processing: {input_parquet}")
        print(f"Output: {output_parquet}")
        if self.debug_mode and self.debug_output_dir:
            print(f"Debug output: {self.debug_output_dir}")
        print(f"{'='*60}")

        # Load input data
        df = pl.read_parquet(input_parquet)
        total_samples = len(df)

        if max_samples:
            total_samples = min(total_samples, max_samples)
            df = df.head(max_samples)

        print(f"Total samples: {total_samples}")

        # Initialize rationales list
        rationales = [""] * total_samples

        # Resume support
        start_idx = 0
        if resume and output_parquet.exists():
            try:
                existing_df = pl.read_parquet(output_parquet)
                if "rationale" in existing_df.columns:
                    for i, r in enumerate(existing_df["rationale"].to_list()):
                        if i < total_samples and r:
                            rationales[i] = r
                    start_idx = sum(1 for r in rationales if r)
                    print(f"Resuming from sample {start_idx}")
            except Exception as e:
                print(f"Warning: Could not load existing output: {e}")

        # Create output directory
        output_parquet.parent.mkdir(parents=True, exist_ok=True)

        # Initialize stats
        stats = GenerationStats()
        stats.successful = start_idx

        # Identify samples to process
        samples_to_process = [
            (df.row(i, named=True), i)
            for i in range(total_samples)
            if not rationales[i]
        ]

        if not samples_to_process:
            print("All samples already processed")
            return stats

        print(f"Samples to process: {len(samples_to_process)}")
        print(f"Workers: {self.max_workers}")

        # Process with thread pool
        processed_since_save = 0

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._process_sample, sample, idx, stats): (sample, idx)
                for sample, idx in samples_to_process
            }

            for future in tqdm(as_completed(futures), total=len(futures), desc="Generating"):
                sample, idx = futures[future]

                with self._stats_lock:
                    stats.total_processed += 1

                try:
                    result = future.result()
                    if result:
                        rationales[result["idx"]] = result["rationale"]
                        processed_since_save += 1
                except Exception as e:
                    print(f"  [ERROR] Future failed: {e}")
                    with self._stats_lock:
                        stats.failed += 1

                # Incremental save
                if processed_since_save >= self.save_interval:
                    self._save_partial(df, rationales, output_parquet)
                    processed_since_save = 0

        # Final save
        self._save_partial(df, rationales, output_parquet)

        print(f"\nGeneration complete!")
        print(f"  Successful: {stats.successful}")
        print(f"  Failed: {stats.failed}")
        print(f"  API errors: {stats.api_errors}")
        print(f"  Validation errors: {stats.validation_errors}")
        print(f"  Task counts: {stats.task_counts}")

        return stats

    def _save_partial(
        self,
        df: pl.DataFrame,
        rationales: List[str],
        output_path: Path,
    ) -> None:
        """Save partial results for resume capability."""
        with self._save_lock:
            output_df = df.with_columns(pl.Series("rationale", rationales))
            output_df.write_parquet(output_path)


def save_generation_metadata(
    output_dir: Path,
    config: OpenAIConfig,
    all_stats: Dict[str, GenerationStats],
) -> None:
    """
    Save generation metadata to JSON file.

    Args:
        output_dir: Directory to save metadata
        config: OpenAI client configuration
        all_stats: Dict mapping file paths to generation stats
    """
    metadata = {
        "generation_date": datetime.now().isoformat(),
        "llm_model": config.model,
        "temperature": config.temperature,
        "stats_by_file": {
            str(path): stats.to_dict()
            for path, stats in all_stats.items()
        },
        "total_samples": sum(s.successful for s in all_stats.values()),
        "total_failed": sum(s.failed for s in all_stats.values()),
    }

    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Metadata saved: {metadata_path}")


if __name__ == "__main__":
    print("=" * 60)
    print("TS-Haystack CoT Generator Test")
    print("=" * 60)

    # Test with a small sample
    try:
        from pathlib import Path

        # Check if test data exists
        test_input = Path("data/capture24/ts_haystack/tasks/100s/existence/test/data.parquet")

        if test_input.exists():
            print(f"\nTest input found: {test_input}")

            # Initialize client and generator
            client = OpenAICoTClient()
            generator = TSHaystackCoTGenerator(
                client,
                include_plot=True,
                annotate_needles=True,
                max_workers=2,
            )

            # Test with single sample
            df = pl.read_parquet(test_input)
            sample = df.head(1).to_dicts()[0]

            print("\nTesting single sample generation...")
            print(f"  Task: {sample.get('task_type')}")
            print(f"  Question: {sample.get('question')}")
            print(f"  Expected answer: {sample.get('answer')}")

            rationale = generator.generate_rationale(sample)

            if rationale:
                print(f"\nGenerated rationale:")
                print(f"  {rationale[:500]}...")
            else:
                print("\nGeneration failed")

        else:
            print(f"\nTest data not found: {test_input}")
            print("Generate datasets first with generate_ts_haystack_dataset.py")

    except Exception as e:
        print(f"\nError: {e}")
        print("Make sure OPENAI_API_KEY environment variable is set.")

    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)
