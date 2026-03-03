# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

"""
Capture24 Chain-of-Thought (CoT) Dataset Generator

This script generates chain-of-thought rationales for Capture24 accelerometer
classification using an LLM (Gemini 2.5 Flash). Following the HAR-CoT paper's
strategy, each sample uses binary classification with the correct label and
a dissimilar label.

Usage:
    python -m ts_haystack.capture24.cot.cot_generator \
        --train-samples 10000 --val-samples 2000 --test-samples 2000

Prerequisites:
    1. Classification data must exist at:
       data/capture24/classification/{window_size}s_{hz}hz/{label_scheme}/
    2. GOOGLE_API_KEY environment variable must be set for Gemini API access
"""

import argparse
import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from google import genai
from PIL import Image
from tqdm import tqdm

from ts_haystack.capture24.classification import (
    load_classification_dataset,
)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATA_DIR = os.environ.get("TS_HAYSTACK_DATA_DIR", os.path.join(_PROJECT_ROOT, "data"))


# ---------------------------
# Constants
# ---------------------------

CAPTURE24_COT_DATA_DIR = os.path.join(DATA_DIR, "capture24_cot")

# Dissimilar label mapping for Walmsley2020 scheme (4 classes)
# Each label maps to labels that represent very different activities
CAPTURE24_DISSIMILAR_MAPPING = {
    "sleep": ["light", "moderate-vigorous"],
    "sedentary": ["light", "moderate-vigorous"],
    "light": ["sleep", "sedentary", "moderate-vigorous"],
    "moderate-vigorous": ["sleep", "sedentary"],
}

# Default configuration - matches HAR CoT format: 2.56s @ 50Hz = 128 samples
DEFAULT_WINDOW_SIZE_S = 2.56
DEFAULT_EFFECTIVE_HZ = 50
DEFAULT_LABEL_SCHEME = "Walmsley2020"
DEFAULT_SAMPLES = {"train": 20, "val": 0, "test": 0} # for testing

# Prompt template for CoT generation (image-based)
COT_PROMPT_TEMPLATE = """You are shown a time-series plot of accelerometer data over a {window_duration:.2f} second window.
This data corresponds to one of two possible activities:
- {correct_activity}
- {dissimilar_activity}

Your task is to classify the activity based on analysis of the data.

Instructions:
- Begin by analyzing the time series without assuming a specific label.
- Think step-by-step about what the observed patterns suggest regarding movement intensity and behavior.
- Write your rationale as a single, natural paragraph - do not use bullet points, numbered steps, or section headings.
- Do not refer back to the plot or to the act of visual analysis in your rationale; the plot is only for reference but you should reason about the time-series data.
- Do **not** assume any answer at the beginning - analyze as if you do not yet know which class is correct.
- Do **not** mention either class label until the final sentence.
- Make sure that your last word is the answer. You MUST end your response with "Answer: {correct_activity}" in the answer field, not in the rationale.
"""


def create_timeseries_plot(
    x_data: np.ndarray,
    y_data: np.ndarray,
    z_data: np.ndarray,
    effective_hz: int,
    window_duration: float,
    figsize: Tuple[int, int] = (10, 8),
    dpi: int = 100,
) -> Image.Image:
    """
    Create a 3x1 subplot figure showing X, Y, Z accelerometer axes.

    Args:
        x_data: X-axis accelerometer readings
        y_data: Y-axis accelerometer readings
        z_data: Z-axis accelerometer readings
        effective_hz: Sampling frequency in Hz
        window_duration: Duration of the window in seconds
        figsize: Figure size as (width, height) tuple
        dpi: Resolution for the output image

    Returns:
        PIL Image containing the plot
    """
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)

    # Create time axis
    n_samples = len(x_data)
    time = np.linspace(0, window_duration, n_samples)

    # Plot each axis
    data = [('X-axis', x_data), ('Y-axis', y_data), ('Z-axis', z_data)]
    for ax, (label, values) in zip(axes, data):
        sns.lineplot(x=time, y=values, ax=ax, linewidth=0.8)
        ax.set_ylabel(f'{label} (g)')
        ax.set_title(f'{label} Accelerometer Data')

    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout()

    # Convert to PIL Image
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf).copy()  # Copy to allow buffer to be closed
    buf.close()
    plt.close(fig)

    return img


# ---------------------------
# Data Classes
# ---------------------------

@dataclass
class GenerationConfig:
    """Configuration for CoT generation."""
    window_size_s: float = DEFAULT_WINDOW_SIZE_S
    effective_hz: int = DEFAULT_EFFECTIVE_HZ
    label_scheme: str = DEFAULT_LABEL_SCHEME
    train_samples: int = DEFAULT_SAMPLES["train"]
    val_samples: int = DEFAULT_SAMPLES["val"]
    test_samples: int = DEFAULT_SAMPLES["test"]
    max_retries: int = 5
    base_retry_delay: float = 1.0
    max_retry_delay: float = 60.0
    output_dir: str = CAPTURE24_COT_DATA_DIR
    seed: int = 42
    max_workers: int = 4
    # Plot configuration
    plot_dpi: int = 100
    plot_figsize: Tuple[int, int] = (10, 8)


@dataclass
class GenerationStats:
    """Statistics for tracking generation progress."""
    total_processed: int = 0
    successful: int = 0
    failed: int = 0
    api_errors: int = 0
    class_counts: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "total_processed": self.total_processed,
            "successful": self.successful,
            "failed": self.failed,
            "api_errors": self.api_errors,
            "class_counts": self.class_counts,
        }


# ---------------------------
# API Client
# ---------------------------

# JSON schema for structured CoT response - I make the LLM answer the answer instead of appending it based on vibe that it will "think" better if it has to give the answer.
COT_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "rationale": {
            "type": "string",
            "description": "Step-by-step reasoning analyzing the accelerometer data patterns without mentioning the activity labels until the end."
        },
        "answer": {
            "type": "string",
            "description": "The classified activity label (one of the two options provided)."
        }
    },
    "required": ["rationale", "answer"]
}


class GeminiClient:
    """Client for Google Gemini API calls with exponential backoff retry."""

    # HTTP status codes that should trigger a retry
    RETRYABLE_STATUS_CODES = (429, 500, 502, 503, 504)

    def __init__(self, config: GenerationConfig, model: str = "gemini-2.5-flash-lite"):
        self.model = model
        self.config = config
        self._client = None

    def _ensure_client(self):
        """Lazily initialize the Gemini client."""
        if self._client is None:
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                raise ValueError(
                    "GEMINI_API_KEY environment variable is not set. "
                    "Please set it to use the Gemini API."
                )
            self._client = genai.Client(api_key=api_key)
            print(f"Initialized Gemini client with model: {self.model}")

    def _is_retryable(self, exception: Exception) -> bool:
        """Check if an exception should trigger a retry."""
        error_str = str(exception).lower()
        # Check for rate limit or server errors in the exception message
        if any(str(code) in error_str for code in self.RETRYABLE_STATUS_CODES):
            return True
        # Also retry on connection/timeout errors
        if any(keyword in error_str for keyword in ["timeout", "connection", "temporarily"]):
            return True
        return False

    def _get_retry_delay(self, attempt: int) -> float:
        """Calculate retry delay with exponential backoff and jitter."""
        base_delay = self.config.base_retry_delay * (2 ** attempt)
        # Add jitter: random value between 0 and 1 second
        jitter = random.uniform(0, 1)
        delay = min(base_delay + jitter, self.config.max_retry_delay)
        return delay

    def generate(
        self,
        prompt: str,
        image: Optional[Image.Image] = None,
        temperature: float = 0.3,
    ) -> Optional[dict]:
        """
        Generate structured response using Gemini API with retry logic.

        Args:
            prompt: The prompt to send to the model
            image: Optional PIL Image to include with the prompt
            temperature: Sampling temperature (0.0-1.0)

        Returns:
            Dict with 'rationale' and 'answer' keys, or None if all retries failed
        """
        self._ensure_client()

        # Build contents: image first (if provided), then prompt
        if image is not None:
            contents = [image, prompt]
        else:
            contents = prompt

        last_exception = None
        for attempt in range(self.config.max_retries):
            try:
                response = self._client.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config={
                        "temperature": temperature,  # 0.3 default as per paper
                        "response_mime_type": "application/json",
                        "response_json_schema": COT_RESPONSE_SCHEMA,
                    }
                )

                if response.text:
                    return json.loads(response.text)
                return None

            except Exception as e:
                last_exception = e
                if self._is_retryable(e) and attempt < self.config.max_retries - 1:
                    delay = self._get_retry_delay(attempt)
                    print(f"  [RETRY] Attempt {attempt + 1}/{self.config.max_retries} failed: {e}. Retrying in {delay:.1f}s...")
                    time.sleep(delay)
                else:
                    # Non-retryable error or last attempt
                    break

        print(f"  [ERROR] Gemini API failed after {self.config.max_retries} attempts: {last_exception}")
        return None


# ---------------------------
# Generator Class
# ---------------------------

class Capture24CoTGenerator:
    """
    Generator for Capture24 Chain-of-Thought dataset.

    This class handles:
    1. Stratified sampling from classification parquet files
    2. Prompt creation with time series data
    3. LLM API calls for rationale generation
    4. Validation and incremental CSV saving
    """

    def __init__(self, config: GenerationConfig):
        self.config = config
        self.api_client = GeminiClient(config)
        self.labels = list(CAPTURE24_DISSIMILAR_MAPPING.keys())
        self._csv_lock = threading.Lock()

        # Example image saving (thread-safe)
        self._example_image_saved = False
        self._example_image_lock = threading.Lock()

        # Set random seed for reproducibility
        random.seed(config.seed)

    def sample_windows(
        self,
        split: str,
        n_samples: int,
        stratified: bool = True
    ) -> pd.DataFrame:
        """
        Sample windows from classification dataset.

        Args:
            split: Dataset split ("train", "val", or "test")
            n_samples: Total number of samples to select
            stratified: If True, sample equal numbers per class

        Returns:
            DataFrame with sampled windows
        """
        # Load classification data
        df = load_classification_dataset(
            window_size_s=self.config.window_size_s,
            effective_hz=self.config.effective_hz,
            label_scheme=self.config.label_scheme,
            split=split
        )

        # Convert to pandas for easier sampling
        pdf = df.to_pandas()

        if stratified:
            # Calculate samples per class
            n_classes = len(self.labels)
            samples_per_class = n_samples // n_classes
            remainder = n_samples % n_classes

            sampled_dfs = []
            for i, label in enumerate(self.labels):
                label_df = pdf[pdf["label"] == label]

                # Add remainder samples to first few classes
                n_to_sample = samples_per_class + (1 if i < remainder else 0)
                n_to_sample = min(n_to_sample, len(label_df))

                if n_to_sample > 0:
                    sampled_dfs.append(label_df.sample(n=n_to_sample, random_state=self.config.seed + i))

            if sampled_dfs:
                result = pd.concat(sampled_dfs, ignore_index=True)
                # Shuffle the combined result
                result = result.sample(frac=1, random_state=self.config.seed).reset_index(drop=True)
                return result
            else:
                return pd.DataFrame()
        else:
            n_to_sample = min(n_samples, len(pdf))
            return pdf.sample(n=n_to_sample, random_state=self.config.seed).reset_index(drop=True)

    def get_dissimilar_label(self, correct_label: str) -> str:
        """
        Select a random dissimilar label for binary classification.

        Args:
            correct_label: The true label for the sample

        Returns:
            A dissimilar label from the mapping
        """
        dissimilar_options = CAPTURE24_DISSIMILAR_MAPPING.get(correct_label, [])
        if not dissimilar_options:
            # Fallback: pick any other label
            other_labels = [l for l in self.labels if l != correct_label]
            return random.choice(other_labels)
        return random.choice(dissimilar_options)

    def create_prompt(self, row: dict, dissimilar_label: str) -> Tuple[str, Image.Image]:
        """
        Build the prompt and time series plot image for binary classification.

        Args:
            row: Dictionary containing window data (x, y, z, label)
            dissimilar_label: The dissimilar label for binary classification

        Returns:
            Tuple of (formatted prompt string, PIL Image of the time series plot)
        """
        x_data = row["x"]
        y_data = row["y"]
        z_data = row["z"]
        correct_label = row["label"]

        x_arr = np.array(x_data)
        y_arr = np.array(y_data)
        z_arr = np.array(z_data)

        # Calculate window duration
        window_duration = len(x_data) / self.config.effective_hz

        # Create the time series plot image
        image = create_timeseries_plot(
            x_data=x_arr,
            y_data=y_arr,
            z_data=z_arr,
            effective_hz=self.config.effective_hz,
            window_duration=window_duration,
            figsize=self.config.plot_figsize,
            dpi=self.config.plot_dpi,
        )

        prompt = COT_PROMPT_TEMPLATE.format(
            window_duration=window_duration,
            correct_activity=correct_label,
            dissimilar_activity=dissimilar_label,
        )

        return prompt, image

    def _init_csv(self, split: str) -> Path:
        """Initialize CSV file with headers for a split."""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / f"capture24_cot_{split}.csv"
        columns = ["x_axis", "y_axis", "z_axis", "label", "prompt", "rationale"]

        # Write header
        df = pd.DataFrame(columns=columns)
        df.to_csv(output_path, index=False)

        return output_path

    def _append_to_csv(self, output_path: Path, row_data: dict):
        """Append a single sample to CSV file (thread-safe)."""
        row = {
            "x_axis": json.dumps(list(row_data["x_axis"]) if hasattr(row_data["x_axis"], "__iter__") else row_data["x_axis"]),
            "y_axis": json.dumps(list(row_data["y_axis"]) if hasattr(row_data["y_axis"], "__iter__") else row_data["y_axis"]),
            "z_axis": json.dumps(list(row_data["z_axis"]) if hasattr(row_data["z_axis"], "__iter__") else row_data["z_axis"]),
            "label": row_data["label"],
            "prompt": row_data["prompt"],
            "rationale": row_data["rationale"],
        }

        df = pd.DataFrame([row])
        with self._csv_lock:
            df.to_csv(output_path, mode='a', header=False, index=False)

    def _get_existing_count(self, output_path: Path) -> int:
        """Count existing samples in CSV for resume support."""
        if not output_path.exists():
            return 0
        try:
            df = pd.read_csv(output_path)
            return len(df)
        except Exception:
            return 0

    def _process_sample(self, row: dict, idx: int) -> Optional[dict]:
        """Process a single sample and return result dict or None if failed."""
        dissimilar_label = self.get_dissimilar_label(row["label"])
        prompt, image = self.create_prompt(row, dissimilar_label)

        # Save first example (plot + data) as reference (thread-safe)
        with self._example_image_lock:
            if not self._example_image_saved:
                output_dir = Path(self.config.output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)

                # Save plot image
                plot_path = output_dir / "example_plot.png"
                image.save(plot_path)
                print(f"  Saved example plot: {plot_path}")

                # Save time series data to text file
                data_path = output_dir / "example_data.txt"
                x_arr = np.array(row["x"])
                y_arr = np.array(row["y"])
                z_arr = np.array(row["z"])
                with open(data_path, "w") as f:
                    f.write(f"Label: {row['label']}\n")
                    f.write(f"Dissimilar label: {dissimilar_label}\n")
                    f.write(f"Window duration: {len(x_arr) / self.config.effective_hz:.2f}s\n")
                    f.write(f"Samples: {len(x_arr)}\n\n")
                    f.write(f"X-axis data:\n{', '.join(f'{v:.4f}' for v in x_arr)}\n\n")
                    f.write(f"Y-axis data:\n{', '.join(f'{v:.4f}' for v in y_arr)}\n\n")
                    f.write(f"Z-axis data:\n{', '.join(f'{v:.4f}' for v in z_arr)}\n")
                print(f"  Saved example data: {data_path}")

                self._example_image_saved = True

        response = self.api_client.generate(prompt, image=image)

        if response is None:
            return None

        rationale_text = response["rationale"]
        answer = response["answer"]
        full_rationale = f"{rationale_text} Answer: {answer}"

        return {
            "idx": idx,
            "x_axis": row["x"],
            "y_axis": row["y"],
            "z_axis": row["z"],
            "label": row["label"],
            "prompt": prompt,
            "rationale": full_rationale,
        }

    def generate_split(self, split: str, n_samples: int) -> GenerationStats:
        """
        Generate CoT rationales for a dataset split using parallel API calls.

        Args:
            split: Dataset split ("train", "val", or "test")
            n_samples: Number of samples to generate

        Returns:
            GenerationStats with counts of successful/failed samples
        """
        print(f"\n{'='*60}")
        print(f"Generating {split} split ({n_samples} samples) with {self.config.max_workers} workers")
        print(f"{'='*60}")

        stats = GenerationStats()

        # Initialize or resume from existing CSV
        output_path = Path(self.config.output_dir) / f"capture24_cot_{split}.csv"
        existing_count = self._get_existing_count(output_path)

        if existing_count > 0:
            print(f"Resuming from existing CSV with {existing_count} samples")
        else:
            output_path = self._init_csv(split)

        # Sample windows
        samples_df = self.sample_windows(split, n_samples)
        print(f"Sampled {len(samples_df)} windows for processing")

        # Skip already processed samples if resuming
        start_idx = existing_count
        stats.successful = existing_count

        samples_to_process = [
            (samples_df.iloc[idx].to_dict(), idx)
            for idx in range(start_idx, len(samples_df))
        ]

        if not samples_to_process:
            print(f"All {n_samples} samples already processed")
            return stats

        # Thread-safe stats update lock
        stats_lock = threading.Lock()

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = {
                executor.submit(self._process_sample, row, idx): (row, idx)
                for row, idx in samples_to_process
            }

            for future in tqdm(as_completed(futures), total=len(futures), desc=f"  {split}"):
                row, idx = futures[future]
                result = future.result()

                with stats_lock:
                    stats.total_processed += 1

                    if result is None:
                        stats.api_errors += 1
                        stats.failed += 1
                    else:
                        self._append_to_csv(output_path, result)
                        stats.successful += 1
                        stats.class_counts[row["label"]] = stats.class_counts.get(row["label"], 0) + 1

        print(f"Saved {split} dataset: {output_path} ({stats.successful} samples)")
        return stats

    def generate_all(self) -> Dict[str, GenerationStats]:
        """
        Generate CoT rationales for all splits.

        Returns:
            Dictionary mapping split names to generation stats
        """
        all_stats = {}

        splits_config = {
            "train": self.config.train_samples,
            "val": self.config.val_samples,
            "test": self.config.test_samples,
        }

        for split, n_samples in splits_config.items():
            if n_samples > 0:
                stats = self.generate_split(split, n_samples)
                all_stats[split] = stats

        # Save metadata
        self._save_metadata(all_stats)

        return all_stats

    def _save_metadata(self, all_stats: Dict[str, GenerationStats]):
        """Save generation metadata."""
        output_dir = Path(self.config.output_dir)
        metadata_path = output_dir / "metadata.json"

        metadata = {
            "window_size_s": self.config.window_size_s,
            "effective_hz": self.config.effective_hz,
            "samples_per_window": int(self.config.window_size_s * self.config.effective_hz),
            "label_scheme": self.config.label_scheme,
            "labels": self.labels,
            "llm_model": self.api_client.model,
            "generation_date": datetime.now().isoformat(),
            "dissimilar_mapping": CAPTURE24_DISSIMILAR_MAPPING,
            "samples": {
                split: stats.successful
                for split, stats in all_stats.items()
            },
            "class_distribution": {
                split: stats.class_counts
                for split, stats in all_stats.items()
            },
            "generation_stats": {
                split: stats.to_dict()
                for split, stats in all_stats.items()
            },
            "config": {
                "max_retries": self.config.max_retries,
                "seed": self.config.seed,
            }
        }

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"\nMetadata saved: {metadata_path}")


# ---------------------------
# CLI
# ---------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate Capture24 Chain-of-Thought dataset"
    )
    parser.add_argument(
        "--window-size-s", "-w",
        type=float,
        default=DEFAULT_WINDOW_SIZE_S,
        help=f"Window size in seconds, e.g., 2.56 for HAR CoT compatibility (default: {DEFAULT_WINDOW_SIZE_S})"
    )
    parser.add_argument(
        "--effective-hz", "-e",
        type=int,
        default=DEFAULT_EFFECTIVE_HZ,
        help=f"Effective sampling frequency in Hz (default: {DEFAULT_EFFECTIVE_HZ})"
    )
    parser.add_argument(
        "--label-scheme", "-l",
        type=str,
        default=DEFAULT_LABEL_SCHEME,
        help=f"Label scheme (default: {DEFAULT_LABEL_SCHEME})"
    )
    parser.add_argument(
        "--train-samples",
        type=int,
        default=DEFAULT_SAMPLES["train"],
        help=f"Number of training samples (default: {DEFAULT_SAMPLES['train']})"
    )
    parser.add_argument(
        "--val-samples",
        type=int,
        default=DEFAULT_SAMPLES["val"],
        help=f"Number of validation samples (default: {DEFAULT_SAMPLES['val']})"
    )
    parser.add_argument(
        "--test-samples",
        type=int,
        default=DEFAULT_SAMPLES["test"],
        help=f"Number of test samples (default: {DEFAULT_SAMPLES['test']})"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Number of parallel workers for API calls (default: 4)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=CAPTURE24_COT_DATA_DIR,
        help=f"Output directory (default: {CAPTURE24_COT_DATA_DIR})"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val", "test", "all"],
        default="all",
        help="Which split to generate (default: all)"
    )

    args = parser.parse_args()

    # Create config
    config = GenerationConfig(
        window_size_s=args.window_size_s,
        effective_hz=args.effective_hz,
        label_scheme=args.label_scheme,
        train_samples=args.train_samples,
        val_samples=args.val_samples,
        test_samples=args.test_samples,
        output_dir=args.output_dir,
        seed=args.seed,
        max_workers=args.max_workers,
    )

    print("=" * 60)
    print("Capture24 CoT Dataset Generator")
    print("=" * 60)
    print(f"Window size: {config.window_size_s}s @ {config.effective_hz}Hz")
    print(f"Label scheme: {config.label_scheme}")
    print(f"Output directory: {config.output_dir}")
    print(f"Samples: train={config.train_samples}, val={config.val_samples}, test={config.test_samples}")
    print(f"Max workers: {config.max_workers}")
    print()

    # Create generator
    generator = Capture24CoTGenerator(config)

    # Generate
    if args.split == "all":
        all_stats = generator.generate_all()
    else:
        samples_map = {
            "train": config.train_samples,
            "val": config.val_samples,
            "test": config.test_samples,
        }
        stats = generator.generate_split(args.split, samples_map[args.split])
        all_stats = {args.split: stats}

    # Print summary
    print("\n" + "=" * 60)
    print("Generation Summary")
    print("=" * 60)
    for split, stats in all_stats.items():
        print(f"\n{split}:")
        print(f"  Total processed: {stats.total_processed}")
        print(f"  Successful: {stats.successful}")
        print(f"  Failed: {stats.failed}")
        print(f"  API errors: {stats.api_errors}")
        print(f"  Class distribution: {stats.class_counts}")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
