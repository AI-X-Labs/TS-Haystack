#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors
# SPDX-License-Identifier: MIT

"""
Download TS-Haystack datasets from HuggingFace Hub.

The HuggingFace repos mirror the directory structure expected by the codebase,
so files are downloaded directly into the correct locations under data/.

HuggingFace datasets:
    - nicozumarraga/capture24-ts-haystack-cot  → data/capture24/ts_haystack/cot/  (CoT QA samples)
    - nicozumarraga/capture24-ts-haystack-cot  → data/capture24/ts_haystack/      (core artifacts)
    - nicozumarraga/capture24-classification    → data/capture24/classification/
    - nicozumarraga/capture24-raw              → data/capture24/

Usage:
    # Download everything
    python scripts/download_from_hf.py

    # Download only ts-haystack-cot QA samples
    python scripts/download_from_hf.py --dataset ts-haystack-cot

    # Download only core artifacts (timelines, bout index, transition matrix)
    python scripts/download_from_hf.py --dataset ts-haystack-core

    # Download only capture24-classification
    python scripts/download_from_hf.py --dataset capture24-classification

    # Download raw Capture24 sensor data
    python scripts/download_from_hf.py --dataset capture24-raw

    # Download specific context lengths / window sizes
    python scripts/download_from_hf.py --dataset ts-haystack-cot --subsets 100s 2_56s

    # Dry run (show what would be downloaded)
    python scripts/download_from_hf.py --dry-run
"""

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent  # project root
DATA_DIR = Path(os.environ.get("TS_HAYSTACK_DATA_DIR", str(BASE_DIR / "data")))

# Core artifact files/dirs in the HF repo (not CoT QA samples)
CORE_ARTIFACT_PREFIXES = {"timelines", "bout_index.parquet", "transition_matrix.json"}

HF_DATASETS = {
    "ts-haystack-cot": {
        "repo_id": "nicozumarraga/capture24-ts-haystack-cot",
        "local_root": DATA_DIR / "capture24" / "ts_haystack" / "cot",
        "description": "TS-Haystack benchmark with chain-of-thought rationales",
        "exclude_prefixes": CORE_ARTIFACT_PREFIXES,
    },
    "ts-haystack-core": {
        "repo_id": "nicozumarraga/capture24-ts-haystack-cot",
        "local_root": DATA_DIR / "capture24" / "ts_haystack",
        "description": "TS-Haystack core artifacts (timelines, bout index, transition matrix)",
        "only_prefixes": CORE_ARTIFACT_PREFIXES,
    },
    "capture24-classification": {
        "repo_id": "nicozumarraga/capture24-classification",
        "local_root": DATA_DIR / "capture24" / "classification",
        "description": "Capture24 activity classification dataset",
    },
    "capture24-raw": {
        "repo_id": "nicozumarraga/capture24-raw",
        "local_root": DATA_DIR / "capture24",
        "description": "Raw Capture24 sensor data (100Hz) + metadata",
    },
}

# Files to skip during download
SKIP_FILES = {".gitattributes", ".DS_Store"}


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def list_repo_files(repo_id: str) -> list[str]:
    """List all files in a HuggingFace dataset repo."""
    api = HfApi()
    return api.list_repo_files(repo_id, repo_type="dataset")


def _get_top_prefix(filepath: str) -> str:
    """Get top-level directory or filename from a repo path."""
    return filepath.split("/")[0] if "/" in filepath else filepath


def filter_files(
    files: list[str],
    subsets: list[str] | None = None,
    exclude_prefixes: set[str] | None = None,
    only_prefixes: set[str] | None = None,
) -> list[str]:
    """
    Filter file list to only include relevant files.

    Args:
        files: All files in the repo
        subsets: If provided, only include files under these top-level directories
        exclude_prefixes: Top-level prefixes to exclude
        only_prefixes: If set, only include files matching these top-level prefixes
    """
    filtered = []
    for f in files:
        basename = os.path.basename(f)

        # Skip system files
        if basename in SKIP_FILES:
            continue

        top_prefix = _get_top_prefix(f)

        # Exclude specific prefixes
        if exclude_prefixes and top_prefix in exclude_prefixes:
            continue

        # Only include specific prefixes
        if only_prefixes and top_prefix not in only_prefixes:
            continue

        # If subsets specified, only include files under those directories
        if subsets:
            if top_prefix not in subsets:
                continue

        filtered.append(f)
    return filtered


def download_dataset(
    dataset_key: str,
    subsets: list[str] | None = None,
    dry_run: bool = False,
    force: bool = False,
) -> None:
    """
    Download a dataset from HuggingFace Hub into the local data directory.

    Args:
        dataset_key: Key into HF_DATASETS dict
        subsets: Optional list of top-level subdirectories to download
        dry_run: If True, only print what would be downloaded
        force: If True, re-download even if files exist locally
    """
    config = HF_DATASETS[dataset_key]
    repo_id = config["repo_id"]
    local_root = config["local_root"]

    print(f"\n{'=' * 60}")
    print(f"Dataset: {config['description']}")
    print(f"HF repo: {repo_id}")
    print(f"Local:   {local_root}")
    print(f"{'=' * 60}")

    # List all files in the repo
    print(f"\nFetching file list from {repo_id}...")
    all_files = list_repo_files(repo_id)
    files = filter_files(
        all_files,
        subsets,
        exclude_prefixes=config.get("exclude_prefixes"),
        only_prefixes=config.get("only_prefixes"),
    )

    if not files:
        print("No files match the filter criteria.")
        return

    # Separate parquet data files from metadata
    parquet_files = [f for f in files if f.endswith(".parquet")]
    other_files = [f for f in files if not f.endswith(".parquet")]

    total_files = len(parquet_files) + len(other_files)
    print(f"Found {total_files} files ({len(parquet_files)} parquet, {len(other_files)} other)")

    if subsets:
        print(f"Filtering to subsets: {subsets}")

    # Show what would be downloaded
    if dry_run:
        print("\n[DRY RUN] Would download:")
        for f in files:
            local_path = local_root / f
            exists = local_path.exists()
            status = " (exists, would skip)" if exists and not force else ""
            print(f"  {f} → {local_path}{status}")
        return

    # Download files
    skipped = 0
    downloaded = 0
    errors = []

    for i, filepath in enumerate(files, 1):
        local_path = local_root / filepath

        # Skip if already exists (unless force)
        if local_path.exists() and not force:
            skipped += 1
            continue

        # Create parent directory
        local_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            print(f"  [{i}/{total_files}] Downloading {filepath}...")
            hf_hub_download(
                repo_id=repo_id,
                filename=filepath,
                repo_type="dataset",
                local_dir=str(local_root),
            )
            downloaded += 1
        except Exception as e:
            print(f"    ERROR: {e}")
            errors.append((filepath, str(e)))

    print(f"\nDone: {downloaded} downloaded, {skipped} skipped (already exist)")
    if errors:
        print(f"Errors ({len(errors)}):")
        for filepath, err in errors:
            print(f"  {filepath}: {err}")


def show_available_subsets(dataset_key: str) -> None:
    """Show available top-level subdirectories for a dataset."""
    config = HF_DATASETS[dataset_key]
    repo_id = config["repo_id"]

    print(f"\nAvailable subsets for {repo_id}:")
    all_files = list_repo_files(repo_id)

    # Get unique top-level directories
    top_dirs = set()
    for f in all_files:
        if "/" in f:
            top_dir = f.split("/")[0]
            if top_dir not in SKIP_FILES and not top_dir.startswith("."):
                top_dirs.add(top_dir)

    for d in sorted(top_dirs):
        n_files = sum(1 for f in all_files if f.startswith(d + "/") and f.endswith(".parquet"))
        print(f"  {d}/ ({n_files} parquet files)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Download TS-Haystack and Capture24 datasets from HuggingFace Hub.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download everything
  python scripts/download_from_hf.py

  # Download only TS-Haystack CoT QA samples
  python scripts/download_from_hf.py --dataset ts-haystack-cot

  # Download core artifacts (timelines, bout index) for dataset generation
  python scripts/download_from_hf.py --dataset ts-haystack-core

  # Download raw Capture24 sensor data
  python scripts/download_from_hf.py --dataset capture24-raw

  # Download specific context lengths
  python scripts/download_from_hf.py --dataset ts-haystack-cot --subsets 100s 2_56s

  # Show available subsets
  python scripts/download_from_hf.py --dataset ts-haystack-cot --list-subsets

  # Dry run
  python scripts/download_from_hf.py --dry-run

  # Force re-download
  python scripts/download_from_hf.py --force
        """,
    )
    parser.add_argument(
        "--dataset",
        choices=list(HF_DATASETS.keys()) + ["all"],
        default="all",
        help="Which dataset to download (default: all)",
    )
    parser.add_argument(
        "--subsets",
        nargs="+",
        default=None,
        help="Only download specific top-level subdirectories "
             "(e.g., '100s 2_56s' for ts-haystack-cot, '10s_50hz' for classification)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be downloaded without downloading",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download files even if they already exist locally",
    )
    parser.add_argument(
        "--list-subsets",
        action="store_true",
        help="List available subsets (top-level directories) and exit",
    )

    args = parser.parse_args()

    # Determine which datasets to process
    if args.dataset == "all":
        dataset_keys = list(HF_DATASETS.keys())
    else:
        dataset_keys = [args.dataset]

    # List subsets mode
    if args.list_subsets:
        for key in dataset_keys:
            show_available_subsets(key)
        return

    # Download
    for key in dataset_keys:
        download_dataset(
            dataset_key=key,
            subsets=args.subsets,
            dry_run=args.dry_run,
            force=args.force,
        )

    print("\n" + "=" * 60)
    print("Download complete!")
    print(f"Data directory: {DATA_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
