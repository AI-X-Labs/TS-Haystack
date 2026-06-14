#!/usr/bin/env python3
"""Download PhysioNet Challenge 2018 (You Snooze You Win) sleep PSG data.

Uses AWS S3 (no credentials needed - public bucket) for fast parallel downloads.

Usage:
    # Download first N training subjects (default 5)
    python scripts/data/download_sleep_psg.py --n-subjects 5

    # Download all training subjects
    python scripts/data/download_sleep_psg.py --all

Data is saved to data/sleep_psg/training/<subject_id>/
"""

import argparse
from pathlib import Path

import boto3
from botocore import UNSIGNED
from botocore.config import Config

S3_BUCKET = "physionet-open"
S3_PREFIX = "challenge-2018/1.0.0"
DEFAULT_DATA_DIR = Path("data/sleep_psg")

REQUIRED_EXTENSIONS = {".mat", ".hea", ".arousal"}


def get_s3_client():
    return boto3.client("s3", config=Config(signature_version=UNSIGNED))


def download_metadata(s3, data_dir: Path):
    files = ["RECORDS", "age-sex.csv"]
    for fname in files:
        dest = data_dir / fname
        if dest.exists() and dest.stat().st_size > 0:
            print(f"  [skip] {dest} already exists")
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        key = f"{S3_PREFIX}/{fname}"
        print(f"Downloading {fname}...")
        s3.download_file(S3_BUCKET, key, str(dest))


def get_training_subjects(data_dir: Path) -> list[str]:
    """Parse RECORDS to get training record paths like 'training/tr03-0005/tr03-0005'."""
    records_path = data_dir / "RECORDS"
    subjects = []
    with open(records_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("training/"):
                subjects.append(line)
    return sorted(subjects)


def download_subject(s3, record_path: str, data_dir: Path):
    """Download all files for a single subject from S3.

    record_path: e.g. 'training/tr03-0005/tr03-0005' (from RECORDS file)
    """
    dir_path = str(Path(record_path).parent)
    subject_id = Path(record_path).name
    dest_dir = data_dir / dir_path
    dest_dir.mkdir(parents=True, exist_ok=True)

    extensions = [".mat", ".hea", ".arousal", "-arousal.mat"]
    for ext in extensions:
        fname = f"{subject_id}{ext}"
        dest = dest_dir / fname
        if dest.exists() and dest.stat().st_size > 0:
            continue
        key = f"{S3_PREFIX}/{dir_path}/{fname}"
        try:
            s3.download_file(S3_BUCKET, key, str(dest))
        except Exception as e:
            print(f"  [warn] Failed to download {fname}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Download PhysioNet 2018 Challenge data from S3")
    parser.add_argument("--n-subjects", type=int, default=5, help="Number of training subjects to download (default: 5)")
    parser.add_argument("--all", action="store_true", help="Download all training subjects (~135 GB)")
    parser.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA_DIR), help="Output directory")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    s3 = get_s3_client()

    print("=== Downloading metadata ===")
    download_metadata(s3, data_dir)

    subjects = get_training_subjects(data_dir)
    print(f"\nFound {len(subjects)} training subjects")

    n = len(subjects) if args.all else min(args.n_subjects, len(subjects))
    print(f"\n=== Downloading {n} training subjects from S3 ===")
    for i, subj in enumerate(subjects[:n]):
        print(f"[{i+1}/{n}] {subj}")
        download_subject(s3, subj, data_dir)

    print(f"\n=== Done! Data saved to {data_dir} ===")
    print(f"Downloaded {n}/{len(subjects)} subjects")


if __name__ == "__main__":
    main()
