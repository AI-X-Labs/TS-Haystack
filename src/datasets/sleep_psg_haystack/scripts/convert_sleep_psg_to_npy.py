#!/usr/bin/env python3
"""Convert Sleep PSG .mat files to .npy for memmap-based loading.

One-time conversion: loads each subject's .mat file (int16, shape 13 x N),
transposes to (N, 13), converts to float32, and saves as .npy.

Usage:
    python scripts/data/convert_sleep_psg_to_npy.py
    python scripts/data/convert_sleep_psg_to_npy.py --n-subjects 10  # test with few
"""

import argparse
from pathlib import Path

import numpy as np
import scipy.io
from tqdm import tqdm


DATA_DIR = Path("data/sleep_psg/training")


def convert_subject(subject_dir: Path) -> bool:
    """Convert a single subject's .mat to .npy. Returns True if converted."""
    subject_id = subject_dir.name
    mat_path = subject_dir / f"{subject_id}.mat"
    npy_path = subject_dir / f"{subject_id}.npy"

    if npy_path.exists() and npy_path.stat().st_size > 0:
        return False

    if not mat_path.exists():
        print(f"  [warn] {mat_path} not found, skipping")
        return False

    data = scipy.io.loadmat(str(mat_path))
    val = data["val"]  # (13, N_samples), int16
    signals = val.T.astype(np.float32)  # (N_samples, 13)
    np.save(str(npy_path), signals)
    return True


def main():
    parser = argparse.ArgumentParser(description="Convert .mat to .npy for memmap loading")
    parser.add_argument("--n-subjects", type=int, default=None, help="Limit subjects (default: all)")
    args = parser.parse_args()

    subject_dirs = sorted(d for d in DATA_DIR.iterdir() if d.is_dir())
    if args.n_subjects:
        subject_dirs = subject_dirs[:args.n_subjects]

    print(f"Converting {len(subject_dirs)} subjects...")
    converted = 0
    for d in tqdm(subject_dirs, desc="Converting"):
        if convert_subject(d):
            converted += 1

    print(f"\nDone. Converted {converted} subjects ({len(subject_dirs) - converted} already existed).")


if __name__ == "__main__":
    main()
