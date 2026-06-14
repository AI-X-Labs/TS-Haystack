#!/usr/bin/env python3
"""Pre-decimate Sleep PSG .npy files from 200 Hz to 100 Hz with IIR anti-alias.

Reads each subject's (N, 13) float32 .npy at 200 Hz, applies
scipy.signal.decimate(q=2, ftype='iir', zero_phase=True) per channel,
and saves the result as a new .npy at 100 Hz in a sibling directory.

After conversion, load_window() can skip runtime decimation entirely.

Usage:
    python -m src.datasets.sleep_psg_haystack.scripts.predecimate_sleep_psg
    python -m src.datasets.sleep_psg_haystack.scripts.predecimate_sleep_psg --n-subjects 10
    python -m src.datasets.sleep_psg_haystack.scripts.predecimate_sleep_psg --delete-originals
"""

import argparse
import multiprocessing as mp
import shutil
from pathlib import Path

import numpy as np
import scipy.signal
from tqdm import tqdm

SOURCE_DIR = Path("data/sleep_psg/training")
TARGET_DIR = Path("data/sleep_psg/training_100hz")
SOURCE_HZ = 200
TARGET_HZ = 100
DECIMATE_Q = SOURCE_HZ // TARGET_HZ  # 2


def _worker_args(subject_dir: Path) -> tuple:
    """Pack args for the multiprocessing worker (must be picklable)."""
    return (str(subject_dir), str(TARGET_DIR / subject_dir.name))


def _predecimate_worker(args: tuple) -> bool:
    """Worker function for multiprocessing — takes string paths."""
    subject_dir, target_subject_dir = Path(args[0]), Path(args[1])
    return predecimate_subject(subject_dir, target_subject_dir)


def predecimate_subject(subject_dir: Path, target_subject_dir: Path) -> bool:
    """Decimate a single subject. Returns True if converted."""
    subject_id = subject_dir.name
    src_npy = subject_dir / f"{subject_id}.npy"
    dst_npy = target_subject_dir / f"{subject_id}.npy"

    if dst_npy.exists() and dst_npy.stat().st_size > 0:
        return False

    if not src_npy.exists():
        return False

    # Load 200 Hz signal: (N_samples, 13) float32
    signal = np.load(str(src_npy), mmap_mode="r")

    # Decimate along time axis: transpose to (13, N) for scipy, then back
    signal_ct = np.ascontiguousarray(signal).T  # (13, N)
    decimated = scipy.signal.decimate(
        signal_ct, q=DECIMATE_Q, ftype="iir", zero_phase=True, axis=-1,
    ).astype(np.float32, copy=False)  # (13, N//2)
    decimated = decimated.T  # (N//2, 13) — same layout as source

    # Copy annotation files, save decimated signal
    target_subject_dir.mkdir(parents=True, exist_ok=True)
    np.save(str(dst_npy), decimated)

    # Symlink annotation files so loader can find them
    for suffix in (".hea", ".arousal", "-arousal.mat"):
        src_file = subject_dir / f"{subject_id}{suffix}"
        dst_file = target_subject_dir / f"{subject_id}{suffix}"
        if src_file.exists() and not dst_file.exists():
            dst_file.symlink_to(src_file.resolve())

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Pre-decimate PSG .npy files from 200 Hz to 100 Hz"
    )
    parser.add_argument(
        "--n-subjects", type=int, default=None,
        help="Limit to first N subjects (default: all)",
    )
    parser.add_argument(
        "--delete-originals", action="store_true",
        help="Delete 200 Hz .npy files after successful conversion to reclaim ~134 GB",
    )
    parser.add_argument(
        "--workers", type=int, default=8,
        help="Number of parallel workers (default: 8)",
    )
    args = parser.parse_args()

    if not SOURCE_DIR.exists():
        print(f"Source directory {SOURCE_DIR} not found.")
        return

    subject_dirs = sorted(
        d for d in SOURCE_DIR.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    )
    if args.n_subjects:
        subject_dirs = subject_dirs[: args.n_subjects]

    print(f"Pre-decimating {len(subject_dirs)} subjects: {SOURCE_HZ} Hz → {TARGET_HZ} Hz")
    print(f"  Source: {SOURCE_DIR}")
    print(f"  Target: {TARGET_DIR}")
    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    worker_args = [_worker_args(d) for d in subject_dirs]
    converted = 0
    with mp.Pool(processes=args.workers) as pool:
        for result in tqdm(
            pool.imap_unordered(_predecimate_worker, worker_args),
            total=len(worker_args),
            desc=f"Decimating ({args.workers} workers)",
        ):
            if result:
                converted += 1

    print(
        f"\nDone. Decimated {converted} subjects "
        f"({len(subject_dirs) - converted} already existed)."
    )

    if args.delete_originals and converted > 0:
        print(f"Deleting original 200 Hz .npy files from {SOURCE_DIR}...")
        deleted = 0
        for subject_dir in subject_dirs:
            npy_200 = subject_dir / f"{subject_dir.name}.npy"
            target_npy = TARGET_DIR / subject_dir.name / f"{subject_dir.name}.npy"
            if npy_200.exists() and target_npy.exists():
                npy_200.unlink()
                deleted += 1
        print(f"Deleted {deleted} files.")


if __name__ == "__main__":
    main()
