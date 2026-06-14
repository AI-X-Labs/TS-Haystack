# SPDX-License-Identifier: CC-BY-NC-4.0

"""
Data loader for PhysioNet 2018 Challenge Sleep PSG data.

Loads 13-channel polysomnography signals and parses WFDB annotations
for sleep stages and arousal events.
"""

import os
from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
import scipy.io
import scipy.signal
import wfdb

SLEEP_PSG_DATA_DIR = Path("data/sleep_psg")

CHANNEL_NAMES = [
    "F3-M2", "F4-M1", "C3-M2", "C4-M1", "O1-M2", "O2-M1",
    "E1-M2", "Chin1-Chin2", "ABD", "CHEST", "AIRFLOW", "SaO2", "ECG",
]

SOURCE_HZ = 200
# Single source of truth: every consumer of load_window() sees this rate.
EFFECTIVE_HZ = 100
_DECIMATE_Q = SOURCE_HZ // EFFECTIVE_HZ  # 2

# Pre-decimated (100 Hz) data directory. Created by predecimate_sleep_psg.py.
# When present, load_window() reads directly at EFFECTIVE_HZ and skips runtime
# decimation — eliminating the ~1.5 s/sample IIR filter cost.
_PREDECIMATED_DIR = SLEEP_PSG_DATA_DIR / "training_100hz"

# Sleep stage labels in the WFDB annotations
SLEEP_STAGE_LABELS = {"W", "N1", "N2", "N3", "R"}

# Mapping from WFDB annotation aux_note to standardized arousal event names
AROUSAL_EVENT_MAP = {
    "arousal_rera": "rera",
    "arousal_spontaneous": "spontaneous",
    "arousal_bruxism": "bruxism",
    "arousal_plm": "periodic_leg_movement",
    "arousal_snore": "snore",
    "arousal_noise": "noise",
    "resp_centralapnea": "central_apnea",
    "resp_obstructiveapnea": "obstructive_apnea",
    "resp_mixedapnea": "mixed_apnea",
    "resp_hypopnea": "hypopnea",
    "resp_hypoventilation": "hypoventilation",
    "resp_cheynestokesbreath": "cheyne_stokes",
    "resp_partialobstructive": "partial_obstruction",
}

# Standardized sleep stage name mapping
SLEEP_STAGE_MAP = {
    "W": "Wake",
    "N1": "N1",
    "N2": "N2",
    "N3": "N3",
    "R": "REM",
}


def get_data_dir() -> Path:
    """Get the Sleep PSG data directory."""
    return SLEEP_PSG_DATA_DIR


def get_training_dir() -> Path:
    """Get the training data directory."""
    return SLEEP_PSG_DATA_DIR / "training"


def get_subject_dir(subject_id: str) -> Path:
    """Get the directory for a specific subject."""
    return get_training_dir() / subject_id


def get_subject_ids() -> List[str]:
    """List available subject IDs from data/sleep_psg/training/."""
    training_dir = get_training_dir()
    if not training_dir.exists():
        return []
    return sorted([
        d.name for d in training_dir.iterdir()
        if d.is_dir() and (d / f"{d.name}.mat").exists()
    ])


def parse_header(subject_id: str) -> Dict:
    """
    Parse the .hea header file for a subject.

    Returns:
        Dict with keys: subject_id, n_channels, sampling_rate, n_samples, channels
    """
    hea_path = get_subject_dir(subject_id) / f"{subject_id}.hea"
    with open(hea_path) as f:
        lines = f.readlines()

    # First line: record_name n_signals sampling_rate n_samples
    parts = lines[0].strip().split()
    n_channels = int(parts[1])
    sampling_rate = int(parts[2])
    n_samples = int(parts[3])

    # Signal lines: extract channel names (last field)
    channels = []
    for line in lines[1 : n_channels + 1]:
        channel_name = line.strip().split()[-1]
        channels.append(channel_name)

    return {
        "subject_id": subject_id,
        "n_channels": n_channels,
        "sampling_rate": sampling_rate,
        "n_samples": n_samples,
        "channels": channels,
    }


def load_subject_signals(subject_id: str) -> np.ndarray:
    """
    Load all 13 PSG channels for a subject.

    Args:
        subject_id: e.g., "tr03-0005"

    Returns:
        np.ndarray of shape (N_samples, 13), dtype float32.
        Channels are in the order defined by CHANNEL_NAMES.
    """
    mat_path = get_subject_dir(subject_id) / f"{subject_id}.mat"
    data = scipy.io.loadmat(str(mat_path))
    # Shape: (13, N_samples), int16
    val = data["val"]
    # Transpose to (N_samples, 13) and convert to float32
    return val.T.astype(np.float32)


def _use_predecimated() -> bool:
    """Return True if the pre-decimated 100 Hz data directory exists."""
    return _PREDECIMATED_DIR.is_dir()


def load_subject_signals_mmap(subject_id: str) -> np.ndarray:
    """
    Load PSG channels as a read-only memory-mapped array.

    Prefers the pre-decimated 100 Hz directory (``training_100hz/``) when
    available; falls back to the original 200 Hz directory (``training/``).

    Args:
        subject_id: e.g., "tr03-0005"

    Returns:
        np.memmap of shape (N_samples, 13), dtype float32, read-only.
    """
    if _use_predecimated():
        npy_path = _PREDECIMATED_DIR / subject_id / f"{subject_id}.npy"
    else:
        npy_path = get_subject_dir(subject_id) / f"{subject_id}.npy"
    return np.load(str(npy_path), mmap_mode="r")


def load_window(
    subject_id: str,
    window_start_ms: int,
    window_end_ms: int,
    channels: Optional[Sequence[int]] = None,
) -> np.ndarray:
    """
    Memmap-slice a window from a subject's PSG, decimate to EFFECTIVE_HZ, and z-score per channel.

    Args:
        subject_id: e.g., "tr03-0005"
        window_start_ms: window start in milliseconds (relative to recording start)
        window_end_ms: window end in milliseconds (exclusive)
        channels: optional iterable of channel indices to keep (default: all 13)

    Returns:
        np.ndarray of shape (C, L) at EFFECTIVE_HZ Hz, dtype float32, per-channel z-scored.
    """
    if window_end_ms <= window_start_ms:
        raise ValueError(
            f"window_end_ms ({window_end_ms}) must be > window_start_ms ({window_start_ms})"
        )

    predecimated = _use_predecimated()
    hz = EFFECTIVE_HZ if predecimated else SOURCE_HZ

    mmap = load_subject_signals_mmap(subject_id)  # (N, 13)
    n_total = mmap.shape[0]

    start_sample = int(round(window_start_ms * hz / 1000))
    end_sample = int(round(window_end_ms * hz / 1000))
    start_sample = max(0, min(start_sample, n_total))
    end_sample = max(start_sample, min(end_sample, n_total))

    # Slice and materialize as a contiguous (C, L) float32 array
    chunk = np.ascontiguousarray(mmap[start_sample:end_sample]).T.astype(np.float32, copy=False)

    if channels is not None:
        chunk = chunk[list(channels)]

    # Decimate only when reading the original 200 Hz data
    if not predecimated:
        if chunk.shape[1] >= _DECIMATE_Q * 8:
            chunk = scipy.signal.decimate(
                chunk, q=_DECIMATE_Q, ftype="iir", zero_phase=True, axis=-1,
            ).astype(np.float32, copy=False)
        else:
            chunk = chunk[:, ::_DECIMATE_Q].astype(np.float32, copy=False)

    mean = chunk.mean(axis=-1, keepdims=True)
    std = chunk.std(axis=-1, keepdims=True)
    chunk = (chunk - mean) / (std + 1e-6)
    return chunk.astype(np.float32, copy=False)


def load_annotations(
    subject_id: str,
    label_class: Literal["sleep_stages", "arousals"] = "sleep_stages",
) -> List[Tuple[int, int, str]]:
    """
    Parse WFDB annotations for a subject.

    Sleep stage annotations are point annotations that mark the START of a stage.
    Each stage continues until the next stage annotation.

    Arousal/event annotations use bracket pairs: "(event_type" marks the start,
    "event_type)" marks the end.

    Args:
        subject_id: e.g., "tr03-0005"
        label_class: "sleep_stages" or "arousals"

    Returns:
        List of (start_sample, end_sample, label) tuples, sorted by start_sample.
    """
    record_path = str(get_subject_dir(subject_id) / subject_id)
    ann = wfdb.rdann(record_path, "arousal")

    header = parse_header(subject_id)
    total_samples = header["n_samples"]

    if label_class == "sleep_stages":
        return _parse_sleep_stages(ann, total_samples)
    elif label_class == "arousals":
        return _parse_arousal_events(ann)
    else:
        raise ValueError(f"Unknown label_class: {label_class}. Use 'sleep_stages' or 'arousals'.")


def _parse_sleep_stages(ann, total_samples: int) -> List[Tuple[int, int, str]]:
    """
    Parse sleep stage annotations into (start_sample, end_sample, label) tuples.

    Each stage annotation marks the START of that stage. The stage continues
    until the next stage annotation (or end of recording).
    """
    # Collect stage annotations in order
    stage_points = []
    for i in range(len(ann.sample)):
        note = ann.aux_note[i]
        if note in SLEEP_STAGE_LABELS:
            label = SLEEP_STAGE_MAP[note]
            stage_points.append((ann.sample[i], label))

    if not stage_points:
        return []

    # Convert point annotations to intervals
    intervals = []
    for i in range(len(stage_points)):
        start = stage_points[i][0]
        label = stage_points[i][1]
        if i + 1 < len(stage_points):
            end = stage_points[i + 1][0]
        else:
            end = total_samples
        intervals.append((int(start), int(end), label))

    return intervals


def _parse_arousal_events(ann) -> List[Tuple[int, int, str]]:
    """
    Parse arousal/respiratory event annotations into (start_sample, end_sample, label) tuples.

    Events use bracket pairs: "(event_type" marks start, "event_type)" marks end.
    """
    events = []
    # Track open events: raw_name -> start_sample
    open_events: Dict[str, int] = {}

    for i in range(len(ann.sample)):
        note = ann.aux_note[i]
        sample = int(ann.sample[i])

        if note.startswith("("):
            # Opening bracket — extract raw event name
            raw_name = note[1:]  # strip leading "("
            open_events[raw_name] = sample
        elif note.endswith(")"):
            # Closing bracket — match with opening
            raw_name = note[:-1]  # strip trailing ")"
            if raw_name in open_events:
                start = open_events.pop(raw_name)
                label = AROUSAL_EVENT_MAP.get(raw_name, raw_name)
                events.append((start, sample, label))

    # Sort by start sample
    events.sort(key=lambda x: x[0])
    return events


def load_arousal_vector(subject_id: str) -> Optional[np.ndarray]:
    """
    Load the binary arousal vector from the -arousal.mat file.

    Returns:
        np.ndarray with values +1 (arousal), 0 (non-arousal), -1 (unscored),
        or None if file not found.
    """
    mat_path = get_subject_dir(subject_id) / f"{subject_id}-arousal.mat"
    if not mat_path.exists():
        return None

    import h5py

    with h5py.File(str(mat_path), "r") as f:
        return np.ravel(f["data"]["arousals"]).astype(np.int8)
