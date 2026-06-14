# SPDX-FileCopyrightText: 2026 Anonymous Authors
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Timeline builder for LTAF-Haystack.

Milestone 1 implementation:
- Discover LTAF records from raw WFDB files
- Parse rhythm transitions from atr annotations
- Build participant timelines as contiguous rhythm bouts
- Persist per-participant timelines to parquet
- Create deterministic patient-level split manifest
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl
from tqdm import tqdm

from src.datasets.ltaf_haystack.core.activity_regimes import BEAT_EVENT_TYPES
from src.datasets.ltaf_haystack.core.data_structures import (
    LTAFBeatEvent,
    LTAFBoutRecord,
    LTAFParticipantTimeline,
)


LTAF_RAW_DIR = Path("data") / "ltafdb" / "raw"
LTAF_HAYSTACK_DIR = Path("data") / "ltafdb" / "ltaf_haystack"
TIMELINES_DIR = LTAF_HAYSTACK_DIR / "timelines"
BEAT_TIMELINES_DIR = LTAF_HAYSTACK_DIR / "beat_timelines"
SPLIT_MANIFEST_PATH = LTAF_HAYSTACK_DIR / "split_manifest.json"

# Rhythm aux-note → canonical regime (matches activity_regimes.py).
_RHYTHM_MAP = {
    "N": "NSR",
    "NSR": "NSR",
    "AF": "AFIB",
    "AFIB": "AFIB",
    "AFL": "AFL",
    "SBR": "SBR",
    "AB": "AB",
    "B": "B",
    "T": "T",
    "SVTA": "SVTA",
    "VT": "VT",
    "IVR": "IVR",
}

_BEAT_SYMBOL_SET = set(BEAT_EVENT_TYPES)


def get_ltaf_raw_dir() -> Path:
    return LTAF_RAW_DIR


def get_ltaf_timelines_dir() -> Path:
    return TIMELINES_DIR


def get_ltaf_beat_timelines_dir() -> Path:
    return BEAT_TIMELINES_DIR


def get_ltaf_timeline_path(participant_id: str) -> Path:
    return get_ltaf_timelines_dir() / f"{participant_id}.parquet"


def get_ltaf_beat_timeline_path(participant_id: str) -> Path:
    return get_ltaf_beat_timelines_dir() / f"{participant_id}.parquet"


def get_ltaf_split_manifest_path() -> Path:
    return SPLIT_MANIFEST_PATH


def _participant_id_from_record_id(record_id: str) -> str:
    return record_id.replace("/", "_")


def _parse_rhythm_aux_note(aux_note: object) -> str | None:
    if aux_note is None:
        return None

    note = aux_note.decode("utf-8", errors="ignore") if isinstance(aux_note, bytes) else str(aux_note)
    note = note.strip()
    if not note:
        return None

    if note.startswith("("):
        note = note[1:]
    note = note.split()[0].replace(")", "").strip().upper()
    if not note:
        return None

    return _RHYTHM_MAP.get(note, note)


def _extract_beat_events(annotation, source_hz: int) -> list[LTAFBeatEvent]:
    """Extract per-beat events (N/A/V/Q) from a WFDB annotation object.

    Skips rhythm-change markers (``+``) and signal-quality markers (``"``) so
    only real beat annotations survive.
    """
    samples = list(getattr(annotation, "sample", []))
    symbols = list(getattr(annotation, "symbol", []))

    events: list[LTAFBeatEvent] = []
    for i, sample in enumerate(samples):
        symbol = symbols[i] if i < len(symbols) else ""
        if symbol not in _BEAT_SYMBOL_SET:
            continue
        smp = int(sample)
        events.append(
            LTAFBeatEvent(
                sample=smp,
                time_ms=int(round(smp * 1000.0 / max(source_hz, 1))),
                symbol=symbol,
            )
        )
    return events


def _extract_rhythm_changes(annotation) -> list[tuple[int, str]]:
    samples = list(getattr(annotation, "sample", []))
    symbols = list(getattr(annotation, "symbol", []))
    aux_notes = list(getattr(annotation, "aux_note", []))

    changes: list[tuple[int, str]] = []
    for i, sample in enumerate(samples):
        symbol = symbols[i] if i < len(symbols) else ""
        if symbol != "+":
            continue

        aux = aux_notes[i] if i < len(aux_notes) else None
        rhythm = _parse_rhythm_aux_note(aux)
        if rhythm is None:
            continue

        changes.append((int(sample), rhythm))

    if not changes:
        return []

    changes.sort(key=lambda x: x[0])
    deduped: list[tuple[int, str]] = []
    for sample, rhythm in changes:
        if deduped and deduped[-1][0] == sample:
            deduped[-1] = (sample, rhythm)
        else:
            deduped.append((sample, rhythm))
    return deduped


def _merge_adjacent_bouts(bouts: list[LTAFBoutRecord]) -> list[LTAFBoutRecord]:
    if not bouts:
        return []

    merged = [bouts[0]]
    for bout in bouts[1:]:
        last = merged[-1]
        if bout.activity == last.activity and bout.start_sample <= last.end_sample:
            new_end = max(last.end_sample, bout.end_sample)
            merged[-1] = LTAFBoutRecord(
                start_sample=last.start_sample,
                end_sample=new_end,
                activity=last.activity,
                duration_samples=new_end - last.start_sample,
            )
        else:
            merged.append(bout)
    return merged


def _build_bouts_from_changes(
    signal_length: int,
    changes: list[tuple[int, str]],
    min_bout_duration_samples: int,
) -> list[LTAFBoutRecord]:
    if signal_length <= 0:
        return []

    default_rhythm = "NSR"
    cursor = 0
    current = default_rhythm
    idx_start = 0

    if changes and changes[0][0] <= 0:
        current = changes[0][1]
        idx_start = 1

    bouts: list[LTAFBoutRecord] = []
    for boundary_raw, next_rhythm in changes[idx_start:]:
        boundary = max(0, min(int(boundary_raw), int(signal_length)))
        if boundary > cursor:
            duration = boundary - cursor
            if duration >= min_bout_duration_samples:
                bouts.append(
                    LTAFBoutRecord(
                        start_sample=cursor,
                        end_sample=boundary,
                        activity=current,
                        duration_samples=duration,
                    )
                )

        cursor = boundary
        current = next_rhythm

    if signal_length > cursor:
        duration = signal_length - cursor
        if duration >= min_bout_duration_samples:
            bouts.append(
                LTAFBoutRecord(
                    start_sample=cursor,
                    end_sample=signal_length,
                    activity=current,
                    duration_samples=duration,
                )
            )

    if not bouts and signal_length >= min_bout_duration_samples:
        bouts = [
            LTAFBoutRecord(
                start_sample=0,
                end_sample=signal_length,
                activity=default_rhythm,
                duration_samples=signal_length,
            )
        ]

    return _merge_adjacent_bouts(bouts)


class LTAFTimelineBuilder:
    """Build participant timelines from LTAF WFDB records and atr annotations."""

    def __init__(
        self,
        source_hz: int = 128,
        min_bout_duration_samples: int = 16,
        raw_data_dir: Path | None = None,
        output_root: Path | None = None,
    ):
        self.source_hz = int(source_hz)
        self.min_bout_duration_samples = int(min_bout_duration_samples)
        self.raw_data_dir = Path(raw_data_dir) if raw_data_dir else get_ltaf_raw_dir()
        self.output_root = Path(output_root) if output_root else LTAF_HAYSTACK_DIR

    def get_timelines_dir(self) -> Path:
        return self.output_root / "timelines"

    def get_timeline_path(self, participant_id: str) -> Path:
        return self.get_timelines_dir() / f"{participant_id}.parquet"

    def get_beat_timelines_dir(self) -> Path:
        return self.output_root / "beat_timelines"

    def get_beat_timeline_path(self, participant_id: str) -> Path:
        return self.get_beat_timelines_dir() / f"{participant_id}.parquet"

    def get_split_manifest_path(self) -> Path:
        return self.output_root / "split_manifest.json"

    def discover_available_records(self) -> list[str]:
        if not self.raw_data_dir.exists():
            return []

        records: set[str] = set()
        for header_file in self.raw_data_dir.rglob("*.hea"):
            record_id = header_file.relative_to(self.raw_data_dir).with_suffix("").as_posix()
            if record_id:
                records.add(record_id)

        return sorted(records)

    def _read_record_and_annotation(self, record_id: str) -> tuple[int, int, "wfdb.Annotation"]:
        import wfdb

        base_path = self.raw_data_dir / record_id
        header = wfdb.rdheader(str(base_path))
        signal_length = int(getattr(header, "sig_len", 0))
        source_hz = int(round(float(getattr(header, "fs", self.source_hz))))

        try:
            annotation = wfdb.rdann(str(base_path), "atr")
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Missing atr annotation for record '{record_id}' at base path {base_path}"
            ) from exc
        return signal_length, source_hz, annotation

    def build_participant_timeline(self, participant_id: str, record_id: str) -> LTAFParticipantTimeline:
        signal_length, source_hz, annotation = self._read_record_and_annotation(record_id)
        changes = _extract_rhythm_changes(annotation)
        bouts = _build_bouts_from_changes(
            signal_length=signal_length,
            changes=changes,
            min_bout_duration_samples=self.min_bout_duration_samples,
        )

        return LTAFParticipantTimeline(
            participant_id=participant_id,
            record_id=record_id,
            source_hz=source_hz,
            timeline=bouts,
        )

    def build_participant_beat_timeline(
        self, participant_id: str, record_id: str
    ) -> list[LTAFBeatEvent]:
        _, source_hz, annotation = self._read_record_and_annotation(record_id)
        return _extract_beat_events(annotation, source_hz=source_hz)

    def build_participant_artifacts(
        self, participant_id: str, record_id: str
    ) -> tuple[LTAFParticipantTimeline, list[LTAFBeatEvent]]:
        """Single-annotation pass producing both rhythm + beat timelines."""
        signal_length, source_hz, annotation = self._read_record_and_annotation(record_id)
        changes = _extract_rhythm_changes(annotation)
        bouts = _build_bouts_from_changes(
            signal_length=signal_length,
            changes=changes,
            min_bout_duration_samples=self.min_bout_duration_samples,
        )
        timeline = LTAFParticipantTimeline(
            participant_id=participant_id,
            record_id=record_id,
            source_hz=source_hz,
            timeline=bouts,
        )
        beats = _extract_beat_events(annotation, source_hz=source_hz)
        return timeline, beats

    def _save_timeline(self, timeline: LTAFParticipantTimeline) -> None:
        out_path = self.get_timeline_path(timeline.participant_id)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        rows = [
            {
                "participant_id": timeline.participant_id,
                "record_id": timeline.record_id,
                "source_hz": timeline.source_hz,
                "start_sample": bout.start_sample,
                "end_sample": bout.end_sample,
                "activity": bout.activity,
                "duration_samples": bout.duration_samples,
            }
            for bout in timeline.timeline
        ]

        if rows:
            df = pl.DataFrame(rows)
        else:
            df = pl.DataFrame(
                {
                    "participant_id": [timeline.participant_id],
                    "record_id": [timeline.record_id],
                    "source_hz": [timeline.source_hz],
                    "start_sample": [0],
                    "end_sample": [0],
                    "activity": ["NSR"],
                    "duration_samples": [0],
                }
            )
        df.write_parquet(out_path, compression="snappy")

    def _save_beat_timeline(
        self, participant_id: str, record_id: str, beats: list[LTAFBeatEvent]
    ) -> None:
        out_path = self.get_beat_timeline_path(participant_id)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if beats:
            df = pl.DataFrame(
                {
                    "record_id": [record_id] * len(beats),
                    "sample": [b.sample for b in beats],
                    "time_ms": [b.time_ms for b in beats],
                    "symbol": [b.symbol for b in beats],
                }
            )
        else:
            df = pl.DataFrame(
                {
                    "record_id": pl.Series([], dtype=pl.Utf8),
                    "sample": pl.Series([], dtype=pl.Int64),
                    "time_ms": pl.Series([], dtype=pl.Int64),
                    "symbol": pl.Series([], dtype=pl.Utf8),
                }
            )
        df.write_parquet(out_path, compression="snappy")

    @staticmethod
    def load_beat_timeline(path: Path) -> list[LTAFBeatEvent]:
        if not path.exists():
            raise FileNotFoundError(f"Beat timeline not found at {path}")
        df = pl.read_parquet(path)
        return [
            LTAFBeatEvent(
                sample=int(row["sample"]),
                time_ms=int(row["time_ms"]),
                symbol=str(row["symbol"]),
            )
            for row in df.iter_rows(named=True)
        ]

    @staticmethod
    def _load_timeline(path: Path) -> LTAFParticipantTimeline:
        df = pl.read_parquet(path)
        if len(df) == 0:
            return LTAFParticipantTimeline(
                participant_id=path.stem,
                record_id=path.stem,
                source_hz=128,
                timeline=[],
            )

        participant_id = str(df["participant_id"][0])
        record_id = str(df["record_id"][0])
        source_hz = int(df["source_hz"][0])

        bouts = [
            LTAFBoutRecord(
                start_sample=int(row["start_sample"]),
                end_sample=int(row["end_sample"]),
                activity=str(row["activity"]),
                duration_samples=int(row["duration_samples"]),
            )
            for row in df.iter_rows(named=True)
            if int(row["duration_samples"]) > 0
        ]

        return LTAFParticipantTimeline(
            participant_id=participant_id,
            record_id=record_id,
            source_hz=source_hz,
            timeline=bouts,
        )

    @staticmethod
    def create_split_manifest(
        participant_ids: list[str],
        seed: int = 42,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
    ) -> dict[str, object]:
        if train_ratio <= 0 or val_ratio < 0 or test_ratio < 0:
            raise ValueError("Invalid split ratios")
        if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
            raise ValueError("Split ratios must sum to 1.0")

        unique_ids = sorted(set(participant_ids))
        rng = np.random.default_rng(int(seed))
        shuffled = list(unique_ids)
        rng.shuffle(shuffled)

        n = len(shuffled)
        if n == 0:
            return {
                "seed": int(seed),
                "ratios": {
                    "train": train_ratio,
                    "validation": val_ratio,
                    "test": test_ratio,
                },
                "n_participants": 0,
                "train": [],
                "validation": [],
                "test": [],
            }

        n_train = int(np.floor(n * train_ratio))
        n_val = int(np.floor(n * val_ratio))
        n_test = n - n_train - n_val

        if n >= 3:
            if n_val == 0:
                if n_train > 1:
                    n_train -= 1
                    n_val = 1
                elif n_test > 1:
                    n_test -= 1
                    n_val = 1
            if n_test == 0:
                if n_train > 1:
                    n_train -= 1
                    n_test = 1
                elif n_val > 1:
                    n_val -= 1
                    n_test = 1

        if n_train == 0:
            n_train = 1
            if n_val > 0:
                n_val -= 1
            elif n_test > 0:
                n_test -= 1

        n_test = n - n_train - n_val

        train = sorted(shuffled[:n_train])
        validation = sorted(shuffled[n_train : n_train + n_val])
        test = sorted(shuffled[n_train + n_val :])

        return {
            "seed": int(seed),
            "ratios": {
                "train": train_ratio,
                "validation": val_ratio,
                "test": test_ratio,
            },
            "n_participants": n,
            "train": train,
            "validation": validation,
            "test": test,
        }

    def save_split_manifest(self, manifest: dict[str, object]) -> Path:
        path = self.get_split_manifest_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)
        return path

    def load_split_manifest(self) -> dict[str, object]:
        path = self.get_split_manifest_path()
        if not path.exists():
            raise FileNotFoundError(f"Split manifest not found at {path}")
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def build_all_timelines(
        self,
        overwrite: bool = False,
        record_ids: Optional[list[str]] = None,
        max_participants: Optional[int] = None,
        split_seed: int = 42,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        write_split_manifest: bool = False,
    ) -> dict[str, object]:
        """Build rhythm + beat timelines for all records.

        ``write_split_manifest`` is kept for backward compatibility; the
        canonical split + paced-ratio computation now lives in
        ``core/participant_splitter.py``. Callers should invoke that module
        after timelines are built so paced ratios can be captured.
        """
        self.get_timelines_dir().mkdir(parents=True, exist_ok=True)
        self.get_beat_timelines_dir().mkdir(parents=True, exist_ok=True)

        if record_ids is None:
            selected_record_ids = self.discover_available_records()
        else:
            selected_record_ids = sorted({rid.strip().replace("\\", "/") for rid in record_ids if rid and rid.strip()})

        if max_participants is not None:
            selected_record_ids = selected_record_ids[:max_participants]

        if not selected_record_ids:
            raise FileNotFoundError(
                f"No LTAF .hea records found under {self.raw_data_dir}. "
                "Run scripts/data/download_ltaf.py first."
            )

        selected_participants = [_participant_id_from_record_id(rid) for rid in selected_record_ids]

        for record_id in tqdm(selected_record_ids, desc="Building LTAF timelines"):
            participant_id = _participant_id_from_record_id(record_id)
            rhythm_path = self.get_timeline_path(participant_id)
            beat_path = self.get_beat_timeline_path(participant_id)
            if rhythm_path.exists() and beat_path.exists() and not overwrite:
                continue

            timeline, beats = self.build_participant_artifacts(
                participant_id=participant_id,
                record_id=record_id,
            )
            self._save_timeline(timeline)
            self._save_beat_timeline(
                participant_id=participant_id,
                record_id=record_id,
                beats=beats,
            )

        available = sorted(
            p.stem
            for p in self.get_timelines_dir().glob("*.parquet")
            if p.stem in set(selected_participants)
        )
        if write_split_manifest:
            manifest = self.create_split_manifest(
                participant_ids=available,
                seed=split_seed,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
            )
            self.save_split_manifest(manifest)
            return manifest
        return {"n_participants": len(available), "available": available}

    def get_available_participants(self) -> list[str]:
        timelines_dir = self.get_timelines_dir()
        if not timelines_dir.exists():
            return []
        return sorted([p.stem for p in timelines_dir.glob("*.parquet")])

    def load_all_timelines(self, max_participants: Optional[int] = None) -> dict[str, LTAFParticipantTimeline]:
        timeline_paths = sorted(self.get_timelines_dir().glob("*.parquet"))
        if max_participants is not None:
            timeline_paths = timeline_paths[:max_participants]

        timelines: dict[str, LTAFParticipantTimeline] = {}
        for path in timeline_paths:
            timeline = LTAFTimelineBuilder._load_timeline(path)
            timelines[timeline.participant_id] = timeline
        return timelines
