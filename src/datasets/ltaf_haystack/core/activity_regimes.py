# SPDX-FileCopyrightText: 2026 Anonymous Authors
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Rhythm and beat taxonomy for the LTAF dataset.

Rhythm codes (PhysioNet LTAF / AHA conventions):
  NSR   — Normal sinus rhythm
  AFIB  — Atrial fibrillation
  SBR   — Sinus bradycardia (<60 bpm, sinus origin)
  AB    — Atrial bigeminy (every other beat is an APC)
  B     — Ventricular bigeminy (every other beat is a PVC)
  T     — Ventricular trigeminy (every third beat is a PVC)
  SVTA  — Supraventricular tachyarrhythmia (≥3 consecutive SV ectopics @ >100 bpm)
  VT    — Ventricular tachycardia (≥3 consecutive PVCs @ >100 bpm)
  IVR   — Idioventricular rhythm (ventricular escape, typically <60 bpm, brief)

Beat codes (AHA):
  N — Normal sinus-origin beat
  A — Atrial premature contraction (APC / PAC / SVE)
  V — Ventricular premature contraction (PVC / VE)
  Q — Unclassifiable or paced beat

Known gaps in this LTAF subset (84 records):
  - AFL (atrial flutter): 0 episodes present. Models trained on this dataset
    will NOT see flutter. This must be documented in the dataset card.
  - IVR: rare; brief transient in most records. Context availability constrained.
"""

from __future__ import annotations

from typing import Dict, List, Set


# canonical order is load-bearing: bit index in window-index masks
_ACTIVITY_SETS: Dict[str, List[str]] = {
    "rhythms": ["NSR", "AFIB", "SBR", "AB", "B", "T", "SVTA", "VT", "IVR"],
}

BEAT_EVENT_TYPES: List[str] = ["N", "A", "V", "Q"]

# Used for distractor selection in ordering / multi_hop tasks.
_RHYTHM_REGIMES: Dict[str, List[str]] = {
    "tachy": ["AFIB", "SVTA", "VT"],
    "brady": ["SBR", "IVR"],
    "ectopic": ["AB", "B", "T"],
    "sinus": ["NSR"],
}


def get_all_activities(label_class: str) -> Set[str]:
    if label_class not in _ACTIVITY_SETS:
        raise ValueError(
            f"Unknown label_class '{label_class}'. Use: {list(_ACTIVITY_SETS.keys())}"
        )
    return set(_ACTIVITY_SETS[label_class])


def get_activities_list(label_class: str) -> List[str]:
    """Return the canonical ordered list (load-bearing for window-index bit masks)."""
    if label_class not in _ACTIVITY_SETS:
        raise ValueError(
            f"Unknown label_class '{label_class}'. Use: {list(_ACTIVITY_SETS.keys())}"
        )
    return list(_ACTIVITY_SETS[label_class])


def get_regime_of(rhythm: str) -> str | None:
    for regime, members in _RHYTHM_REGIMES.items():
        if rhythm in members:
            return regime
    return None


def get_same_regime_activities(rhythm: str) -> List[str]:
    regime = get_regime_of(rhythm)
    if regime is None:
        return []
    return [r for r in _RHYTHM_REGIMES[regime] if r != rhythm]


def get_all_regimes() -> Dict[str, List[str]]:
    return {k: list(v) for k, v in _RHYTHM_REGIMES.items()}
