# SPDX-FileCopyrightText: 2026 Anonymous Authors
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Prompt templates for LTAF-Haystack tasks.

Each task has 8–15 template variants plus a rhythm-aware vocabulary so
the generator can phrase questions with either the canonical rhythm
code (``"AFIB"``), a standard medical abbreviation (``"AF"``), or the
full name (``"atrial fibrillation"``). Beat-symbol tasks pick from a
separate vocabulary (``"V"`` / ``"PVC"`` / ``"ventricular premature
contraction"``).

The ``LTAFPromptTemplateBank.sample`` call keeps a single back-compat
entry point: `bank.sample(task, rng, **kwargs)` → ``(question, answer)``.
``render_template`` offers the new three-argument form in case callers
want to choose the template variant themselves.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np


# --------------------------------------------------------------------- #
# Vocabulary
# --------------------------------------------------------------------- #

ACTIVITY_VOCAB: Dict[str, List[str]] = {
    "NSR":  ["normal sinus rhythm", "sinus rhythm", "NSR"],
    "AFIB": ["atrial fibrillation", "AF", "AFIB"],
    "SBR":  ["sinus bradycardia", "bradycardia", "SBR"],
    "AB":   ["atrial bigeminy", "AB"],
    "B":    ["ventricular bigeminy", "bigeminy", "B"],
    "T":    ["ventricular trigeminy", "trigeminy", "T"],
    "SVTA": ["supraventricular tachyarrhythmia", "SVT", "SVTA"],
    "VT":   ["ventricular tachycardia", "VT"],
    "IVR":  ["idioventricular rhythm", "IVR"],
}

BEAT_VOCAB: Dict[str, List[str]] = {
    "N": ["normal beat", "sinus beat", "N"],
    "A": ["atrial premature beat", "APC", "PAC", "A"],
    "V": ["ventricular premature beat", "PVC", "V"],
    "Q": ["unclassifiable beat", "Q"],
}


def pick_vocab(activity: str, rng: np.random.Generator) -> str:
    """Return a randomly chosen phrasing of ``activity``.

    Accepts rhythm codes (NSR/AFIB/...) and beat symbols (N/A/V/Q).
    Unknown codes are returned unchanged.
    """
    if activity in ACTIVITY_VOCAB:
        opts = ACTIVITY_VOCAB[activity]
    elif activity in BEAT_VOCAB:
        opts = BEAT_VOCAB[activity]
    else:
        return str(activity)
    return opts[int(rng.integers(0, len(opts)))]


# Back-compat alias (older call sites use the underscored name).
_pick_vocab = pick_vocab


# --------------------------------------------------------------------- #
# Templates — 10 variants per task
# --------------------------------------------------------------------- #

TEMPLATES_EXISTENCE: List[str] = [
    "Is there any {activity_name} episode in this ECG window?",
    "Does this ECG contain an episode of {activity_name}?",
    "Did the patient experience {activity_name} during this recording?",
    "Is {activity_name} present anywhere in this ECG window?",
    "Can you identify {activity_name} in this recording?",
    "Does the following ECG include a {activity_name} segment?",
    "Was {activity_name} detected in this window of ECG data?",
    "Is there at least one {activity_name} event in this trace?",
    "Does any portion of this ECG show {activity_name}?",
    "Is {activity_name} observed in this recording window?",
]

TEMPLATES_LOCALIZATION: List[str] = [
    "When did the {nth} {activity_name} occur?",
    "At what time range does the {nth} {activity_name} episode appear?",
    "Identify the time range of the {nth} {activity_name} bout.",
    "Locate the {nth} {activity_name} segment in this recording.",
    "Give the start and end timestamps of the {nth} {activity_name} episode.",
    "What is the time span of the {nth} {activity_name} event?",
    "Where in the recording is the {nth} {activity_name} bout?",
    "Report the time range for the {nth} {activity_name} occurrence.",
    "Between which timestamps does the {nth} {activity_name} episode fall?",
    "Find the {nth} instance of {activity_name} and give its time range.",
]

TEMPLATES_COUNTING: List[str] = [
    "How many {activity_name} bouts are there in this ECG?",
    "Count the {activity_name} episodes in this recording.",
    "How many distinct {activity_name} segments appear in this window?",
    "What is the number of {activity_name} events in this ECG?",
    "Report the count of {activity_name} episodes.",
    "How many times does {activity_name} occur in this recording?",
    "Give the total count of {activity_name} bouts visible here.",
    "Enumerate the {activity_name} episodes — how many are there?",
    "How many separate {activity_name} intervals are in this trace?",
    "Count every {activity_name} bout in this ECG window.",
]

TEMPLATES_ORDERING: List[str] = [
    "Did the {nth} {activity_a_name} occur before the {mth} {activity_b_name}?",
    "In this ECG, did the {nth} {activity_a_name} happen earlier than the {mth} {activity_b_name}?",
    "Was the {nth} {activity_a_name} episode earlier than the {mth} {activity_b_name}?",
    "Does the {nth} {activity_a_name} precede the {mth} {activity_b_name} in this recording?",
    "Did the {nth} {activity_a_name} bout occur first, before the {mth} {activity_b_name}?",
    "Is the {nth} {activity_a_name} earlier in time than the {mth} {activity_b_name}?",
    "Compare timing: did the {nth} {activity_a_name} come before the {mth} {activity_b_name}?",
    "Ordering check — {nth} {activity_a_name} before {mth} {activity_b_name}?",
    "Did the recording see the {nth} {activity_a_name} first, then the {mth} {activity_b_name}?",
    "In chronological order, is the {nth} {activity_a_name} prior to the {mth} {activity_b_name}?",
]

TEMPLATES_STATE_QUERY: List[str] = [
    "What rhythm was the patient in at the {nth} {symbol_name}?",
    "At the moment of the {nth} {symbol_name}, which rhythm is present?",
    "Identify the rhythm covering the {nth} {symbol_name} in this ECG.",
    "Which cardiac rhythm is active at the {nth} {symbol_name}?",
    "What ECG rhythm surrounds the {nth} {symbol_name}?",
    "The {nth} {symbol_name} occurs within which rhythm state?",
    "Which rhythm does the patient exhibit at the {nth} {symbol_name}?",
    "At the {nth} {symbol_name}, what is the underlying rhythm?",
    "Name the rhythm at the time of the {nth} {symbol_name}.",
    "What rhythm is present during the {nth} {symbol_name}?",
]

TEMPLATES_ANTECEDENT: List[str] = [
    "Which rhythm immediately preceded the {nth} {activity_name} episode?",
    "What ECG rhythm came right before the {nth} {activity_name} bout?",
    "Identify the rhythm directly prior to the {nth} {activity_name} event.",
    "Before the {nth} {activity_name} started, which rhythm was present?",
    "What rhythm preceded the {nth} {activity_name} in this recording?",
    "Which rhythm transitioned into the {nth} {activity_name} episode?",
    "What was the cardiac rhythm just before the {nth} {activity_name}?",
    "Name the antecedent rhythm to the {nth} {activity_name} bout.",
    "Prior to the {nth} {activity_name}, the patient was in which rhythm?",
    "Which rhythm state existed immediately before the {nth} {activity_name}?",
]

TEMPLATES_COMPARISON_LONGEST_WITH: List[str] = [
    "What was the longest {activity_name} bout in this recording?",
    "Give the time range of the longest {activity_name} episode.",
    "When did the longest {activity_name} segment occur?",
    "Identify the longest {activity_name} bout — what's its time range?",
    "Report the time span of the single longest {activity_name} episode in this window.",
    "Locate the longest continuous {activity_name} period.",
    "Of all {activity_name} bouts here, which is the longest — give its start/end time.",
    "What is the longest-duration {activity_name} episode in this ECG?",
]

TEMPLATES_COMPARISON_SHORTEST_WITH: List[str] = [
    "What was the shortest {activity_name} bout in this recording?",
    "Give the time range of the shortest {activity_name} episode.",
    "When did the shortest {activity_name} segment occur?",
    "Identify the shortest {activity_name} bout — what's its time range?",
    "Report the time span of the single shortest {activity_name} episode in this window.",
    "Locate the shortest continuous {activity_name} period.",
    "Of all {activity_name} bouts here, which is the shortest — give its start/end time.",
    "What is the shortest-duration {activity_name} episode in this ECG?",
]

TEMPLATES_COMPARISON_LONGEST_WITHOUT: List[str] = [
    "What was the longest period without {activity_name} in this recording?",
    "Give the time range of the longest stretch free of {activity_name}.",
    "When is the longest gap between {activity_name} bouts?",
    "Identify the longest {activity_name}-free interval — what's its time range?",
    "Report the time span of the single longest period not in {activity_name}.",
    "Locate the longest continuous window where no {activity_name} occurs.",
    "Across this ECG, which interval without any {activity_name} is the longest?",
    "What is the longest-duration {activity_name}-free period in this ECG?",
]

TEMPLATES_COMPARISON_SHORTEST_WITHOUT: List[str] = [
    "What was the shortest period without {activity_name} in this recording?",
    "Give the time range of the shortest stretch free of {activity_name}.",
    "When is the shortest gap between {activity_name} bouts?",
    "Identify the shortest {activity_name}-free interval — what's its time range?",
    "Report the time span of the single shortest period not in {activity_name}.",
    "Locate the shortest continuous window where no {activity_name} occurs.",
    "Across this ECG, which interval without any {activity_name} is the shortest?",
    "What is the shortest-duration {activity_name}-free period in this ECG?",
]

COMPARISON_TEMPLATES: Dict[Tuple[str, str], List[str]] = {
    ("longest", "with"):     TEMPLATES_COMPARISON_LONGEST_WITH,
    ("shortest", "with"):    TEMPLATES_COMPARISON_SHORTEST_WITH,
    ("longest", "without"):  TEMPLATES_COMPARISON_LONGEST_WITHOUT,
    ("shortest", "without"): TEMPLATES_COMPARISON_SHORTEST_WITHOUT,
}

# Back-compat: a flat list for callers that iterate over all phrasings.
TEMPLATES_COMPARISON: List[str] = (
    TEMPLATES_COMPARISON_LONGEST_WITH
    + TEMPLATES_COMPARISON_SHORTEST_WITH
    + TEMPLATES_COMPARISON_LONGEST_WITHOUT
    + TEMPLATES_COMPARISON_SHORTEST_WITHOUT
)

TEMPLATES_MULTI_HOP: List[str] = [
    "When did the {kth} {target_name} occur {direction} the {nth} {anchor_name}?",
    "Find the {kth} {target_name} that falls {direction} the {nth} {anchor_name} — give its time range.",
    "Locate the {kth} {target_name} occurring {direction} the {nth} {anchor_name}.",
    "At what time does the {kth} {target_name} {direction} the {nth} {anchor_name} appear?",
    "Give the time range of the {kth} {target_name} {direction} the {nth} {anchor_name}.",
    "{direction} the {nth} {anchor_name}, locate the {kth} {target_name}.",
    "Relative to the {nth} {anchor_name}, where is the {kth} {target_name} {direction}?",
    "Find the {kth} instance of {target_name} that lies {direction} the {nth} {anchor_name}.",
    "Report the time span of the {kth} {target_name} situated {direction} the {nth} {anchor_name}.",
    "{kth} {target_name}, {direction} the {nth} {anchor_name} — what's its time range?",
]

TEMPLATES_ANOMALY_DETECTION: List[str] = [
    "Does this ECG window contain any {symbol_name}?",
    "Is there any {symbol_name} in this recording?",
    "Did the patient have a {symbol_name} during this window?",
    "Is a {symbol_name} visible anywhere in this ECG?",
    "Does this trace include a {symbol_name}?",
    "Was any {symbol_name} detected in the recording?",
    "Is there at least one {symbol_name} present in this ECG?",
    "Does the window show a {symbol_name}?",
    "Anomaly check — does this ECG exhibit a {symbol_name}?",
    "Is there any occurrence of {symbol_name} in this window?",
]

TEMPLATES_ANOMALY_LOCALIZATION: List[str] = [
    "When does the {nth} {symbol_name} occur in this ECG?",
    "At what time does the {nth} {symbol_name} appear?",
    "Locate the {nth} {symbol_name} in this recording.",
    "Give the timestamp of the {nth} {symbol_name}.",
    "Where is the {nth} {symbol_name} in this ECG window?",
    "Identify the time of the {nth} {symbol_name} event.",
    "Report the timestamp for the {nth} {symbol_name}.",
    "Find the {nth} {symbol_name} and give its time.",
    "At what moment is the {nth} {symbol_name}?",
    "What is the occurrence time of the {nth} {symbol_name}?",
]


_TASK_TEMPLATES: Dict[str, List[str]] = {
    "existence": TEMPLATES_EXISTENCE,
    "localization": TEMPLATES_LOCALIZATION,
    "counting": TEMPLATES_COUNTING,
    "ordering": TEMPLATES_ORDERING,
    "state_query": TEMPLATES_STATE_QUERY,
    "antecedent": TEMPLATES_ANTECEDENT,
    "comparison": TEMPLATES_COMPARISON,
    "multi_hop": TEMPLATES_MULTI_HOP,
    "anomaly_detection": TEMPLATES_ANOMALY_DETECTION,
    "anomaly_localization": TEMPLATES_ANOMALY_LOCALIZATION,
}


# --------------------------------------------------------------------- #
# Bank
# --------------------------------------------------------------------- #


class LTAFPromptTemplateBank:
    """Samples natural language question + answer for each LTAF task."""

    def __init__(self) -> None:
        self._templates = _TASK_TEMPLATES

    def get_variants(self, task: str) -> List[str]:
        return list(self._templates.get(task, []))

    def render_template(
        self, task: str, variant_index: int, rng: np.random.Generator, **kwargs: Any
    ) -> str:
        variants = self._templates.get(task)
        if not variants:
            return "Answer the ECG question."
        template = variants[int(variant_index) % len(variants)]
        return self._render(template, rng, **kwargs)

    def sample(
        self, task: str, rng: np.random.Generator, **kwargs: Any
    ) -> Tuple[str, str]:
        variants = self._templates.get(task)
        if not variants:
            question = "Answer the ECG question."
        else:
            template = variants[int(rng.integers(0, len(variants)))]
            question = self._render(template, rng, **kwargs)
        return question, self._format_answer(kwargs.get("answer", ""))

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    @staticmethod
    def _format_answer(answer: Any) -> str:
        if isinstance(answer, bool):
            return "yes" if answer else "no"
        return str(answer)

    @staticmethod
    def _render(template: str, rng: np.random.Generator, **kwargs: Any) -> str:
        # Build vocabulary-aware fields on top of caller kwargs so tasks
        # can just pass `activity=`, `activity_a=`, etc.
        mapping: Dict[str, Any] = dict(kwargs)

        def _vocab_for(code: str) -> str:
            return _pick_vocab(code, rng)

        for key in ("activity", "target_activity", "anchor", "target"):
            if key in mapping and isinstance(mapping[key], str):
                mapping.setdefault(f"{key}_name", _vocab_for(mapping[key]))
        for pair in ("activity_a", "activity_b"):
            if pair in mapping and isinstance(mapping[pair], str):
                mapping.setdefault(f"{pair}_name", _vocab_for(mapping[pair]))
        if "symbol" in mapping and isinstance(mapping["symbol"], str):
            mapping.setdefault("symbol_name", _vocab_for(mapping["symbol"]))
        # Beat-style tasks sometimes pass `activity=` with a beat symbol.
        if "activity" in mapping and mapping.get("activity") in BEAT_VOCAB:
            mapping.setdefault("symbol_name", _vocab_for(mapping["activity"]))

        # Fallbacks: expose canonical code names too so both
        # `{activity}` and `{activity_name}` work in a template.
        for src, dst in (
            ("activity", "activity"),
            ("target", "target_name"),
            ("anchor", "anchor_name"),
        ):
            if src in mapping and isinstance(mapping[src], str):
                mapping.setdefault(dst, mapping[src])

        try:
            return template.format(**mapping)
        except KeyError:
            # If a specific template needs a field the caller did not provide,
            # render with a safe default instead of crashing the pipeline.
            from string import Formatter

            fmt = Formatter()
            missing = [
                fname
                for _, fname, _, _ in fmt.parse(template)
                if fname and fname not in mapping
            ]
            for m in missing:
                mapping[m] = mapping.get("activity", "")
            return template.format(**mapping)


__all__ = [
    "LTAFPromptTemplateBank",
    "ACTIVITY_VOCAB",
    "BEAT_VOCAB",
    "pick_vocab",
    "TEMPLATES_EXISTENCE",
    "TEMPLATES_LOCALIZATION",
    "TEMPLATES_COUNTING",
    "TEMPLATES_ORDERING",
    "TEMPLATES_STATE_QUERY",
    "TEMPLATES_ANTECEDENT",
    "TEMPLATES_COMPARISON",
    "TEMPLATES_COMPARISON_LONGEST_WITH",
    "TEMPLATES_COMPARISON_SHORTEST_WITH",
    "TEMPLATES_COMPARISON_LONGEST_WITHOUT",
    "TEMPLATES_COMPARISON_SHORTEST_WITHOUT",
    "COMPARISON_TEMPLATES",
    "TEMPLATES_MULTI_HOP",
    "TEMPLATES_ANOMALY_DETECTION",
    "TEMPLATES_ANOMALY_LOCALIZATION",
]
