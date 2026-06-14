"""
Natural-language prompt templates for UK-DALE-Haystack.

10-15 templates per task type, each speaking about appliances using common
synonyms (kettle / electric kettle, etc.) so the model can't lock onto a
single phrasing.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

APPLIANCE_VOCAB: dict[str, list[str]] = {
    "kettle":          ["kettle", "electric kettle"],
    "microwave":       ["microwave", "microwave oven"],
    "toaster":         ["toaster"],
    "hair_dryer":      ["hair dryer", "hairdryer"],
    "washing_machine": ["washing machine", "washer"],
    "dishwasher":      ["dishwasher"],
    "washer_dryer":    ["washer-dryer", "washer dryer combo"],
    "oven":            ["oven", "electric oven"],
    "fridge":          ["fridge", "refrigerator"],
    "fridge_freezer":  ["fridge-freezer", "refrigerator-freezer"],
    "freezer":         ["freezer"],
}


def appliance_phrase(canon: str, rng: np.random.Generator) -> str:
    options = APPLIANCE_VOCAB.get(canon, [canon])
    return options[int(rng.integers(0, len(options)))]


def ordinal(k: int) -> str:
    """1 -> '1st', 2 -> '2nd', 11 -> '11th', etc."""
    if 10 <= (k % 100) <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(k % 10, "th")
    return f"{k}{suffix}"


def fmt_time_range(t0_s: float, t1_s: float) -> str:
    """Format a window-relative range like '00:12:34 - 00:13:50'."""
    return f"{_hms(t0_s)} - {_hms(t1_s)}"


def _hms(s: float) -> str:
    s = max(0, int(round(s)))
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


# ---------------------------------------------------------------------------
# Templates per task. Each template is a callable rendering (question, answer).
# ---------------------------------------------------------------------------
# The functions take a `params` dict and an `rng`; the caller (task
# generator) is responsible for filling `params` with the keys each template
# expects (a per-task contract documented below).

TaskRenderer = Callable[[dict, np.random.Generator], tuple[str, str]]


# ----- existence ----------------------------------------------------------
# params: appliance (canon), exists (bool)
def _existence_templates() -> list[TaskRenderer]:
    bool_yes = "Yes"
    bool_no = "No"

    def t1(p, r):
        a = appliance_phrase(p["appliance"], r)
        return (f"Was the {a} on at any point in this recording?",
                bool_yes if p["exists"] else bool_no)

    def t2(p, r):
        a = appliance_phrase(p["appliance"], r)
        return (f"Did the {a} run during this window?",
                bool_yes if p["exists"] else bool_no)

    def t3(p, r):
        a = appliance_phrase(p["appliance"], r)
        return (f"Is there any {a} activity in this trace?",
                bool_yes if p["exists"] else bool_no)

    def t4(p, r):
        a = appliance_phrase(p["appliance"], r)
        return (f"Does this recording contain any {a} usage?",
                bool_yes if p["exists"] else bool_no)

    def t5(p, r):
        a = appliance_phrase(p["appliance"], r)
        return (f"Was the {a} used at all?",
                bool_yes if p["exists"] else bool_no)

    def t6(p, r):
        a = appliance_phrase(p["appliance"], r)
        return (f"Did anyone turn on the {a} in this period?",
                bool_yes if p["exists"] else bool_no)

    def t7(p, r):
        a = appliance_phrase(p["appliance"], r)
        return (f"Is the {a} drawing power somewhere in this signal?",
                bool_yes if p["exists"] else bool_no)

    def t8(p, r):
        a = appliance_phrase(p["appliance"], r)
        return (f"Did the {a} switch on during this recording?",
                bool_yes if p["exists"] else bool_no)

    def t9(p, r):
        a = appliance_phrase(p["appliance"], r)
        return (f"Looking at the mains, was the {a} ever active?",
                bool_yes if p["exists"] else bool_no)

    def t10(p, r):
        a = appliance_phrase(p["appliance"], r)
        return (f"Detect whether the {a} ran in this trace.",
                bool_yes if p["exists"] else bool_no)

    return [t1, t2, t3, t4, t5, t6, t7, t8, t9, t10]


# ----- localization -------------------------------------------------------
# params: appliance, k (ordinal), t0_s, t1_s
def _localization_templates() -> list[TaskRenderer]:
    def t1(p, r):
        a = appliance_phrase(p["appliance"], r)
        ans = fmt_time_range(p["t0_s"], p["t1_s"])
        return (f"When did the {ordinal(p['k'])} {a} bout occur?", ans)

    def t2(p, r):
        a = appliance_phrase(p["appliance"], r)
        return (f"At what time did the {ordinal(p['k'])} {a} cycle take place?",
                fmt_time_range(p["t0_s"], p["t1_s"]))

    def t3(p, r):
        a = appliance_phrase(p["appliance"], r)
        return (f"Locate the {ordinal(p['k'])} use of the {a}.",
                fmt_time_range(p["t0_s"], p["t1_s"]))

    def t4(p, r):
        a = appliance_phrase(p["appliance"], r)
        return (f"Find the time range of the {ordinal(p['k'])} {a} bout.",
                fmt_time_range(p["t0_s"], p["t1_s"]))

    def t5(p, r):
        a = appliance_phrase(p["appliance"], r)
        return (f"What is the time window of the {ordinal(p['k'])} {a} event?",
                fmt_time_range(p["t0_s"], p["t1_s"]))

    def t6(p, r):
        a = appliance_phrase(p["appliance"], r)
        return (f"When was the {a} run for the {ordinal(p['k'])} time?",
                fmt_time_range(p["t0_s"], p["t1_s"]))

    def t7(p, r):
        a = appliance_phrase(p["appliance"], r)
        return (f"Identify when the {ordinal(p['k'])} {a} cycle starts and ends.",
                fmt_time_range(p["t0_s"], p["t1_s"]))

    def t8(p, r):
        a = appliance_phrase(p["appliance"], r)
        return (f"Pinpoint the {ordinal(p['k'])} {a} bout in the recording.",
                fmt_time_range(p["t0_s"], p["t1_s"]))

    return [t1, t2, t3, t4, t5, t6, t7, t8]


# ----- counting -----------------------------------------------------------
# params: appliance, n
def _counting_templates() -> list[TaskRenderer]:
    def t1(p, r):
        a = appliance_phrase(p["appliance"], r)
        return (f"How many {a} bouts are present?", str(p["n"]))

    def t2(p, r):
        a = appliance_phrase(p["appliance"], r)
        return (f"Count the number of {a} cycles in this window.", str(p["n"]))

    def t3(p, r):
        a = appliance_phrase(p["appliance"], r)
        return (f"How many times did the {a} switch on?", str(p["n"]))

    def t4(p, r):
        a = appliance_phrase(p["appliance"], r)
        return (f"How many separate {a} runs do you see?", str(p["n"]))

    def t5(p, r):
        a = appliance_phrase(p["appliance"], r)
        return (f"Total {a} activations in this trace?", str(p["n"]))

    def t6(p, r):
        a = appliance_phrase(p["appliance"], r)
        return (f"Number of {a} usage events?", str(p["n"]))

    def t7(p, r):
        a = appliance_phrase(p["appliance"], r)
        return (f"How many {a} events happened?", str(p["n"]))

    def t8(p, r):
        a = appliance_phrase(p["appliance"], r)
        return (f"Count how many distinct {a} uses occur in this signal.", str(p["n"]))

    return [t1, t2, t3, t4, t5, t6, t7, t8]


# ----- ordering -----------------------------------------------------------
# params: appliance_a, appliance_b, before (bool: a-before-b)
def _ordering_templates() -> list[TaskRenderer]:
    def ans(p): return "Yes" if p["before"] else "No"

    def t1(p, r):
        a = appliance_phrase(p["appliance_a"], r)
        b = appliance_phrase(p["appliance_b"], r)
        return (f"Did the {a} run before the {b}?", ans(p))

    def t2(p, r):
        a = appliance_phrase(p["appliance_a"], r)
        b = appliance_phrase(p["appliance_b"], r)
        return (f"Was the {a} used before the {b}?", ans(p))

    def t3(p, r):
        a = appliance_phrase(p["appliance_a"], r)
        b = appliance_phrase(p["appliance_b"], r)
        return (f"Did the {a} cycle precede the {b} cycle?", ans(p))

    def t4(p, r):
        a = appliance_phrase(p["appliance_a"], r)
        b = appliance_phrase(p["appliance_b"], r)
        return (f"Looking at the trace, did the {a} happen earlier than the {b}?", ans(p))

    def t5(p, r):
        a = appliance_phrase(p["appliance_a"], r)
        b = appliance_phrase(p["appliance_b"], r)
        return (f"Did the {a} switch on before the {b} switched on?", ans(p))

    def t6(p, r):
        a = appliance_phrase(p["appliance_a"], r)
        b = appliance_phrase(p["appliance_b"], r)
        return (f"Is the {a} bout earlier in time than the {b} bout?", ans(p))

    def t7(p, r):
        a = appliance_phrase(p["appliance_a"], r)
        b = appliance_phrase(p["appliance_b"], r)
        return (f"Did the {a} come before the {b} in this recording?", ans(p))

    def t8(p, r):
        a = appliance_phrase(p["appliance_a"], r)
        b = appliance_phrase(p["appliance_b"], r)
        return (f"Order check: did the {a} run prior to the {b}?", ans(p))

    return [t1, t2, t3, t4, t5, t6, t7, t8]


# ----- antecedent ---------------------------------------------------------
# params: target (canon), k (ordinal), antecedent (canon)
# Answers are returned in CANONICAL form (e.g. "fridge_freezer") so the
# training target is a single deterministic label per appliance. Question
# phrasings still randomise across synonym vocab.
def _antecedent_templates() -> list[TaskRenderer]:
    def t1(p, r):
        t = appliance_phrase(p["target"], r)
        return (f"Which appliance ran immediately before the {ordinal(p['k'])} {t}?", p["antecedent"])

    def t2(p, r):
        t = appliance_phrase(p["target"], r)
        return (f"What appliance switched on right before the {ordinal(p['k'])} {t}?", p["antecedent"])

    def t3(p, r):
        t = appliance_phrase(p["target"], r)
        return (f"Looking at the mains, what was the prior appliance event before the {ordinal(p['k'])} {t}?", p["antecedent"])

    def t4(p, r):
        t = appliance_phrase(p["target"], r)
        return (f"What was the most recent bout before the {ordinal(p['k'])} {t}?", p["antecedent"])

    def t5(p, r):
        t = appliance_phrase(p["target"], r)
        return (f"Identify the appliance that ran just before the {ordinal(p['k'])} {t}.", p["antecedent"])

    def t6(p, r):
        t = appliance_phrase(p["target"], r)
        return (f"Which appliance was used in the bout preceding the {ordinal(p['k'])} {t}?", p["antecedent"])

    def t7(p, r):
        t = appliance_phrase(p["target"], r)
        return (f"What ran immediately prior to the {ordinal(p['k'])} {t} cycle?", p["antecedent"])

    def t8(p, r):
        t = appliance_phrase(p["target"], r)
        return (f"Which appliance comes right before the {ordinal(p['k'])} {t} bout?", p["antecedent"])

    return [t1, t2, t3, t4, t5, t6, t7, t8]


# ----- comparison ---------------------------------------------------------
# params: appliance, mode ('longest'/'shortest'/'highest_peak'), t0_s, t1_s
def _comparison_templates() -> list[TaskRenderer]:
    def t1(p, r):
        a = appliance_phrase(p["appliance"], r)
        return (f"When in this trace did the {p['mode']} {a} bout occur?",
                fmt_time_range(p["t0_s"], p["t1_s"]))

    def t2(p, r):
        a = appliance_phrase(p["appliance"], r)
        return (f"Locate the {p['mode']} {a} bout.",
                fmt_time_range(p["t0_s"], p["t1_s"]))

    def t3(p, r):
        a = appliance_phrase(p["appliance"], r)
        return (f"What is the time range of the {a} cycle with the {p['mode']} value.",
                fmt_time_range(p["t0_s"], p["t1_s"]))

    def t4(p, r):
        a = appliance_phrase(p["appliance"], r)
        return (f"When did the {p['mode']} {a} bout occur?",
                fmt_time_range(p["t0_s"], p["t1_s"]))

    def t5(p, r):
        a = appliance_phrase(p["appliance"], r)
        return (f"Identify the time range of the {p['mode']} {a} event.",
                fmt_time_range(p["t0_s"], p["t1_s"]))

    def t6(p, r):
        a = appliance_phrase(p["appliance"], r)
        return (f"What is the time window of the {p['mode']} {a} bout?",
                fmt_time_range(p["t0_s"], p["t1_s"]))

    def t7(p, r):
        a = appliance_phrase(p["appliance"], r)
        return (f"Of the {a} bouts in this window, when did the {p['mode']} one occur?",
                fmt_time_range(p["t0_s"], p["t1_s"]))

    def t8(p, r):
        a = appliance_phrase(p["appliance"], r)
        return (f"Give the time range of the {p['mode']} {a} cycle in this trace.",
                fmt_time_range(p["t0_s"], p["t1_s"]))

    return [t1, t2, t3, t4, t5, t6, t7, t8]


# ----- multi_hop ----------------------------------------------------------
# params: anchor (canon), j (ordinal of anchor), target (canon), k (ordinal of target after anchor),
#         direction ('after'/'before'), t0_s, t1_s
def _multi_hop_templates() -> list[TaskRenderer]:
    def t1(p, r):
        a = appliance_phrase(p["anchor"], r)
        t = appliance_phrase(p["target"], r)
        return (f"When did the {ordinal(p['k'])} {t} occur {p['direction']} the {ordinal(p['j'])} {a}?",
                fmt_time_range(p["t0_s"], p["t1_s"]))

    def t2(p, r):
        a = appliance_phrase(p["anchor"], r)
        t = appliance_phrase(p["target"], r)
        return (f"Find the time range of the {ordinal(p['k'])} {t} bout {p['direction']} the {ordinal(p['j'])} {a} bout.",
                fmt_time_range(p["t0_s"], p["t1_s"]))

    def t3(p, r):
        a = appliance_phrase(p["anchor"], r)
        t = appliance_phrase(p["target"], r)
        return (f"Locate the {ordinal(p['k'])} {t} that occurs {p['direction']} the {ordinal(p['j'])} {a}.",
                fmt_time_range(p["t0_s"], p["t1_s"]))

    def t4(p, r):
        a = appliance_phrase(p["anchor"], r)
        t = appliance_phrase(p["target"], r)
        return (f"What is the time range of the {ordinal(p['k'])} {t} cycle {p['direction']} the {ordinal(p['j'])} {a}?",
                fmt_time_range(p["t0_s"], p["t1_s"]))

    def t5(p, r):
        a = appliance_phrase(p["anchor"], r)
        t = appliance_phrase(p["target"], r)
        return (f"After identifying the {ordinal(p['j'])} {a}, when does the {ordinal(p['k'])} {t} {p['direction'].replace('after','occur after').replace('before','occur before')} it?",
                fmt_time_range(p["t0_s"], p["t1_s"]))

    def t6(p, r):
        a = appliance_phrase(p["anchor"], r)
        t = appliance_phrase(p["target"], r)
        return (f"Identify the time range of the {ordinal(p['k'])} {t} bout that comes {p['direction']} the {ordinal(p['j'])} {a}.",
                fmt_time_range(p["t0_s"], p["t1_s"]))

    return [t1, t2, t3, t4, t5, t6]


# ----- state_query --------------------------------------------------------
# params: anchor (canon), k (ordinal), state_appliance (canon, the answer)
# Answers returned in CANONICAL form (single deterministic training target).
def _state_query_templates() -> list[TaskRenderer]:
    def t1(p, r):
        a = appliance_phrase(p["anchor"], r)
        return (f"What appliance was running while the {ordinal(p['k'])} {a} was on?", p["state_appliance"])

    def t2(p, r):
        a = appliance_phrase(p["anchor"], r)
        return (f"During the {ordinal(p['k'])} {a} bout, which other appliance was active?", p["state_appliance"])

    def t3(p, r):
        a = appliance_phrase(p["anchor"], r)
        return (f"Which appliance overlapped in time with the {ordinal(p['k'])} {a}?", p["state_appliance"])

    def t4(p, r):
        a = appliance_phrase(p["anchor"], r)
        return (f"What was running concurrently with the {ordinal(p['k'])} {a}?", p["state_appliance"])

    def t5(p, r):
        a = appliance_phrase(p["anchor"], r)
        return (f"Identify the appliance that was on at the time of the {ordinal(p['k'])} {a} bout.", p["state_appliance"])

    def t6(p, r):
        a = appliance_phrase(p["anchor"], r)
        return (f"Looking at the state during the {ordinal(p['k'])} {a}, what appliance was drawing power?", p["state_appliance"])

    return [t1, t2, t3, t4, t5, t6]


# ----- anomaly_detection --------------------------------------------------
# params: has_anomaly (bool)
def _anomaly_detection_templates() -> list[TaskRenderer]:
    def t1(p, r):
        return ("Is there an anomalous appliance bout in this window?",
                "Yes" if p["has_anomaly"] else "No")

    def t2(p, r):
        return ("Does any appliance show abnormal behaviour in this trace?",
                "Yes" if p["has_anomaly"] else "No")

    def t3(p, r):
        return ("Is at least one appliance bout suspicious or out-of-distribution here?",
                "Yes" if p["has_anomaly"] else "No")

    def t4(p, r):
        return ("Detect whether any appliance event is anomalous.",
                "Yes" if p["has_anomaly"] else "No")

    def t5(p, r):
        return ("Is there a malfunction-like appliance signature in this window?",
                "Yes" if p["has_anomaly"] else "No")

    def t6(p, r):
        return ("Are any appliance cycles in this window abnormally truncated or over-powered?",
                "Yes" if p["has_anomaly"] else "No")

    return [t1, t2, t3, t4, t5, t6]


# ----- anomaly_localization ----------------------------------------------
# params: t0_s, t1_s, anomaly_class
def _anomaly_localization_templates() -> list[TaskRenderer]:
    def t1(p, r):
        return ("When did the anomalous appliance bout occur?",
                fmt_time_range(p["t0_s"], p["t1_s"]))

    def t2(p, r):
        return ("Locate the time range of the anomalous bout in this window.",
                fmt_time_range(p["t0_s"], p["t1_s"]))

    def t3(p, r):
        return ("At what time does the suspicious appliance event happen?",
                fmt_time_range(p["t0_s"], p["t1_s"]))

    def t4(p, r):
        return ("Identify when the abnormal appliance cycle takes place.",
                fmt_time_range(p["t0_s"], p["t1_s"]))

    def t5(p, r):
        return ("Pinpoint the time window of the malfunction-like bout.",
                fmt_time_range(p["t0_s"], p["t1_s"]))

    def t6(p, r):
        return ("Find the timing of the anomalous appliance event.",
                fmt_time_range(p["t0_s"], p["t1_s"]))

    return [t1, t2, t3, t4, t5, t6]


# ---------------------------------------------------------------------------
# Bank
# ---------------------------------------------------------------------------

class PromptTemplateBank:
    def __init__(self) -> None:
        self._tasks: dict[str, list[TaskRenderer]] = {
            "existence":            _existence_templates(),
            "localization":         _localization_templates(),
            "counting":             _counting_templates(),
            "ordering":             _ordering_templates(),
            "antecedent":           _antecedent_templates(),
            "comparison":           _comparison_templates(),
            "multi_hop":            _multi_hop_templates(),
            "state_query":          _state_query_templates(),
            "anomaly_detection":    _anomaly_detection_templates(),
            "anomaly_localization": _anomaly_localization_templates(),
        }

    def render(
        self, task: str, params: dict[str, Any], rng: np.random.Generator,
    ) -> tuple[str, str]:
        templates = self._tasks[task]
        return templates[int(rng.integers(0, len(templates)))](params, rng)
