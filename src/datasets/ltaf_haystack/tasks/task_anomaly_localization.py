# SPDX-FileCopyrightText: 2026 Anonymous Authors
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Anomaly localization: "Where is the Nth {V/A} beat?" → timestamp."""

from __future__ import annotations

from typing import List

import numpy as np

from src.datasets.ltaf_haystack.core.data_structures import (
    LTAFGeneratedSample,
    LTAFRecordingSample,
)
from src.datasets.ltaf_haystack.tasks.base_task import LTAFBaseTaskGenerator, _ordinal


class LTAFAnomalyLocalizationTaskGenerator(LTAFBaseTaskGenerator):

    @property
    def task_name(self) -> str:
        return "anomaly_localization"

    @property
    def answer_type(self) -> str:
        return "timestamp"

    @classmethod
    def supports_context_length(cls, label_class, context_length_s):
        if context_length_s is None:
            return False
        return context_length_s >= 100

    def _generate(
        self, recording: LTAFRecordingSample, rng: np.random.Generator
    ):
        # Pick a symbol with at least one beat; prefer V then A.
        beats_by_symbol: dict[str, List] = {}
        for b in recording.beats_timeline:
            beats_by_symbol.setdefault(b.symbol, []).append(b)

        for symbol in ("V", "A"):
            if beats_by_symbol.get(symbol):
                break
        else:
            return self._create_invalid_sample(
                "No V/A anomaly beats in window", recording=recording
            )

        beats = beats_by_symbol[symbol]
        n = int(rng.integers(1, len(beats) + 1))
        beat = beats[n - 1]

        hz = recording.source_hz
        t_ms = int(round(beat.sample * 1000.0 / hz))
        answer_str = self._ms_to_timestamp(t_ms)

        question, answer = self.template_bank.sample(
            task="anomaly_localization",
            rng=rng,
            nth=_ordinal(n),
            symbol=symbol,
            target_activity=symbol,
            activity=symbol,
            answer=answer_str,
        )
        return self._build_sample(
            recording,
            question,
            answer,
            metadata={
                "beat_symbol": symbol,
                "ordinal": n,
                "beat_sample": int(beat.sample),
            },
        )

