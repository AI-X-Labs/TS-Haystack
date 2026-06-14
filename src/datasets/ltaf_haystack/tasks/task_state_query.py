# SPDX-FileCopyrightText: 2026 Anonymous Authors
# SPDX-License-Identifier: CC-BY-NC-4.0

"""State query: "What rhythm was present at the Nth {V/A/Q} beat?" → category.

Uses the beat timeline to pick a beat (preferring ectopic events — V/A/Q),
then reads the rhythm timeline for the bout containing that beat's sample.
"""

from __future__ import annotations

from typing import List

import numpy as np

from src.datasets.ltaf_haystack.core.data_structures import (
    LTAFGeneratedSample,
    LTAFRecordingSample,
)
from src.datasets.ltaf_haystack.tasks.base_task import LTAFBaseTaskGenerator, _ordinal


_BEAT_QUERY_SYMBOLS = ("V", "A", "Q")


class LTAFStateQueryTaskGenerator(LTAFBaseTaskGenerator):

    @property
    def task_name(self) -> str:
        return "state_query"

    @property
    def answer_type(self) -> str:
        return "category"

    def _generate(
        self, recording: LTAFRecordingSample, rng: np.random.Generator
    ):
        # Group beats by symbol, prefer ectopic/anomaly types.
        by_symbol: dict[str, List] = {}
        for b in recording.beats_timeline:
            by_symbol.setdefault(b.symbol, []).append(b)
        viable_symbols = [s for s in _BEAT_QUERY_SYMBOLS if by_symbol.get(s)]
        if not viable_symbols:
            # Fall back to any beat
            if not recording.beats_timeline:
                return self._create_invalid_sample(
                    "No beats in window", recording=recording
                )
            symbol = recording.beats_timeline[
                int(rng.integers(0, len(recording.beats_timeline)))
            ].symbol
            beats = by_symbol.setdefault(symbol, [])
        else:
            symbol = viable_symbols[int(rng.integers(0, len(viable_symbols)))]
            beats = by_symbol[symbol]

        if not beats:
            return self._create_invalid_sample(
                "No beats of selected symbol", recording=recording
            )
        n = int(rng.integers(1, len(beats) + 1))
        beat = beats[n - 1]

        rhythm = self._rhythm_at(recording, beat.sample)
        if rhythm is None:
            return self._create_invalid_sample(
                f"No rhythm bout covering sample {beat.sample}", recording=recording
            )

        question, answer = self.template_bank.sample(
            task="state_query",
            rng=rng,
            nth=_ordinal(n),
            activity=symbol,
            symbol=symbol,
            query_sample=int(beat.sample),
            answer=rhythm,
        )
        return self._build_sample(
            recording,
            question,
            answer,
            metadata={
                "beat_symbol": symbol,
                "ordinal": n,
                "beat_sample": int(beat.sample),
                "rhythm": rhythm,
            },
        )

    @staticmethod
    def _rhythm_at(recording: LTAFRecordingSample, sample: int):
        for bout in recording.rhythm_timeline:
            if bout.start_sample <= sample < bout.end_sample:
                return bout.activity
        return None

