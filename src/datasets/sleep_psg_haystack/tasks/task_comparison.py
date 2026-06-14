# SPDX-License-Identifier: CC-BY-NC-4.0

"""
Comparison Task: "Which was the longest/shortest {activity} bout?"

Picks an activity with ≥2 bouts and distinct durations, then asks
for the time range of the extremum (longest or shortest) bout.
"""

import numpy as np

from src.datasets.sleep_psg_haystack.core.data_structures import (
    PSGGeneratedSample,
    PSGRecordingSample,
)
from src.datasets.sleep_psg_haystack.tasks.base_task import PSGBaseTaskGenerator


class PSGComparisonTaskGenerator(PSGBaseTaskGenerator):

    @property
    def task_name(self) -> str:
        return "comparison"

    @property
    def answer_type(self) -> str:
        return "time_range"

    def generate_sample(
        self,
        recording: PSGRecordingSample,
        rng: np.random.Generator,
    ) -> PSGGeneratedSample:
        # Pre-filter to activities that have BOTH a unique longest AND a unique
        # shortest among NON-CLIPPED bouts. Doing this once before the
        # superlative coin-flip removes the longest-vs-shortest imbalance that
        # arises when the asymmetric tie-rejection at sample-time biases away
        # from "shortest" (epoch-aligned bouts have many tied minimums).
        candidates = []  # list of (activity, non_clipped_bouts_sorted_by_dur)
        for activity, bouts in recording.activity_index.items():
            clipped = recording.clipped_keys.get(activity, set())
            usable = [b for i, b in enumerate(bouts) if i not in clipped]
            if len(usable) < 2:
                continue
            durs = sorted(b.duration_ms for b in usable)
            if durs[0] == durs[1]:        # tied minimum
                continue
            if durs[-1] == durs[-2]:      # tied maximum
                continue
            candidates.append((activity, usable))

        if not candidates:
            return self._create_invalid_sample(
                "No activities with both unique longest and unique shortest", recording,
            )

        activity, usable = candidates[rng.integers(0, len(candidates))]

        # 50/50 superlative coin flip — guaranteed valid because we filtered
        superlative = "longest" if rng.random() < 0.5 else "shortest"
        if superlative == "longest":
            target_bout = max(usable, key=lambda b: b.duration_ms)
        else:
            target_bout = min(usable, key=lambda b: b.duration_ms)
        bouts = usable

        start_ts, end_ts = self._bout_time_range(target_bout)

        question, answer = self.template_bank.sample(
            task="comparison", rng=rng,
            superlative=superlative, activity=activity,
            start=start_ts, end=end_ts,
        )

        return PSGGeneratedSample(
            subject_id=recording.subject_id,
            recording_duration_ms=recording.recording_duration_ms,
            label_class=self.label_class,
            task_type=self.task_name,
            **self._window_fields(recording),
            question=question,
            answer=answer,
            answer_type=self.answer_type,
            metadata={
                "activity": activity,
                "superlative": superlative,
                "start_ms": target_bout.start_ms,
                "end_ms": target_bout.end_ms,
                "duration_ms": target_bout.duration_ms,
            },
        )
