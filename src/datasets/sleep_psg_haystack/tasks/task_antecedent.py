# SPDX-License-Identifier: CC-BY-NC-4.0

"""
Antecedent Task:
  Sleep stages: "What stage came before the Nth {stage}?"
  Arousals: "What arousal occurred before the Nth {arousal}?"

Finds the immediately preceding bout in the full sorted timeline.
"""

import numpy as np

from src.datasets.sleep_psg_haystack.core.data_structures import (
    PSGGeneratedSample,
    PSGRecordingSample,
)
from src.datasets.sleep_psg_haystack.core.prompt_templates import ordinal
from src.datasets.sleep_psg_haystack.tasks.base_task import PSGBaseTaskGenerator


class PSGAntecedentTaskGenerator(PSGBaseTaskGenerator):

    @property
    def task_name(self) -> str:
        return "antecedent"

    @property
    def answer_type(self) -> str:
        return "category"

    def generate_sample(
        self,
        recording: PSGRecordingSample,
        rng: np.random.Generator,
    ) -> PSGGeneratedSample:
        # Get the full sorted timeline for the active label_class
        full_timeline = self._get_sorted_timeline(recording, self.label_class)
        if len(full_timeline) < 2:
            return self._create_invalid_sample("Timeline too short for antecedent", recording)

        # Pick an activity with at least 1 bout that is NOT the first in timeline
        candidates = []
        for activity, bouts in recording.activity_index.items():
            for i, bout in enumerate(bouts):
                # Find this bout's position in the full timeline
                for j, tb in enumerate(full_timeline):
                    if tb.start_ms == bout.start_ms and tb.activity == bout.activity:
                        if j > 0:  # not the first bout in the recording
                            candidates.append((activity, i + 1, j))  # (activity, ordinal, timeline_idx)
                        break

        if not candidates:
            return self._create_invalid_sample("No bouts with a preceding bout", recording)

        # Pick a random candidate
        activity, n, timeline_idx = candidates[rng.integers(0, len(candidates))]
        target_bout = full_timeline[timeline_idx]
        antecedent_bout = full_timeline[timeline_idx - 1]
        antecedent_activity = antecedent_bout.activity

        # Use label_class-specific template
        template_key = f"antecedent_{self.label_class}"
        question, answer = self.template_bank.sample(
            task=template_key, rng=rng,
            nth=ordinal(n), activity=activity,
            antecedent_activity=antecedent_activity,
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
                "ordinal": n,
                "antecedent_activity": antecedent_activity,
                "start_ms": target_bout.start_ms,
                "end_ms": target_bout.end_ms,
                "antecedent_start_ms": antecedent_bout.start_ms,
                "antecedent_end_ms": antecedent_bout.end_ms,
            },
        )
