# SPDX-License-Identifier: CC-BY-NC-4.0

"""
Ordering Task: "Did the Nth {A} occur before the Mth {B}?"

Balance-first approach: decides the answer (true/false) first,
then selects ordinals that produce the desired temporal order.
"""

import numpy as np

from src.datasets.sleep_psg_haystack.core.data_structures import (
    PSGGeneratedSample,
    PSGRecordingSample,
)
from src.datasets.sleep_psg_haystack.core.prompt_templates import ordinal
from src.datasets.sleep_psg_haystack.tasks.base_task import PSGBaseTaskGenerator


class PSGOrderingTaskGenerator(PSGBaseTaskGenerator):

    @property
    def task_name(self) -> str:
        return "ordering"

    @property
    def answer_type(self) -> str:
        return "boolean"

    def generate_sample(
        self,
        recording: PSGRecordingSample,
        rng: np.random.Generator,
    ) -> PSGGeneratedSample:
        # Need two distinct activities, both present
        present = [a for a, bouts in recording.activity_index.items() if len(bouts) >= 1]
        if len(present) < 2:
            return self._create_invalid_sample("Need ≥2 present activities", recording)

        idx = rng.choice(len(present), size=2, replace=False)
        activity_a = present[idx[0]]
        activity_b = present[idx[1]]
        bouts_a = recording.activity_index[activity_a]
        bouts_b = recording.activity_index[activity_b]

        # Decide answer first (50/50 balance)
        a_before_b = bool(rng.random() < 0.5)

        # Find ordinal pairs that match the desired temporal order
        valid_pairs = []
        for i, ba in enumerate(bouts_a):
            for j, bb in enumerate(bouts_b):
                if a_before_b and ba.start_ms < bb.start_ms:
                    valid_pairs.append((i + 1, j + 1))
                elif not a_before_b and ba.start_ms > bb.start_ms:
                    valid_pairs.append((i + 1, j + 1))

        if not valid_pairs:
            return self._create_invalid_sample(
                f"No ordinal pairs for desired order (a_before_b={a_before_b})", recording,
            )

        pair_idx = rng.integers(0, len(valid_pairs))
        n_a, n_b = valid_pairs[pair_idx]
        bout_a = bouts_a[n_a - 1]
        bout_b = bouts_b[n_b - 1]

        question, answer = self.template_bank.sample(
            task="ordering_boolean", rng=rng,
            nth_a=ordinal(n_a), activity_a=activity_a,
            nth_b=ordinal(n_b), activity_b=activity_b,
            is_before=a_before_b,
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
                "activity_a": activity_a,
                "activity_b": activity_b,
                "ordinal_a": n_a,
                "ordinal_b": n_b,
                "a_before_b": a_before_b,
                "start_ms": bout_a.start_ms,
                "end_ms": bout_a.end_ms,
                "start_ms_b": bout_b.start_ms,
                "end_ms_b": bout_b.end_ms,
            },
        )
