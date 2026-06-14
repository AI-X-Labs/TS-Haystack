# SPDX-License-Identifier: CC-BY-NC-4.0

"""
Multi-Hop Task: "When did the Kth {target} occur {direction} the Nth {anchor}?"

Picks an anchor activity+ordinal, a direction (before/after),
a target activity, and counts K target bouts in that direction.
"""

import numpy as np

from src.datasets.sleep_psg_haystack.core.data_structures import (
    PSGGeneratedSample,
    PSGRecordingSample,
)
from src.datasets.sleep_psg_haystack.core.prompt_templates import ordinal
from src.datasets.sleep_psg_haystack.tasks.base_task import PSGBaseTaskGenerator


class PSGMultiHopTaskGenerator(PSGBaseTaskGenerator):

    @property
    def task_name(self) -> str:
        return "multi_hop"

    @property
    def answer_type(self) -> str:
        return "time_range"

    def generate_sample(
        self,
        recording: PSGRecordingSample,
        rng: np.random.Generator,
    ) -> PSGGeneratedSample:
        # Need two distinct activities, both present. The ANCHOR doesn't
        # appear in the answer (only its ordinal is referenced in the question)
        # so a clipped anchor is fine — its position semantics still work for
        # direction comparison. The TARGET *is* the answer, so target bouts
        # must be non-clipped.
        anchor_pool = [a for a, bouts in recording.activity_index.items() if bouts]
        target_pool = []
        for a, bouts in recording.activity_index.items():
            clipped = recording.clipped_keys.get(a, set())
            usable_idx = [i for i in range(len(bouts)) if i not in clipped]
            if usable_idx:
                target_pool.append((a, usable_idx))
        if not anchor_pool or not target_pool:
            return self._create_invalid_sample("Empty anchor or target pool", recording)

        # Pick target activity, then a distinct anchor activity
        target_activity, target_usable = target_pool[int(rng.integers(0, len(target_pool)))]
        anchor_candidates = [a for a in anchor_pool if a != target_activity]
        if not anchor_candidates:
            return self._create_invalid_sample("Need ≥2 distinct activities", recording)
        anchor_activity = anchor_candidates[int(rng.integers(0, len(anchor_candidates)))]

        anchor_bouts = recording.activity_index[anchor_activity]
        target_bouts = recording.activity_index[target_activity]

        # Pick any anchor ordinal in the window (clipped allowed)
        anchor_n = int(rng.integers(1, len(anchor_bouts) + 1))
        anchor_bout = anchor_bouts[anchor_n - 1]
        anchor_ms = anchor_bout.start_ms

        # Pick direction
        direction = "after" if rng.random() < 0.5 else "before"

        # Count target bouts in that direction (only non-clipped targets are
        # eligible as the answer).
        target_usable_set = set(target_usable)
        if direction == "after":
            targets_in_dir = [
                (i, b) for i, b in enumerate(target_bouts)
                if i in target_usable_set and b.start_ms > anchor_ms
            ]
        else:
            targets_in_dir = [
                (i, b) for i, b in enumerate(target_bouts)
                if i in target_usable_set and b.end_ms < anchor_ms
            ]
            targets_in_dir = list(reversed(targets_in_dir))  # closest first

        if not targets_in_dir:
            return self._create_invalid_sample(
                f"No non-clipped {target_activity} bouts {direction} the {anchor_n}th {anchor_activity}",
                recording,
            )

        # Pick K ∈ [1, available_count]
        k = rng.integers(1, len(targets_in_dir) + 1)
        _, target_bout = targets_in_dir[int(k) - 1]

        start_ts, end_ts = self._bout_time_range(target_bout)

        question, answer = self.template_bank.sample(
            task="multi_hop", rng=rng,
            kth=ordinal(int(k)), target_activity=target_activity,
            direction=direction,
            nth_anchor=ordinal(int(anchor_n)), anchor_activity=anchor_activity,
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
                "anchor_activity": anchor_activity,
                "anchor_ordinal": int(anchor_n),
                "anchor_start_ms": anchor_bout.start_ms,
                "anchor_end_ms": anchor_bout.end_ms,
                "target_activity": target_activity,
                "target_ordinal": int(k),
                "direction": direction,
                "start_ms": target_bout.start_ms,
                "end_ms": target_bout.end_ms,
            },
        )
