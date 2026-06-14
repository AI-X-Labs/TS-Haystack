"""
Ordering. Insert one A-bout and one B-bout, ask "did A occur before B?".

Balance-by-derivation: positions are sampled uniformly inside the window, so
in expectation A-before-B and B-before-A occur equally often. We *read* the
answer off the actual positions rather than swapping needles into pre-chosen
slots (which would assign each needle to a slot not sized for it).
"""
from __future__ import annotations

import numpy as np

from src.datasets.uk_dale_haystack.core.insertion import (
    insert_needles,
    sample_positions,
)
from src.datasets.uk_dale_haystack.tasks.base_task import BaseTaskGenerator


class OrderingTaskGenerator(BaseTaskGenerator):
    task_name = "ordering"
    answer_type = "boolean"

    def generate_sample(self, ctx_s, rng):
        params = self._ctx_params(ctx_s)
        max_dur_s = params["max_needle_duration_s"]
        margin_samples = params["margin_samples"]
        min_gap_samples = params["min_gap_samples"]

        appliances = list(self.bg_sampler.appliances_with_bouts)
        if len(appliances) < 2:
            return self._invalid(ctx_s, "need >=2 appliances")

        for _ in range(20):
            apps = list(appliances)
            rng.shuffle(apps)
            a, b = apps[0], apps[1]

            bg = self.bg_sampler.sample(
                target=a, context_length_s=ctx_s, rng=rng,
                margin_s=margin_samples * self.dt_s,
                extra_off_targets=[b],
            )
            if bg is None:
                continue

            needle_a = self.needle_sampler.sample(
                appliance=a, max_duration_s=max_dur_s / 2, rng=rng,
                require_house_id=(None if self.allow_cross_house else bg.house_id),
                allow_cross_house=self.allow_cross_house,
            )
            needle_b = self.needle_sampler.sample(
                appliance=b, max_duration_s=max_dur_s / 2, rng=rng,
                require_house_id=(None if self.allow_cross_house else bg.house_id),
                allow_cross_house=self.allow_cross_house,
            )
            if needle_a is None or needle_b is None:
                continue

            ctx_samples = bg.mains_w.shape[0]
            positions = sample_positions(
                ctx_samples,
                [needle_a.submeter_w.shape[0], needle_b.submeter_w.shape[0]],
                margin_samples=margin_samples,
                min_gap_samples=min_gap_samples, rng=rng,
            )
            if positions is None:
                continue

            pos_a, pos_b = positions
            answer_a_before_b = bool(pos_a < pos_b)

            question, answer = self.template_bank.render(
                "ordering",
                {"appliance_a": a, "appliance_b": b, "before": answer_a_before_b},
                rng,
            )

            return insert_needles(
                bg, [needle_a, needle_b], positions,
                task_type=self.task_name,
                question=question, answer=answer,
                answer_type=self.answer_type,
                difficulty_config={
                    "appliance_a": a, "appliance_b": b,
                    "answer_a_before_b": answer_a_before_b,
                    "pos_a_samples": int(pos_a), "pos_b_samples": int(pos_b),
                },
            )
        return self._invalid(ctx_s, "could not place A/B")
