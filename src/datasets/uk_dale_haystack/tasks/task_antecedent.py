"""
Antecedent. Insert two needles of different appliances; ask "what occurred
before the [later one]?". Identity is decided by the sampled positions, not
forced -- this avoids the position/needle size mismatch that would happen if
we tried to swap positions to enforce a pre-decided ordering.
"""
from __future__ import annotations

import numpy as np

from src.datasets.uk_dale_haystack.core.insertion import (
    insert_needles,
    sample_positions,
)
from src.datasets.uk_dale_haystack.tasks.base_task import BaseTaskGenerator


class AntecedentTaskGenerator(BaseTaskGenerator):
    task_name = "antecedent"
    answer_type = "category"

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
            app_x, app_y = apps[0], apps[1]

            # Either app can be the asked-about target -- pick at random
            target_label = app_x if rng.integers(0, 2) == 0 else app_y
            other_label = app_y if target_label == app_x else app_x

            bg = self.bg_sampler.sample(
                target=target_label, context_length_s=ctx_s, rng=rng,
                margin_s=margin_samples * self.dt_s,
                extra_off_targets=[other_label],
            )
            if bg is None:
                continue

            needle_x = self.needle_sampler.sample(
                appliance=app_x, max_duration_s=max_dur_s / 3, rng=rng,
                require_house_id=(None if self.allow_cross_house else bg.house_id),
                allow_cross_house=self.allow_cross_house,
            )
            needle_y = self.needle_sampler.sample(
                appliance=app_y, max_duration_s=max_dur_s / 3, rng=rng,
                require_house_id=(None if self.allow_cross_house else bg.house_id),
                allow_cross_house=self.allow_cross_house,
            )
            if needle_x is None or needle_y is None:
                continue

            ctx_samples = bg.mains_w.shape[0]
            positions = sample_positions(
                ctx_samples,
                [needle_x.submeter_w.shape[0], needle_y.submeter_w.shape[0]],
                margin_samples=margin_samples,
                min_gap_samples=min_gap_samples, rng=rng,
            )
            if positions is None:
                continue

            pos_x, pos_y = positions
            # The asked-about target must be the LATER one (so an antecedent
            # exists). Re-pick target_label based on actual ordering.
            if pos_x < pos_y:
                target_label, other_label = app_y, app_x
            else:
                target_label, other_label = app_x, app_y

            question, answer = self.template_bank.render(
                "antecedent",
                {"target": target_label, "k": 1, "antecedent": other_label},
                rng,
            )

            return insert_needles(
                bg, [needle_x, needle_y], positions,
                task_type=self.task_name,
                question=question, answer=answer,
                answer_type=self.answer_type,
                difficulty_config={
                    "target": target_label, "predecessor": other_label, "k": 1,
                },
            )
        return self._invalid(ctx_s, "could not place pair")
