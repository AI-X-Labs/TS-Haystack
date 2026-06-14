"""
Anomaly localization. Same setup as detection but always positive: ask when
the anomalous bout occurred.
"""
from __future__ import annotations

import numpy as np

from src.datasets.uk_dale_haystack.core.insertion import (
    insert_needles,
    sample_positions,
)
from src.datasets.uk_dale_haystack.tasks.base_task import BaseTaskGenerator
from src.datasets.uk_dale_haystack.tasks.task_anomaly_detection import (
    _anomaly_classes_for,
    make_anomaly,
)


class AnomalyLocalizationTaskGenerator(BaseTaskGenerator):
    task_name = "anomaly_localization"
    answer_type = "time_range"

    def generate_sample(self, ctx_s, rng):
        params = self._ctx_params(ctx_s)
        max_dur_s = params["max_needle_duration_s"]
        margin_samples = params["margin_samples"]
        min_gap_samples = params["min_gap_samples"]

        targets = [a for a in self.bg_sampler.appliances_with_bouts
                   if _anomaly_classes_for(a)]
        rng.shuffle(targets)

        for target in targets:
            bg = self.bg_sampler.sample(
                target=target, context_length_s=ctx_s, rng=rng,
                margin_s=margin_samples * self.dt_s,
            )
            if bg is None:
                continue

            needle = self.needle_sampler.sample(
                appliance=target, max_duration_s=max_dur_s, rng=rng,
                require_house_id=(None if self.allow_cross_house else bg.house_id),
                allow_cross_house=self.allow_cross_house,
            )
            if needle is None:
                continue
            anom = make_anomaly(needle, rng)
            if anom is None:
                continue

            # Optionally insert a nominal distractor of the same appliance
            distractors = []
            distractor_needle = self.needle_sampler.sample(
                appliance=target, max_duration_s=max_dur_s / 2, rng=rng,
                require_house_id=(None if self.allow_cross_house else bg.house_id),
                allow_cross_house=self.allow_cross_house,
            )
            if distractor_needle is not None:
                distractors.append(distractor_needle)

            all_needles = [anom] + distractors
            ctx_samples = bg.mains_w.shape[0]
            positions = sample_positions(
                ctx_samples, [n.submeter_w.shape[0] for n in all_needles],
                margin_samples=margin_samples,
                min_gap_samples=min_gap_samples, rng=rng,
            )
            if positions is None:
                continue

            anom_pos = positions[0]
            t0_s = anom_pos * self.dt_s
            t1_s = (anom_pos + anom.submeter_w.shape[0]) * self.dt_s

            question, answer = self.template_bank.render(
                "anomaly_localization",
                {"t0_s": t0_s, "t1_s": t1_s, "anomaly_class": anom.anomaly_class},
                rng,
            )

            return insert_needles(
                bg, all_needles, positions,
                task_type=self.task_name,
                question=question, answer=answer,
                answer_type=self.answer_type,
                difficulty_config={
                    "target": target, "anomaly_class": anom.anomaly_class,
                    "anomaly_params": anom.anomaly_params,
                    "n_distractor_nominals": len(distractors),
                },
            )
        return self._invalid(ctx_s, "no anomaly-eligible targets fit")
