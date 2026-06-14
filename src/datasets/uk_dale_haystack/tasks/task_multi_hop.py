"""
Multi-hop. Insert J anchors and K target bouts; ask "the K-th {target} after
the J-th {anchor}".
"""
from __future__ import annotations

import numpy as np

from src.datasets.uk_dale_haystack.core.insertion import (
    insert_needles,
    sample_positions,
)
from src.datasets.uk_dale_haystack.tasks.base_task import BaseTaskGenerator


class MultiHopTaskGenerator(BaseTaskGenerator):
    task_name = "multi_hop"
    answer_type = "time_range"

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
            anchor = apps[0]
            target = apps[1]
            n_anchors = int(rng.integers(1, 3))   # 1..2
            n_targets = int(rng.integers(2, 4))   # 2..3
            j = int(rng.integers(1, n_anchors + 1))
            k = int(rng.integers(1, n_targets + 1))
            direction = "after"   # v1: after only

            bg = self.bg_sampler.sample(
                target=anchor, context_length_s=ctx_s, rng=rng,
                margin_s=margin_samples * self.dt_s,
                extra_off_targets=[target],
            )
            if bg is None:
                continue

            needles_anchor = []
            for _ in range(n_anchors):
                nd = self.needle_sampler.sample(
                    appliance=anchor, max_duration_s=max_dur_s / 4, rng=rng,
                    require_house_id=(None if self.allow_cross_house else bg.house_id),
                    allow_cross_house=self.allow_cross_house,
                )
                if nd is not None:
                    needles_anchor.append(nd)
            needles_target = []
            for _ in range(n_targets):
                nd = self.needle_sampler.sample(
                    appliance=target, max_duration_s=max_dur_s / 4, rng=rng,
                    require_house_id=(None if self.allow_cross_house else bg.house_id),
                    allow_cross_house=self.allow_cross_house,
                )
                if nd is not None:
                    needles_target.append(nd)
            if len(needles_anchor) < n_anchors or len(needles_target) < n_targets:
                continue

            all_needles = needles_anchor + needles_target
            ctx_samples = bg.mains_w.shape[0]
            positions = sample_positions(
                ctx_samples, [n.submeter_w.shape[0] for n in all_needles],
                margin_samples=margin_samples,
                min_gap_samples=min_gap_samples, rng=rng,
            )
            if positions is None:
                continue

            anchor_pos = positions[:n_anchors]
            target_pos = positions[n_anchors:]

            # Order
            anchor_order = sorted(range(n_anchors), key=lambda i: anchor_pos[i])
            target_order = sorted(range(n_targets), key=lambda i: target_pos[i])

            j_anchor_pos = anchor_pos[anchor_order[j - 1]]
            after_target_indices = [i for i in target_order if target_pos[i] > j_anchor_pos]
            if len(after_target_indices) < k:
                continue
            chosen_target_local = after_target_indices[k - 1]
            t0_s = target_pos[chosen_target_local] * self.dt_s
            t1_s = (target_pos[chosen_target_local]
                    + needles_target[chosen_target_local].submeter_w.shape[0]) * self.dt_s

            question, answer = self.template_bank.render(
                "multi_hop",
                {"anchor": anchor, "j": j, "target": target, "k": k,
                 "direction": direction, "t0_s": t0_s, "t1_s": t1_s},
                rng,
            )

            return insert_needles(
                bg, all_needles, positions,
                task_type=self.task_name,
                question=question, answer=answer,
                answer_type=self.answer_type,
                difficulty_config={
                    "anchor": anchor, "target": target,
                    "j": j, "k": k, "direction": direction,
                    "n_anchors": n_anchors, "n_targets": n_targets,
                },
            )
        return self._invalid(ctx_s, "could not place anchors/targets")
