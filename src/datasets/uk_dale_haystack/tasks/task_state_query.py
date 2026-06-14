"""
State query. NO needle inserted. Picks a window with >= 2 natural bouts of
distinct appliances overlapping in time, then asks "what appliance was running
when the K-th `{anchor}` bout occurred?".
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from src.datasets.uk_dale_haystack.core.activity_regimes import V1_VOCAB
from src.datasets.uk_dale_haystack.core.data_structures import GeneratedSample
from src.datasets.uk_dale_haystack.core.insertion import insert_needles
from src.datasets.uk_dale_haystack.tasks.base_task import BaseTaskGenerator


class StateQueryTaskGenerator(BaseTaskGenerator):
    task_name = "state_query"
    answer_type = "category"

    def generate_sample(self, ctx_s, rng):
        params = self._ctx_params(ctx_s)
        margin_samples = params["margin_samples"]

        # Pick a "fictional target" the bg sampler should ensure is OFF, then
        # use the natural other_bouts to find an overlap pair.
        targets = list(self.bg_sampler.appliances_with_bouts)
        rng.shuffle(targets)

        for off_target in targets:
            for _ in range(5):
                bg = self.bg_sampler.sample(
                    target=off_target, context_length_s=ctx_s, rng=rng,
                    margin_s=margin_samples * self.dt_s,
                    min_other_bouts=2,
                )
                if bg is None:
                    break
                pair = self._find_overlap_pair(bg.other_bouts)
                if pair is None:
                    continue
                anchor_ref, state_ref, k = pair

                t_mid_s = (anchor_ref.start_sample + anchor_ref.end_sample) * self.dt_s / 2.0

                question, answer = self.template_bank.render(
                    "state_query",
                    {
                        "anchor": anchor_ref.appliance,
                        "k": k,
                        "state_appliance": state_ref.appliance,
                        "t0_s": t_mid_s,
                    },
                    rng,
                )

                return insert_needles(
                    bg, [], [],
                    task_type=self.task_name,
                    question=question, answer=answer,
                    answer_type=self.answer_type,
                    difficulty_config={
                        "anchor": anchor_ref.appliance,
                        "state_appliance": state_ref.appliance,
                        "k": k,
                    },
                )
        return self._invalid(ctx_s, "no overlapping natural bouts found")

    def _find_overlap_pair(self, refs):
        """Return (anchor_ref, state_ref, k) where state overlaps anchor's midpoint
        and is a different appliance. k = anchor's ordinal among same-appliance bouts.
        """
        if len(refs) < 2:
            return None
        # Group by appliance
        by_app = defaultdict(list)
        for r in refs:
            by_app[r.appliance].append(r)
        for app, group in by_app.items():
            group.sort(key=lambda r: r.start_sample)
            for k_idx, anchor in enumerate(group):
                mid = (anchor.start_sample + anchor.end_sample) // 2
                for r in refs:
                    if r.appliance == app:
                        continue
                    if r.start_sample <= mid < r.end_sample:
                        return anchor, r, k_idx + 1
        return None
