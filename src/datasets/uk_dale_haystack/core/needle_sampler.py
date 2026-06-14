"""
Sample real per-appliance ON-events from the bout index for additive insertion.

Default policy: same-house pairing -- a kettle bout from house 5 only ever
appears in a house-5 background. Inter-house imbalance becomes an availability
constraint rather than a learnable transfer shortcut. ``allow_cross_house``
toggles to a parallel ablation set.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from src.datasets.uk_dale_haystack.core.activity_regimes import V1_VOCAB
from src.datasets.uk_dale_haystack.core.data_structures import NeedleSample
from src.datasets.uk_dale_haystack.loader import load_meter_window_grid, NOMINAL_DT_S


class NeedleSampler:
    """Sample one real submeter excerpt per call, filtered by duration / house."""

    def __init__(
        self,
        bout_index: pl.DataFrame,
        split: str,
        dt_s: float = NOMINAL_DT_S,
    ):
        self.split = split
        self.dt_s = float(dt_s)
        self._bouts = bout_index.filter(pl.col("split") == split)
        # Cache numpy views per (house, appliance) for fast filtering.
        self._cache: dict[tuple[int, str], dict[str, np.ndarray]] = {}
        # Per-appliance: which (house) candidates exist
        self._appliance_houses: dict[str, list[int]] = {
            app: sorted({int(h) for h in self._bouts.filter(pl.col("appliance") == app)["house_id"].unique().to_list()})
            for app in V1_VOCAB
        }

    def _cache_key(self, house: int, appliance: str) -> dict[str, np.ndarray]:
        key = (house, appliance)
        if key not in self._cache:
            df = (self._bouts
                  .filter((pl.col("house_id") == house) & (pl.col("appliance") == appliance))
                  .select(["meter_id", "start_ns", "end_ns",
                           "duration_s", "peak_w", "mean_w"])
                  .sort("start_ns"))
            self._cache[key] = {
                "meter_id":   df["meter_id"].to_numpy().astype("int64"),
                "start_ns":   df["start_ns"].to_numpy().astype("int64"),
                "end_ns":     df["end_ns"].to_numpy().astype("int64"),
                "duration_s": df["duration_s"].to_numpy().astype("float32"),
                "peak_w":     df["peak_w"].to_numpy().astype("float32"),
                "mean_w":     df["mean_w"].to_numpy().astype("float32"),
            }
        return self._cache[key]

    def has_bouts(self, appliance: str, house: int | None = None) -> bool:
        if house is None:
            return bool(self._appliance_houses.get(appliance))
        return bool(self._cache_key(house, appliance)["start_ns"].size)

    def sample(
        self,
        appliance: str,
        max_duration_s: float,
        rng: np.random.Generator,
        *,
        require_house_id: int | None = None,
        allow_cross_house: bool = False,
        min_duration_s: float = 0.0,
        exclude_bout_ids: set[tuple[int, int, int]] | None = None,
    ) -> NeedleSample | None:
        # Houses to draw from
        if require_house_id is not None and not allow_cross_house:
            houses = [int(require_house_id)] if require_house_id in self._appliance_houses.get(appliance, []) else []
        else:
            houses = list(self._appliance_houses.get(appliance, []))
        if not houses:
            return None

        rng.shuffle(houses)
        for house in houses:
            cache = self._cache_key(house, appliance)
            durations = cache["duration_s"]
            mask = (durations <= max_duration_s) & (durations >= min_duration_s)
            if exclude_bout_ids:
                bad = np.array([
                    (int(house), int(m), int(s)) in exclude_bout_ids
                    for m, s in zip(cache["meter_id"], cache["start_ns"])
                ], dtype="bool")
                mask &= ~bad
            candidates = np.flatnonzero(mask)
            if candidates.size == 0:
                continue
            idx = int(rng.choice(candidates))
            return self._materialise(house, appliance, cache, idx)
        return None

    def _materialise(
        self, house: int, appliance: str, cache: dict[str, np.ndarray], idx: int,
    ) -> NeedleSample:
        meter = int(cache["meter_id"][idx])
        start_ns = int(cache["start_ns"][idx])
        end_ns = int(cache["end_ns"][idx])
        # Pad the right edge by one dt so the last sample is included.
        dt_ns = int(self.dt_s * 1e9)
        submeter_w = load_meter_window_grid(
            house, meter, start_ns, end_ns + dt_ns, dt_s=self.dt_s,
        )
        return NeedleSample(
            source_house_id=house,
            source_meter_id=meter,
            appliance=appliance,
            start_ns=start_ns,
            end_ns=end_ns,
            duration_s=float(cache["duration_s"][idx]),
            submeter_w=submeter_w.astype("float32", copy=False),
            dt_s=self.dt_s,
            peak_w=float(cache["peak_w"][idx]),
            mean_w=float(cache["mean_w"][idx]),
        )
