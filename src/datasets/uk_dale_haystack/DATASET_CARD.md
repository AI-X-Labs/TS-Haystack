# UK-DALE-Haystack — Dataset Card

A controlled additive-needle benchmark for time-series language models
adapted from UK-DALE (Kelly & Knottenbelt, 2015), the canonical UK domestic
appliance-level + whole-house power demand dataset.

## Source

| Field | Value |
|---|---|
| Source dataset | UK-DALE (NILMTK HDF5 export) |
| Local artifact | `data/uk_dale/ukdale.h5` |
| sha256 | `5cb9c07ec0da0ac4cc9e1a6ee6a7ba76561f4fe4367e6de78addaa2a63432092` |
| Size | 5.896 GB |
| Citation | Kelly J., Knottenbelt W., "The UK-DALE dataset, domestic appliance-level electricity demand and whole-house demand from five UK homes," *Sci. Data* 2:150007, 2015 |

The HDF5 source is pinned by sha256 in `source_manifest.json` so any
regeneration can verify the source has not changed.

## Houses

UK-DALE provides 5 instrumented homes; v1 keeps 3 of them.

| House | Status | Reason |
|---|---|---|
| 1 | **kept** | 53 appliance-bearing meters, 4.5-year recording (2012-W45 → 2017-W17), 233 ISO-weeks |
| 2 | **kept** | 18 appliance meters, 8-month recording (2013-W08 → 2013-W41), 34 weeks |
| 3 | dropped | only 4 appliance meters, 6-week recording — too short, too few meters |
| 4 | dropped | metering is circuit-bundled (`tv_dvd_digibox_lamp`, `washing_machine_microwave_breadmaker`) — no per-appliance ground truth |
| 5 | **kept** | 22 appliance meters, 5-month recording (2014-W27 → 2014-W46), 20 weeks |

**Mains channel:** `meter1` of each house — the 6-second active-power
mains. NILMTK metadata marks `meter1` as `disabled=True` for processing
reasons, but the data table is intact and is the canonical 6 s active-power
channel. The other site meter in each house (`meter54`/`m20`/`m26`) is the
1 s sound-card-derived apparent-power mains; it's a v2 candidate, not v1.

## Activity vocabulary (10 appliances, 4 regimes)

The plan called for 11 appliances; the v1 inventory is 10 because
`freezer` does not exist as a standalone meter in any kept house (only
`fridge_freezer`).

The plan also reserved `oven` as cooking-regime; we additionally **exclude
gas-driven appliances** (`original_name` starts with `gas_*`) because their
electric meter only sees the control circuit, not the heat. House 1 m42 is
labelled `type='oven'` but `original_name='gas_oven'` (max ~1 kW from the
igniter, p95 = 3 W) — semantically a different appliance. After this
filter, oven is house-5-only.

| Regime | Appliances | Per-house availability |
|---|---|---|
| **impulse**   | kettle, microwave, toaster, hair_dryer | kettle: {2,5}, microwave: {1,2,5}, toaster: {2,5}, hair_dryer: {1,5} |
| **long_cycle**| washing_machine, dishwasher, washer_dryer | washing_machine: {2}, dishwasher: {1,2,5}, washer_dryer: {1,5} |
| **cooking**   | oven                                   | oven: {5} |
| **refrig**    | fridge, fridge_freezer                  | fridge: {2}, fridge_freezer: {1,5} |

Per-house imbalances (washing_machine = h2-only, fridge = h2-only,
washer_dryer = h1+h5-only, oven = h5-only) are accepted as availability
constraints. The `same-house pairing` policy (Phase 5) means a kettle bout
from h5 only ever appears in an h5 background — these imbalances do not
introduce learnable cross-house transfer shortcuts but do limit context
diversity for the affected appliances.

## Splits

(house, ISO-week) → 80/10/10 train/val/test, seed=42.

| Split | n weeks | n bouts |
|---|---:|---:|
| train | 232 | 46,286 |
| validation | 29 | 5,471 |
| test | 29 | 5,915 |
| **total** | **290** | **57,672** |

Stored in `split_manifest.json`. The bout index is split-pinned (the
`split` column is computed once at index-build time) so all downstream
samplers see consistent assignments.

## Bout index

`bout_index.parquet` — one row per ON-event after contextual-ON hysteresis
extraction (see `core/bout_extractor.py` §Phase 2).

Per-appliance summary across all 290 weeks:

| Appliance | n bouts | duration p10/50/90 (s) | mean peak (W) | total (kWh) |
|---|---:|---:|---:|---:|
| fridge_freezer | 40,572 | 617 / 1396 / 2379 | 280 | 1518.7 |
| microwave | 8,089 | 25 / 52 / 247 | 1646 | 263.0 |
| fridge | 3,829 | 727 / 918 / 1745 | 328 | 113.6 |
| hair_dryer | 1,981 | 18 / 48 / 141 | 1294 | 40.2 |
| washer_dryer | 1,289 | 3556 / 5397 / 6360 | 2546 | 938.5 |
| kettle | 985 | 78 / 163 / 200 | 3128 | 114.3 |
| dishwasher | 779 | 2955 / 5107 / 6001 | 2655 | 1083.5 |
| oven | 58 | 358 / 950 / 2382 | 2387 | 31.4 |
| washing_machine | 57 | 2179 / 2414 / 2661 | 2417 | 23.1 |
| toaster | 33 | 30 / 98 / 163 | 1117 | 0.9 |

Validated empirically against the plan's reference: contextual-ON applied
to h1 m5 (washer_dryer) over the first 14 days yields 7 bouts of 93–97
min, peak 2.1–3.9 kW, mean ~470 W — matches the plan's expected "8 bouts
of 92–99 min, peak 2.1–3.9 kW, mean ~470 W" within the bout-on-day-edge
tolerance.

## Insertion model

**Additive, no style transfer, no blending.** For each generated sample:

```
mains_with_target = background_mains + needle_submeter
```

- `background_mains` = real 6 s active-power mains from a target-OFF
  window of the same house (sampled by `BackgroundSampler`, no synthesis).
- `needle_submeter` = real per-appliance submeter excerpt of one ON-event,
  kept at its native 6 s rate and true duration (sampled by
  `NeedleSampler`, no trimming, no scaling).
- Sum is computed at the regular 6 s grid (irregular submeter timestamps
  are nearest-neighbour resampled by `loader.load_meter_window_grid`).
- **No mean-shift** is applied: outside the bout, the inserted mains is
  exactly the target-OFF background; inside the bout, it's `background +
  submeter` sample-by-sample.
- **No edge blending**: bout boundaries are a sharp transition in a single
  sample. The plan deliberately drops the cosine cross-fade ("no boundary
  to blend"). Visual confirmation in
  `data/uk_dale/inspect/insertion_validation/`.
- **Same-house pairing** (default): a kettle bout from h5 only ever
  appears in an h5 background. `allow_cross_house=True` is available as
  an explicit ablation flag.
- **Anomaly synthesis** (anomaly_detection / anomaly_localization tasks):
  `truncated_cycle` (long_cycle + cooking, trim to t ∈ [0.20, 0.50] of
  duration) and `abnormal_peak` (impulse + cooking, scale by s ∈ [1.5,
  2.5], capped at the 3.5 kW UK ring-main ceiling).

## Tasks

10 task generators sharing a common `BaseTaskGenerator`:

| Task | Answer type | Mechanism |
|---|---|---|
| existence | boolean | 50/50 yes/no by inserting target (+) or same-regime distractor (-) |
| localization | time_range | Insert N target needles + optional distractor; ask for the K-th |
| counting | integer | Insert N ∈ {0..5} target needles; answer = N |
| ordering | boolean | Insert one A and one B; answer derived from sampled positions |
| antecedent | category | Insert two needles; ask what came before the later one |
| comparison | time_range | Insert ≥ 2 target needles with distinct durations/peaks; ask which is longest/shortest/highest-peak |
| multi_hop | time_range | Insert J anchors + K targets; ask for the K-th target after the J-th anchor |
| state_query | category | No insertion; pick window with ≥ 2 natural overlapping bouts; ask which other appliance was on during the K-th anchor |
| anomaly_detection | boolean | Insert nominal (-) or synthesized anomalous (+) bout |
| anomaly_localization | time_range | Always positive; ask when the anomalous bout occurred |

Generated under `data/uk_dale/uk_dale_haystack/tasks/{ctx}s/{task}/{split}/data.parquet`.
Default context lengths: 900, 3600, 7200, 32400, 86400 s (15 min, 1 h,
2 h, 9 h, 24 h). Default samples per (task, ctx, split): 1000 train, 150
val, 150 test.

The shard is **lightweight metadata only** — `mains_w` is reconstructed
on demand from `background_house_id` + `background_start_ns/end_ns` +
`needles_json` (see `plot_generator.reconstruct_sample_signal`).

## Validation

Five diagnostics live under
`data/uk_dale/uk_dale_haystack/diagnostics/`. The first three measure
how distinguishable the synthetic distribution is from the natural
distribution; the fourth is an attempted-mitigation ablation; the fifth
is the practical question — does a classifier trained on inserted data
recognise appliances in natural data?

### 1. Discrimination probes (XGBoost on raw mains/submeter time series)

Three modes via `validate_insertion.py`:

- **Placebo** (`baseline_mode=natural_natural`) — natural-vs-natural with
  arbitrary labels. Expected AUC = 0.5; deviation surfaces classifier-
  pipeline bias. Result: max |AUC − 0.5| = 0.09 across {kettle,
  microwave, fridge_freezer} × {mains, submeter}. **PASS** — the probe
  pipeline is unbiased.

- **Insertion** (`baseline_mode=insertion`) — natural vs (target-OFF BG +
  bout). Plan's blocking gate: AUC ≤ 0.80. Result on **mains**:
  0.74–0.78. **PASS** the 0.80 gate.

- **Null** (`baseline_mode=empty_bg`) — natural-with-bout vs target-OFF
  BG with no bout. Result on mains: 0.82–0.99 (kettle/microwave ≈ 1.0).
  The insertion probe drops AUC by 0.16 mains relative to this null —
  confirming the additive insertion is doing its work (closes 0.16 AUC of
  the natural-vs-empty gap by putting the same bout in both classes).

The submeter probe is **diagnostic-only** (the model only ever sees
mains). Submeter AUC = 0.85–0.94 is driven by appliance temporal
clustering ("kettle, then more kettle") naturally absent from synthetic
insertions.

### 2. Isolated-insertion probe

`validate_isolated_insertion.py` strips out the temporal-clustering
shortcut by forcing both classes to contain exactly one bout per window.
Even with clustering removed, residual mains AUC = 0.78, submeter ≈ 0.91
— so the gap is more fundamental than "natural windows have repeated
bouts."

### 3. Insertion-fix ablation

`ablate_insertion_fixes.py` tested two hypothesised mitigations against
the isolated-insertion baseline (3 appliances × 2 channels):

| Variant | submeter AUC avg | mains AUC avg | Verdict |
|---|---:|---:|---|
| baseline | 0.91 | 0.78 | reference |
| buffer_30s (load bout ±30 s context) | 0.98 | 0.78 | **harmful** |
| buffer_120s (±120 s context) | 0.99 | 0.86 | **strongly harmful** |
| timematch_30d (BG within 30 d of bout source) | 0.90 | 0.77 | neutral |
| timematch_7d (BG within 7 d) | 0.91 | 0.79 | neutral |
| buffer_30s + timematch_7d | 0.96 | 0.80 | buffer harm dominates |

Neither fix closes the gap. The buffer fix actively makes things worse:
including bout-source-time off-state samples around the bout creates a
discontinuity at the buffer boundary (bout-source-time noise abruptly
ends, BG-time noise begins) that XGBoost picks up. Time-matching is
within ±0.02 of baseline — sensor calibration drift across the 4.5-year
recording is **not** the dominant signal.

The remaining gap appears to be structural: each natural-bout window
encodes the household's causal context at that moment, while inserted
windows stitch a bout into an arbitrary target-OFF period. Closing this
would require generating bouts from a learned model or sourcing BG and
bout from the same recording session — a v2 redesign.

### 4. Cross-distribution classifier transfer

`classifier_transfer_test.py` answers the actionable question:
**despite being statistically distinguishable, do natural and inserted
samples share enough structural appliance signature that a classifier
generalises?** It trains a 6-way XGBoost multi-class classifier on
**inserted** windows (train-split bouts only) and evaluates on:

- `test_inserted` (test-split bouts, same pipeline) — within-distribution
- `test_natural` (test-split bouts, real-mains/submeter window centred on
  the actual bout) — sim-to-real

Results (n=100/class train, n=25/class test, 600-sample windows):

| Appliance | mains inserted → natural | submeter inserted → natural |
|---|---|---|
| kettle | 1.00 → 0.96 | 1.00 → 1.00 |
| fridge_freezer | 1.00 → 0.92 | 1.00 → 1.00 |
| microwave | 0.72 → 0.68 | 1.00 → 0.96 |
| washer_dryer | 0.84 → 0.76 | 0.88 → 0.76 |
| hair_dryer | 0.48 → 0.32 | 0.88 → 0.60 |
| **dishwasher** | 0.96 → **0.00** | 1.00 → **0.00** |
| **OVERALL** | **0.83 → 0.61** | **0.96 → 0.72** |

**Per-regime takeaways:**

- **impulse (kettle, microwave) and refrig (fridge_freezer):** synthetic
  insertion preserves the appliance signature; classifier transfers
  cleanly (Δ ≤ 0.10).
- **hair_dryer:** weak even within-distribution (only 1.9k bouts and
  power-band overlap with microwave); transfer drop is mostly noise.
- **long_cycle (dishwasher, washer_dryer):** mutually confused on
  natural samples. Confusion matrix shows 22/25 natural dishwashers
  predicted as **washer_dryer** (submeter), 25/25 misclassified
  (mains). The trained model learns a brittle dishwasher signature
  from inserted data — natural windows likely capture more of the
  surrounding cycle pattern (the bout extends past the 1 h window edge
  for ~90 min cycles), while inserted windows have the bout cleanly
  centred with target-OFF flanks.

**Implication for v1.x → v2:** prioritise long_cycle representation
(longer per-class windows, multi-segment insertion, or BG selection
matching natural high-baseline mains) before chasing the smaller
impulse/refrig gap.

### Reproducing the validation suite

```bash
.venv/bin/python -m src.datasets.uk_dale_haystack.scripts.validate_insertion
.venv/bin/python -m src.datasets.uk_dale_haystack.scripts.validate_isolated_insertion
.venv/bin/python -m src.datasets.uk_dale_haystack.scripts.ablate_insertion_fixes
.venv/bin/python -m src.datasets.uk_dale_haystack.scripts.classifier_transfer_test
.venv/bin/python -m src.datasets.uk_dale_haystack.scripts.plot_insertion_validation
```
Diagnostics land in `data/uk_dale/uk_dale_haystack/diagnostics/`; visual
companion plots under `data/uk_dale/inspect/insertion_validation/`.

## Known limitations

1. **Per-week splits leak weekly routine.** Splitting at (house, ISO-week)
   rather than calendar-month means train/test contain similar
   day-of-week patterns. A house-level split would address this but
   reduce h2 and h5 to a single split each (their recordings span <1
   year).

2. **Mains gaps > 24 s.** UK-DALE has gaps where the mains meter
   disconnected. The conversion manifest enumerates them; the BG sampler
   rejects windows that span a gap, but bouts at gap boundaries may have
   slightly under-counted total energy. Affects all houses; densest in h1
   (13,397 gaps over 4.5 years).

3. **Appliance-by-house imbalance.** As noted above: washing_machine,
   fridge, and oven are each present in only one house. With same-house
   pairing the inserted-bout pool for these is a single house's
   distribution. For cross-house generalisation studies, use
   `allow_cross_house=True` as an explicit ablation.

4. **Sim-to-real gap (probe AUC ~0.77 mains).** A model trained on this
   synthetic distribution will not generalise cleanly to actual real-world
   UK-DALE mains (where bouts are causally embedded in correlated
   surrounding activity). For the closed benchmark this gap is irrelevant
   — the model is consistent across its own world. The validation suite's
   classifier-transfer test (§Validation #4) shows the gap is **regime-
   dependent**: impulse and refrig classes transfer cleanly (Δ ≤ 0.10),
   but long_cycle classes (dishwasher, washer_dryer) are mutually
   confused on natural data. The insertion-fix ablation (§Validation #3)
   confirmed that per-bout buffer extension and time-matched BG
   selection do **not** close this gap — it is structural, not a tuning
   issue. v2 mitigations (longer windows for long_cycle, multi-segment
   insertion, or BG selection matched to natural high-baseline mains)
   are the actionable next steps for real-world deployment.

5. **`gas_oven` filter loses h1's only oven-like meter.** The
   `EXCLUDED_ORIGINAL_PREFIXES = ('gas_',)` filter in the manifest
   builder rejects it because its electric power signature
   (~50 W controls) does not match the cooking regime semantics. h1 has
   no electric oven; oven is h5-only.

6. **Negative power excursions are not clipped.** The mains active-power
   channel can occasionally show small negative spikes (PV export,
   reactive load mis-calibration). v1 leaves them as-is; the additive
   sum may briefly cross zero. Documented but rare; not currently
   tracked per-sample.

## Reproducibility

```bash
# Phase 0: source pin + splits (~15 s for the sha256 hash)
.venv/bin/python -m scripts.data.uk_dale.build_uk_dale_manifest

# Phase 1: HDF5 -> npy memmap sidecars (~2 min, 1.7 GB output)
.venv/bin/python scripts/data/uk_dale/convert_uk_dale_to_npy.py

# Phase 2.2: build the bout index (~3 s)
.venv/bin/python scripts/data/uk_dale/build_uk_dale_bout_index.py --overwrite

# Phase 8: full task generation (defaults: 5 ctx x 10 tasks x 3 splits)
.venv/bin/python scripts/data/uk_dale/build_uk_dale_haystack.py --overwrite

# Phase 9.2: per-(task, ctx) sample inspection plots
.venv/bin/python -m src.datasets.uk_dale_haystack.scripts.verify_uk_dale_haystack \
    --n-samples 3

# Phase 9.3: shortcut probe (placebo + insertion + null on mains + submeter)
.venv/bin/python -m src.datasets.uk_dale_haystack.scripts.validate_insertion
```

Determinism: every sample is generated with `rng = default_rng(seed)`
where `seed = md5(master_seed | task | ctx_s | split | sample_idx)`. Two
generation runs with the same `master_seed` produce identical parquet
shards (verified by `tests/integration/test_uk_dale_reproducibility.py`).

## File structure

```
src/datasets/uk_dale_haystack/
├── DATASET_CARD.md                 (this file)
├── __init__.py
├── loader.py                       npy memmap accessors + window grid resampling
├── plot_generator.py               per-sample inspection plot
├── core/
│   ├── activity_regimes.py         RAW_TO_CANONICAL, REGIMES, BOUT_DEFAULTS, MAX_POWER_W
│   ├── bout_extractor.py           contextual-ON hysteresis with absorption
│   ├── data_structures.py          BoutRecord/BoutRef/BackgroundSample/NeedleSample/InsertedNeedle/GeneratedSample
│   ├── background_sampler.py       target-OFF (or allow_target_on) windows from bout-index complement
│   ├── needle_sampler.py           same-house pairing (default), allow_cross_house ablation
│   ├── insertion.py                additive sum + position sampler
│   └── prompt_templates.py         APPLIANCE_VOCAB synonyms + 6-10 templates per task
├── tasks/
│   ├── base_task.py                margin/gap defaults, _ctx_params helper
│   ├── task_existence.py
│   ├── task_localization.py
│   ├── task_counting.py
│   ├── task_ordering.py
│   ├── task_antecedent.py
│   ├── task_comparison.py
│   ├── task_multi_hop.py
│   ├── task_state_query.py
│   ├── task_anomaly_detection.py   (truncated_cycle, abnormal_peak synthesis)
│   ├── task_anomaly_localization.py
│   └── __init__.py                 TASK_REGISTRY
├── generation/
│   ├── config.py                   YAML loader -> GenerationConfig
│   ├── defaults.yaml
│   └── generator.py                (task, ctx, split) shard orchestrator
└── scripts/
    ├── verify_uk_dale_haystack.py  per-sample inspection plot generator
    ├── probe_insertion_shortcut.py probe engine (3 baseline modes, 2 channels)
    ├── validate_insertion.py       clean placebo + insertion comparison report
    └── plot_insertion_validation.py natural-vs-inserted pair plots for human review

scripts/data/uk_dale/
├── build_uk_dale_manifest.py
├── convert_uk_dale_to_npy.py
├── build_uk_dale_bout_index.py
└── build_uk_dale_haystack.py       (thin wrapper over generation/generator.py)

tests/
├── unit/test_datasets/test_uk_dale_bout_extractor.py
├── unit/test_datasets/test_uk_dale_tasks.py
├── integration/test_uk_dale_reproducibility.py
└── integration/test_uk_dale_insertion_shortcut.py  (placebo + mains gates)
```
