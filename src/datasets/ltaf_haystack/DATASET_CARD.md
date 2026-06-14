# LTAF-Haystack Dataset Card

LTAF-Haystack is a windowed-natural QA benchmark built on top of the
Long-Term Atrial Fibrillation (LTAF) PhysioNet database. It probes a
time-series language model's ability to answer rhythm- and beat-level
questions over two-lead ECG windows up to two hours. Every sample is an
unmodified slice of a real recording — no synthesis, no splicing, no
style transfer.

**Hosted copies on Hugging Face Hub:**

- [`nz00shuuuu/ltaf-raw`](https://huggingface.co/datasets/nz00shuuuu/ltaf-raw) — per-record `.npy` signal cache (float32 `(N, 2)`) plus `conversion_manifest.json`. Lands at `data/ltafdb/training/` and is memmap-sliced at `__getitem__` time.
- [`nz00shuuuu/ltaf-haystack`](https://huggingface.co/datasets/nz00shuuuu/ltaf-haystack) — QA parquet shards under `rhythms/tasks/`, timelines, beat timelines, window indices, bout index, split manifest. Parquets carry metadata only (`record_id`, `window_start_ms`, `window_end_ms`, question, answer, …) — no signal columns.

## Quickstart

```bash
# 1. Pull raw signal cache (~7 GB, 85 records) → data/ltafdb/training/
.venv/bin/python scripts/data/download_from_hf.py --dataset ltaf-raw

# 2. Pull QA annotations + timelines → data/ltafdb/ltaf_haystack/
.venv/bin/python scripts/data/download_from_hf.py --dataset ltaf-haystack

# 3. Train
.venv/bin/python main.py \
  --config configs/ltaf_haystack/flamingo_chronos2_llama_ltaf_haystack.yaml
```

Both downloads are required: the QA parquets store only window coordinates, and the loader (`LTAFHaystackQADataset._get_text_time_series_prompt_list`) hydrates signals from the `.npy` cache on demand.

## 1. Source and composition

- **Source database.** PhysioNet Long-Term Atrial Fibrillation (LTAF):
  84 long-term (~24 h) two-lead ECG recordings at 128 Hz.
- **Derived artifacts.**
  - `.npy` signal cache (~7 GB) — real waveforms, `float32 (N, 2)`.
  - Per-record rhythm timelines (Parquet) — bout ranges in sample coords.
  - Per-record beat timelines (Parquet) — N/A/V/Q beat annotations with
    sample + time_ms coords.
  - Per-context window indices (JSON) — presence bitmasks + beat counts
    per window, built once per context length.
- **Split.** 80 / 10 / 10 at **record level** (no patient leakage),
  deterministic seed 42 → 67 train / 8 validation / 9 test recordings.
- **Generation.** For each (task, context_length, split) the orchestrator
  draws random `(record, window_start)` pairs from the window index and
  builds a QA from whatever rhythm/beat annotations fall inside. Rare
  rhythms (T, VT, IVR) naturally produce fewer samples at shorter
  contexts; thin shards are accepted rather than topped up with
  synthesis. The generator emits per-shard counts in `metadata.json`.

### Rhythm bout counts in the LTAF subset

| Rhythm | Bouts |
|--------|------:|
| NSR    | 22 879 |
| SBR    | 11 326 |
| AFIB   |  7 358 |
| AB     |  4 472 |
| SVTA   |  3 268 |
| B      |  2 696 |
| VT     |    828 |
| T      |    785 |
| IVR    |    137 |

## 2. Rhythm and beat taxonomy

**Rhythm codes (PhysioNet LTAF / AHA conventions)**

| Code | Expansion |
|------|-----------|
| NSR  | Normal sinus rhythm |
| AFIB | Atrial fibrillation |
| SBR  | Sinus bradycardia (<60 bpm, sinus origin) |
| AB   | Atrial bigeminy (every other beat is an APC) |
| B    | Ventricular bigeminy (every other beat is a PVC) |
| T    | Ventricular trigeminy (every third beat is a PVC) |
| SVTA | Supraventricular tachyarrhythmia (≥3 consecutive SV ectopics @ >100 bpm) |
| VT   | Ventricular tachycardia (≥3 consecutive PVCs @ >100 bpm) |
| IVR  | Idioventricular rhythm (ventricular escape, typically <60 bpm, brief) |

**Beat codes (AHA)**

| Code | Expansion |
|------|-----------|
| N | Normal sinus-origin beat |
| A | Atrial premature contraction (APC / PAC / SVE) |
| V | Ventricular premature contraction (PVC / VE) |
| Q | Unclassifiable or paced beat |

### Known absences

- **AFL (atrial flutter): 0 episodes** in the 84-record subset. Models
  trained on this dataset will **not** learn to recognize flutter. The
  label is retained in the regime enumeration for forward compatibility
  but is never used during generation.
- **Paced records.** The maximum observed `Q` beat ratio across the
  84 records is ~0.06 %, so the paced-records flag in the split manifest
  is effectively empty. Retained for documentation.

## 3. Intended use

- **Evaluation of ECG-aware time-series language models** on rhythm and
  beat QA with ten task types:
  existence, localization, counting, ordering, state_query, antecedent,
  comparison, multi_hop, anomaly_detection, anomaly_localization.
- **Not** a clinical decision-support training set. Not FDA-cleared.
  Not suitable for triage, diagnosis, or any patient-facing use.

### Task × context coverage

`✓` = generated, blank = gated off.

| Task | 100s | 900s | 1h | 2h |
|---|:-:|:-:|:-:|:-:|
| existence            | ✓ | ✓ | ✓ |   |
| localization         | ✓ | ✓ | ✓ | ✓ |
| counting             | ✓ | ✓ | ✓ | ✓ |
| ordering             | ✓ | ✓ | ✓ | ✓ |
| antecedent           | ✓ | ✓ | ✓ | ✓ |
| comparison           | ✓ | ✓ | ✓ | ✓ |
| multi_hop            | ✓ | ✓ | ✓ | ✓ |
| state_query          | ✓ | ✓ | ✓ | ✓ |
| anomaly_detection    | ✓ | ✓ |   |   |
| anomaly_localization | ✓ | ✓ | ✓ | ✓ |

Rationale for the gates: `existence` at 2h saturates (every 2h LTAF
window contains most rhythms); `anomaly_detection` at 1h+ saturates to
"yes" given LTAF ectopy density. 10s was dropped entirely — too few
bouts and beats land inside to ask anything meaningful.

## 4. Known limitations

- **Thin rare-rhythm coverage.** At 100s, rhythms like T, VT, IVR
  produce far fewer samples than NSR/AFIB/SBR. This is surfaced in the
  per-shard counts in `metadata.json`. Downstream evaluators should
  report metrics split by `(task, context, target_rhythm)` and not
  expect uniform sample counts.
- **No hemodynamic channels.** Two leads only; no respiratory belts, no
  SpO₂. Baseline wander and respiratory coupling are present in the
  signal but not tagged.
- **No inter-patient temporal structure.** Each window comes from one
  recording at one time; the benchmark does not test reasoning across
  multiple recordings or across a patient's history.
- **Label noise.** Rhythm bout boundaries in the LTAF annotations were
  produced by the original curators and inherit whatever annotator
  disagreement existed in the source data.

## 5. Validation artifacts

- **Verification plots** (`rhythms/verification/`) — per-task PNGs with
  rhythm bands, beat markers, answer highlights, and Q/A text.
- **Unit + integration tests** (`tests/unit/test_datasets/test_ltaf_*.py`
  and `tests/integration/test_ltaf_*.py`) covering:
  - Task registry + natural-only dispatch.
  - Coverage-table gating per (task, context).
  - Generation reproducibility under a fixed seed (byte-identical
    parquet shards across two runs).
  - Training-smoke integration.

## 6. Reproducibility

- **Conversion manifest** (`data/ltafdb/training/conversion_manifest.json`)
  records SHA-256s for every source record's cached `.npy`.
- **Split seed** 42; **train ratio** 0.8 / 0.1 / 0.1.
- **Stride policy** `max(1s, min(ctx_s/4, 30 min))`.
- **Generation command**:
  ```bash
  .venv/bin/python scripts/data/build_ltaf_haystack.py --overwrite
  ```
- **Determinism.** Running the generator twice with the same config
  produces byte-identical parquet shards (enforced by
  `tests/integration/test_ltaf_generation_reproducibility.py`).

## 7. Citation

When publishing results on LTAF-Haystack, please cite the PhysioNet
LTAF database in addition to the Anon-TSLM project:

```bibtex
@misc{petrutiu2008ltafdb,
  title         = {Abrupt Changes in Fibrillatory Wave Characteristics at the Termination of Paroxysmal Atrial Fibrillation in Humans},
  author        = {Petrutiu, Simona and Sahakian, Alan V. and Swiryn, Steven},
  year          = {2008},
  howpublished  = {PhysioNet},
  url           = {https://physionet.org/content/ltafdb/}
}
```
