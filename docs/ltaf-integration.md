# LTAF Integration (Plan + Status + Test Results)

This is the single source of truth for the LTAF-Haystack integration in Anon-TSLM. It merges the implementation plan, completion status, and validated test results.

## 1. Objective

Integrate Long-Term AF Database (LTAF) as a TS-Haystack-style synthetic ECG benchmark (needle-in-a-haystack), not as a natural-window clinical classification dataset.

Primary goals:

- Build long-context ECG QA tasks via controlled insertion of rhythm/beat events into clean backgrounds.
- Prevent shortcut learning from transition priors and splice artifacts.
- Keep generation reproducible, auditable, and compatible with existing training/evaluation scripts.

## 2. Scope and Non-Goals

In scope:

- Offline artifact pipeline for LTAF ingestion, timeline extraction, and synthetic sample generation.
- Runtime parquet-backed dataset loader integration for train/eval.
- Config, registry, docs, and tests.

Out of scope:

- Medical-grade diagnostic claims.
- Replacing existing TS-Haystack Capture24 code paths.
- Real-time generation during training.

## 3. Architecture and Integration Points

Reference repo architecture:

- `scripts/data/build_core_artifacts.py`
- `scripts/data/build_ts_haystack.py`
- `src/datasets/ts_haystack/core/timeline_builder.py`
- `src/datasets/ts_haystack/core/bout_indexer.py`
- `src/datasets/ts_haystack/core/background_sampler.py`
- `src/datasets/ts_haystack/tasks/base_task.py`
- `src/datasets/qa_base.py`
- `src/datasets/registry.py`

LTAF implementation mirrors this split:

- Phase 1: core artifact construction
- Phase 2: synthetic benchmark generation
- Runtime: thin parquet loader + prompt formatting + evaluation helpers

## 4. Implemented File Layout

Core package:

- `src/datasets/ltaf_haystack/core/data_structures.py`
- `src/datasets/ltaf_haystack/core/ltaf_timeline_builder.py`
- `src/datasets/ltaf_haystack/core/ltaf_bout_indexer.py`
- `src/datasets/ltaf_haystack/core/ltaf_transition_matrix.py`
- `src/datasets/ltaf_haystack/core/ltaf_background_sampler.py`
- `src/datasets/ltaf_haystack/core/ltaf_needle_sampler.py`
- `src/datasets/ltaf_haystack/core/ltaf_splicer.py`
- `src/datasets/ltaf_haystack/core/ltaf_prompt_templates.py`
- `src/datasets/ltaf_haystack/core/ltaf_seed_manager.py`

Generation and tasks:

- `src/datasets/ltaf_haystack/generation/config.py`
- `src/datasets/ltaf_haystack/generation/default_generation_config.yaml`
- `src/datasets/ltaf_haystack/tasks/base_task.py`
- `src/datasets/ltaf_haystack/tasks/task_existence.py`
- `src/datasets/ltaf_haystack/tasks/task_localization.py`
- `src/datasets/ltaf_haystack/tasks/task_counting.py`
- `src/datasets/ltaf_haystack/tasks/task_ordering.py`
- `src/datasets/ltaf_haystack/tasks/task_state_query.py`
- `src/datasets/ltaf_haystack/tasks/task_antecedent.py`
- `src/datasets/ltaf_haystack/tasks/task_comparison.py`
- `src/datasets/ltaf_haystack/tasks/task_multi_hop.py`
- `src/datasets/ltaf_haystack/tasks/task_anomaly_detection.py`
- `src/datasets/ltaf_haystack/tasks/task_anomaly_localization.py`

Runtime integration:

- `src/datasets/ltaf_haystack/qa_loader.py`
- `src/datasets/ltaf_haystack/qa_dataset.py`
- `src/datasets/ltaf_haystack/cot_qa_dataset.py`

Scripts/config/docs:

- `scripts/data/download_ltaf.py`
- `scripts/data/build_ltaf_core_artifacts.py`
- `scripts/data/build_ltaf_haystack.py`
- `configs/ltaf_haystack/default_generation_config.yaml`
- `configs/experiments/ltaf_haystack_qwen.yaml`
- `configs/experiments/ltaf_haystack_llama.yaml`
- `configs/experiments/ltaf_haystack_llama_har_pretrained.yaml`

## 5. Data and Artifact Contracts

Raw data location:

- `data/ltafdb/raw/`

Phase 1 outputs:

- `data/ltafdb/ltaf_haystack/timelines/*.parquet`
- `data/ltafdb/ltaf_haystack/bout_index.parquet`
- `data/ltafdb/ltaf_haystack/transition_matrix.json`
- `data/ltafdb/ltaf_haystack/split_manifest.json`

Phase 2 outputs:

- `data/ltafdb/ltaf_haystack/tasks/{context}/{task}/{split}/data.parquet`
- `data/ltafdb/ltaf_haystack/tasks/metadata.json`

Per-sample parquet schema (minimum):

- `record_id`, `background_pid`, `context_length_samples`, `source_hz`
- `lead_1`, `lead_2`, `question`, `answer`, `answer_type`, `task_type`
- `needles`, `distractors`, `splice_qc`, `difficulty_config`, `is_valid`

## 6. Scientific Controls and Validation Requirements

Implemented controls:

- Harvesting by role: haystack backgrounds vs. target needles vs. distractors.
- Isoelectric/phase-aware splice search with objective-based boundary selection.
- QC rejection policy for splice mismatch thresholds.
- Anti-shortcut controls (distractors, negatives, balancing).

Important policy settings now applied:

- Existence task distractors require at least one distractor in both default and 32Hz oracle generation configs (`min_distractors: 1`).
- LTAF runtime evaluation overrides exact-string matching for time tasks and uses IoU-aware scoring with threshold 0.50.

## 7. Milestone Status

- Milestone 0 scaffolding: complete
- Milestone 1 ingestion + timeline build: complete
- Milestone 2 bout index + transition stats: complete
- Milestone 3 ECG splicer core + unit tests: complete
- Milestone 3 visual QC boundary checks: complete
- Milestone 4 all task generators + parquet materialization: complete
- Milestone 5 runtime parquet loader + dataset wiring: complete
- Milestone 6 reproducibility and statistical hardening: complete
- Full local integration smoke flow: complete

## 8. Paper-Style Insertion Quality Validation

Validation script:

- `scripts/data/validate_ltaf_insertion_quality_paper_style.py`

Reference command:

```bash
python scripts/data/validate_ltaf_insertion_quality_paper_style.py \
  --contexts-seconds 2.0 3.0 \
  --source-hz 100 \
  --train-samples 5000 \
  --test-samples 500
```

Reference local results:

- context=2.0s (200 samples): AUROC=0.492584
- context=3.0s (300 samples): AUROC=0.502280

Generated artifacts:

- `data/ltafdb/ltaf_haystack/qc/insertion_quality_paper_style_roc.png`
- `data/ltafdb/ltaf_haystack/qc/insertion_quality_paper_style_summary.json`

## 9. Test Results (Authoritative)

These are the latest validated local test runs used as release evidence for the integration.

Environment:

- macOS local workspace
- project `.venv`

Unit tests:

- `pytest -q tests/unit/test_datasets/test_ltaf_qa_loader.py` -> 7 passed
- `pytest -q tests/unit/test_datasets/test_ltaf_*` -> 26 passed
- `pytest -q tests/unit/test_datasets/test_ltaf_generation_config.py` -> 1 passed

Integration tests:

- `pytest -q tests/integration/test_ltaf_*` -> 7 passed

Coverage highlights from these runs:

- Loader/path resolution and split alias handling
- Runtime dataset formatting and metadata pass-through
- CoT dataset behavior and answer extraction path
- Config parsing and generation defaults
- Generation reproducibility and statistical validation guards
- End-to-end LTAF training/eval smoke path

## 10. Remaining Follow-Ups (Non-Blocking)

- Keep paper-style insertion AUROC near chance in CI via `tests/integration/test_ltaf_statistical_validation.py`.
- Optionally stabilize XGBoost runtime on macOS pytest flows to support paper-faithful backend as default.
