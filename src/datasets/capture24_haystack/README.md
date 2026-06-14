# TS-Haystack

A semi-synthetic benchmark for testing retrieval and reasoning over long time series (1K–1M+ datapoints) using Capture-24 accelerometer data.

See the paper for the full benchmark-construction design.

## Structure

```
ts_haystack/
├── core/
│   ├── data_structures.py     # Dataclasses (BoutRecord, NeedleSample, BackgroundSample, etc.)
│   ├── seed_manager.py        # Reproducibility & deterministic seeds
│   ├── timeline_builder.py    # Extract activity bouts from Capture24
│   ├── bout_indexer.py        # Cross-participant bout index
│   ├── transition_matrix.py   # Activity transition probabilities
│   ├── background_sampler.py  # Sample background windows
│   ├── needle_sampler.py      # Sample needles from bout index
│   ├── style_transfer.py      # Covariance projection + blending
│   └── prompt_templates.py    # NL templates for Q/A diversity
├── configs/                    # NEW: YAML-based configuration
│   ├── default_generation_config.yaml  # Default generation config
│   └── generation_config.py   # Config dataclasses & loader
├── tasks/
│   ├── base_task.py           # Abstract base class for all tasks
│   ├── task_existence.py      # Task 1: Existence detection
│   ├── task_localization.py   # Task 2: Temporal localization
│   ├── task_counting.py       # Task 3: Bout counting
│   ├── task_ordering.py       # Task 4: Temporal ordering
│   ├── task_state_query.py    # Task 5: Cross-scale state query
│   ├── task_antecedent.py     # Task 6: Temporal antecedent
│   ├── task_comparison.py     # Task 7: Comparison & negation
│   └── task_multi_hop.py      # Task 8: Multi-hop localization
├── utils/
│   ├── timestamp_utils.py     # Timestamp conversion utilities
│   └── position_utils.py      # Position sampling utilities
├── dataset/                    # NEW: QADataset implementations
│   ├── ts_haystack_qa_loader.py       # Load parquet files as HuggingFace Dataset
│   ├── TSHaystackQADataset.py         # QADataset for direct answer training
│   └── TSHaystackCoTQADataset.py      # QADataset for CoT training
├── cot/                        # NEW: Chain-of-thought generation
│   ├── llm_client.py                  # Gemini API client with retry logic
│   ├── plot_generator.py              # Accelerometer plot generation
│   ├── prompt_builder.py              # Task-specific prompt construction
│   └── cot_generator.py               # Main CoT generation class
├── scripts/
│   ├── build_core_artifacts.py           # CLI to build timelines, index, matrix
│   ├── generate_ts_haystack_dataset.py   # Centralized dataset generator
│   ├── generate_ts_haystack_cot.py       # NEW: CoT rationale generator
│   └── generate_ts_haystack_dataset.sbatch  # SLURM job script
└── test/
    └── test_imports.py        # Verify module imports
```

## Status

- [x] Phase 1: Core infrastructure (timelines, bout index, transition matrix)
- [x] Phase 2: Sampling & style transfer (background/needle samplers, style transfer, prompts)
- [x] Phase 3: Task generators (all 8 tasks implemented)
- [x] Phase 4: QADataset integration (TSHaystackQADataset, TSHaystackCoTQADataset)
- [x] Phase 5: CoT generation pipeline (LLM-based rationale generation)

## Task Overview

| Task | Name | Question Type | Answer Type | Description |
|------|------|---------------|-------------|-------------|
| 1 | Existence | "Is there {activity} in this recording?" | boolean | Detect presence/absence of an activity |
| 2 | Localization | "When did the {activity} bout occur?" | time_range | Find temporal location of an activity |
| 3 | Counting | "How many {activity} bouts occurred?" | integer | Count occurrences of an activity |
| 4 | Ordering | "Did {activity_a} occur before {activity_b}?" | boolean/category | Determine temporal order of two activities |
| 5 | State Query | "What was the activity level when {event} occurred?" | category | Cross-scale integration (local event + global state) |
| 6 | Antecedent | "What activity occurred before {target}?" | category | Identify preceding activity |
| 7 | Comparison | "What was the longest/shortest period with/without {activity}?" | time_range | Find extremum periods |
| 8 | Multi-Hop | "When did the Kth {target} occur after {anchor}?" | time_range | Multi-step reasoning with anchor reference |
| 9 | anomaly_detection | "Is there an anomaly in this dataset?" |
| 10 | anomaly_localization | "When did the anomaly happen?" |

## Quick Start

### 1. Build Core Artifacts

Requires Capture24 data extracted (see `capture24/README.md`).

```bash
python -m opentslm.time_series_datasets.ts_haystack.scripts.build_core_artifacts \
    --n-jobs 8 \
    --label-scheme WillettsSpecific2018
```

### 2. Generate Task Datasets (YAML-based - Recommended)

The centralized dataset generator uses YAML configuration for full visibility and reproducibility.

```bash
# Print default config to create a starting point
python -m opentslm.time_series_datasets.ts_haystack.scripts.generate_ts_haystack_dataset \
    --print-default-config > my_config.yaml

# Generate using config file
python -m opentslm.time_series_datasets.ts_haystack.scripts.generate_ts_haystack_dataset \
    --config my_config.yaml

# Dry run to validate config and see plan
python -m opentslm.time_series_datasets.ts_haystack.scripts.generate_ts_haystack_dataset \
    --config my_config.yaml --dry-run

# Override specific tasks/context lengths via CLI
python -m opentslm.time_series_datasets.ts_haystack.scripts.generate_ts_haystack_dataset \
    --config my_config.yaml \
    --tasks existence localization \
    --context-lengths 100 1000
```

### 2b. Generate Task Datasets (Legacy per-task scripts)

Each task can also be run as a standalone script:

```bash
# Generate existence task samples
python -m opentslm.time_series_datasets.ts_haystack.tasks.task_existence \
    --context-lengths 10000 50000 \
    --samples-per-split 1000 100 100 \
    --seed 42 \
    --n-jobs 4

# Generate multi-hop task samples
python -m opentslm.time_series_datasets.ts_haystack.tasks.task_multi_hop \
    --context-lengths 10000 50000 \
    --samples-per-split 1000 100 100 \
    --seed 42 \
    --n-jobs 4 \
    --direction-mode random \
    --n-distractors 1
```

## YAML Configuration

All generation parameters are controlled via YAML configuration files for full visibility and reproducibility.

### Config Structure

```yaml
# Global settings
global:
  seed: 42                    # Master seed for reproducibility
  n_jobs: 4                   # Parallel workers
  output_dir: data/capture24/ts_haystack/tasks
  overwrite: false            # Skip existing files
  source_hz: 100              # Capture24 sampling rate

# Context lengths in SECONDS (more readable than samples)
# 100s @ 100Hz = 10,000 samples
context_lengths_seconds:
  - 100                       # 100s = 10,000 samples
  - 1000                      # 1000s = 100,000 samples (~17 min)

# Samples per split
samples:
  train: 10000
  val: 1000
  test: 1000

# Style transfer settings
style_transfer:
  transfer_mode: mean_only    # "mean_only" or "full"
  blend_mode: cosine          # "cosine" or "linear"
  blend_window_samples: 50

# Per-task configuration
tasks:
  existence:
    enabled: true
    needle_position: random   # "random", "beginning", "middle", "end"
    needle_length_ratio_range: [0.02, 0.10]  # 2-10% of context
    background_purity: pure   # "pure", "mixed", or "any"
    margin_samples: 100       # Task-specific parameter

  counting:
    enabled: true
    needle_length_ratio_range: [0.02, 0.08]
    background_purity: pure
    min_bouts: 1              # Task-specific parameters
    max_bouts: 5
    min_gap_samples: 100
  # ... more tasks
```

### Key Configuration Options

| Parameter | Description |
|-----------|-------------|
| `context_lengths_seconds` | Window sizes in seconds (converted to samples internally) |
| `needle_length_ratio_range` | Needle duration as fraction of context (e.g., 0.02 = 2%) |
| `background_purity` | "pure" (single activity), "mixed" (multiple activities), or "any" (samples random window - naturally pure for short contexts, potentially mixed for longer ones) |
| `needle_position` | "random", "beginning", "middle", or "end" |
| Task-specific | Each task has additional parameters (see default config) |

### SLURM Job Submission

```bash
# Submit with default config
sbatch scripts/generate_ts_haystack_dataset.sbatch

# Submit with custom config
sbatch --export=CONFIG=configs/my_experiment.yaml scripts/generate_ts_haystack_dataset.sbatch
```

## Output

Artifacts are saved to `data/capture24/ts_haystack/`:

```
data/capture24/ts_haystack/
├── timelines/P*.parquet       # Per-participant activity timelines
├── bout_index.parquet         # Cross-participant bout index
├── transition_matrix.json     # Activity transition probabilities
└── tasks/
    ├── 10s/                   # 1000 samples at 100Hz (10 seconds)
    │   ├── existence/
    │   │   ├── train/data.parquet
    │   │   ├── val/data.parquet
    │   │   ├── test/data.parquet
    │   │   └── metadata.json
    │   ├── localization/
    │   └── ...
    ├── 100s/                  # 10000 samples at 100Hz (100 seconds)
    │   ├── existence/
    │   │   └── ...
    │   └── ...
    └── 1000s/                 # 100000 samples at 100Hz (~17 minutes)
        └── ...
```

Directory naming uses `{seconds}s` format for human readability. The structure
groups by context length first, then by task, enabling easy curriculum learning
by context length.

### Parquet Schema

Each `data.parquet` contains:

| Column | Type | Description |
|--------|------|-------------|
| `x_axis` | List[float] | X-axis accelerometer data |
| `y_axis` | List[float] | Y-axis accelerometer data |
| `z_axis` | List[float] | Z-axis accelerometer data |
| `task_type` | str | Task name (e.g., "existence") |
| `context_length_samples` | int | Window size in samples |
| `background_pid` | str | Source participant ID |
| `recording_time_start` | str | Human-readable start time (e.g., "6:00 AM") |
| `recording_time_end` | str | Human-readable end time (e.g., "8:00 AM") |
| `question` | str | Generated question |
| `answer` | str | Ground truth answer |
| `answer_type` | str | Answer type (boolean, timestamp, integer, category, time_range) |
| `needles` | str (JSON) | Inserted needle metadata (positions, activities, timestamps) |
| `difficulty_config` | str (JSON) | Difficulty parameters used for generation |
| `is_valid` | bool | Validation status |
| `validation_notes` | str | Validation notes (if any) |

The `needles` field contains rich metadata for each inserted activity bout:

```json
[
  {
    "activity": "walking",
    "source_pid": "P001",
    "insert_position_samples": 5000,
    "insert_position_frac": 0.5,
    "duration_samples": 800,
    "duration_ms": 8000,
    "timestamp_start": "7:15 AM",
    "timestamp_end": "7:23 AM"
  }
]
```

## Programmatic Usage

### Using Task Generators

```python
from opentslm.time_series_datasets.ts_haystack.tasks import (
    ExistenceTaskGenerator,
    LocalizationTaskGenerator,
    CountingTaskGenerator,
    OrderingTaskGenerator,
    StateQueryTaskGenerator,
    AntecedentTaskGenerator,
    ComparisonTaskGenerator,
    MultiHopTaskGenerator,
    TASK_REGISTRY,
    list_available_tasks,
)
from opentslm.time_series_datasets.ts_haystack.core import DifficultyConfig

# List available tasks
print(list_available_tasks())
# ['antecedent', 'comparison', 'counting', 'existence', 'localization',
#  'multi_hop', 'ordering', 'state_query']

# Create a task generator with loaded artifacts
generator = MultiHopTaskGenerator.create_with_artifacts(seed=42)

# Configure difficulty
difficulty = DifficultyConfig(
    context_length_samples=10000,
    needle_position="random",
    needle_length_ratio_range=(0.03, 0.30),  # 3-30% of context (300-3000 samples)
    background_purity="pure",
    task_specific={
        "k_distribution": [0.4, 0.4, 0.2],  # P(K=1,2,3)
        "direction_mode": "random",
        "n_distractors_opposite": 0,
        "min_gap_samples": 100,
    },
)

# Generate samples for a split
samples = generator.generate_dataset(
    n_samples=100,
    difficulty=difficulty,
    split="train",
    n_jobs=4,
)

# Save to parquet
output_path = generator.save_dataset(
    samples=samples,
    split="train",
    context_length=10000,
)
print(f"Saved to: {output_path}")
```

### Using the Task Registry

```python
from opentslm.time_series_datasets.ts_haystack.tasks import get_task_generator

# Get generator class by name
TaskClass = get_task_generator("multi_hop")
generator = TaskClass.create_with_artifacts(seed=42)
```

### Low-Level Component Usage

```python
from opentslm.time_series_datasets.ts_haystack.core import (
    TimelineBuilder,
    BoutIndexer,
    TransitionMatrix,
    BackgroundSampler,
    NeedleSampler,
    StyleTransfer,
    PromptTemplateBank,
    SeedManager,
)

# Load Phase 1 artifacts
timelines = TimelineBuilder.load_all_timelines()
bout_index = BoutIndexer.load_index()
transition_matrix = TransitionMatrix.load()

# Initialize Phase 2 components
seed_manager = SeedManager(master_seed=42)
background_sampler = BackgroundSampler(timelines, bout_index)
needle_sampler = NeedleSampler(bout_index, transition_matrix)
style_transfer = StyleTransfer(blend_mode="cosine")
template_bank = PromptTemplateBank()

# Sample a pure background
rng = seed_manager.get_sample_rng("existence", 10000, "train", sample_index=0)
background = background_sampler.sample_background(
    context_length_samples=10000,
    purity="pure",
    rng=rng,
)

# Sample a needle (different activity than background)
needle = needle_sampler.sample_needle_for_context(
    context_activities=background.activities_present,
    min_duration_ms=5000,
    rng=rng,
)

# Trim needle to desired length
needle = needle.trim(n_samples=500)

# Apply style transfer
local_stats = style_transfer.compute_local_statistics(
    (background.x, background.y, background.z),
    position=5000,
)
transferred_needle = style_transfer.transfer(needle, local_stats)

# Insert with blending
x, y, z = style_transfer.insert_with_blending(
    (background.x, background.y, background.z),
    (transferred_needle.x, transferred_needle.y, transferred_needle.z),
    position=5000,
)

# Generate Q/A using templates
question, answer = template_bank.sample(
    "existence",
    rng,
    activity=needle.activity,
    exists=True,
)
```

## Distractor Insertion (Existence & Localization)

To prevent models from "cheating" by detecting variance changes in homogeneous backgrounds,
the Existence and Localization tasks insert **multiple needles from the same activity regime**.
This forces the model to distinguish between similar activities rather than just detecting
signal variance changes.

### Activity Regimes

Activities are grouped by signal characteristics:

| Regime | Activities | Signal Characteristics |
|--------|-----------|------------------------|
| **Sedentary** | sleep, sitting, vehicle, standing, household-chores | Low-to-moderate variance, minimal rhythmic patterns |
| **Active** | walking, mixed-activity, bicycling, manual-work, sports | Higher variance, rhythmic/dynamic patterns |

### How It Works

1. **Existence Task**: Inserts N needles from ONE randomly-selected regime
   - Positive: asks about an inserted activity
   - Negative: asks about a non-inserted activity **from the same regime**

2. **Localization Task**: Inserts N needles from ONE randomly-selected regime
   - Asks about a specific inserted needle (target)
   - Other needles serve as distractors with similar signal properties

### Configuration Parameters

Distractor insertion is controlled via `task_specific` in `DifficultyConfig`:

```python
difficulty = DifficultyConfig(
    context_length_samples=10000,
    task_specific={
        "min_distractors": 2,      # Minimum needles to insert
        "max_distractors": 4,      # Maximum needles to insert
        "min_gap_samples": 100,    # Minimum gap between needles
    },
)
```

## Task-Specific Configuration

Each task supports `task_specific` parameters in `DifficultyConfig`:

### Existence
- `min_distractors`: Minimum number of needles to insert (default: 1)
- `max_distractors`: Maximum number of needles to insert (default: 3)
- `min_gap_samples`: Minimum gap between inserted needles (default: 100)
- `margin_samples`: Position margin from window edges (default: 100)

### Localization
- `min_distractors`: Minimum number of needles to insert (default: 2)
- `max_distractors`: Maximum number of needles to insert (default: 4)
- `min_gap_samples`: Minimum gap between inserted needles (default: 100)
- `margin_samples`: Position margin from window edges (default: 100)

### Counting
- `min_bouts`: Minimum bouts to insert (default: 1)
- `max_bouts`: Maximum bouts to insert (default: 5)
- `min_gap_samples`: Gap between bouts (default: 100)

### Ordering
- `min_gap_samples`: Gap between activities (default: 100)
- `question_format`: "boolean" or "category" (default: "boolean")

### State Query
- `min_global_states`: Minimum activity states in background (default: 2)
- `max_global_states`: Maximum activity states (default: 5)
- `position_mode`: "center", "near_boundary", or "random" (default: "random")

### Antecedent
- `adjacency_gap_samples`: Gap between antecedent and target (default: 10)
- `background_mode`: "low_activity" or "mixed" (default: "low_activity")
- `use_transition_probs`: Use transition matrix for pairing (default: False)

### Comparison
- `min_bouts`: Minimum bouts to insert (default: 2)
- `max_bouts`: Maximum bouts (default: 4)
- `min_duration_diff_ms`: Minimum duration difference to avoid ties (default: 2000)

### Multi-Hop
- `k_distribution`: Probability distribution for K values [P(K=1), P(K=2), P(K=3)]
- `direction_mode`: "random", "after_only", or "before_only"
- `n_distractors_opposite`: Distractor targets on opposite side of anchor
- `min_gap_samples`: Gap between bouts (default: 100)

## Test Suite

  Comprehensive tests validate all task generators, with optional plot generation for visual inspection.

  ```bash
  # Run all tests
  pytest src/opentslm/time_series_datasets/ts_haystack/test/ -v

  # Run only task tests (no plots)
  pytest src/opentslm/time_series_datasets/ts_haystack/test/tasks/ -v -k "not visualize"

  # Generate sample plots only
  pytest src/opentslm/time_series_datasets/ts_haystack/test/tasks/ -v -k "visualize"

  Plots are saved to test/plots/<task_name>/.

  Note: Tests require Phase 1 artifacts to be built first.

## OpenTSLM Training Integration

### 3. Create QADataset for Training

After generating the task datasets, use `TSHaystackQADataset` to train OpenTSLM models.

`EOS_TOKEN` is a **required** parameter — the training script passes it from
`model.get_eos_token()`, keeping datasets model-agnostic.

```python
from opentslm.time_series_datasets.ts_haystack import TSHaystackQADataset

# EOS_TOKEN is passed by the training script from model.get_eos_token()
eos_token = model.get_eos_token()

# Single task training
train_dataset = TSHaystackQADataset(
    split="train",
    EOS_TOKEN=eos_token,
    tasks=["existence"],
    context_lengths_seconds=[100],  # 100s = 10000 samples at 100Hz
)

# Multi-task training
train_dataset = TSHaystackQADataset(
    split="train",
    EOS_TOKEN=eos_token,
    tasks=["existence", "localization", "counting", "ordering"],
    context_lengths_seconds=[100, 1000],  # Multiple context lengths
)

val_dataset = TSHaystackQADataset(split="validation", EOS_TOKEN=eos_token)
test_dataset = TSHaystackQADataset(split="test", EOS_TOKEN=eos_token)

print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
print(f"Tasks: {train_dataset.get_tasks()}")
```

### Using with DataLoader

```python
from torch.utils.data import DataLoader
from opentslm.time_series_datasets.util import extend_time_series_to_match_patch_size_and_aggregate

dataloader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
    collate_fn=lambda batch: extend_time_series_to_match_patch_size_and_aggregate(
        batch, patch_size=4
    ),
)

for batch in dataloader:
    # batch is a list of dicts with keys:
    # - pre_prompt, post_prompt, time_series, time_series_text, answer
    # - task_type, answer_type, question, x_axis, y_axis, z_axis
    print(batch[0]["answer"])
    break
```

### Sample Output Format

Each sample from `TSHaystackQADataset` contains:

```python
sample = {
    # Prompt components (for model input)
    "pre_prompt": "You are given accelerometer data... Question: Is there walking...",
    "time_series": [[x_values], [y_values], [z_values]],
    "time_series_text": ["The following is the accelerometer data on the x-axis", ...],
    "post_prompt": "Instructions: ... Answer with 'Yes' or 'No'...",
    "answer": "Yes",

    # Metadata
    "task_type": "existence",
    "answer_type": "boolean",
    "question": "Is there walking in this recording?",
    "context_length_samples": 10000,

    # Raw data (for analysis)
    "x_axis": [0.38, 0.39, ...],
    "y_axis": [0.48, 0.49, ...],
    "z_axis": [-0.79, -0.78, ...],
    "needles": "[{\"activity\": \"walking\", ...}]",  # JSON string
}
```

## Chain-of-Thought (CoT) Generation

### 4. Generate CoT Rationales

Generate LLM-based chain-of-thought rationales for training models to reason step-by-step:

```bash
# Generate CoT for all tasks at 100s context length
python -m opentslm.time_series_datasets.ts_haystack.scripts.generate_ts_haystack_cot \
    --context-lengths 100 \
    --tasks all \
    --max-workers 4

# Generate CoT for specific tasks and splits
python -m opentslm.time_series_datasets.ts_haystack.scripts.generate_ts_haystack_cot \
    --context-lengths 100 \
    --tasks existence localization counting \
    --splits train val \
    --max-workers 8

# Test with a few samples
python -m opentslm.time_series_datasets.ts_haystack.scripts.generate_ts_haystack_cot \
    --context-lengths 100 \
    --tasks existence \
    --splits test \
    --max-samples 10

# Debug mode - saves JSON files with sample data, prompts, rationales, and plot images
python -m opentslm.time_series_datasets.ts_haystack.scripts.generate_ts_haystack_cot \
    --context-lengths 100 \
    --tasks existence \
    --splits test \
    --max-samples 5 \
    --debug \
    --debug-output-dir debug_output/cot
```

**Prerequisites:**
- Task datasets must be generated first (see generate_ts_haystack_dataset.py)
- `GEMINI_API_KEY` environment variable must be set

### CoT Output Structure

```
data/capture24/ts_haystack/cot/
├── 100s/
│   ├── existence/
│   │   ├── train/data.parquet  # Same schema + "rationale" column
│   │   ├── val/data.parquet
│   │   └── test/data.parquet
│   ├── localization/
│   └── ...
├── debug/                      # Debug output (when --debug is used)
│   ├── existence/
│   │   ├── sample_000000.json  # Full sample data + prompt + rationale
│   │   ├── sample_000000_plot.png  # Plot image sent to LLM
│   │   ├── sample_000001.json
│   │   └── ...
│   └── ...
└── metadata.json               # Generation metadata (model, params, stats)
```

### Debug Mode Output

When running with `--debug`, each sample generates:

1. **JSON file** (`sample_NNNNNN.json`):
   - `sample_idx`: Sample index
   - `task_type`: Task type (existence, counting, etc.)
   - `sample_data`: All parquet columns (excluding large arrays, includes length info)
   - `prompt`: Full prompt sent to LLM
   - `rationale`: Generated rationale
   - `plot_base64`: Base64-encoded PNG plot (if plots enabled)
   - `timestamp`: Generation timestamp

2. **PNG file** (`sample_NNNNNN_plot.png`): The accelerometer plot image sent to the LLM for visual analysis

### 5. Train with CoT Rationales

Use `TSHaystackCoTQADataset` to train with chain-of-thought reasoning:

```python
from opentslm.time_series_datasets.ts_haystack import TSHaystackCoTQADataset

# CoT dataset - answer includes full rationale
# EOS_TOKEN is passed by the training script from model.get_eos_token()
train_dataset = TSHaystackCoTQADataset(
    split="train",
    EOS_TOKEN=model.get_eos_token(),
    tasks=["existence", "counting"],
    context_lengths_seconds=[100],
)

sample = train_dataset[0]
print(sample["answer"])         # Full rationale ending with "Answer: ..."
print(sample["direct_answer"])  # Just the final answer (for evaluation)
```

### Example CoT Rationale

```
Looking at the accelerometer data spanning from 6:00 AM to 7:40 AM, I need to
identify all walking bouts. The signal shows predominantly low-variance patterns
consistent with sedentary activity, but I can identify three distinct periods of
rhythmic, moderate-intensity oscillations characteristic of walking gait.

The first walking bout appears around 6:12 AM and continues until approximately
6:18 AM, showing the typical regular patterns of heel-strike and toe-off. A second
walking period begins at 6:35 AM, lasting until 6:42 AM with similar oscillatory
characteristics. Finally, a third walking bout is visible from 7:15 AM to 7:22 AM.

Counting all the distinct walking periods: 3.

Answer: 3
```

## Caching Note

Both `TSHaystackQADataset` and `TSHaystackCoTQADataset` use class-level caching (inherited from `QADataset`). Once data is loaded for a configuration, it's cached for all subsequent instances. If you need different configurations in the same session, restart Python or use separate processes.
