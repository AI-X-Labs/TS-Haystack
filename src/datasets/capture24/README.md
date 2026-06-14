# Capture-24 Dataset

## Dataset Overview

- 151 participants with ~7 days of wrist-worn accelerometer data each
- 100Hz sampling rate = ~60 million rows per participant
- Total: ~9 billion rows of sensor data
- Files in ZIP: PXXX.csv.gz (one per participant), metadata.csv, annotation-label-dictionary.csv

## Getting Started

### 1. Download the Dataset

Download `capture24.zip` from the Oxford Research Archive:
https://ora.ox.ac.uk/objects/uuid:99d7c092-d865-4a19-b096-cc16440cd001/files/rpr76f381b

### 2. Place the ZIP File

Place the downloaded `capture24.zip` in the `data/` directory at the project root:

```
OpenTSLM/
├── data/
│   └── capture24.zip    <-- place here
├── src/
└── ...
```

### 3. Extract and Convert to Parquet

Run the loader to extract the ZIP and convert CSV files to Parquet format:

```bash
# Basic extraction (100Hz, single-threaded)
python -m opentslm.time_series_datasets.capture24.capture24_loader

# With parallel processing (recommended for faster extraction)
python -m opentslm.time_series_datasets.capture24.capture24_loader --n-jobs 4

# Downsample to 25Hz during extraction
python -m opentslm.time_series_datasets.capture24.capture24_loader --downsample-hz 25 --n-jobs 4

# Test with a few participants first
python -m opentslm.time_series_datasets.capture24.capture24_loader --max-participants 3

# Submit as SLURM job (recommended if working on a computing cluster -- adjust partition to your needs)
sbatch src/opentslm/time_series_datasets/capture24/extract_data.sbatch
```

This creates:
```
data/capture24/
├── participants.parquet         # Participant metadata
├── label_mappings.parquet       # Annotation-to-label mappings
└── sensor_data_{hz}hz/          # Sensor data directory
    ├── pid=P001/data.parquet
    ├── pid=P002/data.parquet
    └── ...
```

### 4. Extract Windows

After extraction, split the data into non-overlapping windows for training:

```bash
# Basic window extraction (2.56s windows at 50Hz -- matches the sampling size of the HAR_CoT dataset already in the OpenTSLM datasets)
python -m opentslm.time_series_datasets.capture24.capture24_windows

# 10s windows downsampled to 25Hz with parallel processing
python -m opentslm.time_series_datasets.capture24.capture24_windows \
    --window-size-s 10 \
    --source-hz 100 \
    --downsample-hz 25

# Custom train/val/test split ratios
python -m opentslm.time_series_datasets.capture24.capture24_windows \
    --train-ratio 0.8 \
    --val-ratio 0.1

# Test with a few participants
python -m opentslm.time_series_datasets.capture24.capture24_windows --max-participants 5
```

This creates:
```
data/capture24/windows/{window_size}s_{hz}hz/
├── train/data.parquet
├── val/data.parquet
└── test/data.parquet
```

With the following schema:

```py
# Example window schema
  window = {
      "window_id": "P001_1476676380000",
      "pid": "P001",
      "start_ms": 1476676380000,
      "end_ms": 1476676390000,
      "x": [0.38, 0.39, ...],       # 300 values
      "y": [0.48, 0.49, ...],       # 300 values
      "z": [-0.79, -0.78, ...],     # 300 values
      "annotations": [              # 300 raw annotation strings
          "7030 sleeping;MET 0.95",
          "7030 sleeping;MET 0.95",
          ...
      ]
  }
  ```

### 5. Create Classification Dataset

After window extraction, create labeled classification datasets for training:

```bash
# Basic classification with Walmsley2020 labels (4 classes)
python -m opentslm.time_series_datasets.capture24.capture24_classification

# Specify window config and label scheme
python -m opentslm.time_series_datasets.capture24.capture24_classification \
    --window-size-s 10 \
    --effective-hz 25 \
    --label-scheme Walmsley2020

# Filter low-confidence windows (require 50%+ label agreement)
python -m opentslm.time_series_datasets.capture24.capture24_classification \
    --label-scheme WillettsSpecific2018 \
    --min-confidence 0.5
```

This creates:
```
data/capture24/classification/{window_size}s_{hz}hz/{label_scheme}/
├── train/data.parquet
├── val/data.parquet
├── test/data.parquet
└── metadata.json
```

With the following schema:

```py
# Example classification sample
sample = {
    "window_id": "P001_1476676380000",
    "pid": "P001",
    "start_ms": 1476676380000,
    "end_ms": 1476676390000,
    "x": [0.38, 0.39, ...],       # 250 values (10s @ 25Hz)
    "y": [0.48, 0.49, ...],       # 250 values
    "z": [-0.79, -0.78, ...],     # 250 values
    "label": "sleep",             # Mapped label string
    "label_id": 3,                # Integer encoding (alphabetical order)
    "confidence": 0.95,           # Fraction of samples with this label
}
```

## Programmatic Usage

### Loading Raw Sensor Data

```python
from opentslm.time_series_datasets.capture24.capture24_loader import (
    ensure_capture24_data,
    load_participants,
    load_label_mappings,
    load_participant_sensor_data,
)

# Ensure data is extracted (no-op if already done)
ensure_capture24_data(downsample_hz=100)

# Load metadata
participants = load_participants()
label_mappings = load_label_mappings()

# Load sensor data for a specific participant
df = load_participant_sensor_data("P001", downsample_hz=100)
# Columns: timestamp_ms, x, y, z, annotation
```

### Loading Pre-extracted Windows

```python
from opentslm.time_series_datasets.capture24.capture24_windows import (
    extract_windows,
    load_windows,
)

# Extract windows (no-op if already done)
extract_windows(window_size_s=10, source_hz=100, downsample_hz=25)

# Load windows for a split
train_windows = load_windows(window_size_s=10, effective_hz=25, split="train")
val_windows = load_windows(window_size_s=10, effective_hz=25, split="val")
test_windows = load_windows(window_size_s=10, effective_hz=25, split="test")

# Window schema: window_id, pid, start_ms, end_ms, x, y, z, annotations
```
To **match HAR CoT's 128 samples per window**, here are the Capture-24 configurations:
  ┌─────────────┬──────────────┬─────────┬─────────────────────────────────────────────┐
  │ Window Size │ Effective Hz │ Samples │                    Notes                    │
  ├─────────────┼──────────────┼─────────┼─────────────────────────────────────────────┤
  │ 2.56s       │ 50Hz         │ 128     │ Exact HAR match (50Hz divides 100Hz evenly) │
  └─────────────┴──────────────┴─────────┴─────────────────────────────────────────────┘

  This gives you:
  - Same number of samples (128)
  - Same sampling rate (50Hz)
  - Same temporal duration (2.56 seconds)
  - Clean downsampling from 100Hz (take every 2nd sample)

### Loading Classification Datasets

```python
from opentslm.time_series_datasets.capture24.capture24_classification import (
    create_classification_dataset,
    load_classification_dataset,
    load_classification_metadata,
    get_class_names,
    get_class_distribution,
    LABEL_SCHEMES,
)

# List available label schemes
print(LABEL_SCHEMES.keys())
# ['WillettsSpecific2018', 'WillettsMET2018', 'DohertySpecific2018',
#  'Willetts2018', 'Doherty2018', 'Walmsley2020']

# Create classification dataset (no-op if already done)
create_classification_dataset(
    window_size_s=10,
    effective_hz=25,
    label_scheme="Walmsley2020",
    min_confidence=0.5  # Optional: require 50%+ label agreement
)

# Load classification data for a split
train_df = load_classification_dataset(
    window_size_s=10, effective_hz=25, label_scheme="Walmsley2020", split="train"
)

# Get class names (alphabetically sorted)
classes = get_class_names("Walmsley2020")
# ['light', 'moderate-vigorous', 'sedentary', 'sleep']

# Load metadata with class distribution
metadata = load_classification_metadata(
    window_size_s=10, effective_hz=25, label_scheme="Walmsley2020"
)
print(metadata["class_names"])
print(metadata["class_distribution"]["train"])
```

## Command-Line Options

### capture24_loader.py

| Option | Default | Description |
|--------|---------|-------------|
| `--max-participants, -n` | None | Limit participants (for testing) |
| `--downsample-hz, -d` | 100 | Target sampling frequency |
| `--n-jobs, -j` | 1 | Parallel jobs |
| `--overwrite` | False | Force re-extraction |

### capture24_windows.py

| Option | Default | Description |
|--------|---------|-------------|
| `--window-size-s, -w` | 10 | Window size in seconds |
| `--source-hz` | 100 | Source data frequency |
| `--downsample-hz, -d` | None | Target frequency |
| `--annotation-threshold, -a` | 0.6 | Min annotation coverage |
| `--seed, -s` | 42 | Random seed for splits |
| `--train-ratio, -t` | 0.7 | Training set fraction |
| `--val-ratio, -v` | 0.15 | Validation set fraction |
| `--n-jobs, -j` | 1 | Parallel jobs |
| `--max-participants, -n` | None | Limit participants |
| `--overwrite` | False | Force re-extraction |

### capture24_classification.py

| Option | Default | Description |
|--------|---------|-------------|
| `--window-size-s, -w` | 10 | Window size in seconds |
| `--effective-hz, -e` | 100 | Effective sampling frequency |
| `--label-scheme, -l` | Walmsley2020 | Label scheme to use |
| `--min-confidence, -c` | 0.0 | Min confidence threshold (0.0-1.0) |
| `--overwrite` | False | Force re-creation |

## Label Scheme Summary

| Scheme | Unique Labels | Description |
|--------|---------------|-------------|
| WillettsSpecific2018 | 12 | Most granular activity types |
| WillettsMET2018 | 10 | Activity + intensity combinations |
| DohertySpecific2018 | 11 | Screen/task-based distinction |
| Willetts2018 | 6 | Simplified posture-based |
| Doherty2018 | 6 | Intensity-based |
| Walmsley2020 | 5 | Most simplified |

### Labels per Scheme

**WillettsSpecific2018 (12 labels):**
sleep, sitting, standing, walking, vehicle, bicycling, sports, household-chores, mixed-activity, manual-work

**WillettsMET2018 (10 labels):**
sleep, sitting, sitstand+lowactivity, sitstand+activity, walking, walking+activity, vehicle, bicycling, sports, gym

**DohertySpecific2018 (11 labels):**
sleep, sedentary-screen, sedentary-non-screen, tasks-light, tasks-moderate, walking, vehicle, bicycling, sports-continuous, sport-interrupted

**Willetts2018 (6 labels):**
sleep, sit-stand, mixed, walking, vehicle, bicycling

**Doherty2018 (6 labels):**
sleep, sedentary, tasks-light, moderate, walking

**Walmsley2020 (5 labels):**
sleep, sedentary, light, moderate-vigorous

## OpenTSLM Training Integration

### 6. Create QADataset for Training

After creating the classification dataset, use `Capture24AccQADataset` to train OpenTSLM models:

```python
from opentslm.time_series_datasets.capture24 import Capture24AccQADataset

# Create dataset for training (uses default config: 10s windows, 100Hz, Walmsley2020)
train_dataset = Capture24AccQADataset(
    split="train",
    EOS_TOKEN=tokenizer.eos_token,
    window_size_s=10,
    effective_hz=100,
    label_scheme="Walmsley2020"
)

val_dataset = Capture24AccQADataset(split="validation", EOS_TOKEN=tokenizer.eos_token)
test_dataset = Capture24AccQADataset(split="test", EOS_TOKEN=tokenizer.eos_token)

print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
print(f"Labels: {train_dataset.get_labels()}")
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
    # - label, x_axis, y_axis, z_axis (raw data preserved)
    print(batch[0]["answer"])  # e.g., "sleep"
    break
```

### Integration with CurriculumTrainer

The `Capture24AccQADataset` follows the same interface as `HARCoTQADataset`, so it's compatible with the existing `CurriculumTrainer`. To add a Capture24 training stage:

```python
# In curriculum_learning.py, add a new stage method:
def stage_capture24(self, batch_size: int = None, eval_only: bool = False):
    """Stage: Capture-24 Activity Classification."""
    from opentslm.time_series_datasets.capture24 import Capture24AccQADataset

    return self._train_stage(
        stage_name="stage_capture24",
        dataset_class=Capture24AccQADataset,
        num_epochs=30,
        lr_encoder=2e-4,
        lr_projector=1e-4,
        lr_base=2e-4,
        metric_func=lambda preds, golds: {
            "accuracy": self._calculate_accuracy(preds, golds)
        },
        batch_size=batch_size,
        eval_only=eval_only,
    )
```

### Custom Configuration

For custom window sizes or label schemes, create a wrapper class:

```python
from functools import partial
from opentslm.time_series_datasets.capture24 import Capture24AccQADataset

# Create a partial class with custom config
Capture24Custom = partial(
    Capture24AccQADataset,
    window_size_s=2,        # 2.56s to match HAR
    effective_hz=50,        # 50Hz for 128 samples
    label_scheme="Doherty2018"
)

# Use in training
train_dataset = Capture24Custom(split="train", EOS_TOKEN=tokenizer.eos_token)
```

### Sample Output Format

Each sample from `Capture24AccQADataset` contains:

```python
sample = {
    # Prompt components (for model input)
    "pre_prompt": "You are given accelerometer data in all three dimensions from a wrist-worn sensor...",
    "time_series": [[x_values], [y_values], [z_values]],  # 3 axes
    "time_series_text": ["The following is the accelerometer data on the x-axis", ...],
    "post_prompt": "Instructions: ... The following activities are possible: light, moderate-vigorous, sedentary, sleep ...",
    "answer": "sleep",

    # Raw data (preserved for analysis)
    "label": "sleep",
    "x_axis": [0.38, 0.39, ...],
    "y_axis": [0.48, 0.49, ...],
    "z_axis": [-0.79, -0.78, ...],
}
```

### Caching Note

The `QADataset` base class uses class-level caching. Once data is loaded for a configuration, it's cached for subsequent instances. If you need different configurations in the same session, restart Python or use separate processes.

### Verification

Run the test script to verify the integration:

```bash
python -m opentslm.time_series_datasets.capture24.test.test_capture24_qa
```
