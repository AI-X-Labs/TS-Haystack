# Adding a New Dataset

This guide explains how to integrate a new time series dataset into TSLM-Bench for QA-based training and evaluation.

In the future, this will be integrated with TimeNet.

---

## How Datasets Work

TSLM-Bench datasets produce **question-answering samples** over time series data. Each sample contains raw sensor data, a text prompt with a question, and a ground truth answer. The training loop expects this format (see also [Sample Schema](architecture.md#sample-schema)):

```python
# Guaranteed keys produced by QADataset.__getitem__()
# via PromptWithAnswer.to_dict() (src/prompt/prompt_with_answer.py)
sample = {
    "time_series": list[Tensor],   # one (seq_len,) tensor per channel
    "time_series_text": list[str], # label for each channel
    "pre_prompt": str,             # text before the time series
    "post_prompt": str,            # text/instructions after the time series
    "answer": str,                 # ground truth (with EOS token appended)
}
```

Datasets inherit from `QADataset` (`src/datasets/qa_base.py`), which handles split management, lazy loading, and formatting.

---

## Step-by-Step Guide

### 1. Create a directory for your dataset

```
src/datasets/my_dataset/
    __init__.py
    qa_dataset.py
```

### 2. Implement the QADataset subclass

Create `src/datasets/my_dataset/qa_dataset.py`:

```python
from typing import Callable, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
from datasets import Dataset, load_dataset

from src.prompt.text_time_series_prompt import TextTimeSeriesPrompt
from src.datasets.qa_base import QADataset


class MyQADataset(QADataset):
    """QADataset for my custom time series dataset."""

    def __init__(
        self,
        split: Literal["train", "test", "validation"],
        EOS_TOKEN: str = "",
        format_sample_str: bool = False,
        time_series_format_function: Callable[[np.ndarray], str] | None = None,
        lazy_loading: bool = True,
    ):
        super().__init__(
            split, EOS_TOKEN, format_sample_str,
            time_series_format_function, lazy_loading,
        )

    def _load_splits(self) -> Tuple[Dataset, Dataset, Dataset]:
        """Load train/val/test splits.

        Returns three HuggingFace Dataset objects. Each row must have the
        fields your _get_* methods expect.
        """
        # Option 1: Load from HuggingFace Hub
        ds = load_dataset("your-org/your-dataset")
        return ds["train"], ds["validation"], ds["test"]

        # Option 2: Load from local files
        # import json
        # from pathlib import Path
        # from datasets import Dataset
        # data_dir = Path("data/my_dataset")
        # train = Dataset.from_json(str(data_dir / "train.jsonl"))
        # val = Dataset.from_json(str(data_dir / "val.jsonl"))
        # test = Dataset.from_json(str(data_dir / "test.jsonl"))
        # return train, val, test

    def _get_answer(self, row) -> str:
        """Return the ground truth answer string."""
        return str(row["answer"])

    def _get_pre_prompt(self, row) -> str:
        """Return text that appears BEFORE the time series in the prompt."""
        return (
            f"You are given sensor data from a wearable device.\n\n"
            f"Question: {row['question']}"
        )

    def _get_post_prompt(self, row) -> str:
        """Return text that appears AFTER the time series in the prompt."""
        return (
            "\nAnalyze the data and provide your answer."
        )

    def _get_text_time_series_prompt_list(self, row) -> List[TextTimeSeriesPrompt]:
        """Convert raw data into TextTimeSeriesPrompt objects.

        Each TextTimeSeriesPrompt pairs a text label with the time series
        data for one channel. The model receives these as interleaved
        text + time series tokens.
        """
        # Example: single-channel data
        series = torch.tensor(row["values"], dtype=torch.float32)
        return [
            TextTimeSeriesPrompt("The following is the sensor reading", series.tolist())
        ]

        # Example: multi-channel data (e.g., 3-axis accelerometer)
        # labels = [
        #     "The following is the accelerometer data on the x-axis",
        #     "The following is the accelerometer data on the y-axis",
        #     "The following is the accelerometer data on the z-axis",
        # ]
        # series = torch.as_tensor(
        #     np.stack([row["x_axis"], row["y_axis"], row["z_axis"]]),
        #     dtype=torch.float32,
        # )
        # return [
        #     TextTimeSeriesPrompt(label, ts.tolist())
        #     for label, ts in zip(labels, series)
        # ]
```

**Note on `EOS_TOKEN`**: The default value in `__init__` doesn't matter — the training script always overrides it with `model.get_eos_token()` when creating the dataset. Each model architecture defines its own EOS token (e.g. Flamingo uses `<|endofchunk|>`, other models may use the LLM's native EOS). Your dataset doesn't need to know which model it will be paired with.

### 3. Create `__init__.py`

```python
from src.datasets.my_dataset.qa_dataset import MyQADataset

__all__ = ["MyQADataset"]
```

### 4. Register it

Add your dataset class to `DATASET_REGISTRY` in `src/datasets/registry.py`:

```python
from src.datasets.my_dataset.qa_dataset import MyQADataset

DATASET_REGISTRY: dict[str, type[QADataset]] = {
    # ... existing entries ...
    "my_dataset": MyQADataset,
}
```

That's it — no training script changes needed. Configure it in your experiment YAML:

```yaml
# configs/experiments/my_experiment.yaml
dataset:
  name: my_dataset
  extra_kwargs:
    window_size_s: 30
    effective_hz: 50
```

```bash
python scripts/train.py --config configs/experiments/my_experiment.yaml
```

Any keyword arguments your `__init__` accepts (beyond the base `split`, `EOS_TOKEN`, etc.) can be passed via `dataset.extra_kwargs` in the YAML config.

---

## Abstract Methods Reference

`QADataset` (`src/datasets/qa_base.py`) requires you to implement five methods:

| Method | Returns | Purpose |
|--------|---------|---------|
| `_load_splits()` | `(train, val, test)` | Load the raw data. Return HuggingFace `Dataset` objects. |
| `_get_answer(row)` | `str` | Extract the ground truth answer from a data row. |
| `_get_pre_prompt(row)` | `str` | Text placed before the time series (context + question). |
| `_get_post_prompt(row)` | `str` | Text placed after the time series (instructions). |
| `_get_text_time_series_prompt_list(row)` | `list[TextTimeSeriesPrompt]` | Convert raw time series into labeled prompt objects. |

Additionally, `QADataset` provides two **optional** methods with sensible defaults that you can override for dataset-specific evaluation:

| Method | Returns | Default | Purpose |
|--------|---------|---------|---------|
| `extract_answer(prediction, sample)` | `str` | `prediction.strip()` | Extract the final answer from a model prediction (e.g., parse "Answer: ..." from CoT output). |
| `evaluate_answer(prediction, sample)` | `dict` with `{"correct": bool, ...}` | Exact match against `sample["answer"]` | Compare predicted answer to ground truth. Override for type-aware logic (boolean normalization, IoU for time ranges, etc.). |

The training script calls these methods during validation, so each dataset controls its own evaluation logic without any changes to `scripts/train.py`.

---

## How Prompts Are Assembled

The base class `_format_sample()` (in `src/datasets/qa_base.py`) combines your methods into a sample dict via `PromptWithAnswer.to_dict()` (`src/prompt/prompt_with_answer.py`):

```
[pre_prompt] + [channel_1_label + channel_1_time_series] + [...] + [post_prompt] + [answer + EOS]
```

The sample dict contains the raw text and time series data separately — it is the **model's** `prepare_batch()` method that decides how to interleave them with model-specific special tokens (e.g. Flamingo uses `<image>` and `<|endofchunk|>` markers). Your dataset doesn't need to worry about these tokens.

The `TextTimeSeriesPrompt` objects (`src/prompt/text_time_series_prompt.py`) carry both the label text and the raw tensor data for each channel.

---

## Lazy vs Eager Loading

By default, `lazy_loading=True`: raw data is kept in HuggingFace's memory-mapped format and samples are formatted on-demand in `__getitem__`. This is memory-efficient for large datasets.

Set `lazy_loading=False` to pre-format all samples at initialization (legacy behavior). Useful for debugging or small datasets.

**Cache warning**: `QADataset` caches data at the class level. If you create two instances of the same dataset class with different parameters in the same Python process, the second instance will silently use the first's cached data. Restart the process to change configuration.

---

## Data Format Requirements

Your raw data must provide enough information for the five abstract methods above. A typical row in your HuggingFace Dataset might look like:

```python
{
    "question": "Is there running activity in this recording?",
    "answer": "Yes",
    "values": [0.1, 0.2, ...],       # or separate x_axis, y_axis, z_axis
    "task_type": "existence",          # optional, for per-task evaluation
    "answer_type": "boolean",          # optional, for answer parsing
}
```

Patch alignment and time series padding are handled by the model's `prepare_batch()` method — your dataset just needs to provide the raw data.

---

## Adding a Download Script

For reproducibility, add a download script at `scripts/data/download_my_dataset.py` or extend `scripts/data/download_from_hf.py` with a new `--dataset` option.

---

## Existing Dataset Examples

Study these implementations for reference:

| Dataset | Path | Description |
|---------|------|-------------|
| TS-Haystack CoT | `src/datasets/ts_haystack/cot_qa_dataset.py` | Multi-task QA with chain-of-thought rationales, multi-channel accelerometer data |
| TS-Haystack Plain | `src/datasets/ts_haystack/qa_dataset.py` | Same tasks without CoT (direct answers only) |
| TS-Haystack Oracle | `src/datasets/ts_haystack/oracle_qa_dataset.py` | CoT with ground-truth activity segmentation in prompts |
| Capture24 | `src/datasets/capture24/qa_dataset.py` | Activity classification from wearable accelerometer data |

---

## Key Files

| Purpose | File |
|---------|------|
| QA dataset base class | `src/datasets/qa_base.py` |
| Dataset registry | `src/datasets/registry.py` |
| Sample formatting | `src/prompt/prompt_with_answer.py` |
| TextTimeSeriesPrompt | `src/prompt/text_time_series_prompt.py` |
| Base dataset (non-QA) | `src/datasets/base.py` |
| Training script (dataloader creation) | `scripts/train.py` |

---

## Things to Know

1. **Dataset registry**: `DATASET_REGISTRY` in `src/datasets/registry.py` maps string keys to `QADataset` subclasses. The experiment YAML's `dataset.name` selects a registered dataset and `dataset.extra_kwargs` passes constructor arguments. No training script changes are needed to add a new dataset — just register it.

2. **EOS token is injected by the training script**: The training script calls `model.get_eos_token()` and passes the result as `EOS_TOKEN` to the dataset constructor. This keeps datasets model-agnostic — each model defines its own answer terminator.

3. **Multi-channel handling**: Each channel is represented as a separate `TextTimeSeriesPrompt` returned from `_get_text_time_series_prompt_list()`. If your data is single-channel, return a single-element list. How these channels are processed (e.g. as separate cross-attention inputs in Flamingo) is up to the model's `prepare_batch()`.