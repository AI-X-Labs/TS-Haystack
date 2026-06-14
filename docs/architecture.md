# Architecture Overview

This document describes the internal structure of TSLM-Bench and how the pieces fit together.

---

## Component Diagram

```
configs/experiments/*.yaml
       |
       v
  ExperimentConfig.from_yaml()     scripts/train.py --config <yaml>
  (src/utils/config.py)                    |
       |                                   v
       +----------> ExperimentConfig ------+------+
                                     |             |
                                     v             v
                                load_model()  create_dataloaders()
                                     |             |
                                     v             v
                             MODEL_REGISTRY   DATASET_REGISTRY
                                     |             |
                                     v             v
                              BaseModel       QADataset base class
                               /   |   \           |
                              v    v    v          v
                        Encoder Projector Backbone  (dataset subclasses)
                        (registry) (registry) (registry)
```

The training script (`scripts/train.py`) is fully model-agnostic: it interacts
with models exclusively through the `BaseModel` interface and with datasets
through `QADataset`. Adding a new model or dataset requires no changes to the
training script — only a new class and a registry entry.

---

## Registries

TSLM-Bench uses a simple dictionary-based registry pattern. Each registry maps string keys to classes and provides a lookup function.

### MODEL_REGISTRY (`src/models/registry.py`)

Maps architecture names to `BaseModel` subclasses (`src/models/base.py`).

```python
MODEL_REGISTRY = {
    "flamingo": OpenTSLMFlamingo,
}
```

- `get_model_class(name)` -- returns the class
- `create_model(config)` -- pops `"architecture"` from config, instantiates
- `list_architectures()` -- returns available names

### DATASET_REGISTRY (`src/datasets/registry.py`)

Maps dataset names to `QADataset` subclasses (`src/datasets/qa_base.py`).

```python
DATASET_REGISTRY = {
    "capture24_haystack_classification": TSHaystackQADataset,
    "capture24_haystack_cot": TSHaystackCoTQADataset,
    "capture24_classification": Capture24AccQADataset,
}
```

- `get_dataset_class(name)` -- returns the class
- `list_datasets()` -- returns available names

The experiment YAML's `dataset.name` selects a registered dataset and `dataset.extra_kwargs` passes constructor arguments.

### ENCODER_REGISTRY (`src/models/ts_encoder/__init__.py`)

Maps encoder names to `TimeSeriesEncoderBase` subclasses (`src/models/ts_encoder/base.py`).

```python
ENCODER_REGISTRY = {
    "cnn_tokenizer": CNNTokenizer,
    "transformer_cnn": TransformerCNNEncoder,
    "transformer_mlp": TransformerMLPEncoder,
}
```

### PROJECTOR_REGISTRY (`src/models/projector/__init__.py`)

Maps projector names to projector classes.

```python
PROJECTOR_REGISTRY = {
    "linear": LinearProjector,
    "mlp": MLPProjector,
    "perceiver_resampler": PerceiverResampler,
}
```

### BACKBONE_REGISTRY (`src/backbones/__init__.py`)

Maps backbone names to `BaseBackbone` subclasses (`src/backbones/base.py`).

```python
BACKBONE_REGISTRY = {
    "llama": LlamaBackbone,
}
```

---

## Base Classes

### BaseModel (`src/models/base.py`)

The single base class for all model implementations. Uses standard PyTorch
`forward()` with tensor arguments and separates preprocessing from the model.

```python
class BaseModel(nn.Module, ABC):
    # Abstract (must implement):
    def from_config(config, device) -> BaseModel          # classmethod
    def forward(time_series, input_ids, attention_mask, labels=None) -> dict
    def generate(time_series, input_ids, attention_mask, **kwargs) -> Tensor
    def prepare_batch(batch, training=True) -> dict[str, Tensor]
    def get_trainable_parameters() -> dict[str, list[Parameter]]
    def get_eos_token() -> str

    # Concrete (override if needed):
    def eval_prompt(prompt, max_new_tokens) -> str
    def get_num_parameters(trainable_only) -> dict[str, int]
    def get_backbone_module() -> nn.Module | None          # for LoRA support
```

See [Adding a New Model](adding-models.md) for implementation guidance.

### BaseBackbone (`src/backbones/base.py`)

Backbones handle LLM-specific tokenizer/model loading. They return the LLM's
**native** EOS token from `get_eos_token()`. Architecture-specific special tokens
(e.g. Flamingo's `<|endofchunk|>`, `<image>`) are added by the model, not the backbone.

```python
class BaseBackbone(ABC):
    def load_model(**kwargs) -> PreTrainedModel
    def load_tokenizer() -> PreTrainedTokenizer
    def get_prompt_template() -> str
    def get_eos_token() -> str               # native LLM EOS
    def get_ts_placeholder_token() -> str    # default: '<ts>' (non-abstract)
    def hidden_size -> int                   # property
    def num_layers -> int                    # property
    def apply_peft(model, config)            # LoRA support (built-in)
```

See `src/backbones/llama/backbone.py` for a reference implementation.

### TimeSeriesEncoderBase (`src/models/ts_encoder/base.py`)

```python
class TimeSeriesEncoderBase(nn.Module):
    def __init__(output_dim, dropout)
    def forward(x: Tensor) -> Tensor   # (B, L) -> (B, N, D)
```

### QADataset (`src/datasets/qa_base.py`)

The base class for QA-style datasets. Handles split management, lazy loading,
and prompt formatting via `PromptWithAnswer` (`src/prompt/prompt_with_answer.py`).

```python
class QADataset(Dataset, ABC):
    # Abstract (must implement):
    def _load_splits() -> (train, val, test)
    def _get_answer(row) -> str
    def _get_pre_prompt(row) -> str
    def _get_post_prompt(row) -> str
    def _get_text_time_series_prompt_list(row) -> list[TextTimeSeriesPrompt]

    # Override for dataset-specific evaluation:
    def extract_answer(prediction, sample) -> str
    def evaluate_answer(prediction, sample) -> dict
```

See [Adding a New Dataset](adding-datasets.md) for implementation guidance.

### BaseDataset (`src/datasets/base.py`)

A generic dataset interface available for non-QA tasks (classification,
forecasting, etc.). Not currently wired into the training pipeline but
provided as an extension point.

```python
class BaseDataset(Dataset, ABC):
    def __len__() -> int
    def __getitem__(idx) -> dict
    def get_collate_fn() -> Callable | None
    def download(data_dir) -> None
    def task_type -> str               # property
    def classes -> list[str] | None    # property
```

---

## Sample Schema

The sample dict is the data contract between datasets and models. It flows from
`QADataset.__getitem__()` through the DataLoader (identity collate) into
`model.prepare_batch()`.

### Dataset output (guaranteed keys)

Produced by `PromptWithAnswer.to_dict()` (`src/prompt/prompt_with_answer.py`):

```python
{
    "time_series": list[Tensor],   # one (seq_len,) tensor per channel
    "time_series_text": list[str], # label for each channel
    "pre_prompt": str,             # text before the time series
    "post_prompt": str,            # text/instructions after the time series
    "answer": str,                 # ground truth (EOS token appended)
}
```

Datasets may add extra keys (e.g. `task_type`, `answer_type`, `direct_answer`,
`question`, `context_length_samples`). Models should ignore keys they do not need.

### Model input (prepare_batch output)

`prepare_batch()` converts the sample dicts into tensors for `forward()` / `generate()`:

```python
{
    "time_series": Tensor,       # (B, channels, seq_len)
    "input_ids": Tensor,         # (B, text_len)
    "attention_mask": Tensor,    # (B, text_len)
    "labels": Tensor | None,     # (B, text_len) or None when not training
}
```

The model owns all tokenization, padding, and special-token insertion inside
`prepare_batch()`. The training script calls `model(**batch_inputs)` directly.

---

## Configuration System

All configuration lives in YAML files. `scripts/train.py` accepts a single
`--config` flag pointing to an experiment YAML (plus an optional `--no-wandb`
convenience override).

```bash
python scripts/train.py --config configs/experiments/capture24_haystack_llama.yaml
python scripts/train.py --config configs/experiments/capture24_haystack_llama.yaml --no-wandb
```

`src/utils/config.py` provides:
- `ExperimentConfig` -- top-level dataclass with nested `DatasetConfig`,
  `ModelConfig`, `BackboneConfig`, `TrainingConfig`, and `RuntimeConfig`
- `ExperimentConfig.from_yaml(path)` -- loads YAML, merges defaults,
  resolves `${ENV_VAR}` patterns, returns typed config
- `ExperimentConfig.to_dict()` -- serialises the full config (Path -> str)
  for saving as JSON/YAML in the run directory
- Convenience properties: `use_wandb`, `wandb_project`, `wandb_entity`,
  `early_stop_patience`, `checkpoint_interval`

```yaml
# configs/experiments/capture24_haystack_llama.yaml
name: capture24_haystack_llama3_1b
seed: 42
dataset:
  name: capture24_haystack_cot
model:
  architecture: flamingo
  encoder:
    type: cnn_tokenizer
backbone:
  name: llama
  model_id: meta-llama/Llama-3.2-1B-Instruct
training:
  batch_size: 2
  num_epochs: 30
  learning_rate:
    default: 2.0e-4
runtime:
  output_dir: results
  max_samples: null
```

---

## Training Pipeline

The training flow in `scripts/train.py`:

1. **Load config** from YAML via `ExperimentConfig.from_yaml()`
2. **Resolve context lengths** (auto-discover from filesystem if `"all"`)
3. **Create run directory** (`results/capture24_haystack_cot/run_YYYYMMDD_HHMMSS/`)
4. **Initialize logger** (`ExperimentLogger` -- W&B + local JSONL)
5. **Load model** via `MODEL_REGISTRY` lookup and `model_cls.from_config(config, device)`
6. **Configure training mode** (`freeze` / `lora` / `full`) using `model.get_trainable_parameters()`
7. **Create dataloaders** — the training script bridges the model and datasets
   by passing `model.get_eos_token()` into `create_dataloaders()`, which injects
   it as `EOS_TOKEN` into the dataset kwargs. This keeps datasets model-agnostic.
8. **Train loop** with AdamW + linear warmup + gradient clipping
9. **Validate** each epoch with loss + optional generation/accuracy
10. **Early stopping** on validation loss
11. **Final test evaluation** using best checkpoint

---

## Logging and Results

`ExperimentLogger` (`src/training/logging.py`) writes to:

```
results/capture24_haystack_cot/run_YYYYMMDD/
    config.json           # Experiment config snapshot
    history.json          # Per-epoch train/val losses and accuracy
    metrics.jsonl         # Step-level metrics (loss, LR, etc.)
    artifacts.jsonl       # Checkpoint metadata
    summary.json          # Final results
    experiment.log        # Full debug log
    checkpoints/
        best_model.pt
        latest_checkpoint.pt
    output_logs/
        val_epoch_1.json  # Per-sample predictions and accuracy
        test_epoch_0.json
```

---

## Key Files Quick Reference

| Purpose | File |
|---------|------|
| Training script | `scripts/train.py` |
| Experiment config | `src/utils/config.py` |
| Model base class | `src/models/base.py` |
| Model registry | `src/models/registry.py` |
| Flamingo model | `src/models/ts_llm/flamingo.py` |
| Backbone base class | `src/backbones/base.py` |
| Llama backbone | `src/backbones/llama/backbone.py` |
| Encoder base class | `src/models/ts_encoder/base.py` |
| Encoder registry | `src/models/ts_encoder/__init__.py` |
| Projector registry | `src/models/projector/__init__.py` |
| QA dataset base class | `src/datasets/qa_base.py` |
| Dataset registry | `src/datasets/registry.py` |
| Sample schema (PromptWithAnswer) | `src/prompt/prompt_with_answer.py` |
| Logging | `src/training/logging.py` |
| Default configs | `configs/defaults/` |
| Experiment configs | `configs/experiments/` |

---

# Future Improvements / Things to consider

- **Dataset split decoupling**: Currently the training script loads pre-split datasets. Perhaps we want to have that as a training concern and not a dataset concern.

- **Encoder/projector registries are only consumed by Flamingo**

The ENCODER_REGISTRY and PROJECTOR_REGISTRY exist but are only used inside OpenTSLMFlamingo.from_config() and OpenTSLMFlamingo.__init__(). A new model architecture (e.g., LLaVA-style) would have to replicate the registry lookup logic in its own from_config(). There's no framework-level wiring that automatically resolves config.model.encoder.type into an encoder class for all models.

- **No validation that config + model + dataset are compatible**
_This is more of a future improvement, we cannot overengineer for every problem_
Nothing prevents a user from writing a YAML that pairs a model expecting 3-channel input with a single-channel dataset, or a model that needs specific encoder output dims with the wrong projector. A config validation step would save hours of debugging.

- **Evaluation/ is empty**
_Let's see if this become important, a per dataset evaluation makes sense to me_
The evaluation module (src/evaluation/) has empty __init__.py files. Dataset-level evaluate_answer() and extract_answer() exist on QADataset, but there's no framework for aggregating metrics, computing per-task breakdowns, or standardized evaluation across models. For a bench, this is a significant gap.
