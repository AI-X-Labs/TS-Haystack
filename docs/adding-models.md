# Adding a New Model

This guide walks through adding a new TSLM architecture to TSLM-Bench. A TSLM combines a **time series encoder**, a **projector**, and an **LLM backbone** to answer questions about time series data.

---

## Architecture Overview

Every TSLM in TSLM-Bench follows this data flow:

```
Raw Time Series  -->  [Encoder]  -->  [Projector]  -->  [LLM Backbone]  -->  Text Output
 (B, channels, L)    (B, N, D_enc)   (B, M, D_llm)    (B, vocab_size)
```

The framework provides registries for each component:

| Component | Registry | Location |
|-----------|----------|----------|
| Full model | `MODEL_REGISTRY` | `src/models/registry.py` |
| Encoder | `ENCODER_REGISTRY` | `src/models/ts_encoder/__init__.py` |
| Projector | `PROJECTOR_REGISTRY` | `src/models/projector/__init__.py` |
| Backbone | `BACKBONE_REGISTRY` | `src/backbones/__init__.py` |

You can add a new full architecture, or just swap in a new encoder/projector/backbone and reuse the existing Flamingo wiring.

---

## Option A: Add a New Encoder

The simplest extension. Create a new encoder that converts raw time series patches into embeddings.

### 1. Implement the encoder

Create `src/models/ts_encoder/my_encoder.py`. Inherit from `TimeSeriesEncoderBase` (`src/models/ts_encoder/base.py`):

```python
import torch
import torch.nn as nn
from src.models.ts_encoder.base import TimeSeriesEncoderBase


class MyEncoder(TimeSeriesEncoderBase):
    """A custom time series encoder."""

    def __init__(
        self,
        output_dim: int = 128,
        dropout: float = 0.0,
        patch_size: int = 4,
        # ... your parameters
    ):
        super().__init__(output_dim, dropout)
        self.patch_size = patch_size
        # Define your layers here
        self.layers = nn.Sequential(...)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L) raw time series, where L is divisible by patch_size.

        Returns:
            (B, N, output_dim) patch embeddings, where N = L // patch_size.
        """
        # Your encoding logic
        return self.layers(x)
```

**Key contract**: Input shape is `(B, L)`, output shape is `(B, N, output_dim)`.

### 2. Register it

Edit `src/models/ts_encoder/__init__.py`:

```python
from src.models.ts_encoder.my_encoder import MyEncoder

ENCODER_REGISTRY: dict[str, type[TimeSeriesEncoderBase]] = {
    "cnn_tokenizer": CNNTokenizer,
    "transformer_cnn": TransformerCNNEncoder,
    "transformer_mlp": TransformerMLPEncoder,
    "my_encoder": MyEncoder,  # <-- add this
}
```

### 3. Use it in training

Once registered, select your encoder in the experiment YAML:

```yaml
# configs/experiments/my_experiment.yaml
model:
  encoder:
    type: my_encoder
    patch_size: 16        # your custom params
```

```bash
python scripts/train.py --config configs/experiments/my_experiment.yaml
```

`OpenTSLMFlamingo` looks up the encoder from `ENCODER_REGISTRY` and passes `max_patches` / `trained_patches` as default kwargs. If your encoder needs extra parameters, you can supply them via `encoder_kwargs` in code.

---

## Option B: Add a New LLM Backbone

Add support for a new LLM family (e.g., Qwen, Gemma, Mistral).

### 1. Implement the backbone

Create `src/backbones/qwen/backbone.py`. Inherit from `BaseBackbone` (`src/backbones/base.py`):

```python
from typing import Any
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
from src.backbones.base import BaseBackbone


class QwenBackbone(BaseBackbone):
    """Backbone for Qwen models."""

    def __init__(self, model_id: str = "Qwen/Qwen2.5-1.5B", **kwargs: Any):
        super().__init__(model_id, **kwargs)
        self._model = None
        self._tokenizer = None

    def load_model(self, **kwargs: Any) -> PreTrainedModel:
        defaults = {
            "trust_remote_code": True,
            "attn_implementation": "eager",
        }
        defaults.update(kwargs)
        self._model = AutoModelForCausalLM.from_pretrained(self.model_id, **defaults)
        if self._tokenizer is not None:
            self._model.resize_token_embeddings(len(self._tokenizer))
        return self._model

    def load_tokenizer(self) -> PreTrainedTokenizer:
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, trust_remote_code=True,
        )
        # Only add pad token — architecture-specific tokens are added by the model
        if self._tokenizer.pad_token is None:
            self._tokenizer.add_special_tokens({"pad_token": "<PAD>"})
        return self._tokenizer

    def get_prompt_template(self) -> str:
        return "{user}{assistant}"

    def get_eos_token(self) -> str:
        if self._tokenizer is not None:
            return self._tokenizer.eos_token
        return "<|endoftext|>"  # Qwen default

    @property
    def hidden_size(self) -> int:
        if self._model is not None:
            return self._model.config.hidden_size
        return 1536  # Qwen2.5-1.5B default

    @property
    def num_layers(self) -> int:
        if self._model is not None:
            return self._model.config.num_hidden_layers
        return 28
```

### 2. Register it

Edit `src/backbones/__init__.py`:

```python
from src.backbones.qwen import QwenBackbone

BACKBONE_REGISTRY: dict[str, type[BaseBackbone]] = {
    "llama": LlamaBackbone,
    "qwen": QwenBackbone,  # <-- add this
}
```

### 3. Create `src/backbones/qwen/__init__.py`

```python
from src.backbones.qwen.backbone import QwenBackbone
__all__ = ["QwenBackbone"]
```

**Note**: The backbone is resolved via `BACKBONE_REGISTRY` in `from_config()`. To add a new LLM family, register it and set `backbone.name` in your YAML. LoRA support comes free via `BaseBackbone.apply_peft()` and `default_lora_targets`.

**Important**: Backbones should NOT add architecture-specific special tokens (e.g. `<|endofchunk|>`, `<image>`). Those belong to the model architecture — for example, `OpenTSLMFlamingo.__init__()` adds its own tokens after calling `backbone.load_tokenizer()`. The backbone's `get_eos_token()` should return the LLM's native EOS token.

See `src/backbones/llama/backbone.py` for a complete reference implementation.

---

## Option C: Add a Complete Model Architecture

Add an entirely new TS-LLM design (e.g., LLaVA-style, Mamba-based, or a simple prefix-tuning approach).

### 1. Implement the model

Create `src/models/ts_llm/my_model.py`. Your model must extend `BaseModel` (`src/models/base.py`):

```python
from typing import Any
import torch
import torch.nn as nn
from src.models.base import BaseModel


class MyTSLM(BaseModel):
    def __init__(self, device: str, llm_id: str = "meta-llama/Llama-3.2-1B", **kwargs):
        super().__init__(device)
        # Build your model components here:
        # 1. Load tokenizer  (set self.tokenizer)
        # 2. Load LLM backbone
        # 3. Build encoder + projector

    @classmethod
    def from_config(cls, config: "ExperimentConfig", device: str) -> "MyTSLM":
        """Construct from an ExperimentConfig.

        The training script calls model_cls.from_config(config, device) — this
        is the only entry point. Extract the parameters your model needs from
        config.model, config.backbone, config.training, etc.
        """
        return cls(
            device=device,
            llm_id=config.backbone.model_id,
            # ... extract your params from config.model, config.backbone, etc.
        )

    def prepare_batch(
        self, batch: list[dict[str, Any]], training: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Convert dataset sample dicts into tensors.

        Each item in `batch` is a dict with guaranteed keys (see Sample Schema
        in architecture.md):

            "time_series":      list[Tensor]  — one (seq_len,) tensor per channel
            "time_series_text": list[str]     — label for each channel
            "pre_prompt":       str           — text before the time series
            "post_prompt":      str           — text after the time series
            "answer":           str           — ground truth (EOS appended)

        Datasets may include extra keys (task_type, answer_type, etc.) — ignore
        any keys your model doesn't need.

        Must return:
            {
                "time_series":    Tensor (B, channels, seq_len),
                "input_ids":      Tensor (B, text_len),
                "attention_mask": Tensor (B, text_len),
                "labels":         Tensor (B, text_len) or None,
            }
        """
        # 1. Stack and pad time series across batch
        # 2. Tokenize text (insert your model-specific special tokens)
        # 3. If training: create labels with -100 masking for prompt tokens
        ...

    def forward(
        self,
        time_series: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Training forward pass.

        Returns:
            Dict with 'loss' (scalar tensor).
        """
        # 1. Encode time series -> embeddings
        # 2. Project to LLM embedding space
        # 3. Forward through LLM with labels
        # 4. Return {"loss": loss}
        ...

    def generate(
        self,
        time_series: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **generate_kwargs: Any,
    ) -> torch.Tensor:
        """Generate answer token IDs.

        Returns:
            Token IDs of shape (B, generated_len), input prompt stripped.
        """
        ...

    def get_trainable_parameters(self) -> dict[str, list[nn.Parameter]]:
        """Return parameter groups that should be trainable.

        The training script uses this to selectively unfreeze parameters.
        Each group gets its own optimizer entry with configurable learning
        rate via training.learning_rate.<group_name> in the YAML config.
        """
        return {
            "encoder": list(self.encoder.parameters()),
            "projector": list(self.projector.parameters()),
            # Add more groups as needed
        }

    def get_eos_token(self) -> str:
        """Return the token that terminates generated answers.

        The training script passes this to the dataset as EOS_TOKEN, which
        appends it to ground truth answers. This keeps datasets model-agnostic.
        """
        return self.tokenizer.eos_token
```

### 2. Register it

Edit `src/models/registry.py`:

```python
from src.models.ts_llm.my_model import MyTSLM

MODEL_REGISTRY: dict[str, type[BaseModel]] = {
    "flamingo": OpenTSLMFlamingo,
    "my_model": MyTSLM,  # <-- add this
}
```

### 3. Train

Create an experiment YAML that uses your model:

```yaml
# configs/experiments/my_model_experiment.yaml
model:
  architecture: my_model
backbone:
  model_id: meta-llama/Llama-3.2-1B
```

```bash
python scripts/train.py --config configs/experiments/my_model_experiment.yaml
```

The `model.architecture` field looks up your model in `MODEL_REGISTRY`, and
`from_config()` handles construction.

---

## Sample Schema Reference

The data contract between datasets and models. See [Sample Schema](architecture.md#sample-schema) for full details.

### Dataset output (input to prepare_batch)

Each item in the `batch` list passed to `prepare_batch()` is a dict produced by `QADataset.__getitem__()`:

```python
{
    # --- Guaranteed keys (from PromptWithAnswer.to_dict()) ---
    "time_series": list[Tensor],       # one (seq_len,) tensor per channel
    "time_series_text": list[str],     # label for each channel
    "pre_prompt": str,                 # text before the time series
    "post_prompt": str,                # text/instructions after the time series
    "answer": str,                     # ground truth answer (with EOS token)

    # --- Optional keys (dataset-specific, may or may not be present) ---
    "direct_answer": str,              # short answer for evaluation (CoT datasets)
    "task_type": str,                  # e.g., "existence", "counting"
    "answer_type": str,                # e.g., "boolean", "integer"
    "context_length_samples": int,     # window size in samples
    "question": str,                   # the question text
}
```

### Model output (prepare_batch return value)

```python
{
    "time_series": Tensor,       # (B, channels, seq_len)
    "input_ids": Tensor,         # (B, text_len)
    "attention_mask": Tensor,    # (B, text_len)
    "labels": Tensor | None,     # (B, text_len) or None when not training
}
```

The training script calls `model(**batch_inputs)` for training and `model.generate(time_series, input_ids, attention_mask)` for evaluation.

---

## Backbone Training Modes

The framework manages backbone LLM training via `training.backbone_training` in the experiment YAML. Model authors **do not** need to implement freezing or PEFT logic.

```yaml
training:
  backbone_training: freeze    # freeze | lora | full
```

| Mode | Backbone LLM | Model components |
|------|-------------|------------------|
| `freeze` | Frozen | Trainable (per `get_trainable_parameters()`) |
| `lora` | LoRA adapters injected, trainable | Trainable (per `get_trainable_parameters()`) |
| `full` | All trainable | All trainable |

### How it works

1. **`get_trainable_parameters()`** (required) — your model declares which of its own components should be trainable. The framework freezes everything else and unfreezes these groups. Each group gets its own optimizer entry with configurable learning rate via `training.learning_rate.<group_name>`.

2. **`get_backbone_module()`** (optional) — override this to return the backbone LLM `nn.Module` if you want framework-managed LoRA support. The training script calls `backbone.apply_peft()` on it when `backbone_training: lora`. If not overridden (returns `None`), `lora` mode raises an error.

### LoRA configuration

When `backbone_training: lora`, these settings control the adapters:

```yaml
training:
  backbone_training: lora
  lora:
    r: 16                      # LoRA rank
    alpha: 32                  # LoRA alpha
    dropout: 0.05              # LoRA dropout
    target_modules: null       # null = backbone-specific defaults (e.g. [q_proj, v_proj])
```

`target_modules: null` uses the backbone's `default_lora_targets` property. Override this in your `BaseBackbone` subclass for model-family-specific defaults (e.g., Qwen might target different modules than LLaMA).

---

## Key Files

| Purpose | File |
|---------|------|
| Model base class (interface to implement) | `src/models/base.py` |
| Model registry | `src/models/registry.py` |
| Flamingo model (reference implementation) | `src/models/ts_llm/flamingo.py` |
| Backbone base class | `src/backbones/base.py` |
| Llama backbone (reference implementation) | `src/backbones/llama/backbone.py` |
| Encoder base class | `src/models/ts_encoder/base.py` |
| Encoder registry | `src/models/ts_encoder/__init__.py` |
| Projector registry | `src/models/projector/__init__.py` |
| Sample schema (PromptWithAnswer) | `src/prompt/prompt_with_answer.py` |
| Training script | `scripts/train.py` |
| Config dataclasses | `src/utils/config.py` |

---

## Things to Know

- **`from_config()` is abstract**: `BaseModel` enforces that every model implements `from_config(config, device)`. The training script calls `model_cls.from_config(config, device)` — this is the only construction path.
- **Backbone registry is wired**: `OpenTSLMFlamingo.from_config()` resolves the backbone via `get_backbone(config.backbone.name)` and delegates tokenizer/model loading to the `BaseBackbone` instance. To add a new LLM family, register it in `BACKBONE_REGISTRY` and set `backbone.name` in your YAML config.
- **Encoder params are forwarded from config**: All keys under `model.encoder` (except `type`, `trained_patches`, `max_patches`) are passed through as `encoder_kwargs` to the encoder constructor. For example, `patch_size`, `dropout`, and `transformer_input_dim` in the YAML flow directly into `CNNTokenizer.__init__()`.
- **`PATCH_SIZE` is config-driven**: `scripts/train.py` reads `patch_size` from `config.model.encoder` (default: 4) instead of importing from `model_config.py`.
- **`self.tokenizer` must be set**: The training script uses `model.tokenizer` to decode generated token IDs for logging. Set this attribute in your `__init__()`.
