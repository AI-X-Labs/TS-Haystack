# SPDX-FileCopyrightText: 2026 Anonymous Authors
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Base model class defining the interface for all TS-LLM models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

from src.prompt.full_prompt import FullPrompt

if TYPE_CHECKING:
    from src.utils.config import ExperimentConfig


class BaseModel(nn.Module, ABC):
    """Abstract base class for Time Series Language Models.

    All model implementations must inherit from this class and implement
    the required abstract methods to ensure compatibility with the training
    infrastructure.

    A TS-LLM model typically consists of:
        1. Time series encoder - converts raw time series to embeddings
        2. Projector - maps encoder output to LLM embedding space
        3. LLM backbone - generates text conditioned on time series

    Subclasses must set ``self.tokenizer`` to the tokenizer used by the
    backbone LLM (used by the default ``eval_prompt`` and by the training
    script for decoding generated token IDs).
    """

    def __init__(self, device: str):
        super().__init__()
        self.device = device

    @classmethod
    @abstractmethod
    def from_config(cls, config: ExperimentConfig, device: str) -> "BaseModel":
        """Construct a model from an :class:`ExperimentConfig`.

        Subclasses must implement this to extract the parameters they
        need from the config, keeping architecture-specific details out
        of the training script.
        """
        ...

    # ------------------------------------------------------------------
    # Abstract methods -- every subclass MUST implement these
    # ------------------------------------------------------------------

    @abstractmethod
    def forward(
        self,
        time_series: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Forward pass for training.

        Args:
            time_series: Time series input of shape (batch, channels, seq_len).
            input_ids: Tokenized text input of shape (batch, text_len).
            attention_mask: Attention mask of shape (batch, text_len).
            labels: Target token IDs for loss computation, shape (batch, text_len).
                Positions set to -100 are ignored in the loss.
            **kwargs: Extra model-specific inputs (e.g. ``query_ids``, ``stage``
                for ITFormer). Models that don't need them simply ignore them.

        Returns:
            Dictionary containing at minimum ``'loss'`` (scalar) when labels
            are provided. May also contain ``'logits'``.
        """
        ...

    @abstractmethod
    def generate(
        self,
        time_series: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **generate_kwargs: Any,
    ) -> torch.Tensor:
        """Generate text given time series and prompt.

        Args:
            time_series: Time series input of shape (batch, channels, seq_len).
            input_ids: Tokenized prompt of shape (batch, prompt_len).
            attention_mask: Optional attention mask.
            **generate_kwargs: Additional arguments (e.g. max_new_tokens,
                temperature, do_sample).

        Returns:
            Generated token IDs of shape (batch, generated_len).
            These are *answer-only* tokens (input prompt stripped).
        """
        ...

    @abstractmethod
    def prepare_batch(
        self,
        batch: list[dict[str, Any]],
        training: bool = True,
        normalize: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Convert a list of dataset sample dicts into tensors.

        This method encapsulates all model-specific collation, preprocessing,
        and tokenization logic. The returned dict can be unpacked directly
        into :meth:`forward` (when ``training=True``) or used to call
        :meth:`generate` (when ``training=False``).

        **Input sample schema** — each dict in ``batch`` is produced by
        ``QADataset.__getitem__()`` (via ``PromptWithAnswer.to_dict()``)
        and contains these guaranteed keys:

        .. code-block:: python

            {
                "time_series": list[Tensor],   # one (seq_len,) tensor per channel
                "time_series_text": list[str],  # label for each channel
                "pre_prompt": str,              # text before the time series
                "post_prompt": str,             # text/instructions after the time series
                "answer": str,                  # ground truth (EOS token appended)
            }

        Datasets may add extra keys (e.g. ``task_type``, ``answer_type``,
        ``direct_answer``, ``question``). Your implementation should
        ignore keys it does not need.

        **Output tensor schema** — must return:

        .. code-block:: python

            {
                "time_series": Tensor,       # (B, channels, seq_len)
                "input_ids": Tensor,         # (B, text_len)
                "attention_mask": Tensor,    # (B, text_len)
                "labels": Tensor | None,     # (B, text_len) or None
            }

        Args:
            batch: List of sample dicts from the dataset (see schema above).
            training: If True, include the answer in the tokenized text and
                return ``labels`` for loss computation. If False, tokenize
                the prompt only (for generation) and set ``labels`` to None.
            normalize: If True, normalize each time series to zero mean and
                unit variance before patch alignment.

        Returns:
            Dictionary matching the output tensor schema above.
            All values are tensors on the model device.
        """
        ...

    @abstractmethod
    def get_trainable_parameters(self) -> dict[str, list[nn.Parameter]]:
        """Return named parameter groups for the optimizer.

        This enables per-component learning rates and selective freezing.
        The training script uses this to decide which parameters to unfreeze.

        Returns:
            Dictionary mapping component names to parameter lists, e.g.::

                {
                    "encoder": [...],
                    "projector": [...],
                    "cross_attention": [...],
                }
        """
        ...

    @abstractmethod
    def get_eos_token(self) -> str:
        """Return the end-of-sequence token string used by this model."""
        ...

    # ------------------------------------------------------------------
    # Checkpoint loading
    # ------------------------------------------------------------------

    def load_from_file(self, path: str) -> None:
        """Load model parameters from a training checkpoint.

        Handles the standard checkpoint format produced by ``scripts/train.py``
        (``{"model_state": model.state_dict(), ...}``).  Subclasses may override
        for architecture-specific logic (e.g. legacy formats, key remapping).

        Args:
            path: Path to a ``.pt`` checkpoint file.
        """
        checkpoint = torch.load(path, map_location=self.device)

        if "model_state" in checkpoint:
            model_state = checkpoint["model_state"]
        else:
            # Assume the file is a raw state_dict
            model_state = checkpoint

        missing, unexpected = self.load_state_dict(model_state, strict=False)
        if missing:
            print(f"Warning: Missing keys when loading checkpoint:")
            for key in missing[:10]:
                print(f"   - {key}")
            if len(missing) > 10:
                print(f"   ... and {len(missing) - 10} more keys")
        if unexpected:
            print(f"Warning: Unexpected keys when loading checkpoint:")
            for key in unexpected[:10]:
                print(f"   - {key}")
            if len(unexpected) > 10:
                print(f"   ... and {len(unexpected) - 10} more keys")
        self.to(self.device)

    # ------------------------------------------------------------------
    # Optional overrides for framework-managed features
    # ------------------------------------------------------------------

    def get_backbone_module(self) -> nn.Module | None:
        """Return the backbone LLM module for framework-managed LoRA.

        The training infrastructure calls ``backbone.apply_peft()`` on the
        returned module when ``training.backbone_training`` is ``"lora"``.
        Override this in your model to enable LoRA without implementing
        PEFT logic yourself.

        Returns:
            The backbone ``nn.Module``, or ``None`` (default) if the model
            does not expose a backbone for PEFT.
        """
        return None

    # ------------------------------------------------------------------
    # Concrete methods with sensible defaults
    # ------------------------------------------------------------------

    def eval_prompt(
        self, prompt: FullPrompt, max_new_tokens: int = 1000
    ) -> str:
        """Evaluate a single prompt and return the generated text.

        Subclasses may override for architecture-specific behaviour (e.g.
        normalization, disabling torch.compile).

        Args:
            prompt: A :class:`FullPrompt` instance.
            max_new_tokens: Maximum tokens to generate.

        Returns:
            Generated answer string.
        """
        batch = [prompt.to_dict()]
        self.eval()
        eval_inputs = self.prepare_batch(batch, training=False)
        gen_inputs = {k: v for k, v in eval_inputs.items() if k != "labels"}
        token_ids = self.generate(**gen_inputs, max_new_tokens=max_new_tokens)
        return self.tokenizer.batch_decode(token_ids, skip_special_tokens=True)[0]

    def get_num_parameters(self, trainable_only: bool = False) -> dict[str, int]:
        """Count parameters in the model.

        Args:
            trainable_only: If True, only count trainable parameters.

        Returns:
            Dictionary with ``'total'`` and per-child-module parameter counts.
        """
        counts: dict[str, int] = {"total": 0}

        for name, module in self.named_children():
            count = sum(
                p.numel()
                for p in module.parameters()
                if not trainable_only or p.requires_grad
            )
            counts[name] = count
            counts["total"] += count

        return counts
