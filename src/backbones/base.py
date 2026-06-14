# SPDX-FileCopyrightText: 2026 Anonymous Authors
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Base backbone class defining the interface for LLM backbones."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from transformers import PreTrainedModel, PreTrainedTokenizer

if TYPE_CHECKING:
    import torch.nn as nn


class BaseBackbone(ABC):
    """Abstract base class for LLM backbones.

    Backbones encapsulate the specifics of different LLM families (LLaMA, Qwen, etc.)
    including tokenizer setup, chat templates, special tokens, and PEFT configuration.

    This abstraction allows the training infrastructure to work with any LLM
    without needing to know the specifics of each model family.
    """

    def __init__(self, model_id: str, **kwargs: Any):
        """Initialize the backbone.

        Args:
            model_id: HuggingFace model identifier (e.g., 'meta-llama/Llama-3.1-8B-Instruct')
            **kwargs: Additional backbone-specific configuration.
        """
        self.model_id = model_id
        self.config = kwargs

    @abstractmethod
    def load_model(self, **kwargs: Any) -> PreTrainedModel:
        """Load the pretrained LLM.

        Args:
            **kwargs: Additional arguments for model loading (device_map, torch_dtype, etc.)

        Returns:
            The loaded pretrained model.
        """
        pass

    @abstractmethod
    def load_tokenizer(self) -> PreTrainedTokenizer:
        """Load and configure the tokenizer.

        This should set up any special tokens needed for the model,
        including padding tokens, time series placeholder tokens, etc.

        Returns:
            Configured tokenizer instance.
        """
        pass

    @abstractmethod
    def get_prompt_template(self) -> str:
        """Return the chat/prompt template for this model.

        Returns:
            A string template with placeholders for system, user, and assistant content.
        """
        pass

    @abstractmethod
    def get_eos_token(self) -> str:
        """Return the end-of-sequence token for training.

        Returns:
            The EOS token string.
        """
        pass

    def get_ts_placeholder_token(self) -> str:
        """Return the placeholder token for time series embedding insertion.

        Override if your model architecture uses a different placeholder.
        Default: '<ts>'

        Returns:
            The placeholder token string.
        """
        return "<ts>"

    @property
    @abstractmethod
    def hidden_size(self) -> int:
        """Return the hidden dimension of the LLM.

        Used to configure the projector output dimension.

        Returns:
            Hidden dimension size.
        """
        pass

    @property
    @abstractmethod
    def num_layers(self) -> int:
        """Return the number of transformer layers in the LLM.

        Returns:
            Number of layers.
        """
        pass

    def apply_peft(
        self,
        model: "nn.Module",
        peft_config: dict[str, Any],
    ) -> None:
        """Apply LoRA adapters to the model in-place.

        Uses ``inject_adapter_in_model`` so the module tree is modified
        directly without wrapping in a ``PeftModel`` (which would break
        architectures that access LLM sub-modules by attribute).

        Args:
            model: The LLM module to inject LoRA into.
            peft_config: Dictionary with keys ``r``, ``alpha``,
                ``target_modules`` (list or None for backbone defaults),
                and ``dropout``.
        """
        from peft import LoraConfig, inject_adapter_in_model

        lora_config = LoraConfig(
            r=peft_config.get("r", 16),
            lora_alpha=peft_config.get("alpha", 32),
            target_modules=peft_config.get("target_modules") or self.default_lora_targets,
            lora_dropout=peft_config.get("dropout", 0.05),
            bias="none",
            task_type="CAUSAL_LM",
        )
        inject_adapter_in_model(lora_config, model)

    @property
    def default_lora_targets(self) -> list[str]:
        """Return default LoRA target modules for this backbone.

        Override in subclasses for model-specific defaults.

        Returns:
            List of module names to apply LoRA to.
        """
        return ["q_proj", "v_proj"]

    def format_prompt(
        self,
        pre_prompt: str,
        post_prompt: str,
        answer: str | None = None,
    ) -> str:
        """Format a prompt using this backbone's template.

        Args:
            pre_prompt: Text before the time series placeholder.
            post_prompt: Text/question after the time series placeholder.
            answer: Optional answer to include (for training).

        Returns:
            Formatted prompt string.
        """
        ts_token = self.get_ts_placeholder_token()
        template = self.get_prompt_template()

        user_content = f"{pre_prompt}{ts_token}{post_prompt}"

        if answer is not None:
            return template.format(user=user_content, assistant=answer)
        else:
            return template.format(user=user_content, assistant="")

    def get_model_config(self) -> dict[str, Any]:
        """Return the model configuration for reference.

        Returns:
            Dictionary containing model configuration.
        """
        return {
            "model_id": self.model_id,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "eos_token": self.get_eos_token(),
            "ts_placeholder": self.get_ts_placeholder_token(),
            **self.config,
        }
