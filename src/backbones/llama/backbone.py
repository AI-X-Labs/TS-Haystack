# SPDX-FileCopyrightText: 2026 Anonymous Authors
# SPDX-License-Identifier: CC-BY-NC-4.0
"""LLaMA backbone implementation."""

from typing import Any

from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from src.backbones.base import BaseBackbone


class LlamaBackbone(BaseBackbone):
    """Backbone for Meta LLaMA models (e.g., Llama-3.2-1B, Llama-3.2-3B).

    Handles tokenizer setup and model loading. Architecture-specific special
    tokens (e.g. Flamingo's ``<|endofchunk|>``) are added by the model, not here.
    """

    def __init__(self, model_id: str = "meta-llama/Llama-3.2-1B", **kwargs: Any):
        super().__init__(model_id, **kwargs)
        self._model = None
        self._tokenizer = None

    def load_model(self, **kwargs: Any) -> PreTrainedModel:
        """Load the LLaMA model from HuggingFace.

        Args:
            **kwargs: Passed to AutoModelForCausalLM.from_pretrained().
                Common options: device_map, torch_dtype, attn_implementation.

        Returns:
            The loaded LLaMA model.
        """
        defaults = {
            "local_files_only": False,
            "trust_remote_code": True,
            "cache_dir": None,
            "attn_implementation": "eager",
        }
        defaults.update(kwargs)

        self._model = AutoModelForCausalLM.from_pretrained(self.model_id, **defaults)

        # Resize embeddings if tokenizer has been loaded with extra tokens
        if self._tokenizer is not None:
            self._model.resize_token_embeddings(len(self._tokenizer))

        return self._model

    def load_tokenizer(self) -> PreTrainedTokenizer:
        """Load and configure the LLaMA tokenizer.

        Only adds a pad token if missing. Architecture-specific special tokens
        are added by the model (e.g. Flamingo adds ``<|endofchunk|>``).

        Returns:
            Configured tokenizer.
        """
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            local_files_only=False,
            trust_remote_code=True,
            cache_dir=None,
        )

        # Only add pad token — architecture-specific tokens are added by the model
        if self._tokenizer.pad_token is None:
            self._tokenizer.add_special_tokens({"pad_token": "<PAD>"})
            self._tokenizer.pad_token = "<PAD>"

        return self._tokenizer

    def get_prompt_template(self) -> str:
        """Return a simple prompt template for LLaMA.

        Returns:
            Template string with {user} and {assistant} placeholders.
        """
        return "{user}{assistant}"

    def get_eos_token(self) -> str:
        """Return the native LLaMA end-of-sequence token.

        Returns:
            The tokenizer's EOS token (e.g. '<|end_of_text|>' for LLaMA 3.x).
        """
        if self._tokenizer is not None:
            return self._tokenizer.eos_token
        return "<|end_of_text|>"  # LLaMA 3.x default

    @property
    def hidden_size(self) -> int:
        """Return the hidden dimension of the LLaMA model."""
        if self._model is not None:
            config = self._model.config
            if hasattr(config, "hidden_size"):
                return config.hidden_size
            if hasattr(config, "text_config") and hasattr(config.text_config, "hidden_size"):
                return config.text_config.hidden_size
        # Fallback defaults based on model name
        model_name = self.model_id.lower()
        if "3b" in model_name:
            return 3072
        return 2048  # Llama-3.2-1B default

    @property
    def num_layers(self) -> int:
        """Return the number of transformer layers."""
        if self._model is not None:
            config = self._model.config
            if hasattr(config, "num_hidden_layers"):
                return config.num_hidden_layers
        model_name = self.model_id.lower()
        if "3b" in model_name:
            return 28
        return 16  # Llama-3.2-1B default
