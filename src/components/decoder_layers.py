# SPDX-FileCopyrightText: 2026 Anonymous Authors
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Utilities for accessing and modifying LLM decoder layers by attribute name.

Used by DeepFusion, MambaFusion, FlamingoMamba, and OpenTSLMFlamingo to
wrap or inspect the decoder layers of arbitrary HuggingFace causal-LM models.
"""

from __future__ import annotations

import torch.nn as nn


_KNOWN_DECODER_LAYERS_ATTR_NAMES = {
    "opt": "model.decoder.layers",
    "gptj": "transformer.h",
    "gpt-j": "transformer.h",
    "pythia": "gpt_neox.layers",
    "llama": "model.layers",
    "gptneoxforcausallm": "gpt_neox.layers",
    "mpt": "transformer.blocks",
    "mosaicgpt": "transformer.blocks",
    "gemma": "model.layers",
    "gemma2": "model.layers",
    "gemma3": "model.layers",
    "medgemma": "model.layers",
    "qwen": "model.layers",
}


def _infer_decoder_layers_attr_name(model: nn.Module) -> str:
    model_class_name = model.__class__.__name__
    if "gemma3" in model_class_name.lower():
        if "ConditionalGeneration" in model_class_name:
            return "language_model.layers"
        return "model.layers"

    for key in _KNOWN_DECODER_LAYERS_ATTR_NAMES:
        if key.lower() in model_class_name.lower():
            return _KNOWN_DECODER_LAYERS_ATTR_NAMES[key]

    raise ValueError(
        f"Cannot infer decoder layers attribute for {model_class_name}. "
        "Please add it to _KNOWN_DECODER_LAYERS_ATTR_NAMES."
    )


def _get_decoder_layers(model: nn.Module, attr_name: str) -> nn.ModuleList:
    parts = attr_name.split(".")
    obj = model
    for part in parts:
        obj = getattr(obj, part)
    return obj


def _set_decoder_layers(model: nn.Module, attr_name: str, new_layers: nn.ModuleList) -> None:
    parts = attr_name.split(".")
    obj = model
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], new_layers)
