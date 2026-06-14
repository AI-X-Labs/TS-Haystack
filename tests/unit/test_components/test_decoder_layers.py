# SPDX-FileCopyrightText: 2026 Anonymous Authors
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Tests for decoder layer attribute inference utilities."""

import pytest
import torch.nn as nn

from src.components.decoder_layers import (
    _get_decoder_layers,
    _infer_decoder_layers_attr_name,
    _set_decoder_layers,
)


# ---------------------------------------------------------------------------
# Mock model classes that mimic HuggingFace naming
# ---------------------------------------------------------------------------


class _MockLayers(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(4, 4) for _ in range(3)])


class _MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _MockLayers()


def _make_mock(cls_name: str) -> nn.Module:
    """Create a mock model with a dynamically-named class."""
    model = _MockModel()
    model.__class__ = type(cls_name, (_MockModel,), {})
    return model


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestInferDecoderLayersAttrName:
    def test_llama(self):
        model = _make_mock("LlamaForCausalLM")
        assert _infer_decoder_layers_attr_name(model) == "model.layers"

    def test_qwen(self):
        model = _make_mock("QwenForCausalLM")
        assert _infer_decoder_layers_attr_name(model) == "model.layers"

    def test_gemma3_text(self):
        model = _make_mock("Gemma3ForCausalLM")
        assert _infer_decoder_layers_attr_name(model) == "model.layers"

    def test_gemma3_conditional_generation(self):
        """Gemma3ForConditionalGeneration uses language_model.layers."""
        model = _make_mock("Gemma3ForConditionalGeneration")
        # Build the expected attribute path
        model.language_model = _MockLayers()
        assert _infer_decoder_layers_attr_name(model) == "language_model.layers"

    def test_unknown_raises(self):
        model = _make_mock("UnknownArchitectureXYZ")
        with pytest.raises(ValueError, match="Cannot infer"):
            _infer_decoder_layers_attr_name(model)


class TestGetSetDecoderLayers:
    def test_get_decoder_layers(self):
        model = _MockModel()
        layers = _get_decoder_layers(model, "model.layers")
        assert isinstance(layers, nn.ModuleList)
        assert len(layers) == 3

    def test_set_decoder_layers(self):
        model = _MockModel()
        new_layers = nn.ModuleList([nn.Linear(4, 4) for _ in range(5)])
        _set_decoder_layers(model, "model.layers", new_layers)
        assert len(model.model.layers) == 5
