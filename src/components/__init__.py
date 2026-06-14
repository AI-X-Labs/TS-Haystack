# SPDX-FileCopyrightText: 2026 Anonymous Authors
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Shared architectural primitives for composing TSLM architectures."""

# Decoder layer utilities
from src.components.decoder_layers import (
    _KNOWN_DECODER_LAYERS_ATTR_NAMES,
    _get_decoder_layers,
    _infer_decoder_layers_attr_name,
    _set_decoder_layers,
)

# Attention
from src.components.cross_attention import SteerableCrossAttention
from src.components.itformer_attention import InstructTimeAttention, SeqAttention, VarAttention

# SSM
from src.components.ssm import AttentionPooling, MemoryBank, SelectiveSSMBlock

# Pooling
from src.components.multi_scale_pooling import MultiScalePooling

# Projection / alignment
from src.components.alignment import AlignmentProjector

# Fusion
from src.components.fusion_layer import FusionLayerWrapper
from src.components.gated_ff import GatedFeedForward

# Positional encoding
from src.components.positional_encoding import (
    LearnablePositionalEmbedding,
    RotaryPositionalEncoding,
    SinusoidalPositionalEncoding,
)

# Regularization
from src.components.drop_path import DropPath

__all__ = [
    # Decoder layer utilities
    "_KNOWN_DECODER_LAYERS_ATTR_NAMES",
    "_get_decoder_layers",
    "_infer_decoder_layers_attr_name",
    "_set_decoder_layers",
    # Attention
    "SteerableCrossAttention",
    "InstructTimeAttention",
    "SeqAttention",
    "VarAttention",
    # SSM
    "AttentionPooling",
    "MemoryBank",
    "SelectiveSSMBlock",
    # Pooling
    "MultiScalePooling",
    # Projection / alignment
    "AlignmentProjector",
    # Fusion
    "FusionLayerWrapper",
    "GatedFeedForward",
    # Positional encoding
    "LearnablePositionalEmbedding",
    "RotaryPositionalEncoding",
    "SinusoidalPositionalEncoding",
    # Regularization
    "DropPath",
]
