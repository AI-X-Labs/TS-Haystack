# SPDX-FileCopyrightText: 2026 Anonymous Authors
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Model implementations for TS-Haystack."""

from src.models.base import BaseModel
from src.models.registry import MODEL_REGISTRY, get_model_class, create_model, list_architectures
from src.models.ts_encoder import (
    TimeSeriesEncoderBase,
    CNNTokenizer,
    TransformerCNNEncoder,
    TransformerMLPEncoder,
    ENCODER_REGISTRY,
    get_encoder,
)
from src.models.projector import (
    LinearProjector,
    MLPProjector,
    PROJECTOR_REGISTRY,
    get_projector,
    get_projector_output_dim,
)
from src.models.ts_llm import (
    OpenTSLMFlamingo,
)

__all__ = [
    # Base
    "BaseModel",
    # Registry
    "MODEL_REGISTRY",
    "get_model_class",
    "create_model",
    "list_architectures",
    # Encoders
    "TimeSeriesEncoderBase",
    "CNNTokenizer",
    "TransformerCNNEncoder",
    "TransformerMLPEncoder",
    "ENCODER_REGISTRY",
    "get_encoder",
    # Projectors
    "LinearProjector",
    "MLPProjector",
    "PROJECTOR_REGISTRY",
    "get_projector",
    "get_projector_output_dim",
    # TS-LLM
    "OpenTSLMFlamingo",
]
