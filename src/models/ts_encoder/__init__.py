# SPDX-FileCopyrightText: 2026 Anonymous Authors
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Time series encoder implementations."""

from src.models.ts_encoder.base import TimeSeriesEncoderBase
from src.models.ts_encoder.cnn_tokenizer import CNNTokenizer
from src.models.ts_encoder.transformer_cnn import TransformerCNNEncoder
from src.models.ts_encoder.transformer_mlp import TransformerMLPEncoder
from src.models.ts_encoder.itformer_encoder import ITFormerTSEncoder
from src.models.ts_encoder.chronos2_encoder import Chronos2TSEncoder
from src.models.ts_encoder.timesfm_encoder import TimesFMTSEncoder
from src.models.ts_encoder.mlp_encoder import MLPEncoder

ENCODER_REGISTRY: dict[str, type[TimeSeriesEncoderBase]] = {
    "cnn_tokenizer": CNNTokenizer,
    "transformer_cnn": TransformerCNNEncoder,
    "transformer_mlp": TransformerMLPEncoder,
    "itformer_ts_encoder": ITFormerTSEncoder,
    "chronos2": Chronos2TSEncoder,
    "timesfm": TimesFMTSEncoder,
    "mlp_encoder": MLPEncoder,
}


def get_encoder(name: str) -> type[TimeSeriesEncoderBase]:
    """Look up an encoder class by name.

    Args:
        name: Registry key (e.g. 'cnn_tokenizer').

    Returns:
        The encoder class.

    Raises:
        KeyError: If the name is not registered.
    """
    if name not in ENCODER_REGISTRY:
        raise KeyError(
            f"Unknown encoder '{name}'. Available: {list(ENCODER_REGISTRY.keys())}"
        )
    return ENCODER_REGISTRY[name]


__all__ = [
    "TimeSeriesEncoderBase",
    "CNNTokenizer",
    "TransformerCNNEncoder",
    "TransformerMLPEncoder",
    "ITFormerTSEncoder",
    "Chronos2TSEncoder",
    "TimesFMTSEncoder",
    "MLPEncoder",
    "ENCODER_REGISTRY",
    "get_encoder",
]
