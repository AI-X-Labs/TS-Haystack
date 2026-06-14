# SPDX-FileCopyrightText: 2026 Anonymous Authors
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Unified fusion layer wrapper for injecting cross-attention into LLM decoder layers."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.components.cross_attention import SteerableCrossAttention
from src.components.gated_ff import GatedFeedForward


class FusionLayerWrapper(nn.Module):
    """Wraps an LLM decoder layer with a SteerableCrossAttention and optional GatedFeedForward.

    When ``ff=None``, behaves identically to the original ``DeepFusionLayer``.
    When ``ff`` is provided, behaves identically to the original ``MambaFusionLayer``.

    Set ``self.ts_features`` externally before the forward pass; it is
    cleared by the model-level cleanup (not auto-cleared here to support
    autoregressive generation).
    """

    def __init__(
        self,
        decoder_layer: nn.Module,
        cross_attn: SteerableCrossAttention,
        ff: GatedFeedForward | None = None,
    ):
        super().__init__()
        self.decoder_layer = decoder_layer
        self.cross_attn = cross_attn
        self.ff = ff
        self.ts_features: torch.Tensor | None = None

    @property
    def attention_type(self):
        """Proxy for HF transformers compatibility (cf. Flamingo FlamingoLayer)."""
        return getattr(self.decoder_layer, "attention_type", None)

    def forward(self, *args, **kwargs):
        outputs = self.decoder_layer(*args, **kwargs)

        # decoder layers return a tuple; hidden_states is always the first element
        if isinstance(outputs, tuple):
            hidden_states = outputs[0]
        else:
            hidden_states = outputs

        if self.ts_features is not None:
            hidden_states = self.cross_attn(hidden_states, self.ts_features)
            if self.ff is not None:
                hidden_states = self.ff(hidden_states)

        # NOTE: Do NOT auto-clear ts_features here. During autoregressive
        # generation, each decoder layer is called multiple times (once per
        # generated token). Clearing here would remove TS conditioning after
        # the prefill pass. The model-level _clear_fusion_layers() in the
        # finally block of forward()/generate() handles cleanup instead.

        if isinstance(outputs, tuple):
            return (hidden_states,) + outputs[1:]
        return hidden_states
