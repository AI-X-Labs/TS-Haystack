# SPDX-FileCopyrightText: 2026 Anonymous Authors
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Chronos-2 pretrained time series encoder.

Wraps Amazon's Chronos-2 transformer encoder as a composable
:class:`TimeSeriesEncoderBase` for use with Flamingo (univariate) and
ITFormer (multivariate via ``group_ids``).

Requires ``uv pip install chronos-forecasting>=1.4``.
Registered in ENCODER_REGISTRY as ``chronos2``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.ts_encoder.base import TimeSeriesEncoderBase


class Chronos2TSEncoder(TimeSeriesEncoderBase):
    """Pretrained Chronos-2 encoder usable as a drop-in TS encoder.

    Univariate (Flamingo): ``(B, L) -> (B, N, D)``
    Multivariate (ITFormer): ``(B, L, V) -> (B, V, N, D)``

    Chronos-2's ``GroupSelfAttention`` enables cross-variable information
    sharing when multiple channels share the same ``group_id``.
    """

    # No patch_size attribute — Flamingo skips external patching,
    # Chronos handles its own patching internally.

    def __init__(
        self,
        model_id: str = "amazon/chronos-2",
        output_dim: int = 512,
        dropout: float = 0.0,
        freeze: bool = True,
        multi_variable: bool = False,
        # Accept and ignore Flamingo's default kwargs:
        max_patches: int | None = None,
        trained_patches: int | None = None,
        **kwargs,
    ):
        try:
            from chronos import Chronos2Model
        except ImportError as exc:
            raise ImportError(
                "Chronos-2 encoder requires the chronos-forecasting package. "
                "Install it with: uv pip install 'chronos-forecasting>=1.4'"
            ) from exc

        super().__init__(output_dim=output_dim, dropout=dropout)

        self.chronos_model = Chronos2Model.from_pretrained(model_id)
        # Override output_dim from the actual model config
        self.output_dim = self.chronos_model.config.d_model
        self.multi_variable = multi_variable

        self.freeze = freeze
        if freeze:
            self.chronos_model.requires_grad_(False)

    def get_num_output_patches(self, input_length: int) -> int:
        """Return context patch count based on Chronos-2's internal patch size."""
        patch_size = self.chronos_model.config.chronos_config["input_patch_size"]
        return -(-input_length // patch_size)  # ceil division

    def _encode_single(
        self,
        x: torch.Tensor,
        group_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode univariate batch through Chronos-2.

        Args:
            x: ``(B, L)`` — univariate time series batch.
            group_ids: ``(B,)`` — optional group IDs for cross-variable
                attention via ``GroupSelfAttention``.

        Returns:
            ``(B, N, D)`` — context patch embeddings.
        """
        encoder_outputs, _loc_scale, _fut_mask, num_context_patches = (
            self.chronos_model.encode(
                context=x,
                group_ids=group_ids,
                num_output_patches=1,
            )
        )
        # last_hidden_state: (B, num_context_patches + [REG] + num_output_patches, D)
        # Keep only the context patches.
        hidden = encoder_outputs.last_hidden_state
        return hidden[:, :num_context_patches, :]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.multi_variable:
            # x: (B, L, V) — multi-variable input (ITFormer path)
            B, L, V = x.shape
            # Stack all variables into batch dim: (B*V, L)
            x_flat = x.permute(0, 2, 1).reshape(B * V, L)
            # Variables from the same sample share a group →
            # enables Chronos-2's GroupSelfAttention across channels
            group_ids = torch.arange(B, device=x.device).repeat_interleave(V)
            encoded = self._encode_single(x_flat, group_ids=group_ids)  # (B*V, N, D)
            _, N, D = encoded.shape
            return encoded.reshape(B, V, N, D)
        else:
            # x: (B, L) — single-channel input (Flamingo path)
            return self._encode_single(x)