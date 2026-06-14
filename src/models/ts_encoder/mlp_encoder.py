# SPDX-FileCopyrightText: 2026 Anonymous Authors
# SPDX-License-Identifier: CC-BY-NC-4.0
"""MLP patch encoder for time series — ported from ChatTS.

Registered in ENCODER_REGISTRY as ``mlp_encoder``.

Takes flattened SP-encoded input ``(B, flat_len, num_features)`` and returns a
ragged tensor of patches: ``(total_patches, hidden_size)`` plus per-item patch
counts ``(B,)``.  The model class (ChatTSModel) handles the mapping from ragged
patches back to token positions.

Reference: ChatTS/chatts/vllm/chatts_vllm.py:61-193 (TimeSeriesEmbedding)
"""

import torch
import torch.nn as nn

from src.models.ts_encoder.base import TimeSeriesEncoderBase


class MLPEncoder(TimeSeriesEncoderBase):
    """MLP patch encoder that processes SP-encoded time series.

    Input:  ``(B, flat_len)`` or ``(B, flat_len, 1)`` — SP-encoded time series
            where ``flat_len = seq_len * num_features`` (interleaved values
            and validity markers).
    Output: tuple of ``(features, patch_cnt)``
        - ``features``: ``(total_patches_across_batch, hidden_size)``
        - ``patch_cnt``: ``(B,)`` integer tensor with patch count per item
    """

    def __init__(
        self,
        patch_size: int = 20,
        num_layers: int = 3,
        hidden_size: int = 1024,
        num_features: int = 2,
        max_sequence_length: int = 4096,
        **kwargs,
    ):
        super().__init__(output_dim=hidden_size, dropout=0.0)
        self.patch_size = patch_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.num_features = num_features
        self.max_sequence_length = max_sequence_length

        # MLP input: one value feature per timestep in the patch
        input_size = 1 * self.patch_size

        layers: list[nn.Module] = []
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.GELU())
            input_size = hidden_size
        layers.append(nn.Linear(input_size, hidden_size))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode SP-encoded time series into patch features.

        Args:
            x: ``(B, flat_len)`` or ``(B, flat_len, *)`` — the encoder
               reshapes internally using ``num_features``.

        Returns:
            ``(features, patch_cnt)`` where features is
            ``(total_patches, hidden_size)`` and patch_cnt is ``(B,)``.
        """
        batch_size = x.size(0)
        x = x.reshape(batch_size, -1, self.num_features)

        # Last feature column is the validity mask
        mask = x[:, :, -1].long()
        valid_lengths = mask.sum(dim=1).long()
        patch_cnt = (valid_lengths + self.patch_size - 1) // self.patch_size

        patches_list: list[torch.Tensor] = []

        for i in range(batch_size):
            vl = valid_lengths[i].item()
            pc = patch_cnt[i].item()
            if pc == 0:
                continue

            # Extract value feature only (first column)
            xi = x[i, :vl, :1]  # (vl, 1)
            total_padded = pc * self.patch_size
            pad_len = total_padded - vl

            if pad_len > 0:
                # Pad with last observed value
                last_val = xi[-1:, :]
                padding = last_val.repeat(pad_len, 1)
                xi = torch.cat([xi, padding], dim=0)

            # Reshape into patches: (pc, patch_size)
            xi = xi.reshape(pc, self.patch_size)
            patches_list.append(xi)

        if patches_list:
            x_patches = torch.cat(patches_list, dim=0)
            features = self.mlp(x_patches)
        else:
            features = torch.empty(0, self.hidden_size, device=x.device, dtype=x.dtype)

        return features, patch_cnt
