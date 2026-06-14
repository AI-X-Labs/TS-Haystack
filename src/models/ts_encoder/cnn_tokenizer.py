# SPDX-FileCopyrightText: 2026 Anonymous Authors
# SPDX-License-Identifier: CC-BY-NC-4.0

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.ts_encoder.base import TimeSeriesEncoderBase


class CNNTokenizer(TimeSeriesEncoderBase):
    def __init__(
        self,
        output_dim: int = 128,
        dropout: float = 0.0,
        transformer_input_dim: int = 128,
        patch_size: int = 4,
        max_patches: int = 1024,
        trained_patches: int = 32,  # Number of patches seen during training (2.56s @ 50Hz / 4)
    ):
        """
        Args:
            embed_dim: dimension of patch embeddings
            num_heads: number of attention heads
            num_layers: number of TransformerEncoder layers
            patch_size: length of each patch
            ff_dim: hidden size of the feed-forward network inside each encoder layer
            dropout: dropout probability
            max_patches: maximum number of patches expected per sequence (for pos emb)
        """
        super().__init__(output_dim, dropout)
        self.patch_size = patch_size
        self.trained_patches = trained_patches  # For position interpolation

        # 1) Conv1d patch embedding: (B, 1, L) -> (B, embed_dim, L/patch_size)
        self.patch_embed = nn.Conv1d(
            in_channels=1,
            out_channels=transformer_input_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )

        # 2) Learnable positional embeddings
        #    If trained_patches < max_patches, interpolate once at init so that
        #    forward() only needs a simple slice.  This ensures the same absolute
        #    position always receives the same embedding regardless of sequence
        #    length, and avoids per-batch interpolation cost.
        if trained_patches >= max_patches:
            self.pos_embed = nn.Parameter(
                torch.randn(1, max_patches, transformer_input_dim) * 0.02
            )
        else:
            base = torch.randn(1, trained_patches, transformer_input_dim) * 0.02
            # (1, trained, dim) -> (1, dim, trained) -> interpolate -> (1, dim, max) -> (1, max, dim)
            interpolated = F.interpolate(
                base.transpose(1, 2), size=max_patches, mode="linear", align_corners=True
            ).transpose(1, 2)
            self.pos_embed = nn.Parameter(interpolated)
            print(f"Positional embeddings: interpolated {trained_patches} -> {max_patches} at init")

        # 3) Input norm + dropout
        self.input_norm = nn.LayerNorm(transformer_input_dim)
        self.input_dropout = nn.Dropout(self.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: FloatTensor of shape [B, L], a batch of raw time series.
        Returns:
            FloatTensor of shape [B, N, embed_dim], where N = L // patch_size.
        """

        B, L = x.shape
        if L % self.patch_size != 0:
            raise ValueError(
                f"Sequence length {L} not divisible by patch_size {self.patch_size}"
            )

        # reshape to (B, 1, L)
        x = x.unsqueeze(1)

        # conv patch embedding -> (B, embed_dim, N)
        x = self.patch_embed(x)

        # transpose to (B, N, embed_dim)
        x = x.transpose(1, 2)

        # add positional embeddings (just slice — interpolation done at init)
        N = x.size(1)
        x = x + self.pos_embed[:, :N, :]

        # norm + dropout
        x = self.input_norm(x)
        x = self.input_dropout(x)

        return x
