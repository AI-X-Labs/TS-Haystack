# SPDX-FileCopyrightText: 2026 Anonymous Authors
# SPDX-License-Identifier: CC-BY-NC-4.0

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.ts_encoder.base import TimeSeriesEncoderBase


class TransformerMLPEncoder(TimeSeriesEncoderBase):
    def __init__(
        self,
        output_dim: int = 128,
        dropout: float = 0.0,
        num_heads: int = 8,
        num_layers: int = 6,
        patch_size: int = 4,
        ff_dim: int = 2048,
        max_patches: int = 1024,
        trained_patches: int = 32,
    ):
        """
        Transformer encoder with MLP (Linear) patch embedding.

        Unlike CNNTokenizer/TransformerCNNEncoder which use Conv1d for patching,
        this encoder reshapes the input into (B, N, patch_size) patches and
        projects each patch via a Linear layer.

        Args:
            output_dim: dimension of patch embeddings (and transformer d_model)
            dropout: dropout probability
            num_heads: number of attention heads
            num_layers: number of TransformerEncoder layers
            patch_size: length of each patch
            ff_dim: hidden size of the feed-forward network inside each encoder layer
            max_patches: maximum number of patches expected per sequence (for pos emb)
            trained_patches: number of patches seen during training; longer sequences
                             use interpolated positional embeddings
        """
        super().__init__(output_dim, dropout)
        self.patch_size = patch_size
        self.trained_patches = trained_patches

        # 1) Linear patch embedding: each (patch_size,) patch -> (output_dim,)
        self.patch_embed = nn.Linear(patch_size, output_dim)

        # 2) Learnable positional embeddings (interpolated at init if needed)
        if trained_patches >= max_patches:
            self.pos_embed = nn.Parameter(
                torch.randn(1, max_patches, output_dim) * 0.02
            )
        else:
            base = torch.randn(1, trained_patches, output_dim) * 0.02
            interpolated = F.interpolate(
                base.transpose(1, 2), size=max_patches, mode="linear", align_corners=True
            ).transpose(1, 2)
            self.pos_embed = nn.Parameter(interpolated)
            print(f"Positional embeddings: interpolated {trained_patches} -> {max_patches} at init")

        # 3) Input norm + dropout
        self.input_norm = nn.LayerNorm(output_dim)
        self.input_dropout = nn.Dropout(self.dropout)

        # 4) Stack of TransformerEncoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=output_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=self.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: FloatTensor of shape [B, L], a batch of raw time series.
        Returns:
            FloatTensor of shape [B, N, output_dim], where N = L // patch_size.
        """
        B, L = x.shape
        if L % self.patch_size != 0:
            raise ValueError(
                f"Sequence length {L} not divisible by patch_size {self.patch_size}"
            )

        N = L // self.patch_size

        # Reshape into patches: (B, L) -> (B, N, patch_size)
        x = x.reshape(B, N, self.patch_size)

        # Linear patch embedding: (B, N, patch_size) -> (B, N, output_dim)
        x = self.patch_embed(x)

        # Add positional embeddings (just slice — interpolation done at init)
        x = x + self.pos_embed[:, :N, :]

        # Norm + dropout
        x = self.input_norm(x)
        x = self.input_dropout(x)

        # Apply Transformer encoder
        x = self.encoder(x)

        return x
