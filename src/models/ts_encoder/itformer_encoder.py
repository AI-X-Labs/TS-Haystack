# SPDX-FileCopyrightText: 2026 Anonymous Authors
# SPDX-License-Identifier: CC-BY-NC-4.0
"""ITFormer time series encoder — multi-variable patchified encoder.

Ported from ITFormer-clone/models/TimeSeriesEncoder.py.
Registered in ENCODER_REGISTRY as ``itformer_ts_encoder``.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.components.drop_path import DropPath
from src.components.itformer_attention import SeqAttention, VarAttention
from src.models.ts_encoder.base import TimeSeriesEncoderBase


# ---------------------------------------------------------------------------
# Helper modules
# ---------------------------------------------------------------------------


class GateLayer(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gate(x).sigmoid() * x


class FeedForward(nn.Module):
    """Position-wise feed-forward for 4-D tensors (B, V, N, D)."""

    def __init__(self, dim: int, hidden_features: int | None = None, drop: float = 0.0):
        super().__init__()
        hidden_features = hidden_features or dim * 4
        self.fc1 = nn.Linear(dim, hidden_features)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, dim)
        self.norm = nn.Identity() # changed from nn.LayerNorm(dim), to match original architecture
        self.drop2 = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.norm(x)
        x = self.drop2(x) + x
        return x


class Patchify(nn.Module):
    """Unfold time series into non-overlapping patches.

    Input:  (B, L, V) — multivariate time series.
    Output: (B, V, num_patches, patch_len)
    """

    def __init__(self, patch_len: int, stride: int):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        assert patch_len == stride, "Only non-overlapping patches supported."

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, V) -> transpose to (B, V, L) then unfold
        x = x.transpose(1, 2)  # (B, V, L)
        return x.unfold(dimension=-1, size=self.patch_len, step=self.stride)  # (B, V, P, patch_len)


class SeqAttBlock(nn.Module):
    """Sequence (temporal) attention block."""

    def __init__(self, dim: int, num_heads: int, drop: float = 0.0, drop_path: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = SeqAttention(dim, num_heads=num_heads, qkv_bias=True, attn_drop=drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, V, N, D) — reshape to process each variable independently
        x_in = x
        B, V, N, D = x.shape
        x = self.norm(x)
        x = x.reshape(B * V, N, D)
        x = self.attn(x)
        x = x.reshape(B, V, N, D)
        return x_in + self.drop_path(x)


class VarAttBlock(nn.Module):
    """Variable (cross-channel) attention block."""

    def __init__(self, dim: int, num_heads: int, drop: float = 0.0, drop_path: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = VarAttention(dim, num_heads=num_heads, qkv_bias=True, attn_drop=drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.drop_path(self.attn(self.norm(x)))


class BasicBlock(nn.Module):
    """One encoder layer: VarAtt → SeqAtt → FeedForward."""

    def __init__(self, dim: int, num_heads: int, drop: float = 0.0, drop_path: float = 0.0):
        super().__init__()
        self.var_att = VarAttBlock(dim, num_heads, drop=drop, drop_path=drop_path)
        self.seq_att = SeqAttBlock(dim, num_heads, drop=drop, drop_path=drop_path)
        self.ff = FeedForward(dim, hidden_features=dim * 4, drop=drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.var_att(x)
        x = self.seq_att(x)
        x = self.ff(x)
        return x


# ---------------------------------------------------------------------------
# Main encoder
# ---------------------------------------------------------------------------


class ITFormerTSEncoder(TimeSeriesEncoderBase):
    """Multi-variable time series encoder for ITFormer.

    Input:  (B, L, V) — L timesteps, V variables.
    Output: (B, V, num_patches, output_dim)

    Supports optional MAE-style pretraining via ``pretrain=True``.
    """

    multi_variable = True

    def __init__(
        self,
        output_dim: int = 512,
        dropout: float = 0.1,
        n_heads: int = 8,
        e_layers: int = 4,
        patch_len: int = 60,
        stride: int = 60,
        pretrain: bool = False,
        min_mask_ratio: float = 0.7,
        max_mask_ratio: float = 0.8,
        **kwargs,
    ):
        super().__init__(output_dim=output_dim, dropout=dropout)
        self.patchify = Patchify(patch_len, stride)
        self.patch_embedding = nn.Sequential(
            nn.Linear(patch_len, output_dim, bias=False),
            nn.Dropout(dropout),
        )
        self.layers = nn.ModuleList(
            [BasicBlock(output_dim, n_heads, drop=dropout) for _ in range(e_layers)]
        )
        self.norm = nn.LayerNorm(output_dim)
        self.pretrain = pretrain
        self.min_mask_ratio = min_mask_ratio
        self.max_mask_ratio = max_mask_ratio
        self.proj_head = nn.Linear(output_dim, patch_len)

    def _random_masking(self, x: torch.Tensor) -> torch.Tensor:
        """Generate random binary mask for MAE pretraining."""
        N, V, L, _ = x.shape
        mask_ratio = (self.min_mask_ratio + self.max_mask_ratio) / 2
        total_elements = V * L
        total_keeps = int((1 - mask_ratio) * total_elements)

        noise = torch.rand(N, V, L, device=x.device)
        noise_flat = noise.view(N, V * L)
        ids_shuffle = torch.argsort(noise_flat, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        range_tensor = torch.arange(V * L, device=x.device).unsqueeze(0).expand(N, -1)
        mask_flat = (range_tensor >= total_keeps).gather(dim=1, index=ids_restore)
        return mask_flat.view(N, V, L)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, V) — multivariate time series.

        Returns:
            (B, V, num_patches, output_dim)
            In pretrain mode, returns dict with 'loss' and 'logits'.
        """
        x = self.patchify(x)  # (B, V, P, patch_len)

        if self.pretrain:
            orig_x = x
            mask = self._random_masking(x)
            x = x.masked_fill(mask.unsqueeze(-1), 0)

        x = self.patch_embedding(x)  # (B, V, P, output_dim)

        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)

        if self.pretrain:
            predict_x = self.proj_head(x)
            loss = F.mse_loss(predict_x, orig_x, reduction="mean")
            return {"loss": loss, "logits": x}

        return x
