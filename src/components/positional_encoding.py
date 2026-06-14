# SPDX-FileCopyrightText: 2026 Anonymous Authors
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Positional encoding primitives: sinusoidal, rotary, and learnable."""

import math

import torch
from torch import nn


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding for time dimension (stages 1-2)."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # Shape: (1, 1, max_len, d_model)
        pe = pe.unsqueeze(0).unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """Return positional encoding slice matching x's sequence length.

        Args:
            x: Tensor whose dim-1 length determines the slice size.
            offset: Starting position in the encoding table.

        Returns:
            Positional encoding tensor broadcastable to x.
        """
        return self.pe[0, :, offset : offset + x.size(1), :]


class RotaryPositionalEncoding(nn.Module):
    """Rotary positional encoding for cross-cycle analysis (stages 3-4)."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        assert d_model % 2 == 0, "d_model must be even for RotaryPositionalEncoding."

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        dim = torch.arange(0, d_model // 2, dtype=torch.float)
        div_term = torch.exp(dim * -(math.log(10000.0) / (d_model // 2)))

        angle = position * div_term
        sin_part = torch.sin(angle)
        cos_part = torch.cos(angle)

        # Shape: (1, 1, max_len, d_model)
        pe = torch.cat([sin_part, cos_part], dim=-1).unsqueeze(0).unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """Apply rotary positional encoding to *x*.

        Args:
            x: Input tensor of shape ``(batch, seq_len, d_model)``.
            offset: Starting position.

        Returns:
            Tensor with rotary encoding applied, same shape as *x*.
        """
        seq_len = x.size(1)
        pe = self.pe[0, :, offset : offset + seq_len, :]
        half = x.size(-1) // 2
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat(
            [
                x1 * pe[..., :half] - x2 * pe[..., half:],
                x1 * pe[..., half:] + x2 * pe[..., :half],
            ],
            dim=-1,
        )


class LearnablePositionalEmbedding(nn.Module):
    """Learnable positional embedding for the variable dimension."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(1, 1, max_len, d_model))

        # Initialise with sinusoidal values
        pe_init = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * -(math.log(10000.0) / d_model))
        pe_init[:, 0::2] = torch.sin(position * div_term)
        pe_init[:, 1::2] = torch.cos(position * div_term)
        self.pe.data.copy_(pe_init.unsqueeze(0).unsqueeze(0))

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        return self.pe[0, :, offset : offset + x.size(1), :]
