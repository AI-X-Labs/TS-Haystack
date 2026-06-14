# SPDX-FileCopyrightText: 2026 Anonymous Authors
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Steerable gated cross-attention primitive."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SteerableCrossAttention(nn.Module):
    """Gated cross-attention with ``tanh(gate)`` scaling.

    The gate is initialised to **0** so the model starts as the vanilla LLM.
    Uses ``F.scaled_dot_product_attention`` for automatic flash-attention dispatch.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        inner_dim = num_heads * dim_head
        self.num_heads = num_heads
        self.dim_head = dim_head

        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

        self.gate = nn.Parameter(torch.zeros(1))
        self.dropout = dropout

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """``x + tanh(gate) * cross_attn(norm(x), norm(context))``."""
        B, N, _ = x.shape
        h = self.num_heads

        q = self.to_q(self.norm_q(x))
        k = self.to_k(self.norm_kv(context))
        v = self.to_v(self.norm_kv(context))

        # Reshape to (B, heads, seq, dim_head)
        q = q.view(B, N, h, self.dim_head).transpose(1, 2)
        k = k.view(B, -1, h, self.dim_head).transpose(1, 2)
        v = v.view(B, -1, h, self.dim_head).transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0)

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, N, -1)
        out = self.to_out(attn_out)

        return x + torch.tanh(self.gate) * out
