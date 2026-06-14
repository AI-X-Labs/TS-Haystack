# SPDX-FileCopyrightText: 2026 Anonymous Authors
# SPDX-License-Identifier: CC-BY-NC-4.0
"""ITFormer attention primitives: SeqAttention, VarAttention, InstructTimeAttention.

Canonical implementations with full ``qk_norm`` / ``norm_layer`` parameters.
Used by both the ITFormer encoder and the ITFormer fusion module.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# InstructTimeAttention — used by the ITFormer fusion module
# ---------------------------------------------------------------------------


class InstructTimeAttention(nn.Module):
    """Two-stage instruction-aware time-series attention.

    Stage 1: Channel attention — aggregate across variables guided by query.
    Stage 2: Time attention — attend over temporal dimension.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: type = nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()

        # Channel attention projections
        self.channel_query_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.channel_key_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.channel_value_proj = nn.Linear(dim, dim, bias=qkv_bias)

        # Time attention projections
        self.query_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.key_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.value_proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

    def forward(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            query: (B, L_q, d) — instruction / prefix tokens.
            memory: (B, L_p, V, d) — encoded time series patches x variables.

        Returns:
            (B, L_q, d) — attended output.
        """
        B, L_q, d = query.size()
        _, L_p, V, _ = memory.size()

        # --- Stage 1: Channel attention guided by query ---
        q_ch = self.channel_query_proj(query)  # (B, L_q, d)
        k_ch = self.channel_key_proj(memory).mean(dim=1)  # (B, V, d)
        v_ch = self.channel_value_proj(memory)  # (B, L_p, V, d)

        q_ch = self.q_norm(q_ch.view(B, self.num_heads, L_q, self.head_dim))
        k_ch = self.k_norm(k_ch.view(B, self.num_heads, V, self.head_dim))
        v_ch = v_ch.view(B, self.num_heads, L_p, V, self.head_dim)

        ch_scores = torch.matmul(q_ch, k_ch.transpose(-2, -1)) / self.scale
        ch_weights = F.softmax(ch_scores, dim=-1)  # (B, H, L_q, V)
        ch_out = torch.einsum("bnqv,bnlvd->bnld", ch_weights, v_ch)  # (B, H, L_p, hd)

        memory_agg = ch_out.transpose(1, 2).contiguous().view(B, L_p, d)

        # --- Stage 2: Time attention ---
        Q = self.q_norm(self.query_proj(query).view(B, L_q, self.num_heads, self.head_dim).transpose(1, 2))
        K = self.k_norm(self.key_proj(memory_agg).view(B, L_p, self.num_heads, self.head_dim).transpose(1, 2))
        V_proj = self.value_proj(memory_agg).view(B, L_p, self.num_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, V_proj)

        out = out.transpose(1, 2).contiguous().view(B, L_q, d)
        return self.proj(out)


# ---------------------------------------------------------------------------
# Encoder-level attention blocks
# ---------------------------------------------------------------------------


class SeqAttention(nn.Module):
    """Self-attention over the sequence (patch) dimension."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: type = nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True):
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0)

        x = x.transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x))


class VarAttention(nn.Module):
    """Attention across the variable dimension with mean-reduction for Q/K."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: type = nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, P, C = x.shape  # (batch, variables, patches, dim)

        qkv = (
            self.qkv(x).reshape(B, N, P, 3, self.num_heads, self.head_dim).permute(3, 0, 2, 4, 1, 5)
        )  # (3, B, P, H, N, hd)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        # Mean-reduce across patches for Q and K
        q = q.mean(dim=1, keepdim=False)  # (B, H, N, hd)
        k = k.mean(dim=1, keepdim=False)  # (B, H, N, hd)
        # Reshape V: (B, P, H, N, hd) -> (B, H, N, P*hd)
        v = v.permute(0, 2, 3, 4, 1).reshape(B, self.num_heads, N, -1)

        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True):
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0)

        # Reshape back: x is (B, H, N, P*hd) -> (B, N, P, C)
        expected_last_dim = P * self.head_dim
        actual_last_dim = x.shape[-1]

        if actual_last_dim == expected_last_dim:
            x = x.view(B, self.num_heads, N, self.head_dim, P).permute(0, 2, 4, 1, 3).reshape(B, N, P, -1)
        else:
            # Fallback for dimension mismatch
            x = x.transpose(1, 2)
            flattened_dim = self.num_heads * actual_last_dim
            x = x.reshape(B, N, flattened_dim)
            if flattened_dim % P == 0:
                x = x.reshape(B, N, P, flattened_dim // P)
            else:
                x = x.reshape(B, N, 1, flattened_dim).expand(B, N, P, flattened_dim)

        return self.proj_drop(self.proj(x))
