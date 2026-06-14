# SPDX-FileCopyrightText: 2026 Anonymous Authors
# SPDX-License-Identifier: CC-BY-NC-4.0
"""SSM primitives: SelectiveSSMBlock, AttentionPooling, and MemoryBank."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Selective SSM Block (pure PyTorch, Mamba-style)
# ---------------------------------------------------------------------------


class SelectiveSSMBlock(nn.Module):
    """Pure-PyTorch selective SSM following HuggingFace transformers MambaMixer.slow_forward.

    No CUDA kernels required — works on any device.

    Args:
        dim: Input/output dimension.
        d_state: SSM state expansion factor.
        d_conv: Local convolution width.
        expand: Channel expansion factor.
    """

    def __init__(self, dim: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.dim = dim
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        inner_dim = dim * expand
        self.inner_dim = inner_dim
        dt_rank = math.ceil(dim / 16)
        self.dt_rank = dt_rank

        # Input projection: gate + hidden
        self.in_proj = nn.Linear(dim, inner_dim * 2, bias=False)

        # Depthwise conv
        self.conv1d = nn.Conv1d(
            inner_dim, inner_dim, kernel_size=d_conv, groups=inner_dim, padding=d_conv - 1, bias=True
        )

        # Content-dependent projections
        self.x_proj = nn.Linear(inner_dim, dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(dt_rank, inner_dim, bias=True)

        # S4D real initialization for A
        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(torch.log(A).unsqueeze(0).expand(inner_dim, -1).clone())  # (inner_dim, d_state)

        # Skip connection
        self.D = nn.Parameter(torch.ones(inner_dim))

        # Output projection
        self.out_proj = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: ``(B, L, D)`` input features.

        Returns:
            ``(B, L, D)`` output features.
        """
        B, L, _ = x.shape

        # Project input
        xz = self.in_proj(x)  # (B, L, 2 * inner_dim)
        x_hidden, z = xz.chunk(2, dim=-1)  # each (B, L, inner_dim)

        # Depthwise conv (causal: trim to L)
        x_hidden = x_hidden.transpose(1, 2)  # (B, inner_dim, L)
        x_hidden = self.conv1d(x_hidden)[:, :, :L]  # (B, inner_dim, L)
        x_hidden = x_hidden.transpose(1, 2)  # (B, L, inner_dim)
        x_hidden = F.silu(x_hidden)

        # Content-dependent parameters
        x_dbl = self.x_proj(x_hidden)  # (B, L, dt_rank + 2*d_state)
        dt, B_param, C_param = x_dbl.split([self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj(dt)  # (B, L, inner_dim)
        dt = F.softplus(dt)  # ensure positive

        # Discretize A
        A = -torch.exp(self.A_log)  # (inner_dim, d_state)

        # Sequential scan
        y = self._sequential_scan(x_hidden, dt, A, B_param, C_param)  # (B, L, inner_dim)

        # Skip connection
        y = y + self.D.unsqueeze(0).unsqueeze(0) * x_hidden

        # Gate and project out
        y = y * F.silu(z)
        return self.out_proj(y)

    def _sequential_scan(
        self,
        u: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
    ) -> torch.Tensor:
        """Run the sequential SSM scan.

        Args:
            u: ``(B, L, inner_dim)`` — input signal.
            dt: ``(B, L, inner_dim)`` — discretization step sizes.
            A: ``(inner_dim, d_state)`` — continuous-time state matrix (negative).
            B: ``(B, L, d_state)`` — input-dependent B matrix.
            C: ``(B, L, d_state)`` — input-dependent C matrix.

        Returns:
            ``(B, L, inner_dim)`` — output of the scan.
        """
        batch, L, inner_dim = u.shape
        d_state = A.shape[1]

        # discrete_A: (B, L, inner_dim, d_state) = exp(dt * A)
        dA = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))  # (B, L, inner_dim, d_state)

        # deltaB_u: (B, L, inner_dim, d_state) = dt * B * u
        dB_u = dt.unsqueeze(-1) * B.unsqueeze(2) * u.unsqueeze(-1)  # (B, L, inner_dim, d_state)

        # Sequential scan
        h = torch.zeros(batch, inner_dim, d_state, device=u.device, dtype=u.dtype)  # (B, inner_dim, d_state)
        ys = []
        for t in range(L):
            h = dA[:, t] * h + dB_u[:, t]  # (B, inner_dim, d_state)
            y_t = torch.einsum("bid,bid->bi", h, C[:, t].unsqueeze(1).expand_as(h))  # (B, inner_dim)
            ys.append(y_t)

        return torch.stack(ys, dim=1)  # (B, L, inner_dim)


# ---------------------------------------------------------------------------
# Attention Pooling
# ---------------------------------------------------------------------------


class AttentionPooling(nn.Module):
    """Learned query-based pooling that compresses variable-length features to K tokens.

    Uses K learned query tokens that cross-attend to the input, similar to
    the Perceiver Resampler but as a single lightweight layer per scale.
    Fully differentiable — the model learns which positions to focus on.
    """

    def __init__(self, dim: int, num_queries: int, num_heads: int = 8, dim_head: int = 64):
        super().__init__()
        self.num_queries = num_queries
        self.num_heads = num_heads
        self.dim_head = dim_head
        inner_dim = num_heads * dim_head

        self.queries = nn.Parameter(torch.randn(num_queries, dim) * 0.02)
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, N, D)`` -> ``(B, K, D)`` via learned cross-attention pooling."""
        B = x.shape[0]
        h = self.num_heads
        K = self.num_queries

        q = self.to_q(self.norm_q(self.queries.unsqueeze(0).expand(B, -1, -1)))
        kv_input = self.norm_kv(x)
        k = self.to_k(kv_input)
        v = self.to_v(kv_input)

        q = q.view(B, K, h, self.dim_head).transpose(1, 2)
        k = k.view(B, -1, h, self.dim_head).transpose(1, 2)
        v = v.view(B, -1, h, self.dim_head).transpose(1, 2)

        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).contiguous().view(B, K, -1)
        return self.to_out(out)


# ---------------------------------------------------------------------------
# Memory Bank
# ---------------------------------------------------------------------------


class MemoryBank(nn.Module):
    """Compress per-scale features through SSM blocks and learned attention pooling.

    Each scale gets an independent SSM block for sequential processing,
    followed by learned attention pooling (K query tokens cross-attend to
    the SSM output).

    Args:
        enc_dim: Encoder feature dimension.
        num_tokens_per_scale: K — number of query tokens per scale.
        scales: List of scale factors.
        d_state: SSM state expansion factor.
        d_conv: SSM local convolution width.
        expand: SSM channel expansion factor.
        pool_heads: Number of attention heads in the pooling layer.
        pool_dim_head: Dimension per head in the pooling layer.
    """

    def __init__(
        self,
        enc_dim: int,
        num_tokens_per_scale: int = 32,
        scales: list[int] | None = None,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        pool_heads: int = 8,
        pool_dim_head: int = 64,
    ):
        super().__init__()
        self.scales = scales or [1, 4, 16]
        self.num_tokens_per_scale = num_tokens_per_scale

        self.norms = nn.ModuleList([nn.LayerNorm(enc_dim) for _ in self.scales])
        self.ssm_blocks = nn.ModuleList(
            [SelectiveSSMBlock(enc_dim, d_state=d_state, d_conv=d_conv, expand=expand) for _ in self.scales]
        )
        self.pool_layers = nn.ModuleList(
            [
                AttentionPooling(
                    enc_dim, num_queries=num_tokens_per_scale, num_heads=pool_heads, dim_head=pool_dim_head
                )
                for _ in self.scales
            ]
        )

    def forward(self, scale_features: list[torch.Tensor]) -> torch.Tensor:
        """Compress per-scale features into a fixed-size memory.

        Args:
            scale_features: List of ``(B, N_s, D)`` tensors, one per scale.

        Returns:
            ``(B, num_scales * K, D)`` concatenated memory tokens.
        """
        parts = []
        for feat, norm, ssm, pool in zip(scale_features, self.norms, self.ssm_blocks, self.pool_layers):
            out = ssm(norm(feat))  # (B, N_s, D) — SSM sequential processing
            out = pool(out)  # (B, K, D) — learned attention pooling
            parts.append(out)

        return torch.cat(parts, dim=1)  # (B, num_scales * K, D)
