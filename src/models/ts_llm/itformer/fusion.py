# SPDX-FileCopyrightText: 2026 Anonymous Authors
# SPDX-License-Identifier: CC-BY-NC-4.0
"""ITFormer fusion module — instruction-aware time-series fusion.

Ported from ITFormer-clone/models/ITFormer.py.
"""

import torch
import torch.nn as nn

from src.components.drop_path import DropPath
from src.components.itformer_attention import InstructTimeAttention, SeqAttention
from src.models.ts_llm.itformer.position_encoding import (
    LearnablePositionalEmbedding,
    RotaryPositionalEncoding,
    SinusoidalPositionalEncoding,
)


class SelfAttBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, drop: float = 0.0, drop_path: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = SeqAttention(dim, num_heads=num_heads, qkv_bias=True,
                                 attn_drop=drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.drop_path(self.attn(self.norm(x)))


class ITAttBlock(nn.Module):
    """InstructTime attention wrapper with pre-norm and drop path."""

    def __init__(self, dim: int, num_heads: int, drop: float = 0.0, drop_path: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = InstructTimeAttention(dim, num_heads=num_heads, qkv_bias=True,
                                          attn_drop=drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        return x + self.norm(self.drop_path(self.attn(x, memory)))


class DecoderBasicBlock(nn.Module):
    """One fusion decoder layer: self-attn → IT-attn (prefix only) → FFN."""

    def __init__(self, dim: int, num_heads: int, prefix_num: int = 10,
                 mlp_ratio: float = 4.0, drop: float = 0.0, drop_path: float = 0.0):
        super().__init__()
        self.prefix_num = prefix_num

        self.self_attn = SelfAttBlock(dim, num_heads, drop=drop, drop_path=drop_path)
        self.it_attn = ITAttBlock(dim, num_heads, drop=drop, drop_path=drop_path)

        hidden = int(dim * mlp_ratio)
        self.ff_prefix = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, dim),
        )
        self.ff_instruct = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, dim),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        # Self-attention on full sequence (prefix + query)
        x = x + self.self_attn(x)

        prefix = x[:, : self.prefix_num, :]
        x = self.ff_instruct(x) + x

        # IT-attention: only prefix tokens attend to memory
        prefix = prefix + self.it_attn(prefix, memory)
        prefix = prefix + self.ff_prefix(prefix)

        # Reconstruct full sequence
        x = torch.cat([prefix, x[:, self.prefix_num :, :]], dim=1)
        return x


# ---------------------------------------------------------------------------
# ITFormerFusion — main fusion module
# ---------------------------------------------------------------------------


class ITFormerFusion(nn.Module):
    """Instruction-aware fusion of query embeddings and encoded time series.

    Takes projected query embeddings and projected encoder output, applies
    stage-aware positional encodings, and returns fused prefix tokens.

    Args:
        it_d_model: Internal fusion dimension.
        it_n_heads: Number of attention heads.
        it_layers: Number of decoder blocks.
        it_dropout: Dropout rate.
        prefix_num: Number of learnable prefix tokens (output length).
    """

    def __init__(
        self,
        it_d_model: int = 896,
        it_n_heads: int = 16,
        it_layers: int = 2,
        it_dropout: float = 0.1,
        prefix_num: int = 25,
    ):
        super().__init__()
        self.prefix_num = prefix_num
        self.prefix_token = nn.Parameter(torch.randn(1, prefix_num, it_d_model))

        self.layers = nn.ModuleList(
            [
                DecoderBasicBlock(
                    dim=it_d_model,
                    num_heads=it_n_heads,
                    prefix_num=prefix_num,
                    drop=it_dropout,
                )
                for _ in range(it_layers)
            ]
        )
        self.norm = nn.LayerNorm(it_d_model)

        # Positional encodings
        self.time_pos = SinusoidalPositionalEncoding(it_d_model)
        self.var_pos = LearnablePositionalEmbedding(it_d_model)
        self.instruc_pos = SinusoidalPositionalEncoding(it_d_model)
        self.cycle_pos = RotaryPositionalEncoding(it_d_model)

    def forward(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        stage: list[int] | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            query: (B, Q, it_d_model) — projected query embeddings.
            memory: (B, V, P, it_d_model) — projected encoder output
                    (variables × patches).
            stage: Per-sample stage (1-4). Stages 1-2 use sinusoidal time PE,
                   stages 3-4 use rotary cycle PE. Defaults to stage 1 if None.

        Returns:
            (B, prefix_num, it_d_model) — fused prefix tokens.
        """
        B = query.size(0)

        # Prepend learnable prefix tokens to query
        x = torch.cat([self.prefix_token.expand(B, -1, -1), query], dim=1)
        x = x + self.instruc_pos(x)

        # --- Stage-aware positional encoding for memory ---
        if stage is None:
            stage_list = [1] * B
        elif torch.is_tensor(stage):
            stage_list = stage.tolist()
        else:
            stage_list = list(stage)

        cycle_idx = [i for i, s in enumerate(stage_list) if s not in (3, 4)]
        cross_cycle_idx = [i for i, s in enumerate(stage_list) if s in (3, 4)]

        # Build reorder map
        original_order = cycle_idx + cross_cycle_idx
        reorder = {idx: i for i, idx in enumerate(original_order)}
        reverse = [reorder[i] for i in range(len(stage_list))]

        parts = []
        if cycle_idx:
            sub = memory[cycle_idx]
            b, l, v, d = sub.shape
            sub = sub.view(b * l, v, d)
            sub = sub + self.time_pos(sub)
            sub = sub.view(b, l, v, d)
            parts.append(sub)

        if cross_cycle_idx:
            sub = memory[cross_cycle_idx]
            b, l, v, d = sub.shape
            sub = sub.view(b * v, l, d)
            sub = sub + self.cycle_pos(sub)
            sub = sub.view(b, l, v, d)
            parts.append(sub)

        memory = torch.cat(parts, dim=0)[reverse]

        # Variable positional encoding
        b, l, v, d = memory.shape
        memory = memory.view(b * l, v, d)
        memory = memory + self.var_pos(memory)
        memory = memory.view(b, l, v, d)

        # Decoder layers
        for layer in self.layers:
            x = layer(x, memory)

        x = self.norm(x)
        return x[:, : self.prefix_num, :]
