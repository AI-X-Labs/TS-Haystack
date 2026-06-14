# SPDX-License-Identifier: CC-BY-NC-4.0
"""Rhythm classifier that operates over beat-level features.

Given a 10 s rhythm bout, we:

1. Extract the `n_beats` R-peak samples falling in that window (from
   the LTAF beat-timeline parquets).
2. For each beat, run the frozen `EcgBeatHTFClassifier` to get a
   pre-head 576-d feature vector (time CNN + freq CNN + RR-history MLP
   concatenated).
3. Concatenate the normalized RR-to-previous-beat as an extra scalar
   per beat (577-d total).
4. Run a small Transformer encoder over that variable-length sequence,
   masked-mean pool, and project to a 6-class head.

The intuition: the beat classifier already generalizes well across
patients (test F1 = 0.94), so the per-beat representation is robust
to the patient-distribution shift that limits raw-signal rhythm CNNs
to F1 ≈ 0.66. Rhythm-defining structure (RR variability, ectopy
sequences) lives in the *sequence* of beat features, not in any
single beat's morphology.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.classifiers.ecg.beat_htf import EcgBeatHTFClassifier


def htf_fused_features(
    htf: EcgBeatHTFClassifier,
    x_time: torch.Tensor,
    rr_history: torch.Tensor,
    label_history: torch.Tensor,
) -> torch.Tensor:
    """Run an HTF beats classifier and return the 576-d pre-head feature.

    Mirrors `EcgBeatHTFClassifier.forward` up to (but not including) the
    final classification head.
    """
    time_feat = htf.time_trunk(x_time)
    freq_feat = htf.freq_trunk(htf._compute_freq(x_time))
    if htf.history_use_labels:
        B, K = label_history.shape
        valid = (label_history >= 0).float().unsqueeze(-1)
        idx = label_history.clamp(min=0)
        one_hot = F.one_hot(idx, num_classes=htf.num_classes).float() * valid
        hist_in = torch.cat([rr_history, one_hot.reshape(B, -1)], dim=-1)
    else:
        hist_in = rr_history
    hist_feat = htf.history_net(hist_in)
    return torch.cat([time_feat, freq_feat, hist_feat], dim=-1)


class _Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, mlp_ratio: float = 4.0,
                 dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                          batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, int(d_model * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(d_model * mlp_ratio), d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, key_padding_mask=None) -> torch.Tensor:
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x


class RhythmFromBeats(nn.Module):
    """Sequence-of-beat-features → 6-class rhythm classifier.

    Args:
        num_classes / class_names: rhythm taxonomy.
        beat_feat_dim: 576 for the default HTF (time 256 + freq 256 + hist 64).
        d_model: width of the per-beat transformer.
        n_layers / n_heads: transformer config.
        max_beats: maximum beats per rhythm bout (cap for positional
            embedding table; longer bouts are truncated).
        dropout / head_hidden: standard.
    """

    def __init__(
        self,
        num_classes: int,
        class_names: List[str],
        beat_feat_dim: int = 576,
        rr_extra_dim: int = 1,
        d_model: int = 128,
        n_layers: int = 4,
        n_heads: int = 4,
        mlp_ratio: float = 4.0,
        max_beats: int = 64,
        dropout: float = 0.1,
        head_hidden: int = 128,
    ):
        super().__init__()
        assert len(class_names) == num_classes
        self.num_classes = num_classes
        self.class_names = list(class_names)
        self.beat_feat_dim = beat_feat_dim
        self.rr_extra_dim = rr_extra_dim
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.max_beats = max_beats

        self.input_proj = nn.Linear(beat_feat_dim + rr_extra_dim, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_beats, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.input_drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            _Block(d_model, n_heads, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, num_classes),
        )

    def forward(
        self,
        beat_feats: torch.Tensor,        # (B, T, beat_feat_dim)
        rr_extra: torch.Tensor,          # (B, T, rr_extra_dim)
        valid_mask: torch.Tensor,        # (B, T) bool — True at valid beat positions
    ) -> torch.Tensor:
        B, T, _ = beat_feats.shape
        T = min(T, self.max_beats)
        beat_feats = beat_feats[:, :T]
        rr_extra = rr_extra[:, :T]
        valid_mask = valid_mask[:, :T]

        x = torch.cat([beat_feats, rr_extra], dim=-1)
        x = self.input_proj(x)
        x = x + self.pos_embed[:, :T]
        x = self.input_drop(x)
        # MHA expects key_padding_mask=True at INVALID positions.
        kpm = ~valid_mask
        for blk in self.blocks:
            x = blk(x, key_padding_mask=kpm)
        x = self.norm(x)
        # Masked mean pool.
        m = valid_mask.unsqueeze(-1).float()
        denom = m.sum(dim=1).clamp(min=1.0)
        feat = (x * m).sum(dim=1) / denom
        return self.head(feat)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "arch": "RhythmFromBeats",
            "num_classes": self.num_classes,
            "class_names": self.class_names,
            "beat_feat_dim": self.beat_feat_dim,
            "rr_extra_dim": self.rr_extra_dim,
            "d_model": self.d_model,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "max_beats": self.max_beats,
            "state_dict": self.state_dict(),
        }, path)

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "RhythmFromBeats":
        ckpt = torch.load(path, map_location=device, weights_only=False)
        m = cls(
            num_classes=ckpt["num_classes"],
            class_names=ckpt["class_names"],
            beat_feat_dim=ckpt["beat_feat_dim"],
            rr_extra_dim=ckpt["rr_extra_dim"],
            d_model=ckpt["d_model"],
            n_layers=ckpt["n_layers"],
            n_heads=ckpt["n_heads"],
            max_beats=ckpt["max_beats"],
        )
        m.load_state_dict(ckpt["state_dict"])
        m.to(device).eval()
        return m
