# SPDX-License-Identifier: CC-BY-NC-4.0
"""History-Time-Frequency (HTF) beat classifier for LTAF (N / A / V).

Inspired by the PAC-PVC ensemble in
<https://github.com/alberto-rota/PAC-PVC-Beat-Classifier-for-ECGs>:
three parallel streams whose features are concatenated before the head.

Streams
-------
- **Time** — raw 2-channel R-peak-centered window (B, 2, 256) → 1D-CNN
  trunk → pooled feature vector. This stream sees the QRS morphology.
- **Frequency** — log-magnitude rFFT of the same window
  (B, 2, 129) → 1D-CNN trunk → pooled feature vector. Captures spectral
  signature of PVC vs PAC vs normal.
- **History** — RR intervals to the preceding K beats (B, K) plus
  (optionally, training-time teacher-forced) one-hot of the preceding
  beats' labels (B, K * num_classes). MLP → feature vector. Captures
  rhythm context — bigeminy / trigeminy / coupling intervals can only be
  recognized from beat-to-beat structure, not a single window.

The frequency stream is computed on-device so the dataset only needs to
return the time signal and the history features.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


BEAT_CLASS_NAMES = ["N", "A", "V"]


class _ConvBlock(nn.Module):
    """Conv → BN → ReLU → optional MaxPool."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 7,
                 pool: int = 2, dropout: float = 0.0):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=kernel,
                              padding=kernel // 2, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool1d(pool) if pool > 1 else nn.Identity()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.pool(self.act(self.bn(self.conv(x)))))


class _CNNTrunk(nn.Module):
    """Stack of conv blocks ending in adaptive average pool.

    For 256-sample input: 256 → 128 → 64 → 32 → 16 → 8 → pool → (B, channels).
    """

    def __init__(self, in_channels: int, base_channels: int = 32,
                 n_blocks: int = 5, dropout: float = 0.1):
        super().__init__()
        layers = []
        ch = in_channels
        out_ch = base_channels
        for _ in range(n_blocks):
            layers.append(_ConvBlock(ch, out_ch, kernel=7, pool=2, dropout=dropout))
            ch = out_ch
            out_ch = min(out_ch * 2, 256)
        self.net = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.out_channels = ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x)
        return self.pool(h).squeeze(-1)


class EcgBeatHTFClassifier(nn.Module):
    """HTF ensemble: time + frequency + history → MLP head.

    Args:
        num_classes: number of output classes (default 3 for N/A/V).
        n_channels: ECG leads (default 2 for LTAF).
        window_samples: time window length in samples (default 256 for 2 s @ 128 Hz).
        history_k: how many preceding beats to include in the history stream.
        history_use_labels: include one-hot of preceding beats' labels in history.
        time_base_channels / freq_base_channels: trunk widths.
        head_hidden: hidden width of the post-fusion MLP.
        dropout: dropout in trunks and head.
    """

    def __init__(
        self,
        num_classes: int = 3,
        class_names: List[str] = BEAT_CLASS_NAMES,
        n_channels: int = 2,
        window_samples: int = 256,
        history_k: int = 5,
        history_use_labels: bool = True,
        time_base_channels: int = 32,
        freq_base_channels: int = 32,
        head_hidden: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert len(class_names) == num_classes
        self.num_classes = num_classes
        self.class_names = list(class_names)
        self.n_channels = n_channels
        self.window_samples = window_samples
        self.history_k = history_k
        self.history_use_labels = history_use_labels

        self.time_trunk = _CNNTrunk(n_channels, time_base_channels,
                                    n_blocks=5, dropout=dropout)

        # rFFT magnitude length for 256-sample window is 129.
        self.freq_trunk = _CNNTrunk(n_channels, freq_base_channels,
                                    n_blocks=4, dropout=dropout)

        history_in = history_k  # RR intervals (seconds, normalized)
        if history_use_labels:
            history_in += history_k * num_classes
        self.history_net = nn.Sequential(
            nn.Linear(history_in, 64), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 64), nn.ReLU(inplace=True),
        )

        fused_dim = self.time_trunk.out_channels + self.freq_trunk.out_channels + 64
        self.head = nn.Sequential(
            nn.Linear(fused_dim, head_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, num_classes),
        )

    def _compute_freq(self, x: torch.Tensor) -> torch.Tensor:
        """Log-magnitude rFFT along the time axis. (B, C, L) → (B, C, L//2+1)."""
        spec = torch.fft.rfft(x, dim=-1)
        return torch.log1p(spec.abs())

    def forward(
        self,
        x_time: torch.Tensor,
        rr_history: torch.Tensor,
        label_history: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x_time: (B, C, L) — raw 2-channel ECG window.
            rr_history: (B, K) — RR intervals to preceding K beats, in seconds.
                        Missing beats (record start) get 0.0.
            label_history: (B, K) int64 — preceding K beat labels (0/1/2).
                           Used iff ``history_use_labels``. -1 marks missing.

        Returns:
            (B, num_classes) logits.
        """
        time_feat = self.time_trunk(x_time)
        freq_feat = self.freq_trunk(self._compute_freq(x_time))

        if self.history_use_labels:
            assert label_history is not None
            B, K = label_history.shape
            valid = (label_history >= 0).float().unsqueeze(-1)
            idx = label_history.clamp(min=0)
            one_hot = F.one_hot(idx, num_classes=self.num_classes).float() * valid
            hist_in = torch.cat([rr_history, one_hot.reshape(B, -1)], dim=-1)
        else:
            hist_in = rr_history
        hist_feat = self.history_net(hist_in)

        fused = torch.cat([time_feat, freq_feat, hist_feat], dim=-1)
        return self.head(fused)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "num_classes": self.num_classes,
            "class_names": self.class_names,
            "n_channels": self.n_channels,
            "window_samples": self.window_samples,
            "history_k": self.history_k,
            "history_use_labels": self.history_use_labels,
            "state_dict": self.state_dict(),
        }
        torch.save(state, path)
        print(f"EcgBeatHTFClassifier saved to {path}")

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "EcgBeatHTFClassifier":
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model = cls(
            num_classes=ckpt["num_classes"],
            class_names=ckpt["class_names"],
            n_channels=ckpt["n_channels"],
            window_samples=ckpt["window_samples"],
            history_k=ckpt["history_k"],
            history_use_labels=ckpt["history_use_labels"],
        )
        model.load_state_dict(ckpt["state_dict"])
        model.to(device).eval()
        return model
