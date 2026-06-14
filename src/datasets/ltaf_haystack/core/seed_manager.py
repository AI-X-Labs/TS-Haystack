# SPDX-FileCopyrightText: 2026 Anonymous Authors
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Reproducible random seed management for LTAF-Haystack."""

from dataclasses import dataclass
from hashlib import sha256

import numpy as np


@dataclass(frozen=True)
class LTAFReproducibilityConfig:
    master_seed: int = 42


class LTAFSeedManager:
    """Derives deterministic per-sample seeds from a master seed."""

    def __init__(self, master_seed: int = 42):
        self.master_seed = int(master_seed)

    def _derive_seed(self, key: str) -> int:
        digest = sha256(f"{self.master_seed}:{key}".encode("utf-8")).hexdigest()
        return int(digest[:8], 16)

    def get_sample_seed(self, task: str, context_length: int, split: str, index: int) -> int:
        key = f"{task}:{context_length}:{split}:{index}"
        return self._derive_seed(key)

    def get_sample_rng(self, task: str, context_length: int, split: str, index: int) -> np.random.Generator:
        return np.random.default_rng(self.get_sample_seed(task, context_length, split, index))

    def get_sample_seeds(self, task: str, context_length: int, split: str, n_samples: int) -> list[int]:
        return [
            self.get_sample_seed(task=task, context_length=context_length, split=split, index=i)
            for i in range(n_samples)
        ]

    def get_metadata(self) -> dict[str, int]:
        return {"master_seed": self.master_seed}
