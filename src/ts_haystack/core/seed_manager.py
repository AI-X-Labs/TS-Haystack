# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors
# SPDX-License-Identifier: MIT

"""
Seed management for reproducible TS-Haystack benchmark generation.

This module provides deterministic seed generation to ensure:
- Cross-task consistency (same participant splits across all tasks)
- Per-sample reproducibility (each sample has a deterministic seed)
- Parallel-safe generation (pre-computed seeds enable safe parallelization)
- Full metadata tracking for reproducibility
"""

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class ReproducibilityConfig:
    """
    Configuration for reproducible benchmark generation.

    Attributes:
        master_seed: The root seed from which all other seeds are derived
        participant_split_seed: Derived seed for participant train/val/test splitting
    """

    master_seed: int = 42
    participant_split_seed: int = field(init=False)

    def __post_init__(self):
        """Derive dependent seeds from master seed."""
        self.participant_split_seed = self._derive_seed("participant_split")

    def _derive_seed(self, component: str) -> int:
        """Derive a deterministic seed for a named component."""
        h = hashlib.sha256(f"{self.master_seed}:{component}".encode())
        return int.from_bytes(h.digest()[:4], "big")

    def to_dict(self) -> Dict:
        """Serialize for metadata storage."""
        return {
            "master_seed": self.master_seed,
            "participant_split_seed": self.participant_split_seed,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "ReproducibilityConfig":
        """Deserialize from metadata."""
        return cls(master_seed=d["master_seed"])


class SeedManager:
    """
    Manages deterministic seed generation for reproducible task generation.

    This class provides a hierarchical seed derivation system that ensures:
    1. Consistent participant splits across all tasks
    2. Deterministic per-sample seeds for parallel-safe generation
    3. Full traceability via metadata recording

    Seed Hierarchy:
        master_seed (e.g., 42)
            |
            +-- participant_split_seed --> Train/Val/Test participant assignment
            |       (shared across ALL tasks)
            |
            +-- task_seed(task, context_length, split)
                    |
                    +-- sample_seed[0] --> Background selection, needle selection
                    +-- sample_seed[1]
                    +-- ...

    Example:
        >>> seed_mgr = SeedManager(master_seed=42)
        >>>
        >>> # Split participants (same for all tasks)
        >>> train_pids, val_pids, test_pids = seed_mgr.split_participants(all_pids)
        >>>
        >>> # Generate samples for a specific task/context/split
        >>> sample_seeds = seed_mgr.get_sample_seeds("existence", 1000, "train", n_samples=10000)
        >>> for i, seed in enumerate(sample_seeds):
        ...     rng = np.random.default_rng(seed)
        ...     sample = generate_sample(rng, ...)
    """

    def __init__(self, master_seed: int = 42):
        """
        Initialize the seed manager.

        Args:
            master_seed: The root seed from which all other seeds are derived
        """
        self.config = ReproducibilityConfig(master_seed=master_seed)

    @property
    def master_seed(self) -> int:
        """Return the master seed."""
        return self.config.master_seed

    def _derive_seed(self, *components) -> int:
        """
        Derive a deterministic seed from components.

        Uses SHA-256 hash of "master_seed:component1:component2:..." to get
        a deterministic integer seed.
        """
        key = ":".join(str(c) for c in [self.master_seed, *components])
        h = hashlib.sha256(key.encode())
        return int.from_bytes(h.digest()[:8], "big")

    def get_rng(self, *components) -> np.random.Generator:
        """
        Get a fresh RNG for the given components.

        Each unique combination of components produces a unique,
        deterministic RNG.
        """
        seed = self._derive_seed(*components)
        return np.random.default_rng(seed)

    def get_participant_split_rng(self) -> np.random.Generator:
        """
        Get RNG for participant train/val/test splitting.

        This RNG is shared across ALL tasks to ensure consistent
        participant assignment regardless of which task is generated first.
        """
        return np.random.default_rng(self.config.participant_split_seed)

    def get_task_rng(
        self,
        task: str,
        context_length: int,
        split: str,
    ) -> np.random.Generator:
        """
        Get RNG for a specific (task, context_length, split) combination.

        Use this for sequential sample generation when not parallelizing.
        """
        return self.get_rng("task", task, str(context_length), split)

    def get_sample_seeds(
        self,
        task: str,
        context_length: int,
        split: str,
        n_samples: int,
    ) -> List[int]:
        """
        Pre-generate seeds for each sample in a task/context/split.

        This is the key method for parallel-safe generation.
        """
        rng = self.get_task_rng(task, context_length, split)
        return [int(rng.integers(0, 2**63)) for _ in range(n_samples)]

    def get_sample_rng(
        self,
        task: str,
        context_length: int,
        split: str,
        sample_index: int,
    ) -> np.random.Generator:
        """
        Get RNG for a specific sample by index.

        Useful for regenerating a single sample without generating all prior seeds.
        """
        return self.get_rng("sample", task, str(context_length), split, str(sample_index))

    def split_participants(
        self,
        participant_ids: List[str],
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
    ) -> Tuple[List[str], List[str], List[str]]:
        """
        Split participants into train/val/test sets deterministically.

        Args:
            participant_ids: List of all participant IDs
            train_ratio: Fraction for training (default: 0.70)
            val_ratio: Fraction for validation (default: 0.15)

        Returns:
            Tuple of (train_pids, val_pids, test_pids)
        """
        rng = self.get_participant_split_rng()

        pids = list(participant_ids)
        rng.shuffle(pids)

        n = len(pids)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train_pids = pids[:n_train]
        val_pids = pids[n_train : n_train + n_val]
        test_pids = pids[n_train + n_val :]

        return sorted(train_pids), sorted(val_pids), sorted(test_pids)

    def get_metadata(self) -> Dict:
        """
        Get seed metadata for storage with generated datasets.

        This should be saved alongside generated data to enable reproduction.
        """
        return {
            "seed_config": self.config.to_dict(),
            "seed_manager_version": "1.0",
        }

    def save_metadata(self, path: Path) -> None:
        """Save seed metadata to a JSON file."""
        with open(path, "w") as f:
            json.dump(self.get_metadata(), f, indent=2)

    @classmethod
    def from_metadata(cls, metadata: Dict) -> "SeedManager":
        """Reconstruct SeedManager from saved metadata."""
        config = ReproducibilityConfig.from_dict(metadata["seed_config"])
        return cls(master_seed=config.master_seed)

    @classmethod
    def load_from_file(cls, path: Path) -> "SeedManager":
        """Load SeedManager from a metadata JSON file."""
        with open(path) as f:
            metadata = json.load(f)
        return cls.from_metadata(metadata)
