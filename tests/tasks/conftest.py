# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors
# SPDX-License-Identifier: MIT

"""
Shared pytest fixtures for TS-Haystack task tests.

Extends the parent conftest.py with task-specific fixtures.
"""

import pytest
import numpy as np
from pathlib import Path

# Import from parent test module
from ts_haystack.core import (
    BoutIndexer,
    TimelineBuilder,
    TransitionMatrix,
    BackgroundSampler,
    NeedleSampler,
    StyleTransfer,
    PromptTemplateBank,
    SeedManager,
    DifficultyConfig,
)
from ts_haystack.tasks import (
    ExistenceTaskGenerator,
    LocalizationTaskGenerator,
    CountingTaskGenerator,
    OrderingTaskGenerator,
    StateQueryTaskGenerator,
    AntecedentTaskGenerator,
    ComparisonTaskGenerator,
    MultiHopTaskGenerator,
    AnomalyDetectionTaskGenerator,
    AnomalyLocalizationTaskGenerator,
)


# =============================================================================
# Plot Output Directory
# =============================================================================

PLOT_OUTPUT_DIR = Path(__file__).parent.parent / "plots"


def ensure_plot_dir(task_type: str) -> Path:
    """Create and return the plot output directory for a task type."""
    plot_dir = PLOT_OUTPUT_DIR / task_type
    plot_dir.mkdir(parents=True, exist_ok=True)
    return plot_dir


# =============================================================================
# Data Availability Check
# =============================================================================


def _check_phase1_available() -> bool:
    """Check if Phase 1 artifacts are available."""
    try:
        BoutIndexer.load_index()
        return True
    except Exception:
        return False


PHASE1_AVAILABLE = _check_phase1_available()


# =============================================================================
# Fixtures - Phase 1 Artifacts (module scope for efficiency)
# =============================================================================


@pytest.fixture(scope="module")
def timelines():
    """Load all participant timelines."""
    if not PHASE1_AVAILABLE:
        pytest.skip("Phase 1 artifacts not available")
    return TimelineBuilder.load_all_timelines()


@pytest.fixture(scope="module")
def bout_index():
    """Load the bout index."""
    if not PHASE1_AVAILABLE:
        pytest.skip("Phase 1 artifacts not available")
    return BoutIndexer.load_index()


@pytest.fixture(scope="module")
def transition_matrix():
    """Load the transition matrix."""
    if not PHASE1_AVAILABLE:
        pytest.skip("Phase 1 artifacts not available")
    return TransitionMatrix.load()


# =============================================================================
# Fixtures - Phase 2 Components
# =============================================================================


@pytest.fixture(scope="module")
def seed_manager():
    """Create a SeedManager for reproducible tests."""
    return SeedManager(master_seed=42)


@pytest.fixture(scope="module")
def background_sampler(timelines, bout_index):
    """Create a BackgroundSampler with actual data."""
    return BackgroundSampler(timelines, bout_index, source_hz=100)


@pytest.fixture(scope="module")
def needle_sampler(bout_index, transition_matrix):
    """Create a NeedleSampler with actual data."""
    return NeedleSampler(bout_index, transition_matrix, source_hz=100)


@pytest.fixture(scope="module")
def style_transfer():
    """Create a StyleTransfer instance."""
    return StyleTransfer(transfer_mode="mean_only", blend_mode="cosine")


@pytest.fixture(scope="module")
def template_bank():
    """Create a PromptTemplateBank instance."""
    return PromptTemplateBank()


# =============================================================================
# Fixtures - Task Generators (module scope - expensive to create)
# =============================================================================


@pytest.fixture(scope="module")
def existence_generator():
    """Create ExistenceTaskGenerator with loaded artifacts."""
    if not PHASE1_AVAILABLE:
        pytest.skip("Phase 1 artifacts not available")
    return ExistenceTaskGenerator.create_with_artifacts(seed=42)


@pytest.fixture(scope="module")
def localization_generator():
    """Create LocalizationTaskGenerator with loaded artifacts."""
    if not PHASE1_AVAILABLE:
        pytest.skip("Phase 1 artifacts not available")
    return LocalizationTaskGenerator.create_with_artifacts(seed=42)


@pytest.fixture(scope="module")
def counting_generator():
    """Create CountingTaskGenerator with loaded artifacts."""
    if not PHASE1_AVAILABLE:
        pytest.skip("Phase 1 artifacts not available")
    return CountingTaskGenerator.create_with_artifacts(seed=42)


@pytest.fixture(scope="module")
def ordering_generator():
    """Create OrderingTaskGenerator with loaded artifacts."""
    if not PHASE1_AVAILABLE:
        pytest.skip("Phase 1 artifacts not available")
    return OrderingTaskGenerator.create_with_artifacts(seed=42)


@pytest.fixture(scope="module")
def state_query_generator():
    """Create StateQueryTaskGenerator with loaded artifacts."""
    if not PHASE1_AVAILABLE:
        pytest.skip("Phase 1 artifacts not available")
    return StateQueryTaskGenerator.create_with_artifacts(seed=42)


@pytest.fixture(scope="module")
def antecedent_generator():
    """Create AntecedentTaskGenerator with loaded artifacts."""
    if not PHASE1_AVAILABLE:
        pytest.skip("Phase 1 artifacts not available")
    return AntecedentTaskGenerator.create_with_artifacts(seed=42)


@pytest.fixture(scope="module")
def comparison_generator():
    """Create ComparisonTaskGenerator with loaded artifacts."""
    if not PHASE1_AVAILABLE:
        pytest.skip("Phase 1 artifacts not available")
    return ComparisonTaskGenerator.create_with_artifacts(seed=42)


@pytest.fixture(scope="module")
def multi_hop_generator():
    """Create MultiHopTaskGenerator with loaded artifacts."""
    if not PHASE1_AVAILABLE:
        pytest.skip("Phase 1 artifacts not available")
    return MultiHopTaskGenerator.create_with_artifacts(seed=42)


@pytest.fixture(scope="module")
def anomaly_detection_generator():
    """Create AnomalyDetectionTaskGenerator with loaded artifacts."""
    if not PHASE1_AVAILABLE:
        pytest.skip("Phase 1 artifacts not available")
    return AnomalyDetectionTaskGenerator.create_with_artifacts(seed=42)


@pytest.fixture(scope="module")
def anomaly_localization_generator():
    """Create AnomalyLocalizationTaskGenerator with loaded artifacts."""
    if not PHASE1_AVAILABLE:
        pytest.skip("Phase 1 artifacts not available")
    return AnomalyLocalizationTaskGenerator.create_with_artifacts(seed=42)


# =============================================================================
# Fixtures - Common Test Parameters
# =============================================================================


@pytest.fixture
def rng():
    """Create a seeded RNG for reproducible tests."""
    return np.random.default_rng(42)


@pytest.fixture
def small_difficulty():
    """DifficultyConfig for fast tests with small context."""
    return DifficultyConfig(
        context_length_samples=5000,
        needle_position="random",
        needle_length_ratio_range=(0.06, 0.30),  # 300-1500 samples for 5000 context
        background_purity="pure",
        task_specific={
            # Ratio-based margin/gap (adaptive to context length)
            "margin_ratio": 0.01,  # 1% of context = 50 samples
            "margin_max_samples": 50,
            "min_gap_ratio": 0.01,  # 1% of context = 50 samples
            "min_gap_max_samples": 50,
        },
    )


@pytest.fixture
def medium_difficulty():
    """DifficultyConfig for standard tests."""
    return DifficultyConfig(
        context_length_samples=10000,
        needle_position="random",
        needle_length_ratio_range=(0.03, 0.30),  # 300-3000 samples for 10000 context
        background_purity="pure",
        task_specific={
            # Ratio-based margin/gap (adaptive to context length)
            "margin_ratio": 0.02,  # 2% of context, capped at 100
            "margin_max_samples": 100,
            "min_gap_ratio": 0.02,  # 2% of context, capped at 100
            "min_gap_max_samples": 100,
        },
    )
