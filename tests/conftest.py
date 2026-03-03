# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors
# SPDX-License-Identifier: MIT

"""
Shared pytest fixtures and configuration for TS-Haystack tests.
"""

import pytest


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests (require data)"
    )
    config.addinivalue_line(
        "markers", "visualization: marks tests that generate plots"
    )


def pytest_collection_modifyitems(config, items):
    """Skip integration tests if data is not available."""
    # Check if data is available
    try:
        from ts_haystack.core import BoutIndexer
        BoutIndexer.load_index()
        data_available = True
    except Exception:
        data_available = False

    if not data_available:
        skip_integration = pytest.mark.skip(
            reason="Phase 1 artifacts not available - run build_phase1_artifacts.py first"
        )
        for item in items:
            # Skip tests that need actual data
            if "background_sampler" in item.fixturenames or "needle_sampler" in item.fixturenames:
                if "sample_" in item.name or "integration" in item.name:
                    item.add_marker(skip_integration)


@pytest.fixture(scope="session")
def plots_dir(tmp_path_factory):
    """Create a temporary directory for test plots."""
    return tmp_path_factory.mktemp("plots")
