# SPDX-FileCopyrightText: 2026 Anonymous Authors
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
Tests for TS-Haystack benchmark.

Test modules:
- test_style_transfer.py: StyleTransfer unit tests (synthetic data)
- test_prompt_templates.py: PromptTemplateBank unit tests (no data needed)
- test_needle_sampler.py: NeedleSampler integration tests (requires data)
- test_background_sampler.py: BackgroundSampler integration tests (requires data)
- test_phase2_integration.py: Full pipeline integration tests (requires data)

Running tests:

    # Run all tests (some will skip if data unavailable)
    pytest src/datasets/ts_haystack/test/ -v

    # Run only unit tests (no data required)
    pytest src/datasets/ts_haystack/test/test_style_transfer.py -v
    pytest src/datasets/ts_haystack/test/test_prompt_templates.py -v

    # Run with plot generation
    pytest src/datasets/ts_haystack/test/ -v -k "visualization"

    # Run specific test class
    pytest src/datasets/ts_haystack/test/test_style_transfer.py::TestVisualization -v

Plots are saved to:
    src/datasets/ts_haystack/test/plots/

Prerequisites for integration tests:
1. Capture24 sensor data extracted (run scripts/data/build_core_artifacts.py)
2. Phase 1 artifacts built (run scripts/data/build_ts_haystack.py)
"""
