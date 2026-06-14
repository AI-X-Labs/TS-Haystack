# SPDX-FileCopyrightText: 2026 Anonymous Authors
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
Task generator tests for TS-Haystack benchmark.

This module contains tests for all 8 task generators, including:
- Sample generation validation
- Visualization for human evaluation
- Prompt and answer format verification

Running Tests
-------------
# Run all task tests with pytest output capture:
pytest src/datasets/ts_haystack/test/tasks/ -v

# Run all task tests WITHOUT pytest output capture (most detailed):
pytest src/datasets/ts_haystack/test/tasks/ -v -s

# Run with visualizations (default):
pytest src/datasets/ts_haystack/test/tasks/ -v --visualize

# Skip visualizations:
pytest src/datasets/ts_haystack/test/tasks/ -v --no-visualize

# Run specific task:
pytest src/datasets/ts_haystack/test/tasks/test_existence.py -v

Prerequisites
-------------
1. Capture24 data extracted (see capture24/README.md)
2. Phase 1 artifacts built (run build_core_artifacts.py)

Output
------
Visualizations are saved to: test/plots/{task_type}/
"""
