# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

from .loader import (
    CAPTURE24_DATA_DIR,
    ensure_capture24_data,
    get_sensor_data_dir,
    load_label_mappings,
    load_participant_sensor_data,
    load_participants,
)
from .windows import (
    WINDOWS_DIR,
    extract_windows,
    format_window_size,
    get_windows_path,
    load_windows,
    split_participants,
)
from .classification import (
    CLASSIFICATION_DIR,
    LABEL_SCHEMES,
    create_classification_dataset,
    get_class_distribution,
    get_class_names,
    get_classification_path,
    load_classification_dataset,
    load_classification_metadata,
    load_label_mapping,
)
from .evaluation import (
    evaluate_classification,
    normalize_label,
    extract_predicted_label,
    compute_balanced_accuracy,
    print_classification_summary,
    print_samples_per_activity,
    print_curriculum_summary,
    format_confusion_matrix,
    aggregate_results_by_context_length,
)

__all__ = [
    # Loader
    "ensure_capture24_data",
    "load_participants",
    "load_label_mappings",
    "load_participant_sensor_data",
    "get_sensor_data_dir",
    "CAPTURE24_DATA_DIR",
    # Windows
    "extract_windows",
    "format_window_size",
    "load_windows",
    "get_windows_path",
    "split_participants",
    "WINDOWS_DIR",
    # Classification
    "create_classification_dataset",
    "load_classification_dataset",
    "load_classification_metadata",
    "get_classification_path",
    "get_class_names",
    "get_class_distribution",
    "load_label_mapping",
    "CLASSIFICATION_DIR",
    "LABEL_SCHEMES",
    # Evaluation Utilities
    "evaluate_classification",
    "normalize_label",
    "extract_predicted_label",
    "compute_balanced_accuracy",
    "print_classification_summary",
    "print_samples_per_activity",
    "print_curriculum_summary",
    "format_confusion_matrix",
    "aggregate_results_by_context_length",
]
