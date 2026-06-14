# SPDX-FileCopyrightText: 2026 Anonymous Authors
# SPDX-FileCopyrightText: 2026 Anonymous Authors
#
# SPDX-License-Identifier: CC-BY-NC-4.0

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
from .qa_loader import (
    get_label_list,
    load_capture24_classification_splits,
)
from .qa_dataset import Capture24AccQADataset
from .eval_dataset import Capture24EvalQADataset, load_all_eval_datasets
from .evaluation import (
    WILLETTS_SPECIFIC_2018_LABELS,
    evaluate_classification,
    normalize_label,
    extract_predicted_label,
    compute_balanced_accuracy,
    print_classification_summary,
    print_samples_per_activity,
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
    # QA Dataset
    "load_capture24_classification_splits",
    "get_label_list",
    "Capture24AccQADataset",
    # Evaluation Dataset
    "Capture24EvalQADataset",
    "load_all_eval_datasets",
    "WILLETTS_SPECIFIC_2018_LABELS",
    # Evaluation Utilities
    "evaluate_classification",
    "normalize_label",
    "extract_predicted_label",
    "compute_balanced_accuracy",
    "print_classification_summary",
    "print_samples_per_activity",
    "format_confusion_matrix",
    "aggregate_results_by_context_length",
]
