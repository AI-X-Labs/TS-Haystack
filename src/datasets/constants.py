# SPDX-FileCopyrightText: 2026 Anonymous Authors
# SPDX-License-Identifier: CC-BY-NC-4.0

# constants.py

import os

# Path to this file's directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Path to raw data directory (go up from datasets -> src -> project_root -> data)
RAW_DATA = os.path.join(BASE_DIR, "..", "..", "data")
