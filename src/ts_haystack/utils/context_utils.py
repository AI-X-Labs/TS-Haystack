# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors
# SPDX-License-Identifier: MIT

"""
Context length utilities for TS-Haystack benchmark.

Provides functions for converting context lengths between different formats
and creating filesystem-safe directory names.
"""

from typing import Union


def format_context_dir(context_seconds: Union[int, float]) -> str:
    """
    Convert context length to directory name format.

    For integer context lengths (or floats that equal integers), returns
    the traditional format (e.g., "100s"). For true float values, replaces
    the decimal point with underscore (e.g., "2_56s").

    This maintains backward compatibility with existing integer-based
    directory structures while supporting new float context lengths.

    Args:
        context_seconds: Context length in seconds (int or float)

    Returns:
        Directory name string suitable for filesystem paths

    Examples:
        >>> format_context_dir(100)
        '100s'
        >>> format_context_dir(100.0)
        '100s'
        >>> format_context_dir(2.56)
        '2_56s'
        >>> format_context_dir(1.5)
        '1_5s'
    """
    # Check if it's effectively an integer
    if context_seconds == int(context_seconds):
        return f"{int(context_seconds)}s"

    # Float with decimals - replace dot with underscore
    return str(context_seconds).replace(".", "_") + "s"


def parse_context_dir(dir_name: str) -> float:
    """
    Parse a context directory name back to seconds.

    Inverse of format_context_dir(). Handles both integer format ("100s")
    and float format ("2_56s").

    Args:
        dir_name: Directory name (e.g., "100s" or "2_56s")

    Returns:
        Context length in seconds as float

    Raises:
        ValueError: If directory name format is invalid

    Examples:
        >>> parse_context_dir("100s")
        100.0
        >>> parse_context_dir("2_56s")
        2.56
    """
    if not dir_name.endswith("s"):
        raise ValueError(f"Invalid context directory name: {dir_name} (must end with 's')")

    # Remove the 's' suffix
    value_str = dir_name[:-1]

    # Replace underscore with dot for float parsing
    value_str = value_str.replace("_", ".")

    try:
        return float(value_str)
    except ValueError:
        raise ValueError(f"Invalid context directory name: {dir_name}")
