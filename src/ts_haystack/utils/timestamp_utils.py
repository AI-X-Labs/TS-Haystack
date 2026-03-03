# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors
# SPDX-License-Identifier: MIT

"""
Timestamp conversion utilities for TS-Haystack.

Provides functions to convert between sample indices and human-readable timestamps.
"""

from datetime import datetime, timedelta
from typing import Tuple


def parse_time_string(time_str: str) -> datetime:
    """
    Parse a time string to datetime, supporting various formats including
    those with seconds and milliseconds.

    Args:
        time_str: Time string in formats like:
            - "6:00 AM"
            - "6:00:00 AM"
            - "6:00:00.000 AM"
            - "06:00"
            - "06:00:00"
            - "06:00:00.000"

    Returns:
        datetime object (date will be arbitrary baseline)
    """
    time_str = time_str.strip()

    # Handle milliseconds separately since strptime %f expects microseconds
    ms = 0
    if "." in time_str:
        # Split off milliseconds
        parts = time_str.rsplit(".", 1)
        time_str_base = parts[0]
        # Extract ms part (may have AM/PM attached)
        ms_part = parts[1]
        if " " in ms_part:
            ms_str, am_pm = ms_part.split(" ", 1)
            time_str = f"{time_str_base} {am_pm}"
            ms = int(ms_str)
        else:
            ms = int(ms_part)
            time_str = time_str_base

    # Try common formats
    formats = [
        "%I:%M:%S %p",   # "6:00:00 AM"
        "%I:%M %p",      # "6:00 AM"
        "%H:%M:%S",      # "06:00:00"
        "%H:%M",         # "06:00"
    ]

    for fmt in formats:
        try:
            dt = datetime.strptime(time_str, fmt)
            # Add milliseconds as microseconds
            return dt.replace(microsecond=ms * 1000)
        except ValueError:
            continue

    raise ValueError(f"Could not parse time string: {time_str}")


def format_timestamp(dt: datetime) -> str:
    """
    Format a datetime to human-readable time string with full precision.

    Always includes seconds and milliseconds to ensure distinguishable
    timestamps even for short needle durations (a few seconds).

    Args:
        dt: datetime object

    Returns:
        String like "6:00:00.000 AM" or "2:30:15.500 PM"
    """
    # Format with seconds
    base = dt.strftime("%I:%M:%S")
    # Add milliseconds
    ms = dt.microsecond // 1000
    am_pm = dt.strftime("%p")
    # Remove leading zero from hour
    result = f"{base}.{ms:03d} {am_pm}"
    return result.lstrip("0") or "0" + result[1:]  # Keep at least one digit for hour


def samples_to_timestamp(
    sample_idx: int,
    total_samples: int,
    start_time_str: str,
    end_time_str: str,
) -> str:
    """
    Convert sample index to human-readable timestamp.

    Linearly interpolates between start_time and end_time based on
    the sample position within the total window.

    Args:
        sample_idx: Sample index (0 to total_samples-1)
        total_samples: Total number of samples in the window
        start_time_str: Start time string (e.g., "6:00 AM")
        end_time_str: End time string (e.g., "8:00 AM")

    Returns:
        Human-readable timestamp string (e.g., "6:45 AM")

    Example:
        >>> samples_to_timestamp(5000, 10000, "6:00 AM", "8:00 AM")
        "7:00 AM"
    """
    if total_samples <= 0:
        return start_time_str

    # Clamp sample_idx to valid range
    sample_idx = max(0, min(sample_idx, total_samples - 1))

    # Parse times
    start_dt = parse_time_string(start_time_str)
    end_dt = parse_time_string(end_time_str)

    # Handle day wraparound (e.g., 11 PM to 1 AM)
    if end_dt < start_dt:
        end_dt += timedelta(days=1)

    # Compute interpolation fraction
    frac = sample_idx / max(total_samples - 1, 1)

    # Interpolate
    delta = end_dt - start_dt
    result_dt = start_dt + timedelta(seconds=delta.total_seconds() * frac)

    return format_timestamp(result_dt)


def samples_to_timestamp_from_background(
    sample_idx: int,
    background,  # BackgroundSample
) -> str:
    """
    Convert sample index to timestamp using background context.

    Convenience wrapper that extracts time context from BackgroundSample.

    Args:
        sample_idx: Sample index within the background window
        background: BackgroundSample object with recording_time_context

    Returns:
        Human-readable timestamp string
    """
    return samples_to_timestamp(
        sample_idx=sample_idx,
        total_samples=background.n_samples,
        start_time_str=background.recording_time_context[0],
        end_time_str=background.recording_time_context[1],
    )


def format_time_range(start: str, end: str) -> str:
    """
    Format a time range for display in answers.

    Args:
        start: Start timestamp string
        end: End timestamp string

    Returns:
        Formatted range like "from 6:00 AM to 7:30 AM"
    """
    return f"from {start} to {end}"


def compute_duration_string(duration_samples: int, source_hz: int = 100) -> str:
    """
    Convert sample duration to human-readable string.

    Args:
        duration_samples: Duration in samples
        source_hz: Sampling frequency in Hz

    Returns:
        String like "30 seconds" or "2.5 minutes"
    """
    duration_seconds = duration_samples / source_hz

    if duration_seconds < 60:
        if duration_seconds == int(duration_seconds):
            return f"{int(duration_seconds)} seconds"
        return f"{duration_seconds:.1f} seconds"

    duration_minutes = duration_seconds / 60
    if duration_minutes == int(duration_minutes):
        return f"{int(duration_minutes)} minutes"
    return f"{duration_minutes:.1f} minutes"


def ms_to_timestamp(
    target_ms: int,
    window_start_ms: int,
    window_end_ms: int,
    start_time_str: str,
    end_time_str: str,
) -> str:
    """
    Convert millisecond timestamp to human-readable format.

    Args:
        target_ms: Target timestamp in milliseconds
        window_start_ms: Window start in milliseconds
        window_end_ms: Window end in milliseconds
        start_time_str: Start time string
        end_time_str: End time string

    Returns:
        Human-readable timestamp string
    """
    if window_end_ms <= window_start_ms:
        return start_time_str

    # Clamp to window
    target_ms = max(window_start_ms, min(target_ms, window_end_ms))

    # Compute fraction
    frac = (target_ms - window_start_ms) / (window_end_ms - window_start_ms)

    # Parse times
    start_dt = parse_time_string(start_time_str)
    end_dt = parse_time_string(end_time_str)

    # Handle day wraparound
    if end_dt < start_dt:
        end_dt += timedelta(days=1)

    # Interpolate
    delta = end_dt - start_dt
    result_dt = start_dt + timedelta(seconds=delta.total_seconds() * frac)

    return format_timestamp(result_dt)
