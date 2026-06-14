"""Serializer: converts discrete numeric arrays to/from text representation.

Ported from ChatTime (Wang et al., AAAI 2025). Each numeric value is wrapped in
flag delimiters (default ``###``) and values are separated by spaces, producing
strings like ``###0.1234### ###-0.5000### ###0.2500###``.
"""

from __future__ import annotations

import re

import numpy as np


class Serializer:
    """Serialize / deserialize numeric arrays as delimited text.

    Args:
        prec: Decimal precision for formatting floats.
        time_sep: Separator between tokens.
        time_flag: Delimiter wrapping each value (e.g. ``###``).
        nan_flag: Text representation for NaN values.
    """

    def __init__(
        self,
        prec: int = 4,
        time_sep: str = " ",
        time_flag: str = "###",
        nan_flag: str = "Nan",
    ):
        self.prec = prec
        self.time_sep = time_sep
        self.time_flag = time_flag
        self.nan_flag = nan_flag

    def serialize(self, context: np.ndarray) -> str:
        """Convert a 1-D numeric array to a delimited string.

        Example::

            >>> s = Serializer()
            >>> s.serialize(np.array([0.1, -0.5, np.nan]))
            '###0.1000### ###-0.5000### ###Nan###'

        Args:
            context: 1-D array of (possibly NaN) float values.

        Returns:
            Space-separated string of ``{flag}{value}{flag}`` tokens.
        """
        serialized_context = np.array([f"{self.time_flag}{v:.{self.prec}f}{self.time_flag}" for v in context])
        serialized_context[np.isnan(context)] = f"{self.time_flag}{self.nan_flag}{self.time_flag}"
        return self.time_sep.join(serialized_context)

    def inverse_serialize(self, serialized_context: str) -> np.ndarray:
        """Parse a delimited string back into a 1-D numeric array.

        Args:
            serialized_context: String produced by :meth:`serialize`.

        Returns:
            1-D numpy array of float values (unparseable tokens become NaN).
        """
        pattern = rf"{re.escape(self.time_flag)}(.*?){re.escape(self.time_flag)}"
        matches = re.findall(pattern, serialized_context)

        values: list[float] = []
        for token in matches:
            try:
                values.append(float(token))
            except ValueError:
                values.append(np.nan)

        return np.array(values)
