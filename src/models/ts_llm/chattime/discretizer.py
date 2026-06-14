"""Discretizer: converts continuous time series values into discrete bin centers.

Ported from ChatTime (Wang et al., AAAI 2025). The discretizer fits a MinMaxScaler
on the input, maps scaled values to one of ``n_tokens`` bins, and returns the bin
center values. This allows the time series to be represented as a finite vocabulary
of numeric text tokens.
"""

from __future__ import annotations

import numpy as np
from sklearn.preprocessing import MinMaxScaler


class Discretizer:
    """Bin continuous time series values into discrete centers.

    Args:
        low_limit: Lower bound of the bin range (after scaling).
        high_limit: Upper bound of the bin range (after scaling).
        n_tokens: Number of discrete tokens (creates ``n_tokens - 1`` bin boundaries).
    """

    def __init__(self, low_limit: float = -1, high_limit: float = 1, n_tokens: int = 10002):
        self.scaler = MinMaxScaler()

        self.boundaries = np.linspace(low_limit, high_limit, n_tokens - 1)
        self.centers = (self.boundaries[1:] + self.boundaries[:-1]) / 2
        # Pad first and last center so digitize edge bins map correctly
        self.centers = np.concatenate((self.centers[:1], self.centers, self.centers[-1:]))

    def get_centers(self) -> np.ndarray:
        """Return the array of bin center values."""
        return self.centers

    def discretize(self, context: np.ndarray, fit_length: int | None = None) -> np.ndarray:
        """Discretize a 1-D time series.

        Fits MinMaxScaler on ``context[:fit_length]``, scales to [-0.5, 0.5],
        then assigns each value to the nearest bin center.

        Args:
            context: 1-D array of continuous values.
            fit_length: Number of leading samples to fit the scaler on.
                Defaults to the full length.

        Returns:
            1-D array of discrete bin center values (same length as *context*).
            NaN positions in the input are preserved as NaN.
        """
        fit_length = len(context) if fit_length is None else fit_length
        self.scaler.fit(context[:fit_length].reshape(-1, 1))
        scaled_context = self.scaler.transform(context.reshape(-1, 1)).reshape(-1) - 0.5

        bin_ids = np.digitize(x=scaled_context, bins=self.boundaries, right=True)
        dispersed_context = self.centers[bin_ids]

        dispersed_context[np.isnan(context)] = np.nan
        return dispersed_context

    def inverse_discretize(self, scaled_context: np.ndarray) -> np.ndarray:
        """Map discretized (scaled) values back to the original scale.

        Requires that :meth:`discretize` was called first so the scaler is fitted.

        Args:
            scaled_context: 1-D array of bin center values.

        Returns:
            1-D array in the original value range.
        """
        return self.scaler.inverse_transform(scaled_context.reshape(-1, 1) + 0.5).reshape(-1)
