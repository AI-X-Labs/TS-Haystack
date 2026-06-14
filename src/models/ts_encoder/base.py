# SPDX-FileCopyrightText: 2026 Anonymous Authors
# SPDX-License-Identifier: CC-BY-NC-4.0

from abc import abstractmethod
import torch
import torch.nn as nn


class TimeSeriesEncoderBase(nn.Module):
    """Base class for time series encoders.

    Subclasses with ``multi_variable = False`` (default) take single-channel
    input ``(B, L)`` and return ``(B, N, output_dim)``.

    Subclasses with ``multi_variable = True`` take multi-variable input
    ``(B, L, V)`` and return ``(B, V, N, output_dim)``.
    """

    multi_variable: bool = False

    def __init__(
        self,
        output_dim: int = 128,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.dropout = dropout

    def get_num_output_patches(self, input_length: int) -> int:
        """Return the number of output patches for a given input length.

        Default uses ``patch_size`` (floor division) or ``patch_len``
        (ceiling division).  Subclasses with different patching logic
        should override.
        """
        if hasattr(self, "patch_size"):
            return input_length // self.patch_size
        if hasattr(self, "patch_len"):
            return -(-input_length // self.patch_len)  # ceil division
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement get_num_output_patches()"
        )

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass
