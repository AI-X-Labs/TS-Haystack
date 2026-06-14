# SPDX-FileCopyrightText: 2026 Anonymous Authors
# SPDX-FileCopyrightText: 2026 Anonymous Authors
#
# SPDX-License-Identifier: CC-BY-NC-4.0

from abc import ABC, abstractmethod


class Prompt(ABC):
    @abstractmethod
    def get_text(self):
        pass
