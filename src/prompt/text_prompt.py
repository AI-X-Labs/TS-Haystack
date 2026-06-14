# SPDX-FileCopyrightText: 2026 Anonymous Authors
# SPDX-FileCopyrightText: 2026 Anonymous Authors
#
# SPDX-License-Identifier: CC-BY-NC-4.0

from .prompt import Prompt


class TextPrompt(Prompt):
    def __init__(self, text: str):
        assert isinstance(text, str), "Text must be a string!"
        self.__text = text

    def get_text(self) -> str:
        return self.__text
