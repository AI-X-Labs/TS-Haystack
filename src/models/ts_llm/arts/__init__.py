# SPDX-License-Identifier: CC-BY-NC-4.0
"""ARTS — Agentic Retrieval for Time-Series.

ARTS reframes long-context time-series retrieval as a two-stage process: an
LLM *orchestrator* (a frontier chat model used off the shelf; the paper uses
GPT-5.4) reasons over a symbolic timeline produced by sweeping per-domain
*classifier tools* over the recording, then issues ``<bout>`` tool calls to
re-classify segments at full resolution until it commits to an answer.

This package holds the shared, domain-agnostic machinery:

- ``providers``  — OpenAI / Anthropic adapters + the normalized ``ModelResponse``
- ``query``      — per-recording query-length and tool-round budgets
- ``bout_utils`` — ``<bout>`` parsing, timestamp<->index conversion, OracleClassifier

The per-domain classifier *tools* themselves are the trained models in
``src.models.classifiers`` (capture24 / sleep / ecg / uk_dale). Per-domain
orchestration (prompts, data loading, the agentic run loop, the classifier-tool
execution) lives in the per-domain modules of this package — ``capture24``,
``sleep``, ``ltaf``, plus the oracle pre-pass variants — each runnable as a CLI
via ``python -m src.models.ts_llm.arts.<domain>``.
"""

from src.models.ts_llm.arts.providers import (
    ModelResponse,
    ToolCallResponse,
    OpenAIProvider,
    AnthropicProvider,
)
from src.models.ts_llm.arts.query import (
    get_max_query_seconds,
    get_tool_rounds,
)

__all__ = [
    "ModelResponse",
    "ToolCallResponse",
    "OpenAIProvider",
    "AnthropicProvider",
    "get_max_query_seconds",
    "get_tool_rounds",
]
