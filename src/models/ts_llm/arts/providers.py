# SPDX-License-Identifier: CC-BY-NC-4.0
"""LLM provider adapters for the ARTS orchestrator.

ARTS instantiates its orchestrator with a frontier chat model used off the
shelf (the paper uses GPT-5.4). These adapters normalize the OpenAI Responses
API and the Anthropic Messages API (direct or via Bedrock) behind a single
``ModelResponse`` shape so the agentic loop is provider-agnostic.

Tool schemas are domain-specific (each ARTS domain exposes its own classifier
tool), so ``get_tool_config`` takes an OpenAI-style tool dict and adapts it to
the target provider rather than hard-coding any one domain's tool.
"""

from __future__ import annotations

import json
import time as _time
from dataclasses import dataclass, field
from threading import Lock
from typing import Any


# ---------------------------------------------------------------------------
# Normalized response dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ToolCallResponse:
    call_id: str
    name: str
    arguments: dict  # already parsed


@dataclass
class ModelResponse:
    text_parts: list[str] = field(default_factory=list)
    tool_calls: list[ToolCallResponse] = field(default_factory=list)
    reasoning_summaries: list[str] = field(default_factory=list)
    raw: Any = None  # provider-native object for feeding back into conversation


# ---------------------------------------------------------------------------
# Provider adapters
# ---------------------------------------------------------------------------


class OpenAIProvider:
    """Adapter for OpenAI Responses API."""

    def __init__(self, model: str):
        self.model = model
        self._clients: dict[int, Any] = {}
        self._lock = Lock()

    def _get_client(self):
        import threading
        from openai import OpenAI
        tid = threading.get_ident()
        if tid not in self._clients:
            with self._lock:
                if tid not in self._clients:
                    self._clients[tid] = OpenAI()
        return self._clients[tid]

    def build_messages(self, system_prompt: str, pre_prompt: str, post_prompt: str, img_b64: str | None) -> list:
        if img_b64:
            user_content = [
                {"type": "input_text", "text": pre_prompt},
                {"type": "input_image", "image_url": f"data:image/png;base64,{img_b64}"},
                {"type": "input_text", "text": post_prompt},
            ]
        else:
            user_content = f"{pre_prompt}\n{post_prompt}"
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

    def get_tool_config(self, openai_tool: dict) -> list:
        return [openai_tool]

    def call(self, messages: list, tools: list, reasoning_effort: str, max_tokens: int) -> ModelResponse:
        client = self._get_client()
        response = None
        for _attempt in range(10):
            try:
                kwargs = {
                    "model": self.model,
                    "input": messages,
                    "tools": tools,
                    "max_output_tokens": max_tokens,
                }
                if reasoning_effort != "none":
                    kwargs["reasoning"] = {"effort": reasoning_effort, "summary": "auto"}
                response = client.responses.create(**kwargs)
                break
            except Exception as e:
                err_str = str(e)
                is_retryable = "rate_limit" in err_str or "429" in err_str or "500" in err_str or "503" in err_str
                if is_retryable and _attempt < 9:
                    wait = min(2 ** _attempt + 1, 120)
                    _time.sleep(wait)
                    continue
                raise

        # Parse response into normalized format
        text_parts = []
        tool_calls = []
        reasoning_summaries = []

        for item in response.output:
            if item.type == "reasoning":
                if hasattr(item, "summary") and item.summary:
                    for s in item.summary:
                        if hasattr(s, "text"):
                            reasoning_summaries.append(s.text)
            elif item.type == "message":
                for content_part in item.content:
                    if content_part.type == "output_text":
                        text_parts.append(content_part.text)
            elif item.type == "function_call":
                try:
                    args = json.loads(item.arguments)
                except json.JSONDecodeError:
                    args = {"start_time": "", "end_time": ""}
                tool_calls.append(ToolCallResponse(
                    call_id=item.call_id,
                    name=item.name,
                    arguments=args,
                ))

        return ModelResponse(
            text_parts=text_parts,
            tool_calls=tool_calls,
            reasoning_summaries=reasoning_summaries,
            raw=response.output,
        )

    def append_response(self, messages: list, response: ModelResponse) -> None:
        for item in response.raw:
            messages.append(item)

    def append_tool_result(self, messages: list, call_id: str, output: str) -> None:
        messages.append({
            "type": "function_call_output",
            "call_id": call_id,
            "output": output,
        })

    def append_user_text(self, messages: list, text: str) -> None:
        messages.append({"role": "user", "content": text})


class AnthropicProvider:
    """Adapter for Anthropic Messages API (direct or via Bedrock).

    Uses the same Messages API for both — only the client constructor differs.
    """

    _EFFORT_TO_BUDGET = {"low": 2048, "medium": 8192, "high": 16384}

    def __init__(self, model: str, thinking_budget: int | None = None,
                 bedrock: bool = False, aws_region: str = "us-east-1"):
        self.model = model
        self.thinking_budget = thinking_budget
        self._bedrock = bedrock
        self._aws_region = aws_region
        self._system_prompt: str | None = None
        self._clients: dict[int, Any] = {}
        self._lock = Lock()

    def _get_client(self):
        import threading
        tid = threading.get_ident()
        if tid not in self._clients:
            with self._lock:
                if tid not in self._clients:
                    import anthropic
                    if self._bedrock:
                        self._clients[tid] = anthropic.AnthropicBedrock(aws_region=self._aws_region)
                    else:
                        self._clients[tid] = anthropic.Anthropic()
        return self._clients[tid]

    def build_messages(self, system_prompt: str, pre_prompt: str, post_prompt: str, img_b64: str | None) -> list:
        self._system_prompt = system_prompt
        if img_b64:
            user_content = [
                {"type": "text", "text": pre_prompt},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": img_b64}},
                {"type": "text", "text": post_prompt},
            ]
        else:
            user_content = [{"type": "text", "text": f"{pre_prompt}\n{post_prompt}"}]
        return [{"role": "user", "content": user_content}]

    def get_tool_config(self, openai_tool: dict) -> list:
        return [{
            "name": openai_tool["name"],
            "description": openai_tool["description"],
            "input_schema": openai_tool["parameters"],
        }]

    def call(self, messages: list, tools: list, reasoning_effort: str, max_tokens: int) -> ModelResponse:
        client = self._get_client()

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if self._system_prompt:
            kwargs["system"] = self._system_prompt
        if tools:
            kwargs["tools"] = tools
        if reasoning_effort != "none":
            budget = self.thinking_budget or self._EFFORT_TO_BUDGET.get(reasoning_effort, 16384)
            kwargs["thinking"] = {"type": "enabled", "budget_tokens": budget}
            kwargs["max_tokens"] = budget + max_tokens

        response = None
        for _attempt in range(10):
            try:
                with client.messages.stream(**kwargs) as stream:
                    response = stream.get_final_message()
                break
            except Exception as e:
                err_str = str(e)
                # Daily/monthly quota exhaustion — not retryable
                if "Too many tokens" in err_str or "quota" in err_str.lower():
                    raise
                is_retryable = any(s in err_str for s in [
                    "rate_limit", "overloaded", "429", "500", "503", "529",
                ])
                if is_retryable and _attempt < 9:
                    wait = min(2 ** _attempt + 1, 120)
                    _time.sleep(wait)
                    continue
                raise

        text_parts = []
        tool_calls = []
        reasoning_summaries = []

        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(ToolCallResponse(
                    call_id=block.id,
                    name=block.name,
                    arguments=block.input if isinstance(block.input, dict) else {},
                ))
            elif block.type == "thinking":
                thinking_text = block.thinking
                if thinking_text:
                    summary = thinking_text[:2000] + ("..." if len(thinking_text) > 2000 else "")
                    reasoning_summaries.append(summary)

        # Serialize content blocks with only API-accepted fields
        serialized = []
        for block in response.content:
            if block.type == "thinking":
                serialized.append({"type": "thinking", "thinking": block.thinking, "signature": block.signature})
            elif block.type == "text":
                serialized.append({"type": "text", "text": block.text})
            elif block.type == "tool_use":
                serialized.append({"type": "tool_use", "id": block.id, "name": block.name, "input": block.input})

        return ModelResponse(
            text_parts=text_parts,
            tool_calls=tool_calls,
            reasoning_summaries=reasoning_summaries,
            raw={"role": "assistant", "content": serialized},
        )

    def append_response(self, messages: list, response: ModelResponse) -> None:
        messages.append(response.raw)

    def append_tool_result(self, messages: list, call_id: str, output: str) -> None:
        result_block = {"type": "tool_result", "tool_use_id": call_id, "content": output}
        # Anthropic requires strict role alternation — batch tool results into one user message
        if (messages and messages[-1].get("role") == "user"
                and any(isinstance(c, dict) and c.get("type") == "tool_result"
                        for c in messages[-1].get("content", []))):
            messages[-1]["content"].append(result_block)
        else:
            messages.append({"role": "user", "content": [result_block]})

    def append_user_text(self, messages: list, text: str) -> None:
        text_block = {"type": "text", "text": text}
        if messages and messages[-1].get("role") == "user":
            messages[-1]["content"].append(text_block)
        else:
            messages.append({"role": "user", "content": [text_block]})
