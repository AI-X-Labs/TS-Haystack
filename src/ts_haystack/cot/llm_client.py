# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

"""
OpenAI API client for TS-Haystack CoT generation.

This module provides a client for the OpenAI API with:
- Exponential backoff retry for rate limits and transient errors
- Structured JSON output for consistent rationale + answer format
- Support for multimodal prompts (text + image via base64 encoding)
"""

import base64
import json
import os
import random
import threading
import time
from dataclasses import dataclass
from io import BytesIO
from typing import Optional

from PIL import Image


# JSON schema for structured CoT response
COT_RESPONSE_SCHEMA = {
    "name": "cot_response",
    "schema": {
        "type": "object",
        "properties": {
            "rationale": {
                "type": "string",
                "description": (
                    "Step-by-step reasoning analyzing the accelerometer data patterns. "
                    "Should describe observations about the signal, identify relevant activity bouts, "
                    "and explain how the answer is derived."
                )
            },
            "answer": {
                "type": "string",
                "description": "The final answer to the question."
            }
        },
        "required": ["rationale", "answer"],
        "additionalProperties": False
    },
    "strict": True
}


@dataclass
class OpenAIConfig:
    """Configuration for OpenAI API client."""
    model: str = "gpt-4.1-mini-2025-04-14"
    max_retries: int = 5
    base_retry_delay: float = 1.0
    max_retry_delay: float = 60.0
    temperature: float = 0.3


class OpenAICoTClient:
    """
    Client for OpenAI API calls with exponential backoff retry.

    This client is designed for generating chain-of-thought rationales
    for TS-Haystack benchmark samples.

    Usage:
        client = OpenAICoTClient()
        result = client.generate(prompt, image)
        # result = {"rationale": "...", "answer": "..."}
    """

    # HTTP status codes that should trigger a retry
    RETRYABLE_STATUS_CODES = (429, 500, 502, 503, 504)

    def __init__(self, config: Optional[OpenAIConfig] = None):
        """
        Initialize the OpenAI client.

        Args:
            config: Configuration object. If None, uses defaults.
        """
        self.config = config or OpenAIConfig()
        # Use thread-local storage to ensure each thread gets its own client
        # This prevents issues when using ThreadPoolExecutor
        self._local = threading.local()

    def _ensure_client(self):
        """Lazily initialize the OpenAI client (thread-local)."""
        if not hasattr(self._local, "client") or self._local.client is None:
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError(
                    "openai package is required. "
                    "Install with: pip install openai"
                )

            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "OPENAI_API_KEY environment variable is not set. "
                    "Please set it to use the OpenAI API."
                )
            self._local.client = OpenAI(api_key=api_key)
            print(f"Initialized OpenAI client with model: {self.config.model}")

    def _pil_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64-encoded PNG string."""
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _is_retryable(self, exception: Exception) -> bool:
        """Check if an exception should trigger a retry."""
        error_str = str(exception).lower()

        # Check for rate limit or server errors in the exception message
        if any(str(code) in error_str for code in self.RETRYABLE_STATUS_CODES):
            return True

        # Also retry on connection/timeout/transient errors
        retryable_keywords = [
            "timeout", "connection", "temporarily",
            "closed", "reset", "unavailable", "overloaded",
            "rate_limit", "rate limit"
        ]
        if any(keyword in error_str for keyword in retryable_keywords):
            return True

        return False

    def _get_retry_delay(self, attempt: int) -> float:
        """Calculate retry delay with exponential backoff and jitter."""
        base_delay = self.config.base_retry_delay * (2 ** attempt)
        # Add jitter: random value between 0 and 1 second
        jitter = random.uniform(0, 1)
        delay = min(base_delay + jitter, self.config.max_retry_delay)
        return delay

    def generate(
        self,
        prompt: str,
        image: Optional[Image.Image] = None,
        temperature: Optional[float] = None,
    ) -> Optional[dict]:
        """
        Generate structured response using OpenAI API with retry logic.

        Args:
            prompt: The prompt to send to the model
            image: Optional PIL Image to include with the prompt
            temperature: Sampling temperature (0.0-1.0). If None, uses config default.

        Returns:
            Dict with 'rationale' and 'answer' keys, or None if all retries failed
        """
        self._ensure_client()

        # Build message content
        content = []

        # Add image first if provided (as base64-encoded PNG)
        if image is not None:
            base64_image = self._pil_to_base64(image)
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{base64_image}"
                }
            })

        # Add text prompt
        content.append({
            "type": "text",
            "text": prompt
        })

        messages = [{"role": "user", "content": content}]

        temp = temperature if temperature is not None else self.config.temperature

        last_exception = None
        for attempt in range(self.config.max_retries):
            try:
                response = self._local.client.chat.completions.create(
                    model=self.config.model,
                    messages=messages,
                    temperature=temp,
                    response_format={
                        "type": "json_schema",
                        "json_schema": COT_RESPONSE_SCHEMA
                    }
                )

                if response.choices and response.choices[0].message.content:
                    return json.loads(response.choices[0].message.content)
                return None

            except Exception as e:
                last_exception = e
                if self._is_retryable(e) and attempt < self.config.max_retries - 1:
                    delay = self._get_retry_delay(attempt)
                    print(
                        f"  [RETRY] Attempt {attempt + 1}/{self.config.max_retries} "
                        f"failed: {e}. Retrying in {delay:.1f}s..."
                    )
                    # Reinitialize client on connection errors
                    if "closed" in str(e).lower() or "connection" in str(e).lower():
                        self._local.client = None
                        self._ensure_client()
                    time.sleep(delay)
                else:
                    # Non-retryable error or last attempt
                    break

        print(
            f"  [ERROR] OpenAI API failed after {self.config.max_retries} "
            f"attempts: {last_exception}"
        )
        return None

    def generate_with_validation(
        self,
        prompt: str,
        expected_answer: str,
        image: Optional[Image.Image] = None,
        temperature: Optional[float] = None,
        max_attempts: int = 3,
    ) -> Optional[dict]:
        """
        Generate response and validate that the answer matches expected.

        This is useful for ensuring the LLM's reasoning leads to the correct answer.

        Args:
            prompt: The prompt to send to the model
            expected_answer: The expected answer to validate against
            image: Optional PIL Image
            temperature: Sampling temperature
            max_attempts: Maximum attempts to get matching answer

        Returns:
            Dict with 'rationale' and 'answer' keys if answer matches,
            or None if validation fails after max_attempts
        """
        for attempt in range(max_attempts):
            result = self.generate(prompt, image, temperature)

            if result is None:
                continue

            # Check if answer matches (case-insensitive, stripped)
            generated_answer = result.get("answer", "").strip().lower()
            expected_normalized = expected_answer.strip().lower()

            if generated_answer == expected_normalized:
                return result

            # For boolean answers, also check yes/no variants
            if expected_normalized in ["yes", "no"]:
                if generated_answer in ["yes", "no"] and generated_answer == expected_normalized:
                    return result

            print(
                f"  [VALIDATION] Answer mismatch (attempt {attempt + 1}/{max_attempts}): "
                f"expected '{expected_answer}', got '{result.get('answer', '')}'"
            )

        return None


if __name__ == "__main__":
    print("=" * 60)
    print("OpenAI CoT Client Test")
    print("=" * 60)

    # Test client initialization
    try:
        client = OpenAICoTClient()

        # Test simple generation (no image)
        test_prompt = """
        Analyze this scenario: A person is sitting at a desk for 2 hours.
        Question: What is the primary activity?
        Answer with the activity type.
        """

        print("\nTesting generation...")
        result = client.generate(test_prompt)

        if result:
            print(f"\nResult:")
            print(f"  Rationale: {result.get('rationale', 'N/A')[:200]}...")
            print(f"  Answer: {result.get('answer', 'N/A')}")
        else:
            print("\nGeneration failed")

    except Exception as e:
        print(f"\nError: {e}")
        print("Make sure OPENAI_API_KEY environment variable is set.")

    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)
