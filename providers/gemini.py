"""Google Gemini provider implementation via OpenAI-compatible API."""

import os
from typing import Iterator

import httpx

from .base import BaseProvider, CompletionChunk, CompletionRequest, Message


GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"

SUPPORTED_MODELS = [
    "gemini-2.5-pro-preview-05-06",
    "gemini-2.5-flash-preview-04-17",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-1.5-flash",
    "gemini-1.5-flash-8b",
    "gemini-1.5-pro",
]

# Default model to use when none is specified - prefer the latest stable flash model
DEFAULT_MODEL = "gemini-2.5-flash-preview-04-17"


class GeminiProvider(BaseProvider):
    """Provider for Google Gemini models using the OpenAI-compatible REST API.

    Requires a GEMINI_API_KEY environment variable.
    """

    def __init__(self) -> None:
        api_key = os.getenv("GEMINI_API_KEY", "")
        super().__init__(
            api_key=api_key,
            base_url=GEMINI_BASE_URL,
            supported_models=SUPPORTED_MODELS,
        )

    def is_available(self) -> bool:
        """Return True when a non-empty GEMINI_API_KEY is configured."""
        return bool(self.api_key)

    def complete(self, request: CompletionRequest) -> Iterator[CompletionChunk]:
        """Stream a chat completion from the Gemini OpenAI-compatible endpoint.

        Args:
            request: The completion request containing model, messages, and options.

        Yields:
            CompletionChunk objects with incremental text deltas.

        Raises:
            ValueError: If the requested model is not supported.
            httpx.HTTPStatusError: If the API returns a non-2xx response.
        """
        self.validate_model(request.model)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload: dict = {
            "model": request.model,
            "messages": [
                {"role": msg.role, "content": msg.content}
                for msg in request.messages
            ],
            "stream": True,
        }

        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens

        if request.temperature is not None:
            payload["temperature"] = request.temperature

        if request.system_prompt:
            payload["messages"].insert(
                0, {"role": "system", "content": request.system_prompt}
            )

        with httpx.Client(timeout=120) as client:
            with client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
            ) as response:
                response.raise_for_status()

                for line in response.iter_lines():
                    if not line.startswith("data: "):
                        continue

                    data = line[len("data: "):].strip()
                    if data == "[DONE]":
                        break

                    import json

                    try:
                        c