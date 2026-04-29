"""OpenRouter provider — routes requests to Claude via openrouter.ai."""

import os
from typing import Iterator

import httpx

from .base import BaseProvider, CompletionChunk, CompletionRequest

_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
_DEFAULT_MODEL = "anthropic/claude-3.5-sonnet"


class OpenRouterProvider(BaseProvider):
    """Sends requests to OpenRouter's OpenAI-compatible streaming endpoint."""

    name = "openrouter"
    supported_models = [
        "anthropic/claude-3.5-sonnet",
        "anthropic/claude-3-opus",
        "anthropic/claude-3-haiku",
        "anthropic/claude-3.5-haiku",  # added: cheaper/faster option
    ]

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key or os.getenv("OPENROUTER_API_KEY", "")

    # ------------------------------------------------------------------
    def is_available(self) -> bool:
        return bool(self._api_key)

    # ------------------------------------------------------------------
    def complete(self, request: CompletionRequest) -> Iterator[CompletionChunk]:
        if not self.is_available():
            raise RuntimeError("OPENROUTER_API_KEY is not set.")

        messages = []
        if request.system:
            messages.append({"role": "system", "content": request.system})
        messages += [{"role": m.role, "content": m.content} for m in request.messages]

        payload = {
            "model": request.model if request.model != "claude-3-5-sonnet-20241022" else _DEFAULT_MODEL,
            "messages": messages,
            "max_tokens": request.max_tokens,
            "stream": True,
        }

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            # Using my own repo URL as the referrer
            "HTTP-Referer": "https://github.com/my-fork/free-claude-code",
        }

        with httpx.stream("POST", _BASE_URL, json=payload, headers=headers, timeout=120) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                import json
                chunk = json.loads(data)
                delta = chunk["choices"][0].get("delta", {})
                finish = chunk["choices"][0].get("finish_reason")
                text = delta.get("content", "")
                if text or finish:
                    yield CompletionChunk(text=text, finish_reason=finish, model=chunk.get("model"))
