"""Unit tests for the provider base class and OpenRouter provider."""

from unittest.mock import MagicMock, patch

import pytest

from providers.base import BaseProvider, CompletionChunk, CompletionRequest, Message
from providers.openrouter import OpenRouterProvider


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_request(**kwargs) -> CompletionRequest:
    defaults = dict(
        messages=[Message(role="user", content="Hello")],
        model="claude-3-5-sonnet-20241022",
    )
    defaults.update(kwargs)
    return CompletionRequest(**defaults)


# ---------------------------------------------------------------------------
# BaseProvider contract tests
# ---------------------------------------------------------------------------

class ConcreteProvider(BaseProvider):
    name = "concrete"
    supported_models = ["model-a", "model-b"]

    def complete(self, request):
        yield CompletionChunk(text="ok")

    def is_available(self):
        return True


def test_validate_model_supported():
    p = ConcreteProvider()
    assert p.validate_model("model-a") is True


def test_validate_model_unsupported():
    p = ConcreteProvider()
    assert p.validate_model("model-z") is False


def test_validate_model_empty_list_accepts_all():
    p = ConcreteProvider()
    p.supported_models = []
    assert p.validate_model("anything") is True


def test_repr():
    assert "ConcreteProvider" in repr(ConcreteProvider())


# ---------------------------------------------------------------------------
# OpenRouterProvider tests
# ---------------------------------------------------------------------------

def test_openrouter_unavailable_without_key():
    p = OpenRouterProvider(api_key="")
    assert p.is_available() is False


def test_openrouter_available_with_key():
    p = OpenRouterProvider(api_key="sk-test-123")
    assert p.is_available() is True


def test_openrouter_raises_without_key():
    p = OpenRouterProvider(api_key="")
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        list(p.complete(_make_request()))


def test_openrouter_streams_chunks():
    lines = [
        'data: {"choices":[{"delta":{"content":"Hi"},"finish_reason":null}],"model":"claude"}',
        'data: {"choices":[{"delta":{},"finish_reason":"stop"}],"model":"claude"}',
        "data: [DONE]",
    ]

    mock_resp = MagicMock()
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    mock_resp.iter_lines.return_value = iter(lines)
    mock_resp.raise_for_status = MagicMock()

    with patch("providers.openrouter.httpx.stream", return_value=mock_resp):
        p = OpenRouterProvider(api_key="sk-test")
        chunks = list(p.complete(_make_request()))

    assert chunks[0].text == "Hi"
    assert chunks[1].finish_reason == "stop"
