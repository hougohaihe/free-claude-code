"""Base provider interface for free-claude-code."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterator, Optional


@dataclass
class Message:
    """Represents a chat message."""
    role: str  # 'user' or 'assistant'
    content: str


@dataclass
class CompletionRequest:
    """Encapsulates a completion request."""
    messages: list[Message]
    model: str = "claude-3-5-sonnet-20241022"
    max_tokens: int = 8096
    system: Optional[str] = None
    stream: bool = True
    extra: dict = field(default_factory=dict)


@dataclass
class CompletionChunk:
    """A single streamed chunk from a provider."""
    text: str
    finish_reason: Optional[str] = None
    model: Optional[str] = None


class BaseProvider(ABC):
    """Abstract base class all providers must implement."""

    name: str = "base"
    supported_models: list[str] = []

    @abstractmethod
    def complete(self, request: CompletionRequest) -> Iterator[CompletionChunk]:
        """Yield completion chunks for the given request."""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if the provider is properly configured and reachable."""
        ...

    def validate_model(self, model: str) -> bool:
        """Check whether the requested model is supported."""
        if not self.supported_models:
            return True  # provider accepts any model name
        return model in self.supported_models

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r}>"
