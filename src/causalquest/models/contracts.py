"""Model-generation contracts that keep the agent harness provider independent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class GenerationRequest:
    """One chat-generation request with optional OpenAI-style tool schemas."""

    messages: tuple[dict[str, Any], ...]
    tools: tuple[dict[str, Any], ...] = ()
    temperature: float = 0.0
    max_tokens: int = 2048
    seed: int = 17


@dataclass(frozen=True)
class GenerationResponse:
    """Normalized response returned by local or remote model adapters."""

    content: str | None = None
    tool_calls: tuple[dict[str, Any], ...] = ()
    model: str | None = None
    finish_reason: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)


class ModelAdapter(Protocol):
    """Small boundary shared by static tests, vLLM, and compatible APIs."""

    def generate(self, request: GenerationRequest) -> GenerationResponse: ...

