"""Provider-agnostic interface for a JSON-producing chat model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol


@dataclass(frozen=True)
class ChatMessage:
    role: Literal["user", "assistant"]
    content: str


class LLMProvider(Protocol):
    """Anything that can answer a system prompt + conversation with schema-constrained JSON text.

    Implementations must raise ``app.errors.LLMUnavailableError`` for transport/API failures
    (timeouts, rate limits, auth, 5xx) and return the raw model text otherwise; the caller
    parses and validates it.
    """

    name: str
    model: str

    async def generate_json(
        self,
        system: str,
        messages: list[ChatMessage],
        schema: dict[str, Any],
        timeout_seconds: float,
    ) -> str: ...
