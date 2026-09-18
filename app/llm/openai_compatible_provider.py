"""OpenAI provider (Chat Completions + Structured Outputs), also usable with any OpenAI-compatible server.

Default target is the OpenAI API (LLM_PROVIDER=openai). Pointing LLM_BASE_URL at another
OpenAI-compatible endpoint (Ollama, vLLM, Docker Model Runner, Groq, OpenRouter) uses
the same code path. The same guardrails apply to its output.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from app.errors import LLMOutputInvalidError, LLMUnavailableError
from app.llm.base import ChatMessage

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "https://api.openai.com/v1"
# OpenAI reasoning models take `reasoning_effort` and reject `temperature`.
_REASONING_MODEL_PREFIXES = ("gpt-5", "gpt-6", "o1", "o3", "o4")


def is_reasoning_model(model: str) -> bool:
    return model.startswith(_REASONING_MODEL_PREFIXES)


class OpenAICompatibleProvider:
    name = "openai"

    def __init__(
        self,
        api_key: str | None,
        model: str,
        base_url: str | None = None,
        effort: str | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.model = model
        self._url = (base_url or _DEFAULT_BASE_URL).rstrip("/") + "/chat/completions"
        self._headers = {"Content-Type": "application/json"}
        if api_key:
            self._headers["Authorization"] = f"Bearer {api_key}"
        self._reasoning = is_reasoning_model(model)
        self._effort = effort
        self._client = httpx.AsyncClient(transport=transport)

    def _payload(self, system: str, messages: list[ChatMessage], schema: dict[str, Any]) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}]
            + [{"role": m.role, "content": m.content} for m in messages],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "directive_interpretation", "schema": schema, "strict": True},
            },
        }
        if self._reasoning:
            if self._effort:
                payload["reasoning_effort"] = self._effort
        else:
            payload["temperature"] = 0
        return payload

    async def generate_json(
        self,
        system: str,
        messages: list[ChatMessage],
        schema: dict[str, Any],
        timeout_seconds: float,
    ) -> str:
        payload = self._payload(system, messages, schema)
        try:
            resp = await self._client.post(self._url, json=payload, headers=self._headers, timeout=timeout_seconds)
        except httpx.TimeoutException as exc:
            raise LLMUnavailableError("The language model timed out.") from exc
        except httpx.HTTPError as exc:
            raise LLMUnavailableError("Could not reach the language model provider.") from exc

        if resp.status_code == 429:
            raise LLMUnavailableError("The language model provider is rate limiting requests.")
        if resp.status_code in (401, 403):
            raise LLMUnavailableError("The language model provider rejected the configured credentials.")
        if resp.status_code >= 400:
            # Status only: provider error bodies are not logged or returned.
            logger.warning("openai provider error status=%s request_id=%s", resp.status_code, resp.headers.get("x-request-id"))
            raise LLMUnavailableError(f"The language model provider returned HTTP {resp.status_code}.")
        try:
            choice = resp.json()["choices"][0]
            message = choice["message"]
            text = message.get("content")
        except (ValueError, KeyError, IndexError, TypeError, AttributeError) as exc:
            raise LLMOutputInvalidError("The language model returned an unreadable response.") from exc
        if message.get("refusal"):
            raise LLMOutputInvalidError("The language model declined to interpret the operator notes.")
        if choice.get("finish_reason") == "length":
            raise LLMOutputInvalidError("The language model response was truncated.")
        if not isinstance(text, str) or not text.strip():
            raise LLMOutputInvalidError("The language model returned an empty response.")
        return text
