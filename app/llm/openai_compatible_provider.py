"""Alternative provider for any OpenAI-compatible Chat Completions endpoint.

Lets the service run against other hosted or local models (e.g. an Ollama or vLLM
server, Groq, OpenRouter) by setting LLM_PROVIDER=openai_compatible, LLM_BASE_URL,
LLM_MODEL and LLM_API_KEY. The same guardrails apply to its output.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from app.errors import LLMOutputInvalidError, LLMUnavailableError
from app.llm.base import ChatMessage

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "https://api.openai.com/v1"


class OpenAICompatibleProvider:
    name = "openai_compatible"

    def __init__(self, api_key: str | None, model: str, base_url: str | None = None) -> None:
        self.model = model
        self._url = (base_url or _DEFAULT_BASE_URL).rstrip("/") + "/chat/completions"
        self._headers = {"Content-Type": "application/json"}
        if api_key:
            self._headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.AsyncClient()

    async def generate_json(
        self,
        system: str,
        messages: list[ChatMessage],
        schema: dict[str, Any],
        timeout_seconds: float,
    ) -> str:
        payload = {
            "model": self.model,
            "temperature": 0,
            "messages": [{"role": "system", "content": system}]
            + [{"role": m.role, "content": m.content} for m in messages],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "directive_interpretation", "schema": schema, "strict": True},
            },
        }
        try:
            resp = await self._client.post(self._url, json=payload, headers=self._headers, timeout=timeout_seconds)
        except httpx.TimeoutException as exc:
            raise LLMUnavailableError("The language model timed out.") from exc
        except httpx.HTTPError as exc:
            raise LLMUnavailableError("Could not reach the language model provider.") from exc

        if resp.status_code == 429:
            raise LLMUnavailableError("The language model provider is rate limiting requests.")
        if resp.status_code >= 400:
            logger.warning("openai-compatible provider error status=%s", resp.status_code)
            raise LLMUnavailableError(f"The language model provider returned HTTP {resp.status_code}.")
        try:
            choice = resp.json()["choices"][0]
            text = choice["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise LLMOutputInvalidError("The language model returned an unreadable response.") from exc
        if choice.get("finish_reason") == "length":
            raise LLMOutputInvalidError("The language model response was truncated.")
        if not isinstance(text, str) or not text.strip():
            raise LLMOutputInvalidError("The language model returned an empty response.")
        return text
