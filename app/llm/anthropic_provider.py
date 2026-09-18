"""Claude (Anthropic API) provider using structured JSON outputs."""

from __future__ import annotations

import logging
from typing import Any

import anthropic

from app.errors import LLMOutputInvalidError, LLMUnavailableError
from app.llm.base import ChatMessage

logger = logging.getLogger(__name__)

# Room for adaptive thinking plus a JSON answer for up to three notes.
_MAX_TOKENS = 8000
_FALLBACK_BETA = "server-side-fallback-2026-07-01"


def _supports_effort(model: str) -> bool:
    # Haiku 4.5 rejects output_config.effort.
    return not model.startswith("claude-haiku")


def _supports_server_fallbacks(model: str) -> bool:
    return model.startswith(("claude-opus-5", "claude-fable-5-1"))


class AnthropicProvider:
    name = "anthropic"

    def __init__(
        self,
        api_key: str,
        model: str,
        effort: str | None,
        enable_fallbacks: bool,
        base_url: str | None = None,
        http_client: Any = None,
    ) -> None:
        self.model = model
        self._effort = effort if effort and _supports_effort(model) else None
        self._fallbacks = enable_fallbacks and _supports_server_fallbacks(model)
        # Retries are handled by the interpreter so the whole call stays inside the latency budget.
        self._client = anthropic.AsyncAnthropic(
            api_key=api_key, base_url=base_url, max_retries=0, http_client=http_client
        )

    async def generate_json(
        self,
        system: str,
        messages: list[ChatMessage],
        schema: dict[str, Any],
        timeout_seconds: float,
    ) -> str:
        output_config: dict[str, Any] = {"format": {"type": "json_schema", "schema": schema}}
        if self._effort:
            output_config["effort"] = self._effort
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": _MAX_TOKENS,
            "system": system,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "output_config": output_config,
            "timeout": timeout_seconds,
        }
        if self._fallbacks:
            kwargs["betas"] = [_FALLBACK_BETA]
            kwargs["fallbacks"] = "default"

        try:
            response = await self._client.beta.messages.create(**kwargs)
        except anthropic.APITimeoutError as exc:
            raise LLMUnavailableError("The language model timed out.") from exc
        except anthropic.RateLimitError as exc:
            raise LLMUnavailableError("The language model provider is rate limiting requests.") from exc
        except anthropic.AuthenticationError as exc:
            raise LLMUnavailableError("The language model provider rejected the configured credentials.") from exc
        except anthropic.APIStatusError as exc:
            logger.warning("anthropic api status error status=%s request_id=%s", exc.status_code, exc.request_id)
            raise LLMUnavailableError(f"The language model provider returned HTTP {exc.status_code}.") from exc
        except anthropic.APIConnectionError as exc:
            raise LLMUnavailableError("Could not reach the language model provider.") from exc

        if response.stop_reason == "refusal":
            raise LLMOutputInvalidError("The language model declined to interpret the operator notes.")
        if response.stop_reason == "max_tokens":
            raise LLMOutputInvalidError("The language model response was truncated.")
        text = "".join(block.text for block in response.content if block.type == "text")
        if not text.strip():
            raise LLMOutputInvalidError("The language model returned an empty response.")
        return text
