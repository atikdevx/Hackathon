"""Google Gemini provider (native generateContent REST API with JSON-schema structured output).

Uses `generationConfig.responseMimeType = application/json` plus `responseJsonSchema`, so
Gemini constrains its answer to our interpretation schema. The key travels in the
`x-goog-api-key` header (never in the URL, so it cannot leak into access logs).

Resilience: the provider holds an ordered model list (primary + fallbacks). A model that
is overloaded (503), out of quota (429), erroring (5xx) or timing out is benched for a
cooldown, so the interpreter's next attempt, and later requests, use the next healthy
model instead of hammering the failing one and burning quota.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import Any

import httpx

from app.errors import LLMOutputInvalidError, LLMUnavailableError
from app.llm.base import ChatMessage

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
# Room for internal thinking plus a JSON answer for up to three notes.
_MAX_OUTPUT_TOKENS = 8192
# Gemini 3.x models take thinkingLevel; older 2.5 models use thinkingBudget instead.
_THINKING_LEVELS = {"minimal": "MINIMAL", "low": "LOW", "medium": "MEDIUM", "high": "HIGH"}
_BLOCKED_FINISH_REASONS = {"SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII", "LANGUAGE"}
# How long a model that failed transiently is skipped (seconds), unless Gemini says otherwise.
DEFAULT_COOLDOWN_SECONDS = 60.0
_MAX_COOLDOWN_SECONDS = 300.0
_TRANSIENT_HTTP = {408, 429, 500, 502, 503, 504}


def _uses_thinking_level(model: str) -> bool:
    return model.startswith("gemini-3")


def _error_info(resp: httpx.Response) -> tuple[str | None, float | None]:
    """(Google RPC status, RetryInfo delay in seconds). Only these fields are read; the body is never logged."""
    try:
        error = resp.json().get("error") or {}
    except (ValueError, AttributeError):
        return None, None
    if not isinstance(error, dict):
        return None, None
    retry: float | None = None
    for detail in error.get("details") or []:
        if isinstance(detail, dict) and str(detail.get("@type", "")).endswith("RetryInfo"):
            match = re.fullmatch(r"(\d+(?:\.\d+)?)s", str(detail.get("retryDelay", "")))
            if match:
                retry = float(match.group(1))
    return error.get("status"), retry


class GeminiProvider:
    name = "gemini"

    def __init__(
        self,
        api_key: str,
        model: str,
        effort: str | None = None,
        base_url: str | None = None,
        fallback_models: tuple[str, ...] = (),
        cooldown_seconds: float = DEFAULT_COOLDOWN_SECONDS,
        transport: httpx.AsyncBaseTransport | None = None,
        clock: Any = time.monotonic,
    ) -> None:
        self.model = model  # primary model; also part of the interpreter's cache key
        self._models = [model] + [m for m in dict.fromkeys(fallback_models) if m != model]
        self._base_url = (base_url or _DEFAULT_BASE_URL).rstrip("/")
        self._headers = {"Content-Type": "application/json", "x-goog-api-key": api_key}
        self._effort = effort
        self._cooldown = cooldown_seconds
        self._benched_until: dict[str, float] = {}
        self._clock = clock
        self._client = httpx.AsyncClient(transport=transport)

    # ------------------------------------------------------------------ model selection

    def _pick_model(self) -> str:
        """First model not cooling down; if all are benched, the one that recovers soonest."""
        now = self._clock()
        for m in self._models:
            if self._benched_until.get(m, 0.0) <= now:
                return m
        return min(self._models, key=lambda m: self._benched_until.get(m, 0.0))

    def _bench(self, model: str, reason: str, seconds: float | None = None) -> None:
        duration = min(seconds if seconds is not None else self._cooldown, _MAX_COOLDOWN_SECONDS)
        self._benched_until[model] = self._clock() + max(duration, 1.0)
        if len(self._models) > 1:
            logger.warning("gemini model=%s benched for %.0fs (%s); switching to fallback", model, duration, reason)

    # ------------------------------------------------------------------ request

    def _generation_config(self, model: str, schema: dict[str, Any]) -> dict[str, Any]:
        config: dict[str, Any] = {
            "responseMimeType": "application/json",
            "responseJsonSchema": schema,
            "maxOutputTokens": _MAX_OUTPUT_TOKENS,
        }
        if _uses_thinking_level(model):
            level = _THINKING_LEVELS.get(self._effort or "")
            if level:
                config["thinkingConfig"] = {"thinkingLevel": level}
        else:
            config["temperature"] = 0
        return config

    def _payload(self, model: str, system: str, messages: list[ChatMessage], schema: dict[str, Any]) -> dict[str, Any]:
        return {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [
                {"role": "model" if m.role == "assistant" else "user", "parts": [{"text": m.content}]} for m in messages
            ],
            "generationConfig": self._generation_config(model, schema),
        }

    async def generate_json(
        self,
        system: str,
        messages: list[ChatMessage],
        schema: dict[str, Any],
        timeout_seconds: float,
    ) -> str:
        model = self._pick_model()
        url = f"{self._base_url}/models/{model}:generateContent"
        payload = self._payload(model, system, messages, schema)
        try:
            resp = await self._client.post(url, json=payload, headers=self._headers, timeout=timeout_seconds)
        except httpx.TimeoutException as exc:
            self._bench(model, "timeout")
            raise LLMUnavailableError("The language model timed out.") from exc
        except httpx.HTTPError as exc:
            self._bench(model, "connection error")
            raise LLMUnavailableError("Could not reach the language model provider.") from exc
        except asyncio.CancelledError:
            # The interpreter's hard deadline fired first; treat it like a timeout.
            self._bench(model, "deadline")
            raise

        if resp.status_code >= 400:
            status, retry_after = _error_info(resp)
            # Status codes only: provider error bodies are never logged or returned.
            logger.warning("gemini error model=%s http=%s status=%s", model, resp.status_code, status)
            if resp.status_code in (401, 403) or status in ("UNAUTHENTICATED", "PERMISSION_DENIED"):
                raise LLMUnavailableError("The language model provider rejected the configured credentials.")
            if resp.status_code == 429 or status == "RESOURCE_EXHAUSTED":
                self._bench(model, "quota/rate limit", retry_after)
                raise LLMUnavailableError("The language model quota or rate limit is exhausted.")
            if resp.status_code in _TRANSIENT_HTTP:
                self._bench(model, f"http {resp.status_code}")
            elif resp.status_code == 404:
                # Unknown/retired model id: skip it for the maximum cooldown.
                self._bench(model, "model not found", _MAX_COOLDOWN_SECONDS)
            raise LLMUnavailableError(f"The language model provider returned HTTP {resp.status_code}.")

        self._benched_until.pop(model, None)
        try:
            body = resp.json()
        except ValueError as exc:
            raise LLMOutputInvalidError("The language model returned an unreadable response.") from exc
        if not isinstance(body, dict):
            raise LLMOutputInvalidError("The language model returned an unreadable response.")
        if (body.get("promptFeedback") or {}).get("blockReason"):
            raise LLMOutputInvalidError("The language model declined to interpret the operator notes.")
        try:
            candidate = body["candidates"][0]
            parts = (candidate.get("content") or {}).get("parts") or []
        except (KeyError, IndexError, TypeError, AttributeError) as exc:
            raise LLMOutputInvalidError("The language model returned an unreadable response.") from exc

        finish = candidate.get("finishReason")
        if finish in _BLOCKED_FINISH_REASONS:
            raise LLMOutputInvalidError("The language model declined to interpret the operator notes.")
        if finish == "MAX_TOKENS":
            raise LLMOutputInvalidError("The language model response was truncated.")
        # Skip any thought-summary parts; only the answer text is JSON.
        text = "".join(p.get("text", "") for p in parts if isinstance(p, dict) and not p.get("thought"))
        if not text.strip():
            raise LLMOutputInvalidError("The language model returned an empty response.")
        logger.info("gemini answered model=%s", model)
        return text
