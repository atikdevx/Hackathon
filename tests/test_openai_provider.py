"""OpenAI provider request/response handling, exercised offline with a mocked HTTP transport."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from app.config import get_settings
from app.errors import LLMOutputInvalidError, LLMUnavailableError
from app.llm.base import ChatMessage
from app.llm.factory import build_provider
from app.llm.openai_compatible_provider import OpenAICompatibleProvider
from app.llm.prompt import INTERPRETATION_SCHEMA

ANSWER = {"interpretations": []}


def completion(content=json.dumps(ANSWER), finish_reason="stop", refusal=None):
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content, "refusal": refusal},
                "finish_reason": finish_reason,
            }
        ],
    }


def provider_with(handler, model="gpt-5.6-terra", api_key="test-key", base_url=None, effort="low"):
    return OpenAICompatibleProvider(
        api_key=api_key, model=model, base_url=base_url, effort=effort, transport=httpx.MockTransport(handler)
    )


def call(provider):
    return asyncio.run(
        provider.generate_json("system", [ChatMessage("user", "notes")], INTERPRETATION_SCHEMA, timeout_seconds=5)
    )


def test_openai_reasoning_model_request_shape():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=completion())

    assert json.loads(call(provider_with(handler))) == ANSWER
    body = seen["body"]
    assert seen["url"] == "https://api.openai.com/v1/chat/completions"
    assert seen["auth"] == "Bearer test-key"
    assert body["model"] == "gpt-5.6-terra"
    assert body["messages"][0] == {"role": "system", "content": "system"}
    assert body["messages"][1] == {"role": "user", "content": "notes"}
    assert body["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "directive_interpretation", "schema": INTERPRETATION_SCHEMA, "strict": True},
    }
    # Reasoning models reject temperature and take reasoning_effort instead.
    assert body["reasoning_effort"] == "low"
    assert "temperature" not in body


def test_non_reasoning_or_local_model_uses_temperature_and_no_key():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=completion())

    call(provider_with(handler, model="ai/qwen2.5:7B-Q4_K_M", api_key=None, base_url="http://localhost:12434/engines/v1/"))
    assert seen["url"] == "http://localhost:12434/engines/v1/chat/completions"
    assert seen["auth"] is None
    assert seen["body"]["temperature"] == 0
    assert "reasoning_effort" not in seen["body"]


def test_effort_can_be_disabled():
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=completion())

    call(provider_with(handler, effort=None))
    assert "reasoning_effort" not in seen["body"] and "temperature" not in seen["body"]


@pytest.mark.parametrize("status", [400, 401, 403, 404, 429, 500, 503])
def test_http_errors_become_controlled(status):
    handler = lambda request: httpx.Response(status, json={"error": {"message": "secret-ish detail"}})  # noqa: E731
    with pytest.raises(LLMUnavailableError) as err:
        call(provider_with(handler))
    assert "secret-ish" not in err.value.message


def test_exhausted_credit_is_reported_distinctly():
    body = {"error": {"type": "insufficient_quota", "code": "credit_balance_exhausted", "message": "No credits"}}
    handler = lambda request: httpx.Response(429, json=body)  # noqa: E731
    with pytest.raises(LLMUnavailableError) as err:
        call(provider_with(handler))
    assert "credit" in err.value.message


def test_timeout_and_connection_errors_are_controlled():
    def timeout(request):
        raise httpx.ReadTimeout("slow")

    def down(request):
        raise httpx.ConnectError("down")

    for handler in (timeout, down):
        with pytest.raises(LLMUnavailableError):
            call(provider_with(handler))


@pytest.mark.parametrize(
    "body",
    [
        completion(content=None, refusal="I can't help with that."),
        completion(content='{"interp', finish_reason="length"),
        completion(content="   "),
        {"choices": []},
        {"unexpected": True},
    ],
)
def test_unusable_responses_are_rejected(body):
    handler = lambda request: httpx.Response(200, json=body)  # noqa: E731
    with pytest.raises(LLMOutputInvalidError):
        call(provider_with(handler))


def test_non_json_body_is_rejected():
    handler = lambda request: httpx.Response(200, text="<html>gateway</html>")  # noqa: E731
    with pytest.raises(LLMOutputInvalidError):
        call(provider_with(handler))


def test_openai_provider_selection(monkeypatch):
    for var in ("LLM_MODEL", "LLM_BASE_URL", "LLM_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    get_settings.cache_clear()
    try:
        settings = get_settings()
        assert (settings.llm_provider, settings.llm_model, settings.llm_effort) == ("openai", "gpt-5.6-terra", "low")
        provider = build_provider(settings)
        assert isinstance(provider, OpenAICompatibleProvider) and provider.model == "gpt-5.6-terra"
    finally:
        get_settings.cache_clear()


def test_openai_without_key_is_not_configured(monkeypatch):
    for var in ("LLM_API_KEY", "OPENAI_API_KEY", "LLM_BASE_URL"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    get_settings.cache_clear()
    try:
        assert build_provider(get_settings()) is None
    finally:
        get_settings.cache_clear()
