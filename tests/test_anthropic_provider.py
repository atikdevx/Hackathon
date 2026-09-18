"""AnthropicProvider request/response handling, exercised offline with a mocked HTTP transport."""

from __future__ import annotations

import asyncio
import json

import anthropic
import httpx2
import pytest

from app.errors import LLMOutputInvalidError, LLMUnavailableError
from app.llm.anthropic_provider import AnthropicProvider
from app.llm.base import ChatMessage
from app.llm.prompt import INTERPRETATION_SCHEMA

ANSWER = {"interpretations": []}


def message(stop_reason="end_turn", text=json.dumps(ANSWER)):
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-5",
        "content": [{"type": "text", "text": text}] if text is not None else [],
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }


def provider_with(handler, model="claude-opus-5", effort="low"):
    client = anthropic.DefaultAsyncHttpxClient(transport=httpx2.MockTransport(handler))
    return AnthropicProvider(
        api_key="test-key", model=model, effort=effort, enable_fallbacks=True, http_client=client
    )


def call(provider):
    return asyncio.run(
        provider.generate_json("system", [ChatMessage("user", "notes")], INTERPRETATION_SCHEMA, timeout_seconds=5)
    )


def test_request_shape_and_text_extraction():
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        seen["headers"] = request.headers
        seen["url"] = str(request.url)
        return httpx2.Response(200, json=message())

    assert json.loads(call(provider_with(handler))) == ANSWER
    body = seen["body"]
    assert seen["url"].endswith("/v1/messages?beta=true")
    assert body["model"] == "claude-opus-5"
    assert body["system"] == "system"
    assert body["messages"] == [{"role": "user", "content": "notes"}]
    assert body["output_config"]["format"] == {"type": "json_schema", "schema": INTERPRETATION_SCHEMA}
    assert body["output_config"]["effort"] == "low"
    assert body["fallbacks"] == "default"
    assert "server-side-fallback-2026-07-01" in seen["headers"]["anthropic-beta"]
    assert "temperature" not in body


def test_haiku_omits_effort_and_fallbacks():
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        seen["beta"] = request.headers.get("anthropic-beta")
        return httpx2.Response(200, json=message())

    call(provider_with(handler, model="claude-haiku-4-5"))
    assert "effort" not in seen["body"]["output_config"]
    assert "fallbacks" not in seen["body"]
    assert not seen["beta"]


@pytest.mark.parametrize(
    "status,error",
    [(429, LLMUnavailableError), (401, LLMUnavailableError), (500, LLMUnavailableError), (529, LLMUnavailableError)],
)
def test_http_errors_become_controlled(status, error):
    handler = lambda request: httpx2.Response(  # noqa: E731
        status, json={"type": "error", "error": {"type": "api_error", "message": "x"}}
    )
    with pytest.raises(error):
        call(provider_with(handler))


def test_connection_error_is_controlled():
    def handler(request):
        raise httpx2.ConnectError("down")

    with pytest.raises(LLMUnavailableError):
        call(provider_with(handler))


@pytest.mark.parametrize("stop_reason,text", [("refusal", None), ("max_tokens", '{"interp'), ("end_turn", "  ")])
def test_unusable_responses_are_rejected(stop_reason, text):
    handler = lambda request: httpx2.Response(200, json=message(stop_reason, text))  # noqa: E731
    with pytest.raises(LLMOutputInvalidError):
        call(provider_with(handler))
