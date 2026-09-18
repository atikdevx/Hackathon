"""Gemini provider request/response handling, exercised offline with a mocked HTTP transport."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from app.config import get_settings
from app.errors import LLMOutputInvalidError, LLMUnavailableError
from app.llm.base import ChatMessage
from app.llm.factory import build_provider
from app.llm.gemini_provider import GeminiProvider
from app.llm.prompt import INTERPRETATION_SCHEMA

ANSWER = {"interpretations": []}
KEY = "AIza-test-key"


def response(text=json.dumps(ANSWER), finish="STOP", parts=None, **extra):
    body = {
        "candidates": [
            {"content": {"role": "model", "parts": parts or [{"text": text}]}, "finishReason": finish}
        ],
        "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5},
    }
    body.update(extra)
    return body


def provider_with(handler, model="gemini-3.8-flash", effort="low"):
    return GeminiProvider(api_key=KEY, model=model, effort=effort, transport=httpx.MockTransport(handler))


def call(provider, messages=None):
    messages = messages or [ChatMessage("user", "notes")]
    return asyncio.run(provider.generate_json("system", messages, INTERPRETATION_SCHEMA, timeout_seconds=5))


def test_request_shape_and_text_extraction():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["headers"] = request.headers
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=response())

    messages = [ChatMessage("user", "notes"), ChatMessage("assistant", "{}"), ChatMessage("user", "fix it")]
    assert json.loads(call(provider_with(handler), messages)) == ANSWER
    body = seen["body"]
    assert seen["url"] == "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.8-flash:generateContent"
    # Key in a header, never in the URL.
    assert seen["headers"]["x-goog-api-key"] == KEY and KEY not in seen["url"]
    assert body["systemInstruction"] == {"parts": [{"text": "system"}]}
    assert [c["role"] for c in body["contents"]] == ["user", "model", "user"]
    assert body["contents"][2]["parts"] == [{"text": "fix it"}]
    config = body["generationConfig"]
    assert config["responseMimeType"] == "application/json"
    assert config["responseJsonSchema"] == INTERPRETATION_SCHEMA
    assert config["thinkingConfig"] == {"thinkingLevel": "LOW"}
    assert "temperature" not in config


def test_gemini_2_5_uses_temperature_not_thinking_level():
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=response())

    call(provider_with(handler, model="gemini-2.5-flash"))
    config = seen["body"]["generationConfig"]
    assert config["temperature"] == 0 and "thinkingConfig" not in config


def test_effort_none_sends_no_thinking_config():
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=response())

    call(provider_with(handler, effort=None))
    assert "thinkingConfig" not in seen["body"]["generationConfig"]


def test_thought_parts_are_skipped():
    parts = [{"text": "reasoning summary", "thought": True}, {"text": json.dumps(ANSWER)}]
    handler = lambda request: httpx.Response(200, json=response(parts=parts))  # noqa: E731
    assert json.loads(call(provider_with(handler))) == ANSWER


@pytest.mark.parametrize(
    "status,rpc_status,fragment",
    [
        (429, "RESOURCE_EXHAUSTED", "quota"),
        (400, "INVALID_ARGUMENT", "HTTP 400"),
        (403, "PERMISSION_DENIED", "credentials"),
        (401, "UNAUTHENTICATED", "credentials"),
        (500, "INTERNAL", "HTTP 500"),
        (503, "UNAVAILABLE", "HTTP 503"),
    ],
)
def test_http_errors_become_controlled(status, rpc_status, fragment):
    body = {"error": {"code": status, "status": rpc_status, "message": f"detail mentioning {KEY}"}}
    handler = lambda request: httpx.Response(status, json=body)  # noqa: E731
    with pytest.raises(LLMUnavailableError) as err:
        call(provider_with(handler))
    assert fragment in err.value.message and KEY not in err.value.message


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
        response(finish="SAFETY"),
        response(finish="MAX_TOKENS", text='{"interp'),
        response(text="   "),
        {"promptFeedback": {"blockReason": "SAFETY"}},
        {"candidates": []},
        {"unexpected": True},
        [],
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


@pytest.mark.parametrize("key_var", ["LLM_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"])
def test_gemini_is_the_default_provider(monkeypatch, key_var):
    for var in ("LLM_PROVIDER", "LLM_MODEL", "LLM_BASE_URL", "LLM_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv(key_var, KEY)
    get_settings.cache_clear()
    try:
        settings = get_settings()
        assert (settings.llm_provider, settings.llm_model) == ("gemini", "gemini-3.5-flash-lite")
        assert settings.llm_fallback_models == ("gemini-3.1-flash-lite", "gemini-3.5-flash")
        assert settings.llm_api_key == KEY
        provider = build_provider(settings)
        assert isinstance(provider, GeminiProvider)
    finally:
        get_settings.cache_clear()


def test_gemini_without_key_is_not_configured(monkeypatch):
    for var in ("LLM_PROVIDER", "LLM_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "LLM_BASE_URL"):
        monkeypatch.delenv(var, raising=False)
    get_settings.cache_clear()
    try:
        assert build_provider(get_settings()) is None
    finally:
        get_settings.cache_clear()


# --------------------------------------------------------------------------- model fallback


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def model_of(request) -> str:
    return str(request.url).rsplit("/models/", 1)[1].split(":")[0]


def fallback_provider(handler, clock):
    return GeminiProvider(
        api_key=KEY,
        model="primary",
        fallback_models=("backup-1", "backup-2"),
        transport=httpx.MockTransport(handler),
        clock=clock,
    )


@pytest.mark.parametrize("status", [429, 500, 503, 504])
def test_transient_failure_benches_model_and_next_call_uses_fallback(status):
    clock, calls = FakeClock(), []

    def handler(request):
        calls.append(model_of(request))
        if model_of(request) == "primary":
            return httpx.Response(status, json={"error": {"status": "UNAVAILABLE"}})
        return httpx.Response(200, json=response())

    provider = fallback_provider(handler, clock)
    with pytest.raises(LLMUnavailableError):
        call(provider)
    assert json.loads(call(provider)) == ANSWER
    assert json.loads(call(provider)) == ANSWER  # primary still benched
    assert calls == ["primary", "backup-1", "backup-1"]
    clock.now += 61  # cooldown over: primary is tried again (and still fails here)
    with pytest.raises(LLMUnavailableError):
        call(provider)
    assert calls[-1] == "primary"


def test_timeout_benches_model():
    clock, calls = FakeClock(), []

    def handler(request):
        calls.append(model_of(request))
        if model_of(request) == "primary":
            raise httpx.ReadTimeout("slow")
        return httpx.Response(200, json=response())

    provider = fallback_provider(handler, clock)
    with pytest.raises(LLMUnavailableError):
        call(provider)
    call(provider)
    assert calls == ["primary", "backup-1"]


def test_retry_info_sets_cooldown_length():
    clock, calls = FakeClock(), []
    quota = {"error": {"status": "RESOURCE_EXHAUSTED", "details": [
        {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "5s"}]}}

    def handler(request):
        calls.append(model_of(request))
        if model_of(request) == "primary" and len(calls) == 1:
            return httpx.Response(429, json=quota)
        return httpx.Response(200, json=response())

    provider = fallback_provider(handler, clock)
    with pytest.raises(LLMUnavailableError):
        call(provider)
    clock.now += 6  # Gemini's retryDelay (5 s) has passed
    call(provider)
    assert calls == ["primary", "primary"]


def test_when_all_models_benched_the_soonest_recovering_is_used():
    clock, calls = FakeClock(), []

    def handler(request):
        calls.append(model_of(request))
        return httpx.Response(503, json={"error": {"status": "UNAVAILABLE"}})

    provider = fallback_provider(handler, clock)
    for _ in range(4):
        with pytest.raises(LLMUnavailableError):
            call(provider)
        clock.now += 1
    assert calls == ["primary", "backup-1", "backup-2", "primary"]


def test_bad_credentials_do_not_bench():
    clock, calls = FakeClock(), []

    def handler(request):
        calls.append(model_of(request))
        return httpx.Response(403, json={"error": {"status": "PERMISSION_DENIED"}})

    provider = fallback_provider(handler, clock)
    for _ in range(2):
        with pytest.raises(LLMUnavailableError):
            call(provider)
    assert calls == ["primary", "primary"]


def test_interpreter_recovers_within_one_request_via_fallback():
    """Primary overloaded -> the interpreter's second attempt lands on the fallback model."""
    from app.llm.interpreter import NoteInterpreter
    from tests.conftest import interpretation_payload, raw_entry

    answer = interpretation_payload([raw_entry(0, "no_charge_window", [(2, 5)])])
    calls = []

    def handler(request):
        calls.append(model_of(request))
        if model_of(request) == "primary":
            return httpx.Response(503, json={"error": {"status": "UNAVAILABLE"}})
        return httpx.Response(200, json=response(text=json.dumps(answer)))

    interpreter = NoteInterpreter(
        provider=fallback_provider(handler, FakeClock()), timeout_seconds=5, max_attempts=3, cache_size=0
    )
    _, directives = asyncio.run(interpreter.interpret(["Charger isolated 2-5 AM."], 200.0, 20.0))
    assert calls == ["primary", "backup-1"]
    assert directives[0].hours == (2, 3, 4)


def test_fallback_models_env_override(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("LLM_FALLBACK_MODELS", " gemini-3.5-flash , ,gemini-3.1-flash-lite")
    get_settings.cache_clear()
    try:
        assert get_settings().llm_fallback_models == ("gemini-3.5-flash", "gemini-3.1-flash-lite")
        monkeypatch.setenv("LLM_FALLBACK_MODELS", "")
        get_settings.cache_clear()
        assert get_settings().llm_fallback_models == ()
    finally:
        get_settings.cache_clear()
