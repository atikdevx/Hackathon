"""Interpreter retry/caching and controlled failure handling across the whole API."""

from __future__ import annotations

import asyncio

import pytest

from app.errors import LLMOutputInvalidError, LLMUnavailableError
from app.main import app
from tests.conftest import (
    ScriptedProvider,
    flat_scenario,
    interpretation_payload,
    make_interpreter,
    noop_entries,
    raw_entry,
)

NO_CHARGE = interpretation_payload([raw_entry(0, "no_charge_window", [(2, 5)], explanation="Charger isolated.")])
EMPTY_WINDOW = interpretation_payload([raw_entry(0, "no_charge_window", [(4, 4)])])

SECRET = "sk-ant-THIS-MUST-NEVER-LEAK"


def interpret(provider: ScriptedProvider, notes=None, **kwargs):
    interpreter = make_interpreter(provider, **kwargs)
    return asyncio.run(interpreter.interpret(notes or ["note"], 200.0, 20.0))


def test_repair_retry_after_guardrail_rejection():
    provider = ScriptedProvider(EMPTY_WINDOW, NO_CHARGE)
    interpretations, directives = interpret(provider)
    assert len(provider.calls) == 2
    # The repair turn shows the model its rejected output and the validator's reasons.
    repair = provider.calls[1][-1].content
    assert "empty" in repair
    assert directives[0].hours == (2, 3, 4)


def test_malformed_json_twice_is_controlled_failure():
    provider = ScriptedProvider("{not json", "still not json")
    with pytest.raises(LLMOutputInvalidError):
        interpret(provider)


def test_unsupported_directive_never_applied():
    bad = interpretation_payload([raw_entry(0, "shed_load", [(1, 2)])])
    with pytest.raises(LLMOutputInvalidError) as err:
        interpret(ScriptedProvider(bad))
    assert any("shed_load" in d for d in err.value.details)


def test_provider_error_then_success():
    provider = ScriptedProvider(LLMUnavailableError("rate limited"), NO_CHARGE)
    _, directives = interpret(provider)
    assert len(directives) == 1


def test_provider_down_is_controlled():
    with pytest.raises(LLMUnavailableError):
        interpret(ScriptedProvider(LLMUnavailableError("down")))


def test_timeout_is_controlled_and_bounded():
    provider = ScriptedProvider(NO_CHARGE, delay=3.0)
    interpreter = make_interpreter(provider, timeout_seconds=0.2, max_attempts=2)
    interpreter._budget = 1.0
    with pytest.raises(LLMUnavailableError):
        asyncio.run(interpreter.interpret(["note"], 200.0, 20.0))


def test_cache_avoids_repeat_llm_calls():
    provider = ScriptedProvider(NO_CHARGE)
    interpreter = make_interpreter(provider, cache_size=8)
    first = asyncio.run(interpreter.interpret(["note"], 200.0, 20.0))
    second = asyncio.run(interpreter.interpret(["note"], 200.0, 20.0))
    assert len(provider.calls) == 1
    assert first[0] == second[0]
    # Different battery sizing changes percentage conversions, so it is a different key.
    asyncio.run(interpreter.interpret(["note"], 300.0, 20.0))
    assert len(provider.calls) == 2


def test_repeated_api_requests_are_stable(client, use_provider):
    use_provider(interpretation_payload(noop_entries(1)))
    body = flat_scenario(tariff=[5.0] * 12 + [15.0] * 12)
    results = [client.post("/optimize-energy", json=body) for _ in range(15)]
    assert all(r.status_code == 200 for r in results)
    assert len({r.json()["total_cost_bdt"] for r in results}) == 1


def test_llm_failure_over_api_is_json_500_without_secrets(client, use_provider):
    use_provider(LLMUnavailableError("The language model provider is rate limiting requests."))
    resp = client.post("/optimize-energy", json=flat_scenario())
    assert resp.status_code == 500
    assert resp.json()["error"]["code"] == "llm_unavailable"
    assert "Traceback" not in resp.text


def test_malformed_llm_output_over_api(client, use_provider):
    use_provider("definitely not json")
    resp = client.post("/optimize-energy", json=flat_scenario())
    assert resp.status_code == 500
    assert resp.json()["error"]["code"] == "llm_output_rejected"


def test_unexpected_provider_exception_does_not_leak(client, use_provider, caplog):
    use_provider(RuntimeError(f"boom with key {SECRET}"))
    resp = client.post("/optimize-energy", json=flat_scenario())
    assert resp.status_code == 500
    assert SECRET not in resp.text
    assert SECRET not in caplog.text
    assert resp.json()["error"]["code"] == "llm_unavailable"
    # The service is still healthy afterwards.
    assert client.get("/health").status_code == 200


def test_optimizer_failure_is_controlled(client, use_provider, monkeypatch):
    from app.errors import OptimizerError
    from app.services import pipeline

    def broken(_limits):
        raise OptimizerError("The optimizer did not find an optimal schedule.")

    monkeypatch.setattr(pipeline, "optimize", broken)
    use_provider(interpretation_payload(noop_entries(1)))
    resp = client.post("/optimize-energy", json=flat_scenario())
    assert resp.status_code == 500
    assert resp.json()["error"]["code"] == "optimizer_failure"


def test_replay_failure_blocks_response(client, use_provider, monkeypatch):
    from app.services import pipeline

    monkeypatch.setattr(pipeline, "replay_schedule", lambda *a, **k: ["hour 3: energy balance violated"])
    use_provider(interpretation_payload(noop_entries(1)))
    resp = client.post("/optimize-energy", json=flat_scenario())
    assert resp.status_code == 500
    assert resp.json()["error"]["code"] == "schedule_validation_failed"


def test_infeasible_directives_return_422(client, use_provider):
    use_provider(
        interpretation_payload(
            [raw_entry(0, "max_grid_window", [(5, 6)], max_grid_kwh=0), raw_entry(1, "no_discharge_window", [(5, 6)])]
        )
    )
    resp = client.post("/optimize-energy", json=flat_scenario(notes=["a", "b"]))
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "infeasible_scenario"


def test_unhandled_error_is_generic_500_and_not_logged(client, use_provider, monkeypatch, caplog):
    from app.services import pipeline

    def explode(*_a, **_k):
        raise ValueError(f"internal detail {SECRET}")

    monkeypatch.setattr(pipeline, "build_hourly_limits", explode)
    use_provider(interpretation_payload(noop_entries(1)))
    resp = client.post("/optimize-energy", json=flat_scenario())
    assert resp.status_code == 500
    assert resp.json() == {"error": {"code": "internal_error", "message": "An internal error occurred."}}
    assert SECRET not in resp.text and SECRET not in caplog.text
    assert client.get("/health").status_code == 200


def test_api_key_not_in_settings_repr(monkeypatch):
    from app.config import get_settings

    monkeypatch.setenv("LLM_API_KEY", SECRET)
    get_settings.cache_clear()
    try:
        settings = get_settings()
        assert settings.llm_api_key == SECRET
        assert SECRET not in repr(settings)
    finally:
        get_settings.cache_clear()
    app.state.interpreter = None
