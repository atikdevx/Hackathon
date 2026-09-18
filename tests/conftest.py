"""Shared fixtures.

Offline tests use ``ScriptedProvider``, a stand-in for the LLM that returns canned JSON,
so the pipeline, guardrails and failure handling can be tested deterministically. Tests
that need the real model live in test_llm_live.py and are skipped without an API key.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.llm.base import ChatMessage
from app.llm.interpreter import NoteInterpreter
from app.main import app

ROOT = Path(__file__).resolve().parent.parent
SAMPLES_PATH = ROOT / "BUP_CSE_FEST_2026_Preli_Public_Sample_Cases.json"


def load_cases() -> list[dict[str, Any]]:
    return json.loads(SAMPLES_PATH.read_text())["cases"]


Responder = Callable[[str, list[ChatMessage]], Any]


class ScriptedProvider:
    """Returns successive scripted outputs; each item is a str, a dict (JSON-encoded), an
    Exception instance (raised), or a callable(system, messages) producing one of those."""

    name = "scripted"
    model = "scripted-test-model"

    def __init__(self, *outputs: Any, delay: float = 0.0) -> None:
        self._outputs = list(outputs)
        self._delay = delay
        self.calls: list[list[ChatMessage]] = []

    async def generate_json(self, system: str, messages: list[ChatMessage], schema: dict, timeout_seconds: float) -> str:
        self.calls.append(list(messages))
        if self._delay:
            await asyncio.sleep(self._delay)
        item = self._outputs[min(len(self.calls), len(self._outputs)) - 1]
        if callable(item):
            item = item(system, messages)
        if isinstance(item, Exception):
            raise item
        return item if isinstance(item, str) else json.dumps(item)


def make_interpreter(provider: ScriptedProvider, **kwargs: Any) -> NoteInterpreter:
    options = {"timeout_seconds": 5.0, "max_attempts": 2, "cache_size": 0}
    options.update(kwargs)
    return NoteInterpreter(provider=provider, **options)


def interpretation_payload(entries: list[dict[str, Any]]) -> dict[str, Any]:
    return {"interpretations": entries}


def raw_entry(
    i: int,
    dtype: str,
    windows: list[tuple[int, int]] = (),
    factor: float | None = None,
    minimum_energy_kwh: float | None = None,
    max_grid_kwh: float | None = None,
    explanation: str = "Scripted interpretation.",
) -> dict[str, Any]:
    """One entry in the model's raw output format (see app/llm/prompt.py)."""
    return {
        "explanation": explanation,
        "note_index": i,
        "directive_type": dtype,
        "time_windows": [{"start_hour": a, "end_hour": b} for a, b in windows],
        "factor": factor,
        "minimum_energy_kwh": minimum_energy_kwh,
        "max_grid_kwh": max_grid_kwh,
    }


def hours_to_windows(hours: list[int]) -> list[tuple[int, int]]:
    windows: list[tuple[int, int]] = []
    for h in hours:
        if windows and windows[-1][1] == h:
            windows[-1] = (windows[-1][0], h + 1)
        else:
            windows.append((h, h + 1))
    return windows


def raw_from_official(entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Express reference (official-format) interpretations as the model's raw output."""
    out = []
    for e in entries:
        adj = e["structured_adjustment"] or {}
        values = {k: v for k, v in adj.items() if k != "hours"}
        out.append(
            raw_entry(e["note_index"], e["directive_type"], hours_to_windows(adj.get("hours", [])),
                      explanation=e["explanation"], **values)
        )
    return interpretation_payload(out)


def noop_entries(count: int) -> list[dict[str, Any]]:
    return [raw_entry(i, "no_op", explanation="Unrelated to the energy schedule.") for i in range(count)]


def flat_scenario(
    notes: list[str] | None = None,
    demand: float = 100.0,
    solar: list[float] | None = None,
    tariff: list[float] | None = None,
    battery: dict[str, float] | None = None,
) -> dict[str, Any]:
    solar = solar or [0.0] * 24
    tariff = tariff or [10.0] * 24
    return {
        "scenario_id": "TEST-1",
        "operator_notes": notes or ["The cafeteria menu changes tomorrow."],
        "hours": [
            {"hour": h, "demand_kwh": demand, "solar_kwh": solar[h], "tariff_bdt_per_kwh": tariff[h]} for h in range(24)
        ],
        "battery": battery
        or {
            "capacity_kwh": 200.0,
            "initial_energy_kwh": 100.0,
            "minimum_energy_kwh": 20.0,
            "max_charge_kwh_per_hour": 50.0,
            "max_discharge_kwh_per_hour": 50.0,
        },
    }


@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


@pytest.fixture
def use_provider(client: TestClient) -> Callable[..., ScriptedProvider]:
    """Install a scripted provider into the running app for this test."""

    def install(*outputs: Any, **interpreter_kwargs: Any) -> ScriptedProvider:
        provider = ScriptedProvider(*outputs)
        app.state.interpreter = make_interpreter(provider, **interpreter_kwargs)
        return provider

    return install
