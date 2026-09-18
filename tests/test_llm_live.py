"""Semantic tests against the REAL configured language model.

Skipped unless an API key is configured (LLM_API_KEY or the provider's native variable).
Run with:  LLM_API_KEY=... pytest -m live -v

These are not phrase-matching tests of our code: the notes below are paraphrases the model
has never seen in the prompt, checked against the canonical structured answer.
"""

from __future__ import annotations

import asyncio

import pytest

from app.config import get_settings
from app.llm.factory import build_interpreter
from scripts.judge import judge_case
from tests.conftest import load_cases

pytestmark = pytest.mark.live

_settings = get_settings()
requires_llm = pytest.mark.skipif(not _settings.llm_configured, reason="no LLM API key configured")

CAPACITY, BASE_MIN = 240.0, 40.0

# (note, directive_type, hours, value)
SEMANTIC_CASES = [
    # solar_reduction: remaining fraction, percentages, words, 12h/24h times
    ("PV production will drop to about 20% between 13:00 and 15:00.", "solar_reduction", [13, 14], 0.2),
    ("Panel washing from one until three will leave roughly one-fifth of normal solar output.", "solar_reduction", [13, 14], 0.2),
    ("Expect an 80% reduction in rooftop solar during the 1-3 PM maintenance window.", "solar_reduction", [13, 14], 0.2),
    ("Haze will cut photovoltaic generation by 40% from 9 AM to noon.", "solar_reduction", [9, 10, 11], 0.6),
    ("The solar array is completely offline between 10:00 and 12:00 for inverter replacement.", "solar_reduction", [10, 11], 0.0),
    ("Only half of the forecast sunshine will be usable from 11 in the morning until 2 in the afternoon.", "solar_reduction", [11, 12, 13], 0.5),
    # minimum_battery_reserve: absolute and percentage-of-capacity
    ("Hold no less than 150 kWh in storage from 17:00 to 20:00 as a contingency.", "minimum_battery_reserve", [17, 18, 19], 150.0),
    ("Make sure the battery stays at least 75% full between 7 PM and 9 PM.", "minimum_battery_reserve", [19, 20], 180.0),
    ("Keep at least 120 kWh in reserve from 6 PM until 9 PM.", "minimum_battery_reserve", [18, 19, 20], 120.0),
    # no_charge_window
    ("Do not charge the battery between 2 PM and 4 PM.", "no_charge_window", [14, 15], None),
    ("The battery cannot accept any energy from midnight to 3 AM while the rectifier is serviced.", "no_charge_window", [0, 1, 2], None),
    # no_discharge_window
    ("Battery output must stay at zero from 20:00 to 22:00 during relay calibration.", "no_discharge_window", [20, 21], None),
    ("Nobody may draw power from the storage bank between 5 and 7 in the evening.", "no_discharge_window", [17, 18], None),
    # max_grid_window
    ("Utility import is limited to 120 kWh per hour from 6 PM until 10 PM.", "max_grid_window", [18, 19, 20, 21], 120.0),
    ("The feeder can only carry 0.2 MWh each hour between 18:00 and 20:00.", "max_grid_window", [18, 19], 200.0),
    ("A 90 kW transformer limit applies from 11 PM until midnight.", "max_grid_window", [23], 90.0),
    # no_op distractors
    ("The cafeteria menu changes tomorrow.", "no_op", None, None),
    ("The robotics club meets in lab 3 next Thursday.", "no_op", None, None),
    ("Solar panel cleaning is scheduled for next month.", "no_op", None, None),
]


def _interpret(notes):
    interpreter = build_interpreter(_settings)
    assert interpreter is not None
    return asyncio.run(interpreter.interpret(notes, CAPACITY, BASE_MIN))


@requires_llm
@pytest.mark.parametrize("note,dtype,hours,value", SEMANTIC_CASES, ids=[c[0][:40] for c in SEMANTIC_CASES])
def test_live_semantics(note, dtype, hours, value):
    interpretations, _ = _interpret([note])
    got = interpretations[0]
    assert got.directive_type.value == dtype, got.explanation
    if dtype == "no_op":
        assert got.applies is False and got.structured_adjustment is None
        return
    assert got.applies is True
    assert got.structured_adjustment["hours"] == hours
    if value is not None:
        key = next(k for k in got.structured_adjustment if k != "hours")
        assert got.structured_adjustment[key] == pytest.approx(value, abs=0.01)


@requires_llm
def test_live_three_notes_in_one_call():
    notes = [
        "Cloud cover during panel inspection will leave about half of the forecast solar output from 10 AM until noon.",
        "The charging circuit will be unavailable from 2 PM until 4 PM.",
        "The library is extending book-return hours next week.",
    ]
    interpretations, directives = _interpret(notes)
    assert [i.directive_type.value for i in interpretations] == ["solar_reduction", "no_charge_window", "no_op"]
    assert [i.note_index for i in interpretations] == [0, 1, 2]
    assert len(directives) == 2


CASES = load_cases()


@requires_llm
@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_live_public_samples_end_to_end(client, case):
    from app.main import app

    app.state.interpreter = build_interpreter(_settings)
    resp = client.post("/optimize-energy", json=case["input"])
    assert resp.status_code == 200, resp.text
    result = judge_case(case, resp.json())
    assert result == {"schema": [], "interpretation": [], "validity": [], "cost": []}
