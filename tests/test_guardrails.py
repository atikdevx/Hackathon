"""Deterministic guardrails reject every malformed or unsafe model output."""

from __future__ import annotations

import copy
import math

import pytest

from app.guardrails.validator import validate_interpretation
from app.models.schemas import DirectiveType

CAPACITY = 200.0


def entry(i, dtype, adj, applies=None, explanation="x"):
    return {
        "note_index": i,
        "applies": (dtype != "no_op") if applies is None else applies,
        "directive_type": dtype,
        "structured_adjustment": adj,
        "explanation": explanation,
    }


VALID = {
    "interpretations": [
        entry(0, "solar_reduction", {"hours": [13, 14], "factor": 0.2}),
        entry(1, "minimum_battery_reserve", {"hours": [18, 19, 20], "minimum_energy_kwh": 120}),
        entry(2, "no_op", None),
    ]
}


def run(payload, notes=3):
    return validate_interpretation(payload, notes, CAPACITY)


def test_valid_payload_accepted():
    result = run(VALID)
    assert result.ok, result.errors
    assert [d.type for d in result.directives] == [DirectiveType.SOLAR_REDUCTION, DirectiveType.MINIMUM_BATTERY_RESERVE]
    assert result.interpretations[1].structured_adjustment == {"hours": [18, 19, 20], "minimum_energy_kwh": 120.0}
    assert result.interpretations[2].structured_adjustment is None


@pytest.mark.parametrize(
    "dtype,adj",
    [
        ("no_charge_window", {"hours": [14, 15]}),
        ("no_discharge_window", {"hours": [18, 19]}),
        ("max_grid_window", {"hours": [18, 19, 20], "max_grid_kwh": 155}),
        ("max_grid_window", {"hours": [0], "max_grid_kwh": 0}),
        ("solar_reduction", {"hours": list(range(24)), "factor": 0}),
        ("solar_reduction", {"hours": [12], "factor": 1}),
        ("minimum_battery_reserve", {"hours": [5], "minimum_energy_kwh": CAPACITY}),
    ],
)
def test_each_directive_shape_accepted(dtype, adj):
    assert run({"interpretations": [entry(0, dtype, adj)]}, notes=1).ok


def mutated(fn):
    payload = copy.deepcopy(VALID)
    fn(payload["interpretations"])
    return payload


@pytest.mark.parametrize(
    "name,payload",
    [
        ("unknown type", mutated(lambda e: e[0].__setitem__("directive_type", "load_shedding"))),
        ("duplicate hours", mutated(lambda e: e[0]["structured_adjustment"].__setitem__("hours", [13, 13]))),
        ("unsorted hours", mutated(lambda e: e[0]["structured_adjustment"].__setitem__("hours", [14, 13]))),
        ("hour 24", mutated(lambda e: e[0]["structured_adjustment"].__setitem__("hours", [23, 24]))),
        ("negative hour", mutated(lambda e: e[0]["structured_adjustment"].__setitem__("hours", [-1, 0]))),
        ("float hour", mutated(lambda e: e[0]["structured_adjustment"].__setitem__("hours", [13.0]))),
        ("bool hour", mutated(lambda e: e[0]["structured_adjustment"].__setitem__("hours", [True]))),
        ("empty hours", mutated(lambda e: e[0]["structured_adjustment"].__setitem__("hours", []))),
        ("factor < 0", mutated(lambda e: e[0]["structured_adjustment"].__setitem__("factor", -0.1))),
        ("factor > 1", mutated(lambda e: e[0]["structured_adjustment"].__setitem__("factor", 1.5))),
        ("factor nan", mutated(lambda e: e[0]["structured_adjustment"].__setitem__("factor", math.nan))),
        ("factor string", mutated(lambda e: e[0]["structured_adjustment"].__setitem__("factor", "0.2"))),
        ("reserve > capacity", mutated(lambda e: e[1]["structured_adjustment"].__setitem__("minimum_energy_kwh", 201))),
        ("reserve negative", mutated(lambda e: e[1]["structured_adjustment"].__setitem__("minimum_energy_kwh", -1))),
        ("reserve inf", mutated(lambda e: e[1]["structured_adjustment"].__setitem__("minimum_energy_kwh", math.inf))),
        ("no_op applies true", mutated(lambda e: e[2].__setitem__("applies", True))),
        ("no_op with adjustment", mutated(lambda e: e[2].__setitem__("structured_adjustment", {"hours": [1]}))),
        ("directive applies false", mutated(lambda e: e[0].__setitem__("applies", False))),
        ("directive null adjustment", mutated(lambda e: e[0].__setitem__("structured_adjustment", None))),
        ("applies as string", mutated(lambda e: e[0].__setitem__("applies", "true"))),
        ("missing factor", mutated(lambda e: e[0]["structured_adjustment"].pop("factor"))),
        ("extra adjustment key", mutated(lambda e: e[0]["structured_adjustment"].__setitem__("demand_kwh", 5))),
        ("wrong value key", mutated(lambda e: e[0].__setitem__("structured_adjustment", {"hours": [1], "max_grid_kwh": 5}))),
        ("adjustment is list", mutated(lambda e: e[0].__setitem__("structured_adjustment", [13, 14]))),
        ("extra entry key", mutated(lambda e: e[0].__setitem__("tariff_bdt_per_kwh", 3))),
        ("missing explanation", mutated(lambda e: e[0].pop("explanation"))),
        ("duplicate note mapping", mutated(lambda e: e[1].__setitem__("note_index", 0))),
        ("out of order mapping", mutated(lambda e: e.reverse())),
        ("note index out of range", mutated(lambda e: e[2].__setitem__("note_index", 7))),
        ("missing note mapping", mutated(lambda e: e.pop())),
        ("extra note mapping", mutated(lambda e: e.append(entry(3, "no_op", None)))),
        ("entry not object", mutated(lambda e: e.__setitem__(0, "solar_reduction"))),
    ],
)
def test_invalid_outputs_rejected(name, payload):
    result = run(payload)
    assert not result.ok, name
    assert result.directives == [] and result.interpretations == []


@pytest.mark.parametrize(
    "payload",
    [None, [], "text", {"interpretations": "x"}, {"interpretations": [], "extra": 1}, {"results": []}],
)
def test_invalid_top_level_rejected(payload):
    assert not run(payload).ok


def test_negative_grid_cap_rejected():
    payload = {"interpretations": [entry(0, "max_grid_window", {"hours": [1], "max_grid_kwh": -5})]}
    assert not run(payload, notes=1).ok


def test_rejects_rather_than_repairs():
    """The validator never silently fixes hours; the model must produce them correctly."""
    payload = {"interpretations": [entry(0, "no_charge_window", {"hours": [15, 14]})]}
    result = run(payload, notes=1)
    assert not result.ok
    assert "ascending" in result.errors[0]


def test_blank_explanation_gets_default_text():
    payload = {"interpretations": [entry(0, "no_op", None, explanation="  ")]}
    result = run(payload, notes=1)
    assert result.ok and result.interpretations[0].explanation
