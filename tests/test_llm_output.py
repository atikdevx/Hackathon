"""Guardrail stage 1: raw model extraction -> official directive_interpretation shape."""

from __future__ import annotations

import pytest

from app.guardrails.llm_output import assemble_interpretations, expand_window
from tests.conftest import interpretation_payload, raw_entry


@pytest.mark.parametrize(
    "start,end,hours",
    [
        (13, 15, [13, 14]),  # "1 PM to 3 PM": start inclusive, end exclusive
        (18, 21, [18, 19, 20]),
        (0, 24, list(range(24))),  # all day / midnight to midnight
        (23, 24, [23]),  # "11 PM until midnight"
        (0, 3, [0, 1, 2]),
        (22, 2, [22, 23, 0, 1]),  # crosses midnight
        (17, 18, [17]),  # single hour
    ],
)
def test_expand_window(start, end, hours):
    assert expand_window(start, end) == hours


def assemble(*entries):
    return assemble_interpretations(interpretation_payload(list(entries)), len(entries))


def test_assembles_every_directive_type():
    official, errors = assemble(
        raw_entry(0, "solar_reduction", [(12, 14)], factor=0.25),
        raw_entry(1, "minimum_battery_reserve", [(18, 21)], minimum_energy_kwh=100),
        raw_entry(2, "no_op"),
    )
    assert errors == []
    assert official["interpretations"] == [
        {
            "note_index": 0,
            "applies": True,
            "directive_type": "solar_reduction",
            "structured_adjustment": {"hours": [12, 13], "factor": 0.25},
            "explanation": "Scripted interpretation.",
        },
        {
            "note_index": 1,
            "applies": True,
            "directive_type": "minimum_battery_reserve",
            "structured_adjustment": {"hours": [18, 19, 20], "minimum_energy_kwh": 100},
            "explanation": "Scripted interpretation.",
        },
        {
            "note_index": 2,
            "applies": False,
            "directive_type": "no_op",
            "structured_adjustment": None,
            "explanation": "Scripted interpretation.",
        },
    ]


@pytest.mark.parametrize(
    "dtype,kwargs,shape",
    [
        ("no_charge_window", {}, {"hours": [14, 15]}),
        ("no_discharge_window", {}, {"hours": [14, 15]}),
        ("max_grid_window", {"max_grid_kwh": 155}, {"hours": [14, 15], "max_grid_kwh": 155}),
    ],
)
def test_hours_only_and_grid_shapes(dtype, kwargs, shape):
    official, errors = assemble(raw_entry(0, dtype, [(14, 16)], **kwargs))
    assert errors == [] and official["interpretations"][0]["structured_adjustment"] == shape


def test_multiple_and_overlapping_windows_are_unioned_and_sorted():
    official, _ = assemble(raw_entry(0, "no_charge_window", [(22, 2), (1, 3), (10, 11)]))
    assert official["interpretations"][0]["structured_adjustment"]["hours"] == [0, 1, 2, 10, 22, 23]


@pytest.mark.parametrize(
    "entry",
    [
        raw_entry(0, "no_op", [(1, 2)]),  # no_op with a window
        raw_entry(0, "no_op", factor=0.5),  # no_op with a value
        raw_entry(0, "no_charge_window"),  # applying directive without a window
        raw_entry(0, "no_charge_window", [(4, 4)]),  # empty window
        raw_entry(0, "no_charge_window", [(24, 2)]),  # start out of range
        raw_entry(0, "no_charge_window", [(3, 25)]),  # end out of range
        raw_entry(0, "no_charge_window", [(-1, 2)]),
        raw_entry(0, "no_charge_window", [(1, 2)], max_grid_kwh=10),  # value on a value-less type
        raw_entry(0, "solar_reduction", [(1, 2)]),  # missing factor
        raw_entry(0, "solar_reduction", [(1, 2)], factor=0.5, max_grid_kwh=3),  # extra value
        raw_entry(0, "max_grid_window", [(1, 2)], factor=0.5),  # wrong value field
        raw_entry(0, "reduce_demand", [(1, 2)]),  # unsupported type
        raw_entry(1, "no_op"),  # wrong note_index
    ],
)
def test_invalid_raw_entries_rejected(entry):
    official, errors = assemble(entry)
    assert official is None and errors


def test_non_integer_window_boundaries_rejected():
    entry = raw_entry(0, "no_charge_window", [(1, 2)])
    entry["time_windows"][0]["start_hour"] = 1.5
    assert assemble(entry)[0] is None
    entry["time_windows"][0] = {"start_hour": 1, "end_hour": 2, "note": "x"}
    assert assemble(entry)[0] is None


def test_extra_or_missing_fields_rejected():
    extra = raw_entry(0, "no_op")
    extra["demand_kwh"] = 5
    assert assemble(extra)[0] is None
    missing = raw_entry(0, "no_op")
    del missing["max_grid_kwh"]
    assert assemble(missing)[0] is None


@pytest.mark.parametrize("payload", [None, [], {"interpretations": {}}, {"other": []}])
def test_bad_top_level_rejected(payload):
    assert assemble_interpretations(payload, 1)[0] is None


def test_wrong_entry_count_rejected():
    payload = interpretation_payload([raw_entry(0, "no_op")])
    assert assemble_interpretations(payload, 2)[0] is None
