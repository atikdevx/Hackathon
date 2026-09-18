"""API contract: endpoints, status codes, request validation, response schema."""

from __future__ import annotations

import copy
import json

import pytest

from app.main import app
from scripts.judge import check_schema
from tests.conftest import flat_scenario, interpretation_payload, noop_entries


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_valid_request_returns_exact_schema(client, use_provider):
    use_provider(interpretation_payload(noop_entries(1)))
    body = flat_scenario()
    resp = client.post("/optimize-energy", json=body)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert check_schema(body, data) == []
    assert data["scenario_id"] == "TEST-1"
    assert [p["hour"] for p in data["hourly_plan"]] == list(range(24))
    assert data["directive_interpretation"][0] == {
        "note_index": 0,
        "applies": False,
        "directive_type": "no_op",
        "structured_adjustment": None,
        "explanation": "Unrelated to the energy schedule.",
    }


def test_malformed_json_is_400(client):
    resp = client.post("/optimize-energy", content=b"{not json", headers={"Content-Type": "application/json"})
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "invalid_request"


def test_non_object_json_is_400(client):
    resp = client.post("/optimize-energy", json=[1, 2, 3])
    assert resp.status_code == 400


@pytest.mark.parametrize("field", ["scenario_id", "operator_notes", "hours", "battery"])
def test_missing_top_level_field_is_400(client, field):
    body = flat_scenario()
    del body[field]
    resp = client.post("/optimize-energy", json=body)
    assert resp.status_code == 400
    assert any(field in d["loc"] for d in resp.json()["error"]["details"])


def test_missing_battery_field_is_400(client):
    body = flat_scenario()
    del body["battery"]["max_charge_kwh_per_hour"]
    assert client.post("/optimize-energy", json=body).status_code == 400


@pytest.mark.parametrize("notes", [[], ["a", "b", "c", "d"], ["   "], [""], [1], "just a string"])
def test_invalid_operator_notes_is_400(client, notes):
    body = flat_scenario()
    body["operator_notes"] = notes
    assert client.post("/optimize-energy", json=body).status_code == 400


def test_duplicate_hours_is_400(client):
    body = flat_scenario()
    body["hours"][5]["hour"] = 4
    resp = client.post("/optimize-energy", json=body)
    assert resp.status_code == 400
    assert "duplicate hours [4]" in json.dumps(resp.json())


def test_missing_hour_is_400(client):
    body = flat_scenario()
    body["hours"].pop(10)
    assert client.post("/optimize-energy", json=body).status_code == 400


def test_out_of_range_hour_is_400(client):
    body = flat_scenario()
    body["hours"][23]["hour"] = 24
    assert client.post("/optimize-energy", json=body).status_code == 400


@pytest.mark.parametrize(
    "path,value",
    [
        (("hours", 0, "demand_kwh"), "100"),
        (("hours", 0, "demand_kwh"), None),
        (("hours", 0, "tariff_bdt_per_kwh"), True),
        (("hours", 0, "hour"), 0.5),
        (("hours", 0, "hour"), "0"),
        (("battery", "capacity_kwh"), "big"),
        (("scenario_id",), 42),
    ],
)
def test_invalid_numeric_types_are_400(client, path, value):
    body = flat_scenario()
    target = body
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    assert client.post("/optimize-energy", json=body).status_code == 400


def test_nan_is_400(client):
    raw = json.dumps(flat_scenario()).replace('"demand_kwh": 100.0', '"demand_kwh": NaN', 1)
    resp = client.post("/optimize-energy", content=raw, headers={"Content-Type": "application/json"})
    assert resp.status_code == 400


def test_hours_out_of_order_are_accepted_and_plan_sorted(client, use_provider):
    use_provider(interpretation_payload(noop_entries(1)))
    body = flat_scenario()
    body["hours"].reverse()
    resp = client.post("/optimize-energy", json=body)
    assert resp.status_code == 200
    assert [p["hour"] for p in resp.json()["hourly_plan"]] == list(range(24))


@pytest.mark.parametrize(
    "mutate",
    [
        lambda b: b["hours"][3].__setitem__("demand_kwh", -5),
        lambda b: b["hours"][3].__setitem__("solar_kwh", -1),
        lambda b: b["battery"].__setitem__("minimum_energy_kwh", 500),
        lambda b: b["battery"].__setitem__("initial_energy_kwh", 500),
        lambda b: b["battery"].__setitem__("max_charge_kwh_per_hour", -1),
    ],
)
def test_semantically_invalid_values_are_422(client, use_provider, mutate):
    provider = use_provider(interpretation_payload(noop_entries(1)))
    body = flat_scenario()
    mutate(body)
    resp = client.post("/optimize-energy", json=body)
    assert resp.status_code == 422
    assert provider.calls == []  # rejected before spending an LLM call


def test_missing_llm_configuration_is_controlled_500(client):
    app.state.interpreter = None
    resp = client.post("/optimize-energy", json=flat_scenario())
    assert resp.status_code == 500
    assert resp.json()["error"]["code"] == "llm_unavailable"


def test_extra_fields_are_ignored(client, use_provider):
    use_provider(interpretation_payload(noop_entries(1)))
    body = copy.deepcopy(flat_scenario())
    body["unexpected"] = {"x": 1}
    body["hours"][0]["note"] = "extra"
    assert client.post("/optimize-energy", json=body).status_code == 200


def test_unknown_route_is_json_404(client):
    resp = client.get("/nope")
    assert resp.status_code == 404
    assert resp.headers["content-type"].startswith("application/json")
