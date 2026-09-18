"""Independent re-implementation of the organizer checks (Problem Statement §9-§11).

Deliberately shares no code with ``app/`` so it can catch bugs in the service itself.
Uses only the standard library.
"""

from __future__ import annotations

import math
from typing import Any

TOL = 0.01  # Official absolute tolerance (kWh / BDT).
DIRECTIVES = {
    "solar_reduction": {"hours", "factor"},
    "minimum_battery_reserve": {"hours", "minimum_energy_kwh"},
    "no_charge_window": {"hours"},
    "no_discharge_window": {"hours"},
    "max_grid_window": {"hours", "max_grid_kwh"},
}
TOP_KEYS = {
    "scenario_id",
    "directive_interpretation",
    "hourly_plan",
    "total_grid_kwh",
    "total_cost_bdt",
    "peak_grid_kwh",
    "plan_summary",
}
ENTRY_KEYS = {"note_index", "applies", "directive_type", "structured_adjustment", "explanation"}
PLAN_KEYS = {"hour", "grid_kwh", "solar_used_kwh", "battery_action", "battery_kwh", "battery_energy_after_kwh"}


def _num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def check_schema(request: dict, response: dict) -> list[str]:
    f: list[str] = []
    missing = TOP_KEYS - set(response)
    if missing:
        return [f"response missing keys {sorted(missing)}"]
    if response["scenario_id"] != request["scenario_id"]:
        f.append("scenario_id not echoed")
    for k in ("total_grid_kwh", "total_cost_bdt", "peak_grid_kwh"):
        if not _num(response[k]):
            f.append(f"{k} not a finite number")
    if not isinstance(response["plan_summary"], str):
        f.append("plan_summary not a string")

    di = response["directive_interpretation"]
    notes = request["operator_notes"]
    if not isinstance(di, list) or len(di) != len(notes):
        f.append("directive_interpretation must have one entry per note")
    else:
        for i, e in enumerate(di):
            if not isinstance(e, dict) or set(e) != ENTRY_KEYS:
                f.append(f"interpretation {i} has wrong keys")
                continue
            if e["note_index"] != i:
                f.append(f"interpretation {i} out of order")
            t = e["directive_type"]
            if t == "no_op":
                if e["applies"] is not False or e["structured_adjustment"] is not None:
                    f.append(f"interpretation {i}: no_op semantics")
            elif t in DIRECTIVES:
                adj = e["structured_adjustment"]
                if e["applies"] is not True or not isinstance(adj, dict) or set(adj) != DIRECTIVES[t]:
                    f.append(f"interpretation {i}: {t} shape/applies")
                else:
                    hrs = adj["hours"]
                    if (
                        not isinstance(hrs, list)
                        or not all(isinstance(h, int) and not isinstance(h, bool) and 0 <= h <= 23 for h in hrs)
                        or hrs != sorted(set(hrs))
                    ):
                        f.append(f"interpretation {i}: hours invalid")
            else:
                f.append(f"interpretation {i}: unsupported type {t!r}")
            if not isinstance(e["explanation"], str):
                f.append(f"interpretation {i}: explanation not a string")

    plan = response["hourly_plan"]
    if not isinstance(plan, list) or len(plan) != 24:
        f.append("hourly_plan must have 24 entries")
    else:
        for p in plan:
            if not isinstance(p, dict) or set(p) != PLAN_KEYS:
                f.append("hourly_plan entry has wrong keys")
                break
        else:
            if sorted(p["hour"] for p in plan) != list(range(24)):
                f.append("hourly_plan hours must be 0..23 once each")
    return f


def compare_interpretation(expected: list[dict], actual: list[dict]) -> list[str]:
    """Semantic match against ground truth; explanation text is ignored."""
    f: list[str] = []
    if len(expected) != len(actual):
        return [f"expected {len(expected)} interpretations, got {len(actual)}"]
    for i, (e, a) in enumerate(zip(expected, actual)):
        if e["directive_type"] != a.get("directive_type"):
            f.append(f"note {i}: type {a.get('directive_type')!r} != expected {e['directive_type']!r}")
            continue
        if e["applies"] != a.get("applies"):
            f.append(f"note {i}: applies mismatch")
        ea, aa = e["structured_adjustment"], a.get("structured_adjustment")
        if ea is None:
            if aa is not None:
                f.append(f"note {i}: expected null adjustment")
            continue
        if not isinstance(aa, dict):
            f.append(f"note {i}: missing adjustment")
            continue
        if ea["hours"] != aa.get("hours"):
            f.append(f"note {i}: hours {aa.get('hours')} != expected {ea['hours']}")
        for k, v in ea.items():
            if k != "hours" and not (_num(aa.get(k)) and abs(aa[k] - v) <= TOL):
                f.append(f"note {i}: {k}={aa.get(k)} != expected {v}")
    return f


def replay(request: dict, response: dict, directives: list[dict]) -> list[str]:
    """Replay hourly_plan under the given (ground-truth) directives."""
    f: list[str] = []
    hours = {h["hour"]: h for h in request["hours"]}
    b = request["battery"]
    solar = {h: hours[h]["solar_kwh"] for h in hours}
    reserve = {h: b["minimum_energy_kwh"] for h in hours}
    no_charge: set[int] = set()
    no_discharge: set[int] = set()
    grid_cap: dict[int, float] = {}
    for d in directives:
        t, adj = d["directive_type"], d["structured_adjustment"]
        if t == "no_op":
            continue
        for h in adj["hours"]:
            if t == "solar_reduction":
                solar[h] = solar[h] * adj["factor"]
            elif t == "minimum_battery_reserve":
                reserve[h] = max(reserve[h], adj["minimum_energy_kwh"])
            elif t == "no_charge_window":
                no_charge.add(h)
            elif t == "no_discharge_window":
                no_discharge.add(h)
            elif t == "max_grid_window":
                grid_cap[h] = min(grid_cap.get(h, math.inf), adj["max_grid_kwh"])

    energy = b["initial_energy_kwh"]
    plan = sorted(response["hourly_plan"], key=lambda p: p["hour"])
    for p in plan:
        h = p["hour"]
        for k in ("grid_kwh", "solar_used_kwh", "battery_kwh", "battery_energy_after_kwh"):
            if not _num(p[k]) or p[k] < -TOL:
                f.append(f"hour {h}: {k} invalid/negative")
        act, kwh = p["battery_action"], p["battery_kwh"]
        if act not in ("charge", "discharge", "idle"):
            f.append(f"hour {h}: bad action")
            continue
        if act == "idle" and abs(kwh) > TOL:
            f.append(f"hour {h}: idle with nonzero battery_kwh")
        ch = kwh if act == "charge" else 0.0
        dis = kwh if act == "discharge" else 0.0
        if ch > b["max_charge_kwh_per_hour"] + TOL:
            f.append(f"hour {h}: charge rate")
        if dis > b["max_discharge_kwh_per_hour"] + TOL:
            f.append(f"hour {h}: discharge rate")
        if abs(p["grid_kwh"] + p["solar_used_kwh"] + dis - hours[h]["demand_kwh"] - ch) > TOL:
            f.append(f"hour {h}: energy balance")
        if p["solar_used_kwh"] > solar[h] + TOL:
            f.append(f"hour {h}: effective solar exceeded")
        energy = energy + ch - dis
        if abs(p["battery_energy_after_kwh"] - energy) > TOL:
            f.append(f"hour {h}: battery transition")
        energy = p["battery_energy_after_kwh"]
        if energy > b["capacity_kwh"] + TOL:
            f.append(f"hour {h}: above capacity")
        if energy < reserve[h] - TOL:
            f.append(f"hour {h}: below minimum/reserve")
        if h in no_charge and act == "charge":
            f.append(f"hour {h}: charged in no_charge window")
        if h in no_discharge and act == "discharge":
            f.append(f"hour {h}: discharged in no_discharge window")
        if h in grid_cap and p["grid_kwh"] > grid_cap[h] + TOL:
            f.append(f"hour {h}: grid cap exceeded")
    if plan and abs(plan[-1]["battery_energy_after_kwh"] - b["initial_energy_kwh"]) > TOL:
        f.append("end-of-day neutrality")

    grid = [p["grid_kwh"] for p in plan]
    cost = sum(p["grid_kwh"] * hours[p["hour"]]["tariff_bdt_per_kwh"] for p in plan)
    if abs(response["total_grid_kwh"] - sum(grid)) > TOL:
        f.append("total_grid_kwh mismatch")
    if abs(response["total_cost_bdt"] - cost) > TOL:
        f.append("total_cost_bdt mismatch")
    if abs(response["peak_grid_kwh"] - max(grid)) > TOL:
        f.append("peak_grid_kwh mismatch")
    return f


def judge_case(case: dict, response: dict) -> dict[str, list[str]]:
    """Full check of one public sample: schema, interpretation, validity under ground truth, cost."""
    request, expected = case["input"], case["expected_output"]
    result = {"schema": check_schema(request, response), "interpretation": [], "validity": [], "cost": []}
    if result["schema"]:
        return result
    result["interpretation"] = compare_interpretation(
        expected["directive_interpretation"], response["directive_interpretation"]
    )
    result["validity"] = replay(request, response, expected["directive_interpretation"])
    recalculated = sum(
        p["grid_kwh"] * {h["hour"]: h for h in request["hours"]}[p["hour"]]["tariff_bdt_per_kwh"]
        for p in response["hourly_plan"]
    )
    if recalculated > expected["total_cost_bdt"] + TOL:
        result["cost"].append(f"cost {recalculated:.2f} above reference optimum {expected['total_cost_bdt']:.2f}")
    return result
