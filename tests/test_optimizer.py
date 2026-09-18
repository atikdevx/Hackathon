"""Optimizer correctness: every schedule is replayed by the independent judge."""

from __future__ import annotations

import pytest

from app.directives.model import Directive, build_hourly_limits
from app.errors import InfeasibleScenarioError
from app.models.schemas import DirectiveType, OptimizeRequest
from app.optimizer.lp import optimize
from app.services.schedule import build_hourly_plan, compute_totals
from app.validation.replay import replay_schedule
from scripts.judge import replay as judge_replay
from tests.conftest import flat_scenario

T = DirectiveType
PEAK_TARIFF = [5.0] * 8 + [10.0] * 10 + [20.0] * 4 + [5.0] * 2  # cheap night, pricey evening


def solve(body: dict, directives: list[Directive] = ()):
    req = OptimizeRequest.model_validate(body)
    limits = build_hourly_limits(req.hours, req.battery, list(directives))
    plan = build_hourly_plan(optimize(limits), limits)
    totals = compute_totals(plan, limits.tariff)
    assert replay_schedule(plan, totals, req.hours, req.battery, list(directives), limits) == []
    response = {
        "hourly_plan": [p.model_dump() for p in plan],
        "total_grid_kwh": totals.total_grid_kwh,
        "total_cost_bdt": totals.total_cost_bdt,
        "peak_grid_kwh": totals.peak_grid_kwh,
    }
    as_json = [
        {"directive_type": d.type.value, "structured_adjustment": {"hours": list(d.hours), **_value(d)}} for d in directives
    ]
    assert judge_replay(body, response, as_json) == []
    return plan, totals


def _value(d: Directive) -> dict:
    key = {T.SOLAR_REDUCTION: "factor", T.MINIMUM_BATTERY_RESERVE: "minimum_energy_kwh", T.MAX_GRID_WINDOW: "max_grid_kwh"}
    return {key[d.type]: d.value} if d.type in key else {}


def test_flat_tariff_needs_no_battery():
    plan, totals = solve(flat_scenario())
    assert all(p.battery_action == "idle" for p in plan)
    assert totals.total_cost_bdt == pytest.approx(100 * 10 * 24)


def test_shifts_energy_from_expensive_to_cheap_hours():
    plan, totals = solve(flat_scenario(tariff=PEAK_TARIFF))
    assert any(p.battery_action == "charge" and PEAK_TARIFF[p.hour] == 5.0 for p in plan)
    assert all(p.battery_action != "discharge" or PEAK_TARIFF[p.hour] == 20.0 for p in plan)
    baseline = sum(100 * t for t in PEAK_TARIFF)
    assert totals.total_cost_bdt < baseline
    # Battery fills to 200 kWh overnight, can discharge 180 kWh (down to the 20 kWh minimum)
    # across the 20-BDT hours, and all of it is bought at 5 BDT: saving 180 * (20 - 5).
    assert totals.total_cost_bdt == pytest.approx(baseline - 180 * 15)


def test_no_solar_all_from_grid_and_battery():
    plan, _ = solve(flat_scenario(tariff=PEAK_TARIFF))
    assert all(p.solar_used_kwh == 0 for p in plan)


def test_high_solar_is_curtailed_not_exported():
    solar = [0.0] * 8 + [400.0] * 8 + [0.0] * 8
    plan, _ = solve(flat_scenario(solar=solar, tariff=PEAK_TARIFF))
    for p in plan:
        assert p.solar_used_kwh <= solar[p.hour] + 1e-9
        assert p.grid_kwh >= 0
    # Midday demand fully met by solar; surplus limited to what the battery can take.
    assert all(p.grid_kwh == 0 for p in plan if 8 <= p.hour < 16)


def test_battery_minimum_respected():
    battery = {
        "capacity_kwh": 300.0,
        "initial_energy_kwh": 150.0,
        "minimum_energy_kwh": 120.0,
        "max_charge_kwh_per_hour": 100.0,
        "max_discharge_kwh_per_hour": 100.0,
    }
    plan, _ = solve(flat_scenario(tariff=PEAK_TARIFF, battery=battery))
    assert min(p.battery_energy_after_kwh for p in plan) >= 120 - 1e-6


def test_rate_limits_respected():
    battery = {
        "capacity_kwh": 1000.0,
        "initial_energy_kwh": 500.0,
        "minimum_energy_kwh": 0.0,
        "max_charge_kwh_per_hour": 30.0,
        "max_discharge_kwh_per_hour": 40.0,
    }
    plan, _ = solve(flat_scenario(tariff=PEAK_TARIFF, battery=battery))
    assert max(p.battery_kwh for p in plan if p.battery_action == "charge") <= 30 + 1e-9
    assert max(p.battery_kwh for p in plan if p.battery_action == "discharge") <= 40 + 1e-9


def test_no_charge_window():
    d = Directive(T.NO_CHARGE_WINDOW, tuple(range(0, 8)))
    plan, _ = solve(flat_scenario(tariff=PEAK_TARIFF), [d])
    assert all(p.battery_action != "charge" for p in plan if p.hour < 8)


def test_no_discharge_window():
    d = Directive(T.NO_DISCHARGE_WINDOW, (18, 19))
    plan, _ = solve(flat_scenario(tariff=PEAK_TARIFF), [d])
    assert all(p.battery_action != "discharge" for p in plan if p.hour in (18, 19))


def test_minimum_reserve_window():
    d = Directive(T.MINIMUM_BATTERY_RESERVE, (18, 19, 20), 150.0)
    plan, _ = solve(flat_scenario(tariff=PEAK_TARIFF), [d])
    assert all(p.battery_energy_after_kwh >= 150 - 1e-6 for p in plan if p.hour in (18, 19, 20))


def test_grid_cap_window():
    d = Directive(T.MAX_GRID_WINDOW, (18, 19, 20, 21), 60.0)
    plan, _ = solve(flat_scenario(tariff=[10.0] * 24), [d])
    assert all(p.grid_kwh <= 60 + 1e-6 for p in plan if 18 <= p.hour <= 21)


def test_solar_reduction_changes_effective_solar():
    solar = [0.0] * 10 + [80.0] * 4 + [0.0] * 10
    d = Directive(T.SOLAR_REDUCTION, (10, 11), 0.25)
    plan, _ = solve(flat_scenario(solar=solar), [d])
    assert plan[10].solar_used_kwh <= 20 + 1e-9 and plan[11].solar_used_kwh <= 20 + 1e-9
    assert plan[12].solar_used_kwh == pytest.approx(80)


def test_multiple_simultaneous_directives():
    solar = [0.0] * 9 + [150.0] * 6 + [0.0] * 9
    directives = [
        Directive(T.SOLAR_REDUCTION, (10, 11, 12), 0.5),
        Directive(T.NO_CHARGE_WINDOW, (13, 14)),
        Directive(T.NO_DISCHARGE_WINDOW, (17,)),
        Directive(T.MINIMUM_BATTERY_RESERVE, (18, 19, 20, 21), 90.0),
        Directive(T.MAX_GRID_WINDOW, (19, 20), 70.0),
    ]
    solve(flat_scenario(solar=solar, tariff=PEAK_TARIFF), directives)


def test_end_of_day_neutrality_exact():
    battery = {
        "capacity_kwh": 333.3,
        "initial_energy_kwh": 123.456,
        "minimum_energy_kwh": 11.1,
        "max_charge_kwh_per_hour": 47.77,
        "max_discharge_kwh_per_hour": 52.31,
    }
    tariff = [3.3 + (h % 7) * 1.7 for h in range(24)]
    plan, _ = solve(flat_scenario(demand=87.65, tariff=tariff, battery=battery))
    assert plan[-1].battery_energy_after_kwh == pytest.approx(123.456, abs=1e-9)


def test_overlapping_directives_compose_conservatively():
    solar = [100.0] * 24
    directives = [
        Directive(T.SOLAR_REDUCTION, (5, 6), 0.5),
        Directive(T.SOLAR_REDUCTION, (6, 7), 0.5),
        Directive(T.MAX_GRID_WINDOW, (3,), 80.0),
        Directive(T.MAX_GRID_WINDOW, (3,), 60.0),
    ]
    plan, _ = solve(flat_scenario(solar=solar, tariff=PEAK_TARIFF), directives)
    assert plan[6].solar_used_kwh <= 25 + 1e-9
    assert plan[3].grid_kwh <= 60 + 1e-9


def test_infeasible_scenario_raises_controlled_error():
    # Grid capped at zero with no solar and the battery unable to discharge.
    directives = [Directive(T.MAX_GRID_WINDOW, (2,), 0.0), Directive(T.NO_DISCHARGE_WINDOW, (2,))]
    req = OptimizeRequest.model_validate(flat_scenario())
    with pytest.raises(InfeasibleScenarioError):
        optimize(build_hourly_limits(req.hours, req.battery, directives))


def test_zero_capacity_battery():
    battery = {
        "capacity_kwh": 0.0,
        "initial_energy_kwh": 0.0,
        "minimum_energy_kwh": 0.0,
        "max_charge_kwh_per_hour": 0.0,
        "max_discharge_kwh_per_hour": 0.0,
    }
    plan, totals = solve(flat_scenario(tariff=PEAK_TARIFF, battery=battery))
    assert all(p.battery_action == "idle" for p in plan)
