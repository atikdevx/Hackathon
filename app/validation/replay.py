"""Independent hour-by-hour replay of a finished schedule (Problem Statement §9, §11).

Checks the plan against the base GridWise rules *and* re-checks every validated
directive on its own terms, so a mistake in directive composition or in the optimizer
cannot produce a response that looks successful but is invalid.
"""

from __future__ import annotations

import math

from app.directives.model import Directive, HourlyLimits
from app.models.schemas import HORIZON_HOURS, Battery, DirectiveType, HourEntry, HourlyPlanEntry
from app.services.schedule import Totals

# Internal tolerance, 100x stricter than the judge's 0.01 kWh / 0.01 BDT.
TOLERANCE = 1e-4


def replay_schedule(
    plan: list[HourlyPlanEntry],
    totals: Totals,
    hours: list[HourEntry],
    battery: Battery,
    directives: list[Directive],
    limits: HourlyLimits,
) -> list[str]:
    """Return every violation found (empty list means the schedule is valid)."""
    errors: list[str] = []
    tol = TOLERANCE

    if len(plan) != HORIZON_HOURS or sorted(p.hour for p in plan) != list(range(HORIZON_HOURS)):
        return ["hourly_plan must contain exactly 24 unique hours 0..23"]
    by_hour = {h.hour: h for h in hours}
    entries = sorted(plan, key=lambda p: p.hour)

    energy = battery.initial_energy_kwh
    for p in entries:
        h = p.hour
        values = (p.grid_kwh, p.solar_used_kwh, p.battery_kwh, p.battery_energy_after_kwh)
        if not all(math.isfinite(v) for v in values):
            errors.append(f"hour {h}: non-finite value")
            continue
        if any(v < -tol for v in values):
            errors.append(f"hour {h}: negative value")

        charge = p.battery_kwh if p.battery_action == "charge" else 0.0
        discharge = p.battery_kwh if p.battery_action == "discharge" else 0.0
        if p.battery_action == "idle" and abs(p.battery_kwh) > tol:
            errors.append(f"hour {h}: idle action must have battery_kwh = 0")

        demand = by_hour[h].demand_kwh
        if abs(p.grid_kwh + p.solar_used_kwh + discharge - demand - charge) > tol:
            errors.append(f"hour {h}: energy balance violated")
        if p.solar_used_kwh > limits.effective_solar[h] + tol:
            errors.append(f"hour {h}: solar used exceeds effective solar")
        if p.solar_used_kwh > by_hour[h].solar_kwh + tol:
            errors.append(f"hour {h}: solar used exceeds base solar forecast")

        expected = energy + charge - discharge
        if abs(p.battery_energy_after_kwh - expected) > tol:
            errors.append(f"hour {h}: battery transition inconsistent")
        energy = p.battery_energy_after_kwh

        if energy > battery.capacity_kwh + tol:
            errors.append(f"hour {h}: battery above capacity")
        if energy < battery.minimum_energy_kwh - tol:
            errors.append(f"hour {h}: battery below base minimum")
        if energy < limits.min_energy[h] - tol:
            errors.append(f"hour {h}: battery below active reserve")
        if charge > battery.max_charge_kwh_per_hour + tol:
            errors.append(f"hour {h}: charge rate exceeded")
        if discharge > battery.max_discharge_kwh_per_hour + tol:
            errors.append(f"hour {h}: discharge rate exceeded")
        if p.grid_kwh > limits.grid_cap[h] + tol:
            errors.append(f"hour {h}: grid cap exceeded")

    if abs(entries[-1].battery_energy_after_kwh - battery.initial_energy_kwh) > tol:
        errors.append("end-of-day battery energy differs from initial energy")

    # Each directive, re-checked directly against the plan.
    plan_by_hour = {p.hour: p for p in entries}
    for d in directives:
        for h in d.hours:
            p = plan_by_hour[h]
            if d.type is DirectiveType.SOLAR_REDUCTION:
                if p.solar_used_kwh > by_hour[h].solar_kwh * float(d.value) + tol:  # type: ignore[arg-type]
                    errors.append(f"hour {h}: solar_reduction directive violated")
            elif d.type is DirectiveType.MINIMUM_BATTERY_RESERVE:
                if p.battery_energy_after_kwh < float(d.value) - tol:  # type: ignore[arg-type]
                    errors.append(f"hour {h}: minimum_battery_reserve directive violated")
            elif d.type is DirectiveType.NO_CHARGE_WINDOW:
                if p.battery_action == "charge":
                    errors.append(f"hour {h}: no_charge_window directive violated")
            elif d.type is DirectiveType.NO_DISCHARGE_WINDOW:
                if p.battery_action == "discharge":
                    errors.append(f"hour {h}: no_discharge_window directive violated")
            elif d.type is DirectiveType.MAX_GRID_WINDOW:
                if p.grid_kwh > float(d.value) + tol:  # type: ignore[arg-type]
                    errors.append(f"hour {h}: max_grid_window directive violated")

    grid = [p.grid_kwh for p in entries]
    if abs(totals.total_grid_kwh - sum(grid)) > tol:
        errors.append("total_grid_kwh does not match hourly_plan")
    cost = sum(p.grid_kwh * by_hour[p.hour].tariff_bdt_per_kwh for p in entries)
    if abs(totals.total_cost_bdt - cost) > tol:
        errors.append("total_cost_bdt does not match hourly_plan")
    if abs(totals.peak_grid_kwh - max(grid)) > tol:
        errors.append("peak_grid_kwh does not match hourly_plan")
    return errors
