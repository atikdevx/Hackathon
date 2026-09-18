"""Validated directives and how they change the optimization model (Problem Statement §5.3)."""

from __future__ import annotations

import math
from dataclasses import dataclass

from app.models.schemas import HORIZON_HOURS, Battery, DirectiveType, HourEntry


@dataclass(frozen=True)
class Directive:
    """A directive that has already passed the guardrails. ``value`` is the factor / kWh amount."""

    type: DirectiveType
    hours: tuple[int, ...]
    value: float | None = None


@dataclass(frozen=True)
class HourlyLimits:
    """Per-hour constraint set consumed by both the optimizer and the replay validator."""

    demand: tuple[float, ...]
    tariff: tuple[float, ...]
    effective_solar: tuple[float, ...]
    min_energy: tuple[float, ...]
    capacity: float
    initial_energy: float
    max_charge: tuple[float, ...]
    max_discharge: tuple[float, ...]
    grid_cap: tuple[float, ...]  # math.inf when uncapped


def build_hourly_limits(hours: list[HourEntry], battery: Battery, directives: list[Directive]) -> HourlyLimits:
    """Apply every validated directive to the base scenario.

    Overlapping directives compose conservatively so the result satisfies each one:
    solar factors multiply, reserves take the maximum, grid caps take the minimum.
    """
    by_hour = sorted(hours, key=lambda h: h.hour)
    solar_factor = [1.0] * HORIZON_HOURS
    min_energy = [battery.minimum_energy_kwh] * HORIZON_HOURS
    charge_allowed = [True] * HORIZON_HOURS
    discharge_allowed = [True] * HORIZON_HOURS
    grid_cap = [math.inf] * HORIZON_HOURS

    for d in directives:
        for h in d.hours:
            if d.type is DirectiveType.SOLAR_REDUCTION:
                solar_factor[h] *= float(d.value)  # type: ignore[arg-type]
            elif d.type is DirectiveType.MINIMUM_BATTERY_RESERVE:
                min_energy[h] = max(min_energy[h], float(d.value))  # type: ignore[arg-type]
            elif d.type is DirectiveType.NO_CHARGE_WINDOW:
                charge_allowed[h] = False
            elif d.type is DirectiveType.NO_DISCHARGE_WINDOW:
                discharge_allowed[h] = False
            elif d.type is DirectiveType.MAX_GRID_WINDOW:
                grid_cap[h] = min(grid_cap[h], float(d.value))  # type: ignore[arg-type]

    return HourlyLimits(
        demand=tuple(h.demand_kwh for h in by_hour),
        tariff=tuple(h.tariff_bdt_per_kwh for h in by_hour),
        effective_solar=tuple(h.solar_kwh * f for h, f in zip(by_hour, solar_factor)),
        min_energy=tuple(min_energy),
        capacity=battery.capacity_kwh,
        initial_energy=battery.initial_energy_kwh,
        max_charge=tuple(battery.max_charge_kwh_per_hour if ok else 0.0 for ok in charge_allowed),
        max_discharge=tuple(battery.max_discharge_kwh_per_hour if ok else 0.0 for ok in discharge_allowed),
        grid_cap=tuple(grid_cap),
    )
