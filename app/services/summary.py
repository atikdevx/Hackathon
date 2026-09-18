"""Deterministic plan_summary text (no second model call: faster, cheaper, reproducible)."""

from __future__ import annotations

from app.directives.model import Directive, HourlyLimits
from app.models.schemas import DirectiveType, HourlyPlanEntry
from app.services.schedule import Totals


def _hours_label(hours: tuple[int, ...]) -> str:
    """[13, 14, 15] -> '13:00-16:00'; non-contiguous hours are listed as ranges."""
    ranges: list[tuple[int, int]] = []
    for h in hours:
        if ranges and h == ranges[-1][1] + 1:
            ranges[-1] = (ranges[-1][0], h)
        else:
            ranges.append((h, h))
    return ", ".join(f"{a:02d}:00-{b + 1:02d}:00" for a, b in ranges)


def _describe(d: Directive) -> str:
    window = _hours_label(d.hours)
    if d.type is DirectiveType.SOLAR_REDUCTION:
        return f"solar limited to {d.value:.0%} of forecast during {window}"
    if d.type is DirectiveType.MINIMUM_BATTERY_RESERVE:
        return f"battery kept at or above {d.value:g} kWh during {window}"
    if d.type is DirectiveType.NO_CHARGE_WINDOW:
        return f"no battery charging during {window}"
    if d.type is DirectiveType.NO_DISCHARGE_WINDOW:
        return f"no battery discharging during {window}"
    return f"grid import capped at {d.value:g} kWh/h during {window}"


def build_summary(
    plan: list[HourlyPlanEntry],
    totals: Totals,
    directives: list[Directive],
    ignored_notes: int,
    limits: HourlyLimits,
) -> str:
    parts: list[str] = []
    if directives:
        parts.append("Applied " + "; ".join(_describe(d) for d in directives) + ".")
    else:
        parts.append("No operator note changed the schedule constraints.")
    if ignored_notes:
        parts.append(f"Ignored {ignored_notes} note(s) that do not affect today's energy schedule.")

    charge = [p for p in plan if p.battery_action == "charge"]
    discharge = [p for p in plan if p.battery_action == "discharge"]
    if charge or discharge:
        avg = lambda ps: sum(limits.tariff[p.hour] * p.battery_kwh for p in ps) / max(  # noqa: E731
            sum(p.battery_kwh for p in ps), 1e-9
        )
        text = "Battery"
        if charge:
            text += f" charges {sum(p.battery_kwh for p in charge):g} kWh in cheaper hours (avg {avg(charge):.2f} BDT/kWh)"
        if discharge:
            text += (" and" if charge else "") + (
                f" discharges {sum(p.battery_kwh for p in discharge):g} kWh in pricier hours (avg {avg(discharge):.2f} BDT/kWh)"
            )
        parts.append(text + f", ending at its initial {limits.initial_energy:g} kWh.")
    else:
        parts.append("Battery stays idle; no cost-saving shift was available within the constraints.")

    parts.append(
        f"Grid import {totals.total_grid_kwh:g} kWh, cost {totals.total_cost_bdt:.2f} BDT, peak {totals.peak_grid_kwh:g} kWh."
    )
    return " ".join(parts)
