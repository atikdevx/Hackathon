"""Semantic checks on a structurally valid request (HTTP 422 when violated)."""

from __future__ import annotations

from app.errors import SemanticRequestError
from app.models.schemas import OptimizeRequest


def check_semantics(req: OptimizeRequest) -> None:
    problems: list[str] = []
    for h in req.hours:
        if h.demand_kwh < 0:
            problems.append(f"hour {h.hour}: demand_kwh must be non-negative")
        if h.solar_kwh < 0:
            problems.append(f"hour {h.hour}: solar_kwh must be non-negative")

    b = req.battery
    for name in (
        "capacity_kwh",
        "initial_energy_kwh",
        "minimum_energy_kwh",
        "max_charge_kwh_per_hour",
        "max_discharge_kwh_per_hour",
    ):
        if getattr(b, name) < 0:
            problems.append(f"battery.{name} must be non-negative")
    if b.minimum_energy_kwh > b.capacity_kwh:
        problems.append("battery.minimum_energy_kwh must not exceed capacity_kwh")
    if not b.minimum_energy_kwh <= b.initial_energy_kwh <= b.capacity_kwh:
        problems.append(
            "battery.initial_energy_kwh must lie between minimum_energy_kwh and capacity_kwh "
            "(end-of-day neutrality requires the final state to equal it)"
        )
    if not req.scenario_id.strip():
        problems.append("scenario_id must be a non-empty string")

    if problems:
        raise SemanticRequestError("The request is well-formed but not physically valid.", problems)
