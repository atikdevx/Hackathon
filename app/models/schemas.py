"""Typed request/response models for the public API contract (Problem Statement §7, §10)."""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, field_validator

HORIZON_HOURS = 24
MIN_NOTES = 1
MAX_NOTES = 3

# JSON numbers only: ints and floats are accepted, strings/bools/NaN/Infinity are not.
FiniteNumber = Annotated[float, Field(strict=True, allow_inf_nan=False)]


class DirectiveType(str, Enum):
    SOLAR_REDUCTION = "solar_reduction"
    MINIMUM_BATTERY_RESERVE = "minimum_battery_reserve"
    NO_CHARGE_WINDOW = "no_charge_window"
    NO_DISCHARGE_WINDOW = "no_discharge_window"
    MAX_GRID_WINDOW = "max_grid_window"
    NO_OP = "no_op"


# Exact structured_adjustment keys required for each directive type (Problem Statement §4.1).
ADJUSTMENT_KEYS: dict[DirectiveType, frozenset[str]] = {
    DirectiveType.SOLAR_REDUCTION: frozenset({"hours", "factor"}),
    DirectiveType.MINIMUM_BATTERY_RESERVE: frozenset({"hours", "minimum_energy_kwh"}),
    DirectiveType.NO_CHARGE_WINDOW: frozenset({"hours"}),
    DirectiveType.NO_DISCHARGE_WINDOW: frozenset({"hours"}),
    DirectiveType.MAX_GRID_WINDOW: frozenset({"hours", "max_grid_kwh"}),
}


# --------------------------------------------------------------------------- request


class HourEntry(BaseModel):
    model_config = ConfigDict(extra="ignore")

    hour: StrictInt
    demand_kwh: FiniteNumber
    solar_kwh: FiniteNumber
    tariff_bdt_per_kwh: FiniteNumber


class Battery(BaseModel):
    model_config = ConfigDict(extra="ignore")

    capacity_kwh: FiniteNumber
    initial_energy_kwh: FiniteNumber
    minimum_energy_kwh: FiniteNumber
    max_charge_kwh_per_hour: FiniteNumber
    max_discharge_kwh_per_hour: FiniteNumber


class OptimizeRequest(BaseModel):
    """Structural validation only. Semantic checks live in services.request_checks."""

    model_config = ConfigDict(extra="ignore")

    scenario_id: Annotated[str, Field(strict=True)]
    operator_notes: Annotated[list[Annotated[str, Field(strict=True)]], Field(min_length=MIN_NOTES, max_length=MAX_NOTES)]
    hours: Annotated[list[HourEntry], Field(min_length=HORIZON_HOURS, max_length=HORIZON_HOURS)]
    battery: Battery

    @field_validator("operator_notes")
    @classmethod
    def _notes_not_blank(cls, notes: list[str]) -> list[str]:
        for i, note in enumerate(notes):
            if not note.strip():
                raise ValueError(f"operator_notes[{i}] must be a non-empty string")
        return notes

    @field_validator("hours")
    @classmethod
    def _hours_cover_horizon(cls, hours: list[HourEntry]) -> list[HourEntry]:
        seen = [h.hour for h in hours]
        if sorted(seen) != list(range(HORIZON_HOURS)):
            duplicates = sorted({h for h in seen if seen.count(h) > 1})
            missing = sorted(set(range(HORIZON_HOURS)) - set(seen))
            out_of_range = sorted({h for h in seen if not 0 <= h < HORIZON_HOURS})
            problems = []
            if duplicates:
                problems.append(f"duplicate hours {duplicates}")
            if missing:
                problems.append(f"missing hours {missing}")
            if out_of_range:
                problems.append(f"out-of-range hours {out_of_range}")
            raise ValueError("hours must contain each integer 0..23 exactly once: " + "; ".join(problems))
        return sorted(hours, key=lambda h: h.hour)


# --------------------------------------------------------------------------- response


class DirectiveInterpretation(BaseModel):
    note_index: StrictInt
    applies: StrictBool
    directive_type: DirectiveType
    structured_adjustment: Optional[dict[str, Any]]
    explanation: str


BatteryAction = Literal["charge", "discharge", "idle"]


class HourlyPlanEntry(BaseModel):
    hour: int
    grid_kwh: float
    solar_used_kwh: float
    battery_action: BatteryAction
    battery_kwh: float
    battery_energy_after_kwh: float


class OptimizeResponse(BaseModel):
    scenario_id: str
    directive_interpretation: list[DirectiveInterpretation]
    hourly_plan: list[HourlyPlanEntry]
    total_grid_kwh: float
    total_cost_bdt: float
    peak_grid_kwh: float
    plan_summary: str


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"

