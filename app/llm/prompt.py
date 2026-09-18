"""Prompt and JSON schema used to turn operator notes into structured directives.

The model reads the language: it decides relevance, the directive type, numeric values,
and the time windows exactly as the note states them (24-hour clock boundaries).
Deterministic code (app/guardrails/llm_output.py) then expands those windows into hour
lists using the spec's start-inclusive / end-exclusive rule, so the model never does
off-by-one hour arithmetic.

The model sees only the notes plus the battery capacity / base minimum (needed to convert
"keep 50% of capacity" into kWh). It never sees or returns demand, solar or tariff data.
"""

from __future__ import annotations

import json
from typing import Any

from app.models.schemas import DirectiveType

PROMPT_VERSION = "gridwise-interpreter-v3"

SYSTEM_PROMPT = """\
You are the operator-note interpreter for GridWise, a smart-campus energy scheduler.
The schedule covers ONE day of 24 whole hours on a 24-hour clock (00:00 to 24:00).
Operators send short natural-language notes. Map EACH note to EXACTLY ONE directive. Your JSON is validated by strict code and applied as hard optimization constraints, so precision matters more than anything else.

## Directive types (the only allowed values)
- solar_reduction: usable rooftop solar / PV output is reduced for a period. Set "factor" = fraction of forecast solar that REMAINS usable (0..1).
- minimum_battery_reserve: battery stored energy must stay at or above a level for a period. Set "minimum_energy_kwh".
- no_charge_window: the battery cannot be charged for a period (charger isolated/offline, charging disabled, battery must not absorb or store energy).
- no_discharge_window: the battery cannot discharge / supply energy for a period (discharge disabled, must not draw from the battery, protection or relay testing that blocks battery output).
- max_grid_window: grid import may not exceed a limit in each hour of a period (feeder, transformer, substation or utility import cap). Set "max_grid_kwh" (kWh per hour).
- no_op: the note does not change today's 24-hour energy schedule.
Battery direction matters: energy flowing OUT of the battery (drawing from it, using or tapping stored energy, the battery supplying the load, discharging) is discharge; energy flowing INTO it (charging, storing, topping up, absorbing surplus) is charge.

## Fields for each note
- explanation: FIRST, one short sentence naming the directive, the stated time window converted to the 24-hour clock, and the value you extracted.
- note_index: the note's zero-based index.
- directive_type: one of the values above.
- time_windows: the period(s) the note states, as {"start_hour", "end_hour"} on the 24-hour clock. Copy the boundaries the note gives; do NOT enumerate hours yourself. Use [] for no_op.
- factor, minimum_energy_kwh, max_grid_kwh: fill ONLY the one field belonging to the directive type; set the others to null. All three are null for no_op, no_charge_window and no_discharge_window.

## Time windows
- "from 1 PM to 3 PM", "between 13:00 and 15:00", "1-3 PM", "from one until three" in the afternoon -> {"start_hour": 13, "end_hour": 15}.
- start_hour is 0..23 and end_hour is 1..24. Noon = 12. Midnight as a start = 0; midnight as an end = 24.
- A single hour ("at 5 PM", "during the 17:00 hour") -> {"start_hour": 17, "end_hour": 18}.
- A duration ("from 6 PM for three hours") -> {"start_hour": 18, "end_hour": 21}.
- "after 8 PM" / "from 8 PM onward" / "for the rest of the day from 20:00" -> {"start_hour": 20, "end_hour": 24}. "until 6 AM" / "before 6 AM" -> {"start_hour": 0, "end_hour": 6}.
- A window crossing midnight keeps its stated boundaries: "10 PM to 2 AM" -> {"start_hour": 22, "end_hour": 2}.
- "all day", "the whole day", "around the clock", or a constraint that clearly applies with no time given -> {"start_hour": 0, "end_hour": 24}.
- If AM/PM is omitted, infer it from context: solar work, daylight or "afternoon" means daytime; "evening" or "tonight" means PM; "overnight" or "early morning" means AM.

## Numbers
- Solar factor is what REMAINS: "drop to 20%", "about one-fifth of normal", "20% of the forecast" -> 0.2; "an 80% reduction", "cut by 80%", "lose 80%" -> 0.2; "half" -> 0.5; "reduced by a quarter" -> 0.75; "no solar at all", "panels fully offline" -> 0.0.
- Battery reserve given as a share of the battery ("at least 50% of capacity", "keep it 40% charged") -> multiply by battery capacity_kwh from the context. Absolute amounts ("120 kWh") are used directly. Convert MWh to kWh (0.15 MWh -> 150).
- Grid caps in kW over one-hour intervals equal the same number of kWh per hour. Convert MW/MWh to kW/kWh.

## Relevance (no_op)
Use no_op when the note is unrelated to today's energy operation (menus, registration deadlines, club notices, room bookings, library hours), refers only to a different day outside this schedule ("next week", "next month", "last year"), is purely informational without a constraint of the supported kinds, or asks to change demand, tariffs or anything no directive type covers. Only use a non-no_op type when the note clearly states that constraint for today's schedule.
"""

_NULLABLE_NUMBER: dict[str, Any] = {"anyOf": [{"type": "number"}, {"type": "null"}]}

INTERPRETATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "interpretations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "explanation": {"type": "string"},
                    "note_index": {"type": "integer"},
                    "directive_type": {"type": "string", "enum": [t.value for t in DirectiveType]},
                    "time_windows": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "start_hour": {"type": "integer"},
                                "end_hour": {"type": "integer"},
                            },
                            "required": ["start_hour", "end_hour"],
                            "additionalProperties": False,
                        },
                    },
                    "factor": _NULLABLE_NUMBER,
                    "minimum_energy_kwh": _NULLABLE_NUMBER,
                    "max_grid_kwh": _NULLABLE_NUMBER,
                },
                "required": [
                    "explanation",
                    "note_index",
                    "directive_type",
                    "time_windows",
                    "factor",
                    "minimum_energy_kwh",
                    "max_grid_kwh",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["interpretations"],
    "additionalProperties": False,
}


def build_user_message(notes: list[str], capacity_kwh: float, minimum_energy_kwh: float) -> str:
    context = {
        "battery_context": {"capacity_kwh": capacity_kwh, "base_minimum_energy_kwh": minimum_energy_kwh},
        "operator_notes": [{"note_index": i, "text": note} for i, note in enumerate(notes)],
    }
    return (
        f"Interpret these {len(notes)} operator note(s). Return exactly {len(notes)} entries, "
        f"note_index 0..{len(notes) - 1} in order.\n\n" + json.dumps(context, ensure_ascii=False, indent=2)
    )


def build_repair_message(errors: list[str]) -> str:
    bullet_list = "\n".join(f"- {e}" for e in errors[:20])
    return (
        "Your previous JSON was rejected by the deterministic validator:\n"
        f"{bullet_list}\n\nReturn the corrected JSON for ALL notes, following every rule."
    )
