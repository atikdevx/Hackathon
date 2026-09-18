"""First guardrail stage: validate the model's raw extraction and assemble the official shape.

The model returns, per note, the directive type, the time windows exactly as stated
(24-hour clock boundaries) and the one numeric value its type needs. This module checks
that raw structure strictly, expands each window with the Problem Statement's whole-hour
rule (start inclusive, end exclusive, wrapping past midnight), and emits entries in the
official ``directive_interpretation`` format. Those entries then pass through the second,
contract-level stage in ``validator.py`` before anything reaches the optimizer.

Nothing is guessed: any missing, extra, ill-typed or contradictory field is an error.
"""

from __future__ import annotations

import math
from typing import Any

from app.models.schemas import HORIZON_HOURS, DirectiveType

_RAW_KEYS = frozenset(
    {"explanation", "note_index", "directive_type", "time_windows", "factor", "minimum_energy_kwh", "max_grid_kwh"}
)
_VALUE_FIELDS = ("factor", "minimum_energy_kwh", "max_grid_kwh")
_VALUE_FIELD_FOR: dict[DirectiveType, str] = {
    DirectiveType.SOLAR_REDUCTION: "factor",
    DirectiveType.MINIMUM_BATTERY_RESERVE: "minimum_energy_kwh",
    DirectiveType.MAX_GRID_WINDOW: "max_grid_kwh",
}


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def expand_window(start: int, end: int) -> list[int]:
    """Whole-hour window, start inclusive and end exclusive; end <= start wraps past midnight."""
    if end > start:
        return list(range(start, end))
    return list(range(start, HORIZON_HOURS)) + list(range(0, end))


def _windows_to_hours(windows: Any, where: str, errors: list[str]) -> list[int] | None:
    if not isinstance(windows, list) or not windows:
        errors.append(f"{where}.time_windows must be a non-empty array for an applying directive")
        return None
    hours: set[int] = set()
    for j, window in enumerate(windows):
        w = f"{where}.time_windows[{j}]"
        if not isinstance(window, dict) or set(window) != {"start_hour", "end_hour"}:
            errors.append(f"{w} must be an object with exactly start_hour and end_hour")
            return None
        start, end = window["start_hour"], window["end_hour"]
        if not (_is_int(start) and _is_int(end)):
            errors.append(f"{w} boundaries must be integers")
            return None
        if not (0 <= start <= 23 and 1 <= end <= 24):
            errors.append(f"{w} needs start_hour in 0..23 and end_hour in 1..24")
            return None
        if start == end:
            errors.append(f"{w} is empty (start_hour equals end_hour); use 0..24 for a whole day")
            return None
        hours.update(expand_window(start, end))
    return sorted(hours)


def assemble_interpretations(payload: Any, note_count: int) -> tuple[dict[str, Any] | None, list[str]]:
    """Validate the raw model JSON and convert it to ``{"interpretations": [official entries]}``."""
    errors: list[str] = []
    if not isinstance(payload, dict) or set(payload) != {"interpretations"}:
        return None, ['output must be an object with exactly one key "interpretations"']
    entries = payload["interpretations"]
    if not isinstance(entries, list):
        return None, ["interpretations must be an array"]
    if len(entries) != note_count:
        return None, [f"expected exactly {note_count} interpretation entries (one per note), got {len(entries)}"]

    official: list[dict[str, Any]] = []
    for i, raw in enumerate(entries):
        where = f"interpretations[{i}]"
        if not isinstance(raw, dict) or set(raw) != _RAW_KEYS:
            got = sorted(raw) if isinstance(raw, dict) else type(raw).__name__
            errors.append(f"{where} must have exactly the fields {sorted(_RAW_KEYS)} (got {got})")
            continue
        if not _is_int(raw["note_index"]) or raw["note_index"] != i:
            errors.append(f"{where}.note_index must be {i} (entries must follow note order 0..N-1)")
            continue
        try:
            dtype = DirectiveType(raw["directive_type"])
        except ValueError:
            errors.append(f"{where}.directive_type {raw['directive_type']!r} is not a supported directive type")
            continue

        value_field = _VALUE_FIELD_FOR.get(dtype)
        stray = [f for f in _VALUE_FIELDS if f != value_field and raw[f] is not None]
        if stray:
            errors.append(f"{where}: {dtype.value} must leave {stray} null")
            continue

        explanation = raw["explanation"]
        if dtype is DirectiveType.NO_OP:
            if raw["time_windows"] != []:
                errors.append(f"{where}: no_op must have an empty time_windows list")
                continue
            official.append(
                {
                    "note_index": i,
                    "applies": False,
                    "directive_type": dtype.value,
                    "structured_adjustment": None,
                    "explanation": explanation,
                }
            )
            continue

        hours = _windows_to_hours(raw["time_windows"], where, errors)
        if hours is None:
            continue
        adjustment: dict[str, Any] = {"hours": hours}
        if value_field is not None:
            value = raw[value_field]
            if value is None or isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                errors.append(f"{where}.{value_field} must be a finite number for {dtype.value}")
                continue
            adjustment[value_field] = value
        official.append(
            {
                "note_index": i,
                "applies": True,
                "directive_type": dtype.value,
                "structured_adjustment": adjustment,
                "explanation": explanation,
            }
        )

    if errors:
        return None, errors
    return {"interpretations": official}, []
