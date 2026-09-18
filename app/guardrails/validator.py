"""Deterministic guardrails for untrusted LLM output (Problem Statement §8).

Nothing the model returns is applied unless every check here passes. The validator
never repairs, re-orders, or guesses: it either returns fully validated directives
or a list of human-readable violations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from app.directives.model import Directive
from app.models.schemas import ADJUSTMENT_KEYS, HORIZON_HOURS, DirectiveInterpretation, DirectiveType

_ENTRY_KEYS = frozenset({"note_index", "applies", "directive_type", "structured_adjustment", "explanation"})
_VALUE_KEY: dict[DirectiveType, str] = {
    DirectiveType.SOLAR_REDUCTION: "factor",
    DirectiveType.MINIMUM_BATTERY_RESERVE: "minimum_energy_kwh",
    DirectiveType.MAX_GRID_WINDOW: "max_grid_kwh",
}
_MAX_EXPLANATION_CHARS = 500


@dataclass
class GuardrailResult:
    interpretations: list[DirectiveInterpretation] = field(default_factory=list)
    directives: list[Directive] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _check_hours(hours: Any, where: str, errors: list[str]) -> tuple[int, ...] | None:
    if not isinstance(hours, list):
        errors.append(f"{where}.hours must be an array of integers")
        return None
    if not hours:
        errors.append(f"{where}.hours must not be empty for an applying directive")
        return None
    if not all(_is_int(h) for h in hours):
        errors.append(f"{where}.hours must contain only integers")
        return None
    if any(not 0 <= h < HORIZON_HOURS for h in hours):
        errors.append(f"{where}.hours values must be within 0..23")
        return None
    if len(set(hours)) != len(hours):
        errors.append(f"{where}.hours must not contain duplicates")
        return None
    if hours != sorted(hours):
        errors.append(f"{where}.hours must be in ascending order")
        return None
    return tuple(hours)


def _check_value(dtype: DirectiveType, value: Any, capacity_kwh: float, where: str, errors: list[str]) -> float | None:
    key = _VALUE_KEY[dtype]
    if not _is_number(value):
        errors.append(f"{where}.{key} must be a finite number")
        return None
    value = float(value)
    if dtype is DirectiveType.SOLAR_REDUCTION and not 0.0 <= value <= 1.0:
        errors.append(f"{where}.factor must be between 0 and 1 inclusive (fraction of solar that remains)")
        return None
    if dtype is DirectiveType.MINIMUM_BATTERY_RESERVE and not 0.0 <= value <= capacity_kwh:
        errors.append(f"{where}.minimum_energy_kwh must be between 0 and battery capacity {capacity_kwh:g} kWh")
        return None
    if dtype is DirectiveType.MAX_GRID_WINDOW and value < 0.0:
        errors.append(f"{where}.max_grid_kwh must be non-negative")
        return None
    return value


def _validate_entry(
    position: int, entry: Any, capacity_kwh: float, errors: list[str]
) -> tuple[DirectiveInterpretation, Directive | None] | None:
    where = f"interpretations[{position}]"
    if not isinstance(entry, dict):
        errors.append(f"{where} must be an object")
        return None
    keys = set(entry)
    if keys != _ENTRY_KEYS:
        missing, extra = sorted(_ENTRY_KEYS - keys), sorted(keys - _ENTRY_KEYS)
        errors.append(f"{where} has wrong fields (missing={missing}, unexpected={extra})")
        return None

    note_index = entry["note_index"]
    if not _is_int(note_index) or note_index != position:
        errors.append(f"{where}.note_index must be {position} (entries must follow note order 0..N-1)")
        return None

    raw_type = entry["directive_type"]
    try:
        dtype = DirectiveType(raw_type)
    except ValueError:
        errors.append(f"{where}.directive_type {raw_type!r} is not a supported directive type")
        return None

    applies = entry["applies"]
    if not isinstance(applies, bool):
        errors.append(f"{where}.applies must be a boolean")
        return None

    explanation = entry["explanation"]
    if not isinstance(explanation, str):
        errors.append(f"{where}.explanation must be a string")
        return None
    explanation = explanation.strip()[:_MAX_EXPLANATION_CHARS] or f"Interpreted as {dtype.value}."

    adjustment = entry["structured_adjustment"]
    if dtype is DirectiveType.NO_OP:
        if applies is not False:
            errors.append(f"{where}: no_op requires applies = false")
            return None
        if adjustment is not None:
            errors.append(f"{where}: no_op requires structured_adjustment = null")
            return None
        return DirectiveInterpretation(
            note_index=position, applies=False, directive_type=dtype, structured_adjustment=None, explanation=explanation
        ), None

    if applies is not True:
        errors.append(f"{where}: {dtype.value} requires applies = true")
        return None
    if not isinstance(adjustment, dict):
        errors.append(f"{where}: {dtype.value} requires a structured_adjustment object")
        return None
    required = ADJUSTMENT_KEYS[dtype]
    if set(adjustment) != required:
        errors.append(
            f"{where}.structured_adjustment for {dtype.value} must have exactly the keys {sorted(required)}"
        )
        return None

    hours = _check_hours(adjustment["hours"], f"{where}.structured_adjustment", errors)
    if hours is None:
        return None
    value: float | None = None
    clean_adjustment: dict[str, Any] = {"hours": list(hours)}
    if dtype in _VALUE_KEY:
        value = _check_value(dtype, adjustment[_VALUE_KEY[dtype]], capacity_kwh, f"{where}.structured_adjustment", errors)
        if value is None:
            return None
        clean_adjustment[_VALUE_KEY[dtype]] = value

    return (
        DirectiveInterpretation(
            note_index=position,
            applies=True,
            directive_type=dtype,
            structured_adjustment=clean_adjustment,
            explanation=explanation,
        ),
        Directive(type=dtype, hours=hours, value=value),
    )


def validate_interpretation(payload: Any, note_count: int, capacity_kwh: float) -> GuardrailResult:
    """Validate the model's parsed JSON for ``note_count`` notes."""
    result = GuardrailResult()
    if not isinstance(payload, dict) or set(payload) != {"interpretations"}:
        result.errors.append('output must be an object with exactly one key "interpretations"')
        return result
    entries = payload["interpretations"]
    if not isinstance(entries, list):
        result.errors.append("interpretations must be an array")
        return result
    if len(entries) != note_count:
        result.errors.append(f"expected exactly {note_count} interpretation entries (one per note), got {len(entries)}")
        return result

    for position, entry in enumerate(entries):
        validated = _validate_entry(position, entry, capacity_kwh, result.errors)
        if validated is None:
            continue
        interpretation, directive = validated
        result.interpretations.append(interpretation)
        if directive is not None:
            result.directives.append(directive)

    if result.errors:
        result.interpretations.clear()
        result.directives.clear()
    return result
