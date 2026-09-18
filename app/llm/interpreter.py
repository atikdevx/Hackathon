"""LLM operator-note interpreter: model call -> JSON parse -> deterministic guardrails.

One model call interprets all 1-3 notes. Its raw extraction is assembled into the
official shape (guardrails/llm_output.py) and then contract-checked (guardrails/validator.py).
If the output fails either stage, the model
gets one repair turn listing the violations; nothing is ever applied until the
validator accepts it, and no hard-coded phrase matching is used as a fallback.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from collections import OrderedDict

from app.directives.model import Directive
from app.errors import LLMOutputInvalidError, LLMUnavailableError
from app.guardrails.llm_output import assemble_interpretations
from app.guardrails.validator import validate_interpretation
from app.llm.base import ChatMessage, LLMProvider
from app.llm.prompt import (
    INTERPRETATION_SCHEMA,
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    build_repair_message,
    build_user_message,
)
from app.models.schemas import DirectiveInterpretation

logger = logging.getLogger(__name__)

# Leaves headroom under the 30 s per-request limit for optimization and serialization.
TOTAL_LLM_BUDGET_SECONDS = 25.0
_MIN_ATTEMPT_SECONDS = 2.0

Interpretation = tuple[list[DirectiveInterpretation], list[Directive]]


class NoteInterpreter:
    def __init__(
        self,
        provider: LLMProvider,
        timeout_seconds: float,
        max_attempts: int,
        cache_size: int,
        total_budget_seconds: float = TOTAL_LLM_BUDGET_SECONDS,
    ) -> None:
        self._provider = provider
        self._timeout = timeout_seconds
        self._max_attempts = max_attempts
        self._budget = total_budget_seconds
        self._cache_size = cache_size
        self._cache: OrderedDict[str, Interpretation] = OrderedDict()

    @property
    def provider_name(self) -> str:
        return self._provider.name

    @property
    def model(self) -> str:
        return self._provider.model

    def _cache_key(self, notes: list[str], capacity_kwh: float, minimum_energy_kwh: float) -> str:
        # The interpretation depends only on the notes and the battery sizing the prompt exposes.
        material = json.dumps(
            [PROMPT_VERSION, self._provider.name, self._provider.model, notes, capacity_kwh, minimum_energy_kwh]
        )
        return hashlib.sha256(material.encode()).hexdigest()

    async def interpret(self, notes: list[str], capacity_kwh: float, minimum_energy_kwh: float) -> Interpretation:
        key = self._cache_key(notes, capacity_kwh, minimum_energy_kwh)
        if key in self._cache:
            self._cache.move_to_end(key)
            interpretations, directives = self._cache[key]
            return [i.model_copy(deep=True) for i in interpretations], list(directives)

        result = await self._interpret_uncached(notes, capacity_kwh, minimum_energy_kwh)
        if self._cache_size:
            self._cache[key] = result
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        interpretations, directives = result
        return [i.model_copy(deep=True) for i in interpretations], list(directives)

    async def _interpret_uncached(
        self, notes: list[str], capacity_kwh: float, minimum_energy_kwh: float
    ) -> Interpretation:
        deadline = time.monotonic() + self._budget
        messages = [ChatMessage("user", build_user_message(notes, capacity_kwh, minimum_energy_kwh))]
        last_problem: Exception | None = None

        for attempt in range(1, self._max_attempts + 1):
            remaining = deadline - time.monotonic()
            if remaining < _MIN_ATTEMPT_SECONDS:
                break
            timeout = min(self._timeout, remaining)
            started = time.monotonic()
            try:
                raw = await asyncio.wait_for(
                    self._provider.generate_json(SYSTEM_PROMPT, messages, INTERPRETATION_SCHEMA, timeout),
                    timeout=timeout + 1.0,
                )
            except asyncio.TimeoutError:
                last_problem = LLMUnavailableError("The language model timed out.")
                logger.warning("llm attempt=%d timed out", attempt)
                continue
            except (LLMUnavailableError, LLMOutputInvalidError) as exc:
                last_problem = exc
                logger.warning("llm attempt=%d failed: %s", attempt, exc.message)
                continue
            except Exception as exc:  # noqa: BLE001 - provider bugs must not crash the request
                # Only the type is logged: third-party messages could echo request material.
                last_problem = LLMUnavailableError("The language model provider failed unexpectedly.")
                logger.warning("llm attempt=%d unexpected provider error type=%s", attempt, type(exc).__name__)
                continue

            elapsed = time.monotonic() - started
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                errors = ["response was not valid JSON"]
            else:
                # Stage 1: raw extraction -> official shape. Stage 2: contract-level guardrails.
                assembled, errors = assemble_interpretations(payload, len(notes))
                check = validate_interpretation(assembled, len(notes), capacity_kwh) if assembled else None
                if check is not None and check.ok:
                    logger.info(
                        "llm interpretation accepted attempt=%d latency=%.2fs types=%s",
                        attempt,
                        elapsed,
                        [i.directive_type.value for i in check.interpretations],
                    )
                    return check.interpretations, check.directives
                if check is not None:
                    errors = check.errors

            logger.warning("llm attempt=%d rejected by guardrails: %s", attempt, errors)
            last_problem = LLMOutputInvalidError(
                "The language model output failed deterministic validation; no directive was applied.", errors
            )
            messages = messages + [ChatMessage("assistant", raw), ChatMessage("user", build_repair_message(errors))]

        if isinstance(last_problem, (LLMUnavailableError, LLMOutputInvalidError)):
            raise last_problem
        raise LLMUnavailableError("The language model did not respond within the time budget.")
