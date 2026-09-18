"""End-to-end pipeline: validate -> LLM interpret -> guardrails -> apply -> optimize -> replay -> respond."""

from __future__ import annotations

import logging
import time

from app.directives.model import build_hourly_limits
from app.errors import LLMUnavailableError, ScheduleValidationError
from app.llm.interpreter import NoteInterpreter
from app.models.schemas import DirectiveType, OptimizeRequest, OptimizeResponse
from app.optimizer.lp import optimize
from app.services.request_checks import check_semantics
from app.services.schedule import build_hourly_plan, compute_totals
from app.services.summary import build_summary
from app.validation.replay import replay_schedule

logger = logging.getLogger(__name__)


async def run_pipeline(req: OptimizeRequest, interpreter: NoteInterpreter | None) -> OptimizeResponse:
    started = time.monotonic()
    check_semantics(req)
    if interpreter is None:
        raise LLMUnavailableError("No language model is configured for operator-note interpretation.")

    # 1-2. LLM interpretation, accepted only after deterministic guardrails.
    interpretations, directives = await interpreter.interpret(
        req.operator_notes, req.battery.capacity_kwh, req.battery.minimum_energy_kwh
    )
    t_llm = time.monotonic()

    # 3-4. Apply directives to the model, then solve.
    limits = build_hourly_limits(req.hours, req.battery, directives)
    raw = optimize(limits)
    plan = build_hourly_plan(raw, limits)
    totals = compute_totals(plan, limits.tariff)

    # 5. Independent replay before anything is returned.
    violations = replay_schedule(plan, totals, req.hours, req.battery, directives, limits)
    if violations:
        logger.error("replay validation failed scenario=%s violations=%s", req.scenario_id, violations[:10])
        raise ScheduleValidationError("The computed schedule failed final validation and was not returned.")

    ignored = sum(1 for i in interpretations if i.directive_type is DirectiveType.NO_OP)
    summary = build_summary(plan, totals, directives, ignored, limits)
    logger.info(
        "scenario=%s directives=%d llm=%.2fs total=%.2fs cost=%.2f",
        req.scenario_id,
        len(directives),
        t_llm - started,
        time.monotonic() - started,
        totals.total_cost_bdt,
    )
    return OptimizeResponse(
        scenario_id=req.scenario_id,
        directive_interpretation=interpretations,
        hourly_plan=plan,
        total_grid_kwh=totals.total_grid_kwh,
        total_cost_bdt=totals.total_cost_bdt,
        peak_grid_kwh=totals.peak_grid_kwh,
        plan_summary=summary,
    )
