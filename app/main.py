"""FastAPI application: GET /health and POST /optimize-energy."""

from __future__ import annotations

import json
import logging
import traceback
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from app.config import get_settings
from app.errors import InvalidRequestError, ServiceError
from app.llm.factory import build_interpreter
from app.models.schemas import HealthResponse, OptimizeRequest, OptimizeResponse
from app.services.pipeline import run_pipeline

logger = logging.getLogger("gridwise")


def _configure_logging(level: str) -> None:
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    # The HTTP client logs full request lines at INFO; keep it quiet.
    for noisy in ("httpx", "httpx2", "anthropic"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    _configure_logging(settings.log_level)
    app.state.interpreter = build_interpreter(settings)
    logger.info(
        "GridWise ready provider=%s model=%s llm_configured=%s",
        settings.llm_provider,
        settings.llm_model,
        app.state.interpreter is not None,
    )
    yield


app = FastAPI(
    title="GridWise LLM Energy Optimizer",
    version="1.0.0",
    description="LLM-interpreted operator directives + LP-optimized 24-hour campus energy schedule.",
    lifespan=lifespan,
)


def _error(status: int, body: dict[str, Any]) -> JSONResponse:
    return JSONResponse(status_code=status, content=body)


def _validation_details(exc: ValidationError) -> list[dict[str, str]]:
    # Location + message only; the offending input values are never echoed.
    return [
        {"loc": ".".join(str(p) for p in err.get("loc", ())), "msg": str(err.get("msg", "invalid value"))}
        for err in exc.errors(include_url=False, include_input=False, include_context=False)
    ][:30]


@app.exception_handler(ServiceError)
async def _service_error_handler(_: Request, exc: ServiceError) -> JSONResponse:
    return _error(exc.status_code, exc.to_body())


@app.exception_handler(Exception)
async def _unhandled_error_handler(_: Request, exc: Exception) -> JSONResponse:
    # Frames only (no exception message), so no request data or secret can reach the logs.
    frames = "".join(traceback.format_tb(exc.__traceback__))
    logger.error("unhandled error type=%s\n%s", type(exc).__name__, frames)
    return _error(500, {"error": {"code": "internal_error", "message": "An internal error occurred."}})


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse()


@app.post("/optimize-energy", response_model=OptimizeResponse)
async def optimize_energy(request: Request) -> OptimizeResponse:
    raw = await request.body()
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise InvalidRequestError("Request body must be valid JSON.") from exc
    if not isinstance(payload, dict):
        raise InvalidRequestError("Request body must be a JSON object.")
    try:
        req = OptimizeRequest.model_validate(payload)
    except ValidationError as exc:
        raise InvalidRequestError("Request does not match the required schema.", _validation_details(exc)) from exc
    return await run_pipeline(req, request.app.state.interpreter)
