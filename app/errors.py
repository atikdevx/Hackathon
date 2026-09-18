"""Controlled, client-safe error types.

Every error raised inside the pipeline is converted to a structured JSON body
``{"error": {"code": ..., "message": ..., "details": [...]}}``. Messages are
written by us and never include secrets, prompts, or stack traces.
"""

from __future__ import annotations

from typing import Any


class ServiceError(Exception):
    status_code: int = 500
    code: str = "internal_error"

    def __init__(self, message: str, details: list[Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or []

    def to_body(self) -> dict[str, Any]:
        body: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.details:
            body["details"] = self.details
        return {"error": body}


class InvalidRequestError(ServiceError):
    """Malformed JSON or structurally invalid request (HTTP 400)."""

    status_code = 400
    code = "invalid_request"


class SemanticRequestError(ServiceError):
    """Well-formed request with impossible values (HTTP 422)."""

    status_code = 422
    code = "semantically_invalid_request"


class InfeasibleScenarioError(ServiceError):
    """No schedule satisfies the scenario plus its validated directives (HTTP 422)."""

    status_code = 422
    code = "infeasible_scenario"


class LLMUnavailableError(ServiceError):
    """Provider missing, timed out, rate limited, or returned an API error."""

    status_code = 500
    code = "llm_unavailable"


class LLMOutputInvalidError(ServiceError):
    """Model output failed deterministic guardrails; nothing was applied."""

    status_code = 500
    code = "llm_output_rejected"


class OptimizerError(ServiceError):
    status_code = 500
    code = "optimizer_failure"


class ScheduleValidationError(ServiceError):
    """The optimized schedule failed independent replay validation."""

    status_code = 500
    code = "schedule_validation_failed"
