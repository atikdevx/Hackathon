"""Runtime configuration, read once from environment variables.

Secrets (API keys) are held only in memory and are never logged or returned.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache

OPENAI_PROVIDERS = ("openai", "openai_compatible")


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    # LLM provider selection: "gemini" (default), "openai" / "openai_compatible", or "anthropic".
    llm_provider: str
    llm_model: str
    llm_fallback_models: tuple[str, ...]
    llm_api_key: str | None = field(repr=False)
    llm_base_url: str | None
    llm_timeout_seconds: float
    llm_max_attempts: int
    llm_effort: str | None
    llm_enable_fallbacks: bool
    llm_cache_size: int
    # HTTP / logging.
    port: int
    log_level: str

    @property
    def llm_configured(self) -> bool:
        keyless_local = self.llm_provider in OPENAI_PROVIDERS and bool(self.llm_base_url)
        return bool(self.llm_api_key) or keyless_local


def _resolve_api_key(provider: str) -> str | None:
    key = os.getenv("LLM_API_KEY")
    if key and key.strip():
        return key.strip()
    # Provider-native fallbacks so standard env names also work.
    native = {
        "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
        "anthropic": ("ANTHROPIC_API_KEY",),
    }.get(provider, ("OPENAI_API_KEY",))
    for name in native:
        key = os.getenv(name)
        if key and key.strip():
            return key.strip()
    return None


DEFAULT_MODELS = {
    "gemini": "gemini-3.5-flash-lite",
    "openai": "gpt-5.6-terra",
    "openai_compatible": "gpt-5.6-terra",
    "anthropic": "claude-opus-5",
}


# Tried in order when the primary model is overloaded, out of quota, or timing out.
DEFAULT_FALLBACK_MODELS = {
    "gemini": ("gemini-3.1-flash-lite", "gemini-3.5-flash"),
}


def _env_list(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw = os.getenv(name)
    if raw is None:
        return default
    return tuple(item.strip() for item in raw.split(",") if item.strip())


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    provider = _env_str("LLM_PROVIDER", "gemini").lower()
    effort = _env_str("LLM_EFFORT", "low").lower()
    return Settings(
        llm_provider=provider,
        llm_model=_env_str("LLM_MODEL", DEFAULT_MODELS.get(provider, DEFAULT_MODELS["gemini"])),
        llm_fallback_models=_env_list("LLM_FALLBACK_MODELS", DEFAULT_FALLBACK_MODELS.get(provider, ())),
        llm_api_key=_resolve_api_key(provider),
        llm_base_url=os.getenv("LLM_BASE_URL") or None,
        llm_timeout_seconds=_env_float("LLM_TIMEOUT_SECONDS", 8.0),
        llm_max_attempts=max(1, _env_int("LLM_MAX_ATTEMPTS", 3)),
        llm_effort=None if effort in {"", "none", "off"} else effort,
        llm_enable_fallbacks=_env_bool("LLM_ENABLE_FALLBACKS", True),
        llm_cache_size=max(0, _env_int("LLM_CACHE_SIZE", 256)),
        port=_env_int("PORT", 8000),
        log_level=_env_str("LOG_LEVEL", "INFO").upper(),
    )
