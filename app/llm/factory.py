"""Build the configured LLM provider and interpreter from Settings."""

from __future__ import annotations

import logging

from app.config import OPENAI_PROVIDERS, Settings
from app.llm.anthropic_provider import AnthropicProvider
from app.llm.base import LLMProvider
from app.llm.interpreter import NoteInterpreter
from app.llm.openai_compatible_provider import OpenAICompatibleProvider

logger = logging.getLogger(__name__)

SUPPORTED_PROVIDERS = (*OPENAI_PROVIDERS, "anthropic")


def build_provider(settings: Settings) -> LLMProvider | None:
    # A self-hosted OpenAI-compatible server (Ollama, vLLM, Docker Model Runner) needs no key.
    keyless_local = settings.llm_provider in OPENAI_PROVIDERS and bool(settings.llm_base_url)
    if not settings.llm_api_key and not keyless_local:
        logger.warning("no LLM API key configured; /optimize-energy will return llm_unavailable")
        return None
    if settings.llm_provider == "anthropic":
        return AnthropicProvider(
            api_key=settings.llm_api_key,
            model=settings.llm_model,
            effort=settings.llm_effort,
            enable_fallbacks=settings.llm_enable_fallbacks,
            base_url=settings.llm_base_url,
        )
    if settings.llm_provider in OPENAI_PROVIDERS:
        return OpenAICompatibleProvider(
            api_key=settings.llm_api_key,
            model=settings.llm_model,
            base_url=settings.llm_base_url,
            effort=settings.llm_effort,
        )
    logger.error("unsupported LLM_PROVIDER=%r (supported: %s)", settings.llm_provider, SUPPORTED_PROVIDERS)
    return None


def build_interpreter(settings: Settings) -> NoteInterpreter | None:
    provider = build_provider(settings)
    if provider is None:
        return None
    return NoteInterpreter(
        provider=provider,
        timeout_seconds=settings.llm_timeout_seconds,
        max_attempts=settings.llm_max_attempts,
        cache_size=settings.llm_cache_size,
    )
