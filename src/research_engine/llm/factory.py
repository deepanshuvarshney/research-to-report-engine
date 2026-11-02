"""Build the configured provider. Construction does not open a network connection."""

from __future__ import annotations

from research_engine.config import Settings
from research_engine.llm.base import LLMProvider
from research_engine.llm.gemini import GeminiLLM
from research_engine.llm.groq import GroqLLM
from research_engine.llm.mock import MockLLM
from research_engine.llm.ollama import OllamaLLM


def build_llm(settings: Settings) -> LLMProvider:
    settings.require_llm_credentials()
    model = settings.resolved_model()
    timeout = settings.llm_timeout_seconds
    retries = settings.llm_max_retries
    if settings.llm_provider == "mock":
        return MockLLM(model=model)
    if settings.llm_provider == "groq":
        return GroqLLM(
            api_key=settings.groq_api_key,
            model=model,
            timeout=timeout,
            max_retries=retries,
        )
    if settings.llm_provider == "gemini":
        return GeminiLLM(
            api_key=settings.gemini_api_key,
            model=model,
            timeout=timeout,
            max_retries=retries,
        )
    if settings.llm_provider == "ollama":
        return OllamaLLM(
            base_url=settings.ollama_base_url,
            model=model,
            timeout=timeout,
            max_retries=retries,
        )
    raise RuntimeError(f"Unknown LLM provider {settings.llm_provider}")
