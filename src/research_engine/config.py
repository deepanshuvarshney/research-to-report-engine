"""Typed runtime configuration. Every knob is an environment variable."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

LLMProviderName = Literal["groq", "gemini", "ollama", "mock"]
EmbeddingProviderName = Literal["fastembed", "hash"]

_DEFAULT_MODELS: dict[str, str] = {
    "groq": "llama-3.1-8b-instant",
    "gemini": "gemini-2.0-flash",
    "ollama": "llama3.2",
    "mock": "mock-extractive",
}


class Settings(BaseSettings):
    """Process configuration loaded from the environment and an optional .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        env_ignore_empty=True,
    )

    app_env: str = "development"
    log_level: str = "INFO"
    log_json: bool = True

    database_url: str = "postgresql+asyncpg://vellum:vellum@localhost:5432/vellum"
    redis_url: str = "redis://localhost:6379/0"

    llm_provider: LLMProviderName = "mock"
    llm_model: str = ""
    groq_api_key: str = Field(default="", repr=False)
    gemini_api_key: str = Field(default="", repr=False)
    ollama_base_url: str = "http://localhost:11434"
    llm_timeout_seconds: float = 60.0
    llm_max_retries: int = 3
    llm_temperature: float = 0.2
    llm_max_tokens: int = 4096

    embedding_provider: EmbeddingProviderName = "hash"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384

    max_subquestions: int = 3
    max_results_per_provider: int = 3
    max_sources_total: int = 12
    max_sources_for_writer: int = 8
    max_revision_loops: int = 2
    chunk_size: int = 800
    chunk_overlap: int = 120
    retrieval_k: int = 6
    source_store_chars: int = 4000
    source_prompt_chars: int = 1600

    search_rate_limit_rps: float = 1.0
    search_timeout_seconds: float = 20.0
    cache_ttl_seconds: int = 86_400
    http_user_agent: str = (
        "VellumResearchBot/0.1 (educational research-to-report; "
        "+https://github.com/deepanshuvarshney/research-to-report-engine)"
    )
    semantic_scholar_api_key: str = Field(default="", repr=False)

    job_timeout_seconds: int = 600
    worker_max_jobs: int = 2
    api_rate_limit_per_minute: int = 60
    job_rate_limit_per_minute: int = 6

    otel_enabled: bool = False
    otel_exporter_otlp_endpoint: str = ""
    otel_service_name: str = "vellum"

    frontend_dir: str = ""

    @field_validator("embedding_dim")
    @classmethod
    def embedding_dim_matches_schema(cls, value: int) -> int:
        if value != 384:
            raise ValueError("EMBEDDING_DIM must be 384 to match the pgvector column")
        return value

    @field_validator("max_revision_loops")
    @classmethod
    def revision_bounds(cls, value: int) -> int:
        if value < 0 or value > 4:
            raise ValueError("MAX_REVISION_LOOPS must be between 0 and 4")
        return value

    @field_validator("search_rate_limit_rps")
    @classmethod
    def positive_rate(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("SEARCH_RATE_LIMIT_RPS must be greater than 0")
        return value

    @field_validator("max_subquestions", "max_results_per_provider", "max_sources_total")
    @classmethod
    def positive_int(cls, value: int) -> int:
        if value < 1:
            raise ValueError("limits must be at least 1")
        return value

    def resolved_model(self) -> str:
        if self.llm_model.strip():
            return self.llm_model.strip()
        return _DEFAULT_MODELS[self.llm_provider]

    def require_llm_credentials(self) -> None:
        if self.llm_provider == "groq" and not self.groq_api_key:
            raise RuntimeError("GROQ_API_KEY is required when LLM_PROVIDER=groq")
        if self.llm_provider == "gemini" and not self.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY is required when LLM_PROVIDER=gemini")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
