"""Typed runtime configuration. Every knob is an environment variable."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

LLMProviderName = Literal["groq", "gemini", "ollama", "mock"]
EmbeddingProviderName = Literal["fastembed", "hash"]
JobExecutorName = Literal["arq", "inline"]

_DEFAULT_MODELS: dict[str, str] = {
    "groq": "openai/gpt-oss-20b",
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
    fulltext_prompt_chars: int = 2200

    search_rate_limit_rps: float = 1.0
    search_timeout_seconds: float = 20.0
    cache_ttl_seconds: int = 86_400
    http_user_agent: str = (
        "VellumResearchBot/0.1 (educational research-to-report; "
        "+https://github.com/deepanshuvarshney/research-to-report-engine)"
    )
    semantic_scholar_api_key: str = Field(default="", repr=False)
    unpaywall_email: str = ""

    fulltext_enabled: bool = True
    fulltext_max_pdfs: int = 4
    fulltext_max_attempts: int = 6
    fulltext_max_pages: int = 20
    fulltext_max_bytes: int = 12_000_000
    fulltext_max_chars: int = 60_000
    fulltext_max_chunks: int = 36
    fulltext_timeout_seconds: float = 25.0
    fulltext_failure_ttl_seconds: int = 3600

    job_timeout_seconds: int = 600
    worker_max_jobs: int = 2
    job_executor: JobExecutorName = "arq"
    api_rate_limit_per_minute: int = 60
    job_rate_limit_per_minute: int = 6

    demo_mode: bool = False
    demo_max_jobs_per_hour: int = 8
    demo_max_jobs_per_ip_per_hour: int = 3

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

    @field_validator("fulltext_max_pdfs", "fulltext_max_attempts", "fulltext_max_chunks")
    @classmethod
    def non_negative_cap(cls, value: int) -> int:
        if value < 0 or value > 40:
            raise ValueError("full-text caps must be between 0 and 40")
        return value

    @field_validator("fulltext_max_pages")
    @classmethod
    def page_cap(cls, value: int) -> int:
        if value < 1 or value > 40:
            raise ValueError("FULLTEXT_MAX_PAGES must be between 1 and 40")
        return value

    @field_validator("fulltext_max_bytes")
    @classmethod
    def byte_cap(cls, value: int) -> int:
        if value < 50_000 or value > 20_000_000:
            raise ValueError("FULLTEXT_MAX_BYTES must be between 50000 and 20000000")
        return value

    @field_validator("fulltext_max_chars", "fulltext_prompt_chars")
    @classmethod
    def char_cap(cls, value: int) -> int:
        if value < 400 or value > 200_000:
            raise ValueError("full-text character caps are out of range")
        return value

    @field_validator("fulltext_timeout_seconds")
    @classmethod
    def fulltext_timeout(cls, value: float) -> float:
        if value < 3 or value > 60:
            raise ValueError("FULLTEXT_TIMEOUT_SECONDS must be between 3 and 60")
        return value

    @field_validator("demo_max_jobs_per_hour", "demo_max_jobs_per_ip_per_hour")
    @classmethod
    def demo_quota(cls, value: int) -> int:
        if value < 1 or value > 100:
            raise ValueError("demo job quotas must be between 1 and 100")
        return value

    @model_validator(mode="after")
    def clamp_public_demo(self) -> Self:
        """Public demo ceilings stay in code so a loose env file cannot lift them."""

        if not self.demo_mode:
            return self
        self.worker_max_jobs = min(self.worker_max_jobs, 1)
        self.max_subquestions = min(self.max_subquestions, 2)
        self.max_results_per_provider = min(self.max_results_per_provider, 2)
        self.max_sources_total = min(self.max_sources_total, 8)
        self.max_sources_for_writer = min(self.max_sources_for_writer, 4, self.max_sources_total)
        self.max_revision_loops = min(self.max_revision_loops, 1)
        self.job_timeout_seconds = min(self.job_timeout_seconds, 300)
        self.job_rate_limit_per_minute = min(self.job_rate_limit_per_minute, 2)
        self.api_rate_limit_per_minute = min(self.api_rate_limit_per_minute, 30)
        self.llm_max_tokens = min(self.llm_max_tokens, 2048)
        self.fulltext_max_pdfs = min(self.fulltext_max_pdfs, 2)
        self.fulltext_max_attempts = min(self.fulltext_max_attempts, 4)
        self.fulltext_max_pages = min(self.fulltext_max_pages, 16)
        self.fulltext_max_bytes = min(self.fulltext_max_bytes, 8_000_000)
        self.fulltext_max_chars = min(self.fulltext_max_chars, 40_000)
        self.fulltext_max_chunks = min(self.fulltext_max_chunks, 24)
        self.fulltext_timeout_seconds = min(self.fulltext_timeout_seconds, 20.0)
        self.demo_max_jobs_per_hour = min(self.demo_max_jobs_per_hour, 12)
        self.demo_max_jobs_per_ip_per_hour = min(self.demo_max_jobs_per_ip_per_hour, 4)
        return self

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
