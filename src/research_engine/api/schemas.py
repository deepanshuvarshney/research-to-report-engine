"""Request and response models for the job API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


class CreateJobRequest(BaseModel):
    question: str = Field(min_length=12, max_length=800)

    @field_validator("question")
    @classmethod
    def normalize_question(cls, value: str) -> str:
        if "\x00" in value:
            raise ValueError("question contains an invalid character")
        text = " ".join(value.split())
        if len(text) < 12:
            raise ValueError("question is too short")
        return text


class JobResponse(BaseModel):
    id: str
    question: str
    status: str
    error: str | None = None
    provider: str = ""
    model: str = ""
    support_score: float | None = None
    stats: dict[str, Any] = Field(default_factory=dict)
    report_md: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    report_ready: bool = False


class MetaResponse(BaseModel):
    name: str
    version: str
    llm_provider: str
    llm_model: str
    embedding_provider: str
    embedding_model: str
    max_revision_loops: int
    demo_mode: bool = False
