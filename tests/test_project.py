from __future__ import annotations

from pathlib import Path

import pytest

from research_engine.config import Settings
from research_engine.db.session import load_schema_sql, split_sql
from research_engine.eval.harness import evaluate_cases, load_cases, main


def test_schema_covers_jobs_and_vectors() -> None:
    statements = split_sql(load_schema_sql())
    joined = "\n".join(statements)
    assert statements[0].startswith("CREATE EXTENSION")
    for table in ("jobs", "job_events", "sources", "chunks"):
        assert f"CREATE TABLE IF NOT EXISTS {table}" in joined
    assert "vector(384)" in joined
    assert "hnsw" in joined


def test_settings_reject_bad_embedding_dim_and_missing_keys() -> None:
    with pytest.raises(ValueError):
        Settings(_env_file=None, embedding_dim=768)
    settings = Settings(_env_file=None, llm_provider="groq", groq_api_key="")
    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        settings.require_llm_credentials()
    gemini = Settings(_env_file=None, llm_provider="gemini")
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        gemini.require_llm_credentials()
    mock = Settings(_env_file=None, llm_provider="mock", llm_model="")
    mock.require_llm_credentials()
    assert mock.resolved_model() == "mock-extractive"
    assert Settings(_env_file=None, llm_provider="ollama").resolved_model() == "llama3.2"


def test_eval_harness_scores_every_fixture() -> None:
    report = evaluate_cases(load_cases())
    assert report["total"] >= 5
    assert report["accuracy"] == 1.0
    assert main([]) == 0


def test_compose_and_ci_exist() -> None:
    root = Path(__file__).resolve().parents[1]
    compose = (root / "docker-compose.yml").read_text()
    for name in ("postgres", "redis", "api", "worker", "pgvector/pgvector"):
        assert name in compose
    dockerfile = (root / "Dockerfile").read_text()
    assert "research_engine.main:app" in dockerfile
    workflow = (root / ".github" / "workflows" / "ci.yml").read_text()
    assert "ruff check" in workflow
    assert "mypy" in workflow
    assert "pytest" in workflow
    assert (root / ".env.example").is_file()
