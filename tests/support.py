"""Helpers shared by tests. Kept out of conftest so imports do not depend on how pytest was launched."""

from __future__ import annotations

from typing import Any

from research_engine.config import Settings


def make_settings(**overrides: Any) -> Settings:
    """Build Settings without reading a developer .env file."""

    params: dict[str, Any] = {
        "llm_provider": "mock",
        "embedding_provider": "hash",
        "search_rate_limit_rps": 1000,
        "max_revision_loops": 2,
    }
    params.update(overrides)
    return Settings(_env_file=None, **params)
