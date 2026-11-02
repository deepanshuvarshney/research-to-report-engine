"""Engine, session factory, and idempotent schema apply."""

from __future__ import annotations

import asyncio
from importlib.resources import files

import structlog
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

log = structlog.get_logger(__name__)


def split_sql(sql: str) -> list[str]:
    return [part.strip() for part in sql.split(";") if part.strip()]


def create_db_engine(url: str) -> AsyncEngine:
    if not url.startswith("postgresql+asyncpg://"):
        raise RuntimeError("DATABASE_URL must start with postgresql+asyncpg://")
    return create_async_engine(url, pool_pre_ping=True, pool_size=5, max_overflow=5)


def create_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False)


def load_schema_sql() -> str:
    return files("research_engine.db").joinpath("schema.sql").read_text(encoding="utf-8")


async def ensure_schema(engine: AsyncEngine) -> None:
    statements = split_sql(load_schema_sql())
    async with engine.begin() as connection:
        # Serialize startup when the API and the worker boot together.
        await connection.execute(text("SELECT pg_advisory_xact_lock(84261001)"))
        for statement in statements:
            await connection.execute(text(statement))
    log.info("schema_ready", statements=len(statements))


async def wait_for_postgres(engine: AsyncEngine, attempts: int = 20) -> None:
    last: Exception | None = None
    for _ in range(attempts):
        try:
            async with engine.connect() as connection:
                await connection.execute(text("SELECT 1"))
            return
        except Exception as exc:
            last = exc
            await asyncio.sleep(1)
    raise RuntimeError(f"database unavailable: {last}")
