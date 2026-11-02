"""arq worker entrypoint: `arq research_engine.jobs.worker.WorkerSettings`."""

from __future__ import annotations

import os
from typing import Any, ClassVar

import structlog
from arq.connections import RedisSettings
from redis.asyncio import Redis

from research_engine.config import get_settings
from research_engine.db.repository import PostgresJobRepository
from research_engine.db.session import (
    create_db_engine,
    create_session_factory,
    ensure_schema,
    wait_for_postgres,
)
from research_engine.jobs.runner import build_and_run
from research_engine.logging import configure_logging
from research_engine.observability.tracing import setup_tracing

log = structlog.get_logger(__name__)


async def startup(ctx: dict[str, Any]) -> None:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)
    setup_tracing(settings)
    engine = create_db_engine(settings.database_url)
    await wait_for_postgres(engine)
    await ensure_schema(engine)
    redis = Redis.from_url(settings.redis_url, decode_responses=True)
    await redis.ping()
    repo = PostgresJobRepository(create_session_factory(engine))
    abandoned = await repo.fail_abandoned_jobs(settings.job_timeout_seconds)
    if abandoned:
        log.warning("abandoned_jobs_failed", count=abandoned)
    ctx["settings"] = settings
    ctx["engine"] = engine
    ctx["redis"] = redis
    ctx["session_factory"] = create_session_factory(engine)
    ctx["repo"] = repo
    log.info(
        "worker_started",
        provider=settings.llm_provider,
        model=settings.resolved_model(),
        embeddings=settings.embedding_provider,
    )


async def shutdown(ctx: dict[str, Any]) -> None:
    redis = ctx.get("redis")
    engine = ctx.get("engine")
    if redis is not None:
        await redis.aclose()
    if engine is not None:
        await engine.dispose()


async def run_research(ctx: dict[str, Any], job_id: str) -> None:
    await build_and_run(
        job_id,
        settings=ctx["settings"],
        repo=ctx["repo"],
        redis=ctx["redis"],
        session_factory=ctx["session_factory"],
    )


class WorkerSettings:
    functions: ClassVar[list[Any]] = [run_research]
    on_startup = startup
    on_shutdown = shutdown
    redis_settings = RedisSettings.from_dsn(os.environ.get("REDIS_URL", "redis://localhost:6379/0"))
    max_jobs = int(os.environ.get("WORKER_MAX_JOBS", "2"))
    job_timeout = int(os.environ.get("JOB_TIMEOUT_SECONDS", "600"))
    max_tries = 1
    keep_result = 3600
    health_check_interval = 30
