"""FastAPI application: job API, health checks, and the static research UI."""

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import structlog
from arq.connections import RedisSettings, create_pool
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from redis.asyncio import Redis
from sqlalchemy import text

from research_engine import __version__
from research_engine.api.routes import router
from research_engine.config import Settings, get_settings
from research_engine.db.repository import PostgresJobRepository
from research_engine.db.session import (
    create_db_engine,
    create_session_factory,
    ensure_schema,
    wait_for_postgres,
)
from research_engine.jobs.inline import InlineJobQueue
from research_engine.jobs.queue import ArqJobQueue
from research_engine.jobs.service import JobService
from research_engine.logging import configure_logging
from research_engine.observability.tracing import setup_tracing
from research_engine.search.cache import RedisCache

log = structlog.get_logger(__name__)


def resolve_frontend(settings: Settings) -> Path | None:
    if settings.frontend_dir:
        configured = Path(settings.frontend_dir)
        return configured if configured.is_dir() else None
    candidate = Path(__file__).resolve().parents[2] / "frontend"
    return candidate if candidate.is_dir() else None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = app.state.settings
    configure_logging(settings.log_level, settings.log_json)
    setup_tracing(settings)
    if app.state.job_service is not None:
        yield
        return

    engine = create_db_engine(settings.database_url)
    await wait_for_postgres(engine)
    await ensure_schema(engine)
    redis = Redis.from_url(settings.redis_url, decode_responses=True)
    await redis.ping()
    session_factory = create_session_factory(engine)
    repo = PostgresJobRepository(session_factory)
    pool = None
    if settings.job_executor == "inline":
        queue: ArqJobQueue | InlineJobQueue = InlineJobQueue(
            settings=settings,
            repo=repo,
            redis=redis,
            session_factory=session_factory,
        )
    else:
        pool = await create_pool(RedisSettings.from_dsn(settings.redis_url))
        queue = ArqJobQueue(pool)
    service = JobService(
        settings=settings,
        repo=repo,
        queue=queue,
        cache=RedisCache(redis),
        redis=redis,
    )
    app.state.engine = engine
    app.state.redis = redis
    app.state.arq_pool = pool
    app.state.job_queue = queue
    app.state.job_service = service
    log.info(
        "api_started",
        provider=settings.llm_provider,
        model=settings.resolved_model(),
        embeddings=settings.embedding_provider,
    )
    try:
        yield
    finally:
        job_queue = getattr(app.state, "job_queue", None)
        if isinstance(job_queue, InlineJobQueue):
            await job_queue.aclose()
        if pool is not None:
            await _close_pool(pool)
        await redis.aclose()
        await engine.dispose()


async def _close_pool(pool: object) -> None:
    close = getattr(pool, "close", None) or getattr(pool, "aclose", None)
    if close is None:
        return
    result = close()
    if hasattr(result, "__await__"):
        await result


def create_app(
    settings: Settings | None = None,
    job_service: JobService | None = None,
) -> FastAPI:
    app = FastAPI(
        title="Vellum",
        version=__version__,
        summary="Multi-agent research-to-report engine",
        lifespan=lifespan,
    )
    app.state.settings = settings or get_settings()
    app.state.job_service = job_service
    app.state.redis = None
    app.state.engine = None

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(RequestContextMiddleware)
    app.include_router(router)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/ready")
    async def ready() -> JSONResponse:
        if app.state.job_service is not None and app.state.engine is None:
            return JSONResponse({"status": "ready", "mode": "injected"})
        engine = app.state.engine
        redis = app.state.redis
        try:
            if engine is None or redis is None:
                raise RuntimeError("dependencies are not initialized")
            async with engine.connect() as connection:
                await connection.execute(text("SELECT 1"))
            await redis.ping()
        except Exception as exc:
            return JSONResponse(
                {"status": "not_ready", "error": str(exc)[:200]},
                status_code=503,
            )
        return JSONResponse({"status": "ready"})

    frontend = resolve_frontend(app.state.settings)
    if frontend is not None:
        app.mount("/", StaticFiles(directory=frontend, html=True), name="frontend")
    return app


class RequestContextMiddleware:
    """Pure ASGI middleware so server-sent events are not buffered."""

    def __init__(self, app: Callable[..., Awaitable[None]]) -> None:
        self.app = app

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        request_id = _request_id(scope)
        structlog.contextvars.bind_contextvars(request_id=request_id)
        started = time.perf_counter()
        path = str(scope.get("path", ""))

        async def send_with_id(message: dict[str, Any]) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers") or [])
                headers.append((b"x-request-id", request_id.encode()))
                message["headers"] = headers
            await send(message)

        try:
            await self.app(scope, receive, send_with_id)
        finally:
            if path not in {"/health", "/ready"}:
                log.info(
                    "request",
                    method=scope.get("method", ""),
                    path=path,
                    duration_ms=round((time.perf_counter() - started) * 1000, 2),
                )
            structlog.contextvars.unbind_contextvars("request_id")


def _request_id(scope: dict[str, Any]) -> str:
    for key, value in scope.get("headers") or []:
        if (
            isinstance(key, bytes)
            and key.lower() == b"x-request-id"
            and isinstance(value, bytes)
            and value
        ):
            return value.decode()[:64]
    return uuid.uuid4().hex[:12]


app = create_app()
