"""Run one research job off the request path and persist the report."""

from __future__ import annotations

import time
from typing import Any

import structlog
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from research_engine.agents.graph import GraphDeps, run_research_graph
from research_engine.config import Settings
from research_engine.db.repository import PostgresJobRepository
from research_engine.jobs.demo import release_demo_lock
from research_engine.jobs.events import publish_event
from research_engine.llm.base import LLMProvider
from research_engine.llm.factory import build_llm
from research_engine.rag.embeddings import Embedder, build_embedder
from research_engine.rag.store import PgVectorStore, VectorStore
from research_engine.search.cache import RedisCache
from research_engine.search.limiter import RedisRateLimiter
from research_engine.search.service import SearchService

log = structlog.get_logger(__name__)

_SEARCH_PROVIDERS = (
    "arxiv",
    "semantic_scholar",
    "wikipedia",
    "web",
    "arxiv-pdf",
    "unpaywall",
    "oa-pdf",
)


def build_search(settings: Settings, redis: Redis) -> SearchService:
    limiters = {
        name: RedisRateLimiter(redis, f"vellum:rlsearch:{name}", settings.search_rate_limit_rps)
        for name in _SEARCH_PROVIDERS
    }
    return SearchService(settings=settings, cache=RedisCache(redis), limiters=limiters)


async def execute_job(
    job_id: str,
    *,
    settings: Settings,
    repo: PostgresJobRepository,
    redis: Redis,
    llm: LLMProvider,
    search: SearchService,
    embeddings: Embedder,
    vectors: VectorStore,
) -> None:
    structlog.contextvars.bind_contextvars(job_id=job_id)
    started = time.perf_counter()
    try:
        job = await repo.get_job(job_id)
        if job is None:
            log.warning("job_missing")
            return
        if job["status"] == "completed":
            return
        await repo.update_job(job_id, status="running", error=None)
        await publish_event(repo, redis, job_id, "job", "job.started", "Research job started")

        async def emit(
            agent: str,
            kind: str,
            message: str,
            payload: dict[str, Any] | None = None,
        ) -> None:
            await publish_event(repo, redis, job_id, agent, kind, message, payload or {})

        deps = GraphDeps(
            settings=settings,
            llm=llm,
            search=search,
            vectors=vectors,
            embeddings=embeddings,
            emit=emit,
        )
        result = await run_research_graph(job["question"], job_id, deps)
        selected = list(result.get("selected_sources") or [])
        await vectors.prune(
            job_id, [str(source.get("uid")) for source in selected if source.get("uid")]
        )
        await repo.replace_sources(job_id, selected)
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        verdicts = list(result.get("verdicts") or [])
        supported = sum(1 for verdict in verdicts if verdict.get("supported"))
        stats = {
            "sub_questions": list(result.get("sub_questions") or []),
            "plan_rationale": result.get("plan_rationale") or "",
            "source_count": len(selected),
            "fulltext_count": sum(1 for source in selected if source.get("text_origin") == "pdf"),
            "chunk_count": int(result.get("chunk_count") or 0),
            "revisions": int(result.get("revision") or 0),
            "claim_count": len(verdicts),
            "supported_count": supported,
            "search_notes": list(result.get("search_notes") or []),
            "elapsed_ms": elapsed_ms,
        }
        score = result.get("support_score")
        await repo.update_job(
            job_id,
            status="completed",
            error=None,
            report_md=result.get("report_md") or "",
            report_html=result.get("report_html") or "",
            support_score=score,
            stats=stats,
        )
        await publish_event(
            repo,
            redis,
            job_id,
            "report",
            "job.completed",
            "Report ready",
            {"support_score": score, "elapsed_ms": elapsed_ms},
        )
        log.info("job_completed", elapsed_ms=elapsed_ms, support_score=score)
    except Exception as exc:
        log.exception("job_failed")
        message = str(exc).strip() or exc.__class__.__name__
        try:
            await repo.update_job(job_id, status="failed", error=message[:2000])
            await publish_event(repo, redis, job_id, "job", "job.failed", message[:500])
        except Exception:
            log.exception("job_failure_record_failed")
    finally:
        if settings.demo_mode:
            try:
                await release_demo_lock(redis, job_id)
            except Exception:
                log.warning("demo_lock_release_failed")
        structlog.contextvars.unbind_contextvars("job_id")


async def build_and_run(
    job_id: str,
    *,
    settings: Settings,
    repo: PostgresJobRepository,
    redis: Redis,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    llm = build_llm(settings)
    search = build_search(settings, redis)
    embeddings = build_embedder(
        settings.embedding_provider,
        settings.embedding_model,
        settings.embedding_dim,
    )
    vectors = PgVectorStore(session_factory)
    try:
        await execute_job(
            job_id,
            settings=settings,
            repo=repo,
            redis=redis,
            llm=llm,
            search=search,
            embeddings=embeddings,
            vectors=vectors,
        )
    finally:
        await llm.aclose()
        await search.aclose()
