"""Run research inside the API process. The public demo has no second container."""

from __future__ import annotations

import asyncio

from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from research_engine.config import Settings
from research_engine.db.repository import PostgresJobRepository
from research_engine.jobs.runner import build_and_run


class InlineJobQueue:
    """Enqueue returns immediately. The graph runs on a background task."""

    def __init__(
        self,
        *,
        settings: Settings,
        repo: PostgresJobRepository,
        redis: Redis,
        session_factory: async_sessionmaker[AsyncSession],
    ) -> None:
        self._settings = settings
        self._repo = repo
        self._redis = redis
        self._session_factory = session_factory
        self._tasks: set[asyncio.Task[None]] = set()
        self._sem = asyncio.Semaphore(max(1, settings.worker_max_jobs))

    async def enqueue_research(self, job_id: str) -> None:
        task = asyncio.create_task(self._run(job_id), name=f"research-{job_id}")
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def aclose(self) -> None:
        if not self._tasks:
            return
        await asyncio.gather(*list(self._tasks), return_exceptions=True)

    async def _run(self, job_id: str) -> None:
        async with self._sem:
            await build_and_run(
                job_id,
                settings=self._settings,
                repo=self._repo,
                redis=self._redis,
                session_factory=self._session_factory,
            )
