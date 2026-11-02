"""arq enqueue wrapper. The API never runs the graph on the request path."""

from __future__ import annotations

from typing import Any, Protocol

from arq.connections import ArqRedis


class JobQueue(Protocol):
    async def enqueue_research(self, job_id: str) -> None: ...


class ArqJobQueue:
    def __init__(self, pool: ArqRedis) -> None:
        self._pool = pool

    async def enqueue_research(self, job_id: str) -> None:
        job: Any = await self._pool.enqueue_job(
            "run_research",
            job_id,
            _job_id=f"research-{job_id}",
        )
        if job is None:
            raise RuntimeError(f"research job {job_id} is already queued")
