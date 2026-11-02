"""Create jobs, stream their events, and hand report bytes to the API."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any, Protocol

from redis.asyncio import Redis

from research_engine.agents.render import render_html, render_pdf
from research_engine.config import Settings
from research_engine.jobs.demo import (
    DemoLimited,
    DemoSlot,
    adopt_demo_lock,
    begin_demo_job,
    cancel_demo_job,
)
from research_engine.jobs.events import channel_for, publish_event
from research_engine.jobs.queue import JobQueue
from research_engine.search.cache import KeyValueCache

TERMINAL_STATUSES = {"completed", "failed"}
TERMINAL_KINDS = {"job.completed", "job.failed"}


class QueueUnavailable(Exception):
    """Redis or arq rejected the enqueue."""


class JobStore(Protocol):
    async def create_job(self, *, question: str, provider: str, model: str) -> dict[str, Any]: ...

    async def get_job(self, job_id: str) -> dict[str, Any] | None: ...

    async def list_jobs(self, limit: int) -> list[dict[str, Any]]: ...

    async def update_job(self, job_id: str, **fields: Any) -> None: ...

    async def add_event(
        self,
        job_id: str,
        *,
        agent: str,
        kind: str,
        message: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]: ...

    async def list_events(self, job_id: str) -> list[dict[str, Any]]: ...


class JobService:
    def __init__(
        self,
        *,
        settings: Settings,
        repo: JobStore,
        queue: JobQueue,
        cache: KeyValueCache,
        redis: Redis | None,
    ) -> None:
        self.settings = settings
        self.repo = repo
        self.queue = queue
        self.cache = cache
        self.redis = redis

    async def submit(
        self,
        question: str,
        idempotency_key: str | None,
        client_ip: str = "",
    ) -> dict[str, Any]:
        self.settings.require_llm_credentials()
        if idempotency_key:
            existing_id = await self.cache.get(_idem_key(idempotency_key))
            if existing_id:
                existing = await self.repo.get_job(existing_id)
                if existing is not None:
                    return existing
        slot: DemoSlot | None = None
        if self.settings.demo_mode:
            if self.redis is None:
                raise DemoLimited("Demo mode requires Redis.")
            slot = await begin_demo_job(self.redis, self.settings, client_ip)
        try:
            job = await self._create_and_enqueue(question, idempotency_key, slot)
        except Exception:
            if slot is not None and self.redis is not None:
                await cancel_demo_job(self.redis, slot)
            raise
        if slot is not None:
            slot.settled = True
        return job

    async def _create_and_enqueue(
        self,
        question: str,
        idempotency_key: str | None,
        slot: DemoSlot | None,
    ) -> dict[str, Any]:
        job = await self.repo.create_job(
            question=question,
            provider=self.settings.llm_provider,
            model=self.settings.resolved_model(),
        )
        if slot is not None and self.redis is not None:
            await adopt_demo_lock(self.redis, slot, job["id"], self.settings.job_timeout_seconds)
        await publish_event(
            self.repo,
            self.redis,
            job["id"],
            "job",
            "job.queued",
            "Queued for a worker",
        )
        if idempotency_key:
            await self.cache.set(_idem_key(idempotency_key), job["id"], ttl=3600)
        try:
            await self.queue.enqueue_research(job["id"])
        except Exception as exc:
            if idempotency_key:
                await self.cache.delete(_idem_key(idempotency_key))
            await self.repo.update_job(job["id"], status="failed", error=f"Queue error: {exc}")
            await publish_event(
                self.repo,
                self.redis,
                job["id"],
                "job",
                "job.failed",
                "The job queue is unavailable",
            )
            raise QueueUnavailable(str(exc)) from exc
        return job

    async def get(self, job_id: str) -> dict[str, Any] | None:
        return await self.repo.get_job(job_id)

    async def list_recent(self, limit: int) -> list[dict[str, Any]]:
        return await self.repo.list_jobs(limit)

    async def list_events(self, job_id: str) -> list[dict[str, Any]]:
        return await self.repo.list_events(job_id)

    async def iter_events(self, job_id: str) -> AsyncIterator[dict[str, Any]]:
        """Replay stored events, then follow Redis until the job reaches a terminal state."""

        job = await self.repo.get_job(job_id)
        if job is None:
            raise KeyError(job_id)

        pubsub: Any = None
        if self.redis is not None and job["status"] not in TERMINAL_STATUSES:
            pubsub = self.redis.pubsub()
            await pubsub.subscribe(channel_for(job_id))
        try:
            last = 0
            for event in await self.repo.list_events(job_id):
                last = max(last, int(event["seq"]))
                yield event
            current = await self.repo.get_job(job_id)
            if current is None or current["status"] in TERMINAL_STATUSES or pubsub is None:
                return
            while True:
                message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=15.0)
                if message is None:
                    current = await self.repo.get_job(job_id)
                    if current is not None and current["status"] in TERMINAL_STATUSES:
                        for event in await self.repo.list_events(job_id):
                            if int(event["seq"]) > last:
                                last = int(event["seq"])
                                yield event
                        return
                    yield {
                        "kind": "keepalive",
                        "seq": 0,
                        "agent": "system",
                        "message": "",
                        "payload": {},
                    }
                    continue
                if message.get("type") != "message":
                    continue
                data = message.get("data")
                if isinstance(data, bytes):
                    data = data.decode()
                event = json.loads(data)
                seq = int(event.get("seq") or 0)
                if seq <= last:
                    continue
                last = seq
                yield event
                if event.get("kind") in TERMINAL_KINDS:
                    return
        finally:
            if pubsub is not None:
                await pubsub.unsubscribe(channel_for(job_id))
                await pubsub.aclose()

    async def report(self, job_id: str, fmt: str) -> tuple[bytes, str, str]:
        job = await self.repo.get_job(job_id)
        if job is None or not job.get("report_md"):
            raise KeyError(job_id)
        markdown = str(job["report_md"])
        if fmt == "md":
            return markdown.encode(), "text/markdown; charset=utf-8", "report.md"
        if fmt == "html":
            html = str(job.get("report_html") or render_html(markdown))
            return html.encode(), "text/html; charset=utf-8", "report.html"
        if fmt == "pdf":
            return render_pdf(markdown), "application/pdf", "report.pdf"
        raise ValueError(f"unsupported format {fmt}")


def _idem_key(key: str) -> str:
    return f"vellum:idem:{key}"
