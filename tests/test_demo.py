"""Public demo quotas. No network and no Postgres."""

from __future__ import annotations

import uuid
from typing import Any

import pytest

from research_engine.jobs.demo import DemoLimited, release_demo_lock
from research_engine.jobs.service import JobService, QueueUnavailable
from research_engine.search.cache import MemoryCache
from tests.support import make_settings


class FakeRedis:
    def __init__(self) -> None:
        self.values: dict[str, Any] = {}

    async def incr(self, key: str) -> int:
        self.values[key] = int(self.values.get(key, 0)) + 1
        return int(self.values[key])

    async def decr(self, key: str) -> int:
        self.values[key] = int(self.values.get(key, 0)) - 1
        return int(self.values[key])

    async def expire(self, key: str, seconds: int) -> bool:
        del key, seconds
        return True

    async def get(self, key: str) -> str | None:
        value = self.values.get(key)
        return None if value is None else str(value)

    async def set(self, key: str, value: str, nx: bool = False, ex: int | None = None) -> bool:
        del ex
        if nx and key in self.values:
            return False
        self.values[key] = value
        return True

    async def delete(self, key: str) -> int:
        return 1 if self.values.pop(key, None) is not None else 0

    async def publish(self, channel: str, message: str) -> int:
        del channel, message
        return 1


class Repo:
    def __init__(self) -> None:
        self.jobs: dict[str, dict[str, Any]] = {}
        self.events: list[dict[str, Any]] = []

    async def create_job(self, *, question: str, provider: str, model: str) -> dict[str, Any]:
        job_id = str(uuid.uuid4())
        job = {
            "id": job_id,
            "question": question,
            "status": "queued",
            "error": None,
            "provider": provider,
            "model": model,
            "support_score": None,
            "stats": {},
            "report_md": None,
            "created_at": None,
            "updated_at": None,
            "report_ready": False,
        }
        self.jobs[job_id] = job
        return job

    async def get_job(self, job_id: str) -> dict[str, Any] | None:
        return self.jobs.get(job_id)

    async def list_jobs(self, limit: int) -> list[dict[str, Any]]:
        return list(self.jobs.values())[:limit]

    async def update_job(self, job_id: str, **fields: Any) -> None:
        self.jobs[job_id].update(fields)

    async def add_event(
        self,
        job_id: str,
        *,
        agent: str,
        kind: str,
        message: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        event = {
            "job_id": job_id,
            "seq": 1 + len(self.events),
            "agent": agent,
            "kind": kind,
            "message": message,
            "payload": payload or {},
            "created_at": None,
        }
        self.events.append(event)
        return event

    async def list_events(self, job_id: str) -> list[dict[str, Any]]:
        return [event for event in self.events if event["job_id"] == job_id]


class Queue:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.ids: list[str] = []

    async def enqueue_research(self, job_id: str) -> None:
        if self.fail:
            raise ConnectionError("redis down")
        self.ids.append(job_id)


def _service(redis: FakeRedis, queue: Queue, **overrides: Any) -> JobService:
    params: dict[str, Any] = {
        "demo_mode": True,
        "demo_max_jobs_per_hour": 4,
        "demo_max_jobs_per_ip_per_hour": 4,
    }
    params.update(overrides)
    settings = make_settings(**params)
    return JobService(
        settings=settings,
        repo=Repo(),
        queue=queue,
        cache=MemoryCache(),
        redis=redis,  # type: ignore[arg-type]
    )


def test_demo_mode_clamps_public_caps() -> None:
    settings = make_settings(
        demo_mode=True,
        worker_max_jobs=4,
        max_sources_total=12,
        max_revision_loops=2,
        fulltext_max_pages=30,
        fulltext_max_pdfs=4,
        job_rate_limit_per_minute=6,
        demo_max_jobs_per_hour=40,
    )
    assert settings.worker_max_jobs == 1
    assert settings.max_sources_total == 8
    assert settings.max_sources_for_writer <= 4
    assert settings.max_revision_loops == 1
    assert settings.fulltext_max_pages == 16
    assert settings.fulltext_max_pdfs == 2
    assert settings.job_rate_limit_per_minute == 2
    assert settings.demo_max_jobs_per_hour == 12
    plain = make_settings(max_sources_total=12, worker_max_jobs=2)
    assert plain.demo_mode is False
    assert plain.max_sources_total == 12
    assert plain.worker_max_jobs == 2


@pytest.mark.asyncio
async def test_demo_lock_hourly_cap_and_refund() -> None:
    redis = FakeRedis()
    queue = Queue()
    service = _service(redis, queue)
    question = "How does retrieval-augmented generation reduce hallucinations?"
    first = await service.submit(question, None, "203.0.113.4")
    with pytest.raises(DemoLimited, match="already running"):
        await service.submit(question, None, "203.0.113.5")
    await release_demo_lock(redis, first["id"])  # type: ignore[arg-type]
    second = await service.submit(question, "same-key", "203.0.113.4")
    replay = await service.submit(question, "same-key", "203.0.113.4")
    assert replay["id"] == second["id"]
    await release_demo_lock(redis, second["id"])  # type: ignore[arg-type]

    failing = _service(FakeRedis(), Queue(fail=True), demo_max_jobs_per_hour=1)
    with pytest.raises(QueueUnavailable):
        await failing.submit(question, None, "198.51.100.8")
    failing.queue = Queue()
    recovered = await failing.submit(question, None, "198.51.100.8")
    assert recovered["status"] == "queued"

    capped = _service(
        FakeRedis(),
        Queue(),
        demo_max_jobs_per_hour=1,
        demo_max_jobs_per_ip_per_hour=4,
    )
    only = await capped.submit(question, None, "198.51.100.9")
    await release_demo_lock(capped.redis, only["id"])  # type: ignore[arg-type]
    with pytest.raises(DemoLimited, match="hourly"):
        await capped.submit(question, None, "198.51.100.10")
