"""Persist a progress event and publish it for SSE subscribers."""

from __future__ import annotations

import json
from typing import Any, Protocol

from redis.asyncio import Redis


class EventStore(Protocol):
    async def add_event(
        self,
        job_id: str,
        *,
        agent: str,
        kind: str,
        message: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]: ...


async def publish_event(
    repo: EventStore,
    redis: Redis | None,
    job_id: str,
    agent: str,
    kind: str,
    message: str,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    event = await repo.add_event(
        job_id,
        agent=agent,
        kind=kind,
        message=message,
        payload=payload or {},
    )
    if redis is not None:
        await redis.publish(channel_for(job_id), json.dumps(event, ensure_ascii=False))
    return event


def channel_for(job_id: str) -> str:
    return f"job:{job_id}"
