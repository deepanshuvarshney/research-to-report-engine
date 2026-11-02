"""Fixed-window limits in Redis. If Redis is down, requests are allowed and a warning is logged."""

from __future__ import annotations

from typing import Protocol

import structlog
from fastapi import HTTPException, Request

log = structlog.get_logger(__name__)


class Counter(Protocol):
    async def incr(self, key: str) -> int: ...

    async def expire(self, key: str, seconds: int) -> object: ...


def client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client is None:
        return "unknown"
    return request.client.host


async def enforce_rate_limit(request: Request, bucket: str, limit: int) -> None:
    if limit <= 0:
        return
    redis = getattr(request.app.state, "redis", None)
    if redis is None:
        return
    key = f"vellum:rl:{bucket}:{client_ip(request)}"
    try:
        count = int(await redis.incr(key))
        if count == 1:
            await redis.expire(key, 60)
    except Exception as exc:
        log.warning("rate_limit_unavailable", error=str(exc)[:160])
        return
    if count > limit:
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again in a minute.")
