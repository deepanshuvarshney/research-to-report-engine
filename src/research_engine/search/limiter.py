"""Polite pacing. In-process for tests, Redis so every worker shares one budget."""

from __future__ import annotations

import asyncio
import time
from typing import Protocol

from redis.asyncio import Redis


class RateLimiter(Protocol):
    async def acquire(self) -> None: ...


class AsyncRateLimiter:
    """Serialize callers so a host sees at most `rps` requests."""

    def __init__(self, rps: float) -> None:
        self._min_interval = 1.0 / rps
        self._lock = asyncio.Lock()
        self._next = 0.0

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            wait = self._next - now
            if wait > 0:
                await asyncio.sleep(wait)
                now = time.monotonic()
            self._next = now + self._min_interval


class RedisRateLimiter:
    """A coarse global lock: one request per interval per provider, across workers."""

    def __init__(self, redis: Redis, key: str, rps: float) -> None:
        self._redis = redis
        self._key = key
        self._interval_ms = max(int(1000 / rps), 1)

    async def acquire(self) -> None:
        delay = self._interval_ms / 2000
        while True:
            acquired = await self._redis.set(self._key, "1", nx=True, px=self._interval_ms)
            if acquired:
                return
            await asyncio.sleep(delay)
