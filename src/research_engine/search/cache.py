"""Cache raw search payloads. Redis in the app, a dict in tests."""

from __future__ import annotations

import hashlib
import time
from typing import Protocol

from redis.asyncio import Redis


def cache_key(provider: str, query: str, limit: int) -> str:
    raw = f"{provider}|{limit}|{query.strip().lower()}".encode()
    digest = hashlib.sha256(raw).hexdigest()[:32]
    return f"vellum:search:{digest}"


class KeyValueCache(Protocol):
    async def get(self, key: str) -> str | None: ...

    async def set(self, key: str, value: str, ttl: int) -> None: ...

    async def delete(self, key: str) -> None: ...


class MemoryCache:
    def __init__(self) -> None:
        self._data: dict[str, tuple[float, str]] = {}

    async def get(self, key: str) -> str | None:
        item = self._data.get(key)
        if item is None:
            return None
        expires, value = item
        if expires < time.time():
            del self._data[key]
            return None
        return value

    async def set(self, key: str, value: str, ttl: int) -> None:
        self._data[key] = (time.time() + ttl, value)

    async def delete(self, key: str) -> None:
        self._data.pop(key, None)


class RedisCache:
    def __init__(self, redis: Redis) -> None:
        self._redis = redis

    async def get(self, key: str) -> str | None:
        value = await self._redis.get(key)
        if value is None:
            return None
        if isinstance(value, bytes):
            return value.decode()
        return str(value)

    async def set(self, key: str, value: str, ttl: int) -> None:
        await self._redis.set(key, value, ex=ttl)

    async def delete(self, key: str) -> None:
        await self._redis.delete(key)
