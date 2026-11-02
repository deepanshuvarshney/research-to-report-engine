"""Admission control for the public demo. Fail closed if the counters cannot be read."""

from __future__ import annotations

import secrets
import time
from dataclasses import dataclass

import structlog
from redis.asyncio import Redis

from research_engine.config import Settings

log = structlog.get_logger(__name__)

LOCK_KEY = "vellum:demo:running"


class DemoLimited(RuntimeError):
    """The public demo refused this job to protect free-tier quotas."""


@dataclass
class DemoSlot:
    token: str
    global_key: str
    ip_key: str
    job_id: str = ""
    settled: bool = False


async def begin_demo_job(redis: Redis, settings: Settings, client_ip: str) -> DemoSlot:
    """Take the single running slot and charge the hourly counters."""

    token = secrets.token_hex(8)
    try:
        acquired = await redis.set(LOCK_KEY, token, nx=True, ex=settings.job_timeout_seconds)
    except Exception as exc:
        log.warning("demo_lock_failed", error=type(exc).__name__)
        raise DemoLimited("Demo limits are unavailable. Try again shortly.") from exc
    if not acquired:
        raise DemoLimited("A demo job is already running. Try again in a few minutes.")
    try:
        global_key, ip_key = await _charge(redis, settings, client_ip.strip())
    except DemoLimited:
        await _delete_token(redis, token)
        raise
    except Exception as exc:
        await _delete_token(redis, token)
        log.warning("demo_counter_failed", error=type(exc).__name__)
        raise DemoLimited("Demo limits are unavailable. Try again shortly.") from exc
    return DemoSlot(token=token, global_key=global_key, ip_key=ip_key)


async def adopt_demo_lock(redis: Redis, slot: DemoSlot, job_id: str, ttl: int) -> None:
    """Replace the reservation token with the job id so completion can release it."""

    current = await _read(redis, LOCK_KEY)
    if current != slot.token:
        raise DemoLimited("The demo slot was lost. Try again.")
    await redis.set(LOCK_KEY, job_id, ex=ttl)
    slot.job_id = job_id


async def cancel_demo_job(redis: Redis, slot: DemoSlot) -> None:
    """Return the slot and the hourly charge when a job never starts."""

    if slot.settled:
        return
    slot.settled = True
    await _delete_if(redis, {slot.token, slot.job_id or None})
    await _refund(redis, slot)


async def release_demo_lock(redis: Redis, job_id: str) -> None:
    """Drop the running lock after the job finishes. The hourly charge stays spent."""

    current = await _read(redis, LOCK_KEY)
    if current == job_id:
        await redis.delete(LOCK_KEY)


async def _charge(redis: Redis, settings: Settings, client_ip: str) -> tuple[str, str]:
    hour = _hour()
    global_key = f"vellum:demo:hour:{hour}"
    count = int(await redis.incr(global_key))
    if count == 1:
        await redis.expire(global_key, 7200)
    if count > settings.demo_max_jobs_per_hour:
        await redis.decr(global_key)
        raise DemoLimited("The demo has reached its hourly job limit. Try again later.")
    if not client_ip:
        return global_key, ""
    ip_key = f"vellum:demo:hour:{hour}:{client_ip}"
    ip_count = int(await redis.incr(ip_key))
    if ip_count == 1:
        await redis.expire(ip_key, 7200)
    if ip_count > settings.demo_max_jobs_per_ip_per_hour:
        await redis.decr(ip_key)
        await redis.decr(global_key)
        raise DemoLimited("This network has used its demo jobs for the hour.")
    return global_key, ip_key


def _hour() -> int:
    return int(time.time()) // 3600


async def _refund(redis: Redis, slot: DemoSlot) -> None:
    if slot.global_key:
        await redis.decr(slot.global_key)
    if slot.ip_key:
        await redis.decr(slot.ip_key)


async def _delete_token(redis: Redis, token: str) -> None:
    current = await _read(redis, LOCK_KEY)
    if current == token:
        await redis.delete(LOCK_KEY)


async def _delete_if(redis: Redis, expected: set[str | None]) -> None:
    current = await _read(redis, LOCK_KEY)
    if current is not None and current in expected:
        await redis.delete(LOCK_KEY)


async def _read(redis: Redis, key: str) -> str | None:
    value = await redis.get(key)
    if isinstance(value, bytes):
        return value.decode()
    if value is None:
        return None
    return str(value)
