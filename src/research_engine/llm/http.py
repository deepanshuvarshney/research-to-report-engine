"""Timeouts and retries for hosted model calls. Secrets are scrubbed from errors."""

from __future__ import annotations

import asyncio
import random

import httpx

RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


class ProviderError(RuntimeError):
    """A model call failed after retries, or the provider rejected the request."""


def scrub(text: str, secret: str) -> str:
    if secret and secret in text:
        return text.replace(secret, "***")
    return text


async def request_with_retries(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    *,
    max_retries: int,
    headers: dict[str, str] | None = None,
    json: dict[str, object] | None = None,
    secret: str = "",
) -> httpx.Response:
    """Retry timeouts, transport errors, and transient HTTP statuses with jitter."""

    attempt = 0
    while True:
        try:
            response = await client.request(method, url, headers=headers, json=json)
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            if attempt >= max_retries:
                raise ProviderError(f"request failed after {attempt + 1} attempts: {exc}") from exc
            await asyncio.sleep(min(2**attempt, 8) + random.random() * 0.2)
            attempt += 1
            continue

        if response.status_code in RETRYABLE_STATUS and attempt < max_retries:
            await asyncio.sleep(min(2**attempt, 8) + random.random() * 0.2)
            attempt += 1
            continue

        if response.status_code >= 400:
            detail = scrub(response.text[:400], secret)
            raise ProviderError(f"HTTP {response.status_code}: {detail}")
        return response
