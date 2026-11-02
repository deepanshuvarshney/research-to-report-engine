"""Fan out a question to every free source, with caching and per-host pacing.

Result pages are not crawled. Each adapter returns text the API already gave us.
That keeps the worker free of SSRF from untrusted URLs and stays inside public rate limits.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping

import httpx
import structlog

from research_engine.config import Settings
from research_engine.search.arxiv import search_arxiv
from research_engine.search.cache import KeyValueCache, cache_key
from research_engine.search.duckduckgo import search_duckduckgo
from research_engine.search.limiter import AsyncRateLimiter, RateLimiter
from research_engine.search.models import SearchBatch, Source, dedupe_sources
from research_engine.search.semantic_scholar import search_semantic_scholar
from research_engine.search.terms import content_terms, overlaps_topic
from research_engine.search.wikipedia import search_wikipedia

log = structlog.get_logger(__name__)


class SearchService:
    def __init__(
        self,
        *,
        settings: Settings,
        cache: KeyValueCache,
        client: httpx.AsyncClient | None = None,
        limiters: Mapping[str, RateLimiter] | None = None,
    ) -> None:
        self.settings = settings
        self.cache = cache
        self.limiters = dict(limiters or {})
        self._fallback = AsyncRateLimiter(settings.search_rate_limit_rps)
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(settings.search_timeout_seconds, connect=10.0),
            headers={
                "User-Agent": settings.http_user_agent,
                "Accept": "application/json, application/xml, text/html;q=0.9",
            },
            follow_redirects=True,
        )

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def gather(self, queries: list[str]) -> SearchBatch:
        notes: list[str] = []
        found: list[Source] = []
        for name in ("arxiv", "semantic_scholar", "wikipedia", "web"):
            batch, note = await self._run_provider(name, queries)
            found.extend(batch)
            notes.append(note)
        kept = dedupe_sources(found, self.settings.max_sources_total)
        terms: list[str] = []
        seen_terms: set[str] = set()
        for query in queries:
            for term in content_terms(query, 6):
                key = term.lower()
                if key in seen_terms:
                    continue
                seen_terms.add(key)
                terms.append(term)
        focused = [source for source in kept if overlaps_topic(source, terms)]
        return SearchBatch(sources=focused or kept, notes=notes)

    async def _run_provider(self, name: str, queries: list[str]) -> tuple[list[Source], str]:
        found: list[Source] = []
        failures = 0
        hits = 0
        for query in queries:
            try:
                batch = await self._cached(name, query)
            except Exception as exc:
                failures += 1
                log.warning(
                    "search_provider_failed",
                    provider=name,
                    error=str(exc)[:240],
                )
                continue
            hits += len(batch)
            found.extend(batch)
        if failures and not found:
            return [], f"{name}: failed ({failures} queries)"
        suffix = f", {failures} failed" if failures else ""
        return found, f"{name}: {hits} results{suffix}"

    async def _cached(self, provider: str, query: str) -> list[Source]:
        limit = self.settings.max_results_per_provider
        key = cache_key(provider, query, limit)
        try:
            raw = await self.cache.get(key)
        except Exception as exc:
            log.warning("cache_read_failed", provider=provider, error=str(exc)[:160])
            raw = None
        if raw is not None:
            data = json.loads(raw)
            return [Source.model_validate(item) for item in data]

        await self._acquire(provider)
        try:
            sources = await self._fetch(provider, query)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code != 429:
                raise
            await asyncio.sleep(2)
            await self._acquire(provider)
            sources = await self._fetch(provider, query)
        try:
            payload = json.dumps([source.model_dump() for source in sources])
            await self.cache.set(key, payload, self.settings.cache_ttl_seconds)
        except Exception as exc:
            log.warning("cache_write_failed", provider=provider, error=str(exc)[:160])
        return sources

    async def _acquire(self, provider: str) -> None:
        limiter = self.limiters.get(provider, self._fallback)
        await limiter.acquire()

    async def _fetch(self, provider: str, query: str) -> list[Source]:
        limit = self.settings.max_results_per_provider
        content_limit = self.settings.source_store_chars
        client = self._client
        if provider == "arxiv":
            return await search_arxiv(client, query, limit=limit, content_limit=content_limit)
        if provider == "semantic_scholar":
            return await search_semantic_scholar(
                client,
                query,
                limit=limit,
                content_limit=content_limit,
                api_key=self.settings.semantic_scholar_api_key,
            )
        if provider == "wikipedia":
            return await search_wikipedia(client, query, limit=limit, content_limit=content_limit)
        if provider == "web":
            return await search_duckduckgo(client, query, limit=limit, content_limit=content_limit)
        raise RuntimeError(f"unknown search provider {provider}")
