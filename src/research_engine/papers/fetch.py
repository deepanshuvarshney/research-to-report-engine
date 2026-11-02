"""Download allowlisted open-access PDFs, with cache, size caps, and abstract fallback."""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Awaitable, Callable
from urllib.parse import quote, urljoin

import httpx
import structlog

from research_engine.config import Settings
from research_engine.papers.pdf import ParsedPdf, parse_pdf_bytes
from research_engine.papers.urls import (
    UNPAYWALL_HOST,
    arxiv_pdf_url,
    host_of,
    limiter_name,
    url_is_allowed,
)
from research_engine.search.cache import KeyValueCache
from research_engine.search.models import Passage, Source

log = structlog.get_logger(__name__)

_REDIRECTS = {301, 302, 303, 307, 308}
_MAX_REDIRECTS = 3


class PdfRejected(Exception):
    """A permanent reason not to retry this URL inside the failure cache window."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class _Skip(Exception):
    """This source has no open-access PDF candidate."""


async def enrich_with_fulltext(
    sources: list[Source],
    *,
    client: httpx.AsyncClient,
    cache: KeyValueCache,
    settings: Settings,
    acquire: Callable[[str], Awaitable[None]],
) -> tuple[list[Source], str]:
    """Attach parsed passages where a PDF is allowed. Failures keep the abstract."""

    if not settings.fulltext_enabled or settings.fulltext_max_pdfs <= 0:
        return sources, "fulltext: disabled"
    updated = list(sources)
    success = 0
    fallback = 0
    attempts = 0
    for index, source in enumerate(sources):
        if success >= settings.fulltext_max_pdfs or attempts >= settings.fulltext_max_attempts:
            break
        if source.source_type not in {"arxiv", "semantic_scholar"}:
            continue
        try:
            enriched = await _enrich_one(
                source,
                client=client,
                cache=cache,
                settings=settings,
                acquire=acquire,
            )
        except _Skip:
            continue
        except Exception as exc:
            attempts += 1
            fallback += 1
            log.warning("fulltext_error", error=type(exc).__name__, detail=str(exc)[:160])
            continue
        attempts += 1
        if enriched is None:
            fallback += 1
            continue
        updated[index] = enriched
        success += 1
    return updated, f"fulltext: {success} pdfs, {fallback} abstract fallbacks"


def direct_pdf_url(source: Source) -> str:
    if source.pdf_url and url_is_allowed(source.pdf_url):
        return source.pdf_url
    from_id = arxiv_pdf_url(source.arxiv_id)
    if from_id and url_is_allowed(from_id):
        return from_id
    from_abs = arxiv_pdf_url(source.url)
    if from_abs and url_is_allowed(from_abs):
        return from_abs
    return ""


def valid_unpaywall_email(value: str) -> bool:
    text = value.strip()
    if not text or text.lower().startswith("unused") or len(text) > 200:
        return False
    if " " in text or text.count("@") != 1:
        return False
    local, domain = text.split("@", 1)
    return bool(local and "." in domain and not domain.startswith("."))


async def _enrich_one(
    source: Source,
    *,
    client: httpx.AsyncClient,
    cache: KeyValueCache,
    settings: Settings,
    acquire: Callable[[str], Awaitable[None]],
) -> Source | None:
    url = direct_pdf_url(source)
    if not url and source.doi and valid_unpaywall_email(settings.unpaywall_email):
        url = await resolve_unpaywall(
            client,
            cache,
            doi=source.doi,
            email=settings.unpaywall_email.strip(),
            settings=settings,
            acquire=acquire,
        )
    if not url:
        raise _Skip()
    parsed = await load_pdf(
        client,
        cache,
        url,
        settings=settings,
        acquire=acquire,
    )
    if parsed is None or not parsed.passages:
        return None
    return source.model_copy(
        update={
            "text_origin": "pdf",
            "pdf_url": url,
            "page_count": parsed.page_count,
            "truncated": parsed.truncated,
            "passages": parsed.passages,
        }
    )


async def resolve_unpaywall(
    client: httpx.AsyncClient,
    cache: KeyValueCache,
    *,
    doi: str,
    email: str,
    settings: Settings,
    acquire: Callable[[str], Awaitable[None]],
) -> str:
    """Return an allowlisted PDF URL from Unpaywall, or an empty string."""

    key = "vellum:oa:" + hashlib.sha256(doi.strip().lower().encode()).hexdigest()[:32]
    try:
        cached = await cache.get(key)
    except Exception:
        cached = None
    if cached is not None:
        try:
            payload = json.loads(cached)
        except json.JSONDecodeError:
            payload = {}
        url = str(payload.get("url") or "") if isinstance(payload, dict) else ""
        return url if url_is_allowed(url) else ""

    await acquire("unpaywall")
    timeout = _timeout(settings.fulltext_timeout_seconds)
    request_url = "https://" + UNPAYWALL_HOST + "/v2/" + quote(doi.strip(), safe="")
    try:
        response = await client.get(
            request_url,
            params={"email": email},
            timeout=timeout,
            follow_redirects=False,
        )
        response.raise_for_status()
        body = response.json()
    except Exception as exc:
        log.warning("unpaywall_failed", error=type(exc).__name__)
        return ""
    url = select_oa_pdf(body if isinstance(body, dict) else {})
    try:
        await cache.set(key, json.dumps({"url": url}), settings.cache_ttl_seconds)
    except Exception as exc:
        log.warning("unpaywall_cache_failed", error=str(exc)[:120])
    return url


def select_oa_pdf(payload: dict[str, object]) -> str:
    candidates: list[str] = []
    best = payload.get("best_oa_location")
    if isinstance(best, dict):
        candidates.append(str(best.get("url_for_pdf") or ""))
    locations = payload.get("oa_locations")
    if isinstance(locations, list):
        for item in locations:
            if isinstance(item, dict):
                candidates.append(str(item.get("url_for_pdf") or ""))
    for url in candidates:
        if url_is_allowed(url):
            return url
    return ""


async def load_pdf(
    client: httpx.AsyncClient,
    cache: KeyValueCache,
    url: str,
    *,
    settings: Settings,
    acquire: Callable[[str], Awaitable[None]],
) -> ParsedPdf | None:
    if not url_is_allowed(url):
        return None
    key = "vellum:pdf:v1:" + hashlib.sha256(url.encode()).hexdigest()[:32]
    try:
        cached = await cache.get(key)
    except Exception:
        cached = None
    if cached is not None:
        return _parsed_from_cache(cached)

    await acquire(limiter_name(url))
    try:
        data = await download_pdf(
            client,
            url,
            max_bytes=settings.fulltext_max_bytes,
            timeout_seconds=settings.fulltext_timeout_seconds,
        )
        parsed = await asyncio.to_thread(
            parse_pdf_bytes,
            data,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            max_pages=settings.fulltext_max_pages,
            max_chars=settings.fulltext_max_chars,
            max_chunks=settings.fulltext_max_chunks,
        )
    except PdfRejected as exc:
        await _cache_failure(cache, key, exc.reason, settings)
        log.info("fulltext_fallback", host=host_of(url), reason=exc.reason)
        return None
    except ValueError as exc:
        await _cache_failure(cache, key, "unreadable", settings)
        log.info("fulltext_fallback", host=host_of(url), reason=str(exc)[:80])
        return None
    except Exception as exc:
        log.warning("fulltext_fetch_failed", host=host_of(url), error=type(exc).__name__)
        return None
    if not parsed.passages:
        await _cache_failure(cache, key, "no_text", settings)
        log.info("fulltext_fallback", host=host_of(url), reason="no_text")
        return None
    try:
        await cache.set(key, json.dumps(_parsed_payload(parsed)), settings.cache_ttl_seconds)
    except Exception as exc:
        log.warning("fulltext_cache_failed", error=str(exc)[:120])
    log.info(
        "fulltext_ready",
        host=host_of(url),
        pages=parsed.page_count,
        passages=len(parsed.passages),
    )
    return parsed


async def download_pdf(
    client: httpx.AsyncClient,
    url: str,
    *,
    max_bytes: int,
    timeout_seconds: float,
) -> bytes:
    """GET a PDF, following only allowlisted HTTPS redirects, and stop past the size cap."""

    current = url
    timeout = _timeout(timeout_seconds)
    for _ in range(_MAX_REDIRECTS + 1):
        if not url_is_allowed(current):
            raise PdfRejected("url_not_allowed")
        async with client.stream(
            "GET",
            current,
            follow_redirects=False,
            timeout=timeout,
            headers={"Accept": "application/pdf,*/*;q=0.1"},
        ) as response:
            if response.status_code in _REDIRECTS:
                location = response.headers.get("location")
                if not location:
                    raise PdfRejected("redirect_missing")
                current = urljoin(current, location.strip())
                continue
            if response.status_code != 200:
                raise PdfRejected(f"http_{response.status_code}")
            advertised = response.headers.get("content-length", "")
            if advertised.isdigit() and int(advertised) > max_bytes:
                raise PdfRejected("too_large")
            buf = bytearray()
            async for chunk in response.aiter_bytes():
                buf.extend(chunk)
                if len(buf) > max_bytes:
                    raise PdfRejected("too_large")
        data = bytes(buf)
        if not data.startswith(b"%PDF"):
            raise PdfRejected("not_pdf")
        return data
    raise PdfRejected("too_many_redirects")


def _timeout(seconds: float) -> httpx.Timeout:
    budget = float(seconds)
    return httpx.Timeout(budget, connect=min(10.0, budget))


def _parsed_payload(parsed: ParsedPdf) -> dict[str, object]:
    return {
        "ok": True,
        "page_count": parsed.page_count,
        "pages_read": parsed.pages_read,
        "truncated": parsed.truncated,
        "passages": [passage.model_dump() for passage in parsed.passages],
    }


def _parsed_from_cache(raw: str) -> ParsedPdf | None:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict) or not payload.get("ok"):
        return None
    passages_raw = payload.get("passages")
    if not isinstance(passages_raw, list):
        return None
    passages = [
        Passage.model_validate(item)
        for item in passages_raw
        if isinstance(item, dict) and str(item.get("text") or "").strip()
    ]
    if not passages:
        return None
    return ParsedPdf(
        passages=passages,
        page_count=int(payload.get("page_count") or 0),
        pages_read=int(payload.get("pages_read") or 0),
        truncated=bool(payload.get("truncated")),
    )


async def _cache_failure(cache: KeyValueCache, key: str, reason: str, settings: Settings) -> None:
    try:
        await cache.set(
            key,
            json.dumps({"ok": False, "reason": reason[:80]}),
            settings.fulltext_failure_ttl_seconds,
        )
    except Exception as exc:
        log.warning("fulltext_cache_failed", error=str(exc)[:120])
