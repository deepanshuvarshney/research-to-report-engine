"""Semantic Scholar Graph API. Unauthenticated use is allowed and rate limited."""

from __future__ import annotations

from typing import Any

import httpx

from research_engine.papers.urls import arxiv_pdf_url, url_is_allowed
from research_engine.search.models import Source, clean_text, make_uid

FIELDS = "title,abstract,url,year,authors,externalIds,openAccessPdf"


def parse_semantic_scholar(payload: dict[str, Any], *, content_limit: int) -> list[Source]:
    sources: list[Source] = []
    for item in payload.get("data") or []:
        if not isinstance(item, dict):
            continue
        title = clean_text(str(item.get("title") or ""))
        abstract = clean_text(str(item.get("abstract") or ""), content_limit)
        if not title or not abstract:
            continue
        external_ids = item.get("externalIds")
        external: dict[str, Any] = external_ids if isinstance(external_ids, dict) else {}
        url = str(item.get("url") or "")
        doi = external.get("DOI")
        if not url and isinstance(doi, str) and doi:
            url = f"https://doi.org/{doi}"
        authors_field = item.get("authors")
        authors_raw = authors_field if isinstance(authors_field, list) else []
        authors = ", ".join(
            str(author.get("name") or "")
            for author in authors_raw
            if isinstance(author, dict) and author.get("name")
        )
        year = str(item.get("year") or "")
        doi = external.get("DOI")
        doi_text = doi.strip() if isinstance(doi, str) else ""
        arxiv_raw = external.get("ArXiv")
        arxiv_id = arxiv_raw.strip() if isinstance(arxiv_raw, str) else ""
        open_access = item.get("openAccessPdf")
        oa_url = ""
        if isinstance(open_access, dict):
            candidate = str(open_access.get("url") or "")
            if url_is_allowed(candidate):
                oa_url = candidate
        pdf_url = oa_url or arxiv_pdf_url(arxiv_id)
        sources.append(
            Source(
                uid=make_uid(url, title),
                source_type="semantic_scholar",
                title=title,
                url=url,
                authors=authors[:240],
                year=year[:16],
                content=abstract,
                pdf_url=pdf_url,
                arxiv_id=arxiv_id,
                doi=doi_text,
            )
        )
    return sources


async def search_semantic_scholar(
    client: httpx.AsyncClient,
    query: str,
    *,
    limit: int,
    content_limit: int,
    api_key: str = "",
) -> list[Source]:
    headers = {"x-api-key": api_key} if api_key else {}
    response = await client.get(
        "https://api.semanticscholar.org/graph/v1/paper/search",
        params={
            "query": query[:240],
            "limit": limit,
            "fields": FIELDS,
        },
        headers=headers,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        return []
    return parse_semantic_scholar(payload, content_limit=content_limit)[:limit]
