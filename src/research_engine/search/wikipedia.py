"""Wikipedia search plus plain-text extracts. No article scraping."""

from __future__ import annotations

from typing import Any
from urllib.parse import quote

import httpx

from research_engine.search.models import Source, clean_text, make_uid


def parse_wikipedia(payload: dict[str, Any], *, content_limit: int) -> list[Source]:
    query = payload.get("query")
    if not isinstance(query, dict):
        return []
    pages = query.get("pages")
    if not isinstance(pages, dict):
        return []
    sources: list[Source] = []
    for page in pages.values():
        if not isinstance(page, dict) or page.get("missing"):
            continue
        title = clean_text(str(page.get("title") or ""))
        extract = clean_text(str(page.get("extract") or ""), content_limit)
        if not title or not extract:
            continue
        url = "https://en.wikipedia.org/wiki/" + quote(title.replace(" ", "_"))
        sources.append(
            Source(
                uid=make_uid(url, title),
                source_type="wikipedia",
                title=title,
                url=url,
                content=extract,
            )
        )
    return sources


async def search_wikipedia(
    client: httpx.AsyncClient,
    query: str,
    *,
    limit: int,
    content_limit: int,
) -> list[Source]:
    response = await client.get(
        "https://en.wikipedia.org/w/api.php",
        params={
            "action": "query",
            "format": "json",
            "generator": "search",
            "gsrsearch": query[:240],
            "gsrlimit": limit,
            "prop": "extracts",
            "explaintext": 1,
            "exchars": min(content_limit, 2500),
            "exlimit": limit,
        },
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        return []
    return parse_wikipedia(payload, content_limit=content_limit)[:limit]
