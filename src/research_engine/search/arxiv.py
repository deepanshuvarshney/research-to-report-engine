"""arXiv Atom API. The PDF URL is recorded here; the download happens later, if allowed."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import httpx

from research_engine.papers.urls import arxiv_id_from_url, arxiv_pdf_url
from research_engine.search.models import Source, clean_text, make_uid
from research_engine.search.terms import content_terms

_ATOM = {"a": "http://www.w3.org/2005/Atom"}


def build_query(question: str) -> str:
    """Quoted phrase of the content words. arXiv treats the spaces as AND-ish phrase search."""

    terms = content_terms(question, 5)
    if not terms:
        return "all:research"
    return 'all:"' + " ".join(terms) + '"'


def backup_query(question: str) -> str:
    terms = content_terms(question, 4)
    if len(terms) < 2:
        return ""
    return " AND ".join(f"all:{term}" for term in terms)


def parse_arxiv(xml_text: str, *, content_limit: int) -> list[Source]:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []
    sources: list[Source] = []
    for entry in root.findall("a:entry", _ATOM):
        title = clean_text(entry.findtext("a:title", default="", namespaces=_ATOM) or "")
        summary = clean_text(
            entry.findtext("a:summary", default="", namespaces=_ATOM) or "",
            content_limit,
        )
        url = (entry.findtext("a:id", default="", namespaces=_ATOM) or "").strip()
        if not title or not summary:
            continue
        published = entry.findtext("a:published", default="", namespaces=_ATOM) or ""
        authors = ", ".join(
            clean_text(author.findtext("a:name", default="", namespaces=_ATOM) or "")
            for author in entry.findall("a:author", _ATOM)
            if (author.findtext("a:name", default="", namespaces=_ATOM) or "").strip()
        )
        arxiv_id = arxiv_id_from_url(url)
        sources.append(
            Source(
                uid=make_uid(url, title),
                source_type="arxiv",
                title=title,
                url=url,
                authors=authors[:240],
                year=published[:4],
                content=summary,
                arxiv_id=arxiv_id,
                pdf_url=arxiv_pdf_url(arxiv_id),
            )
        )
    return sources


async def _query_arxiv(
    client: httpx.AsyncClient,
    search_query: str,
    *,
    limit: int,
    content_limit: int,
) -> list[Source]:
    response = await client.get(
        "https://export.arxiv.org/api/query",
        params={"search_query": search_query, "start": 0, "max_results": limit},
    )
    response.raise_for_status()
    return parse_arxiv(response.text, content_limit=content_limit)[:limit]


async def search_arxiv(
    client: httpx.AsyncClient,
    query: str,
    *,
    limit: int,
    content_limit: int,
) -> list[Source]:
    sources = await _query_arxiv(
        client, build_query(query), limit=limit, content_limit=content_limit
    )
    if sources:
        return sources
    fallback = backup_query(query)
    if not fallback or fallback == build_query(query):
        return []
    return await _query_arxiv(client, fallback, limit=limit, content_limit=content_limit)
