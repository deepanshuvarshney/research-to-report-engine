"""DuckDuckGo HTML results. Snippets only — result URLs are never fetched (no SSRF)."""

from __future__ import annotations

from html.parser import HTMLParser
from urllib.parse import parse_qs, unquote, urlparse

import httpx

from research_engine.search.models import Source, clean_text, make_uid


def decode_ddg_url(href: str) -> str:
    if href.startswith("//"):
        href = "https:" + href
    parsed = urlparse(href)
    target = parse_qs(parsed.query).get("uddg")
    if target:
        return unquote(target[0])
    return href


class _ResultParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.results: list[dict[str, str]] = []
        self._capture: str | None = None
        self._href = ""
        self._buf: list[str] = []
        self._current: dict[str, str] = {}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr = {key: value or "" for key, value in attrs}
        classes = attr.get("class", "")
        if tag == "a" and "result__a" in classes:
            self._capture = "title"
            self._href = attr.get("href", "")
            self._buf = []
        elif tag in {"a", "td", "div", "span"} and "result__snippet" in classes:
            self._capture = "snippet"
            self._buf = []

    def handle_endtag(self, tag: str) -> None:
        if self._capture is None:
            return
        if self._capture == "title" and tag == "a":
            text = clean_text("".join(self._buf))
            if text:
                self._current = {"title": text, "url": decode_ddg_url(self._href)}
            self._capture = None
            self._buf = []
        elif self._capture == "snippet" and tag in {"a", "td", "div", "span"}:
            text = clean_text("".join(self._buf))
            if text and self._current.get("title"):
                self._current["snippet"] = text
                self.results.append(self._current)
                self._current = {}
            self._capture = None
            self._buf = []

    def handle_data(self, data: str) -> None:
        if self._capture:
            self._buf.append(data)


def parse_duckduckgo(html: str, *, content_limit: int) -> list[Source]:
    parser = _ResultParser()
    parser.feed(html)
    sources: list[Source] = []
    for item in parser.results:
        title = item.get("title", "")
        snippet = clean_text(item.get("snippet", ""), content_limit)
        url = item.get("url", "")
        if not title or not snippet:
            continue
        sources.append(
            Source(
                uid=make_uid(url, title),
                source_type="web",
                title=title,
                url=url,
                content=snippet,
            )
        )
    return sources


async def search_duckduckgo(
    client: httpx.AsyncClient,
    query: str,
    *,
    limit: int,
    content_limit: int,
) -> list[Source]:
    response = await client.post(
        "https://html.duckduckgo.com/html/",
        data={"q": query[:240], "kl": "us-en"},
    )
    response.raise_for_status()
    return parse_duckduckgo(response.text, content_limit=content_limit)[:limit]
