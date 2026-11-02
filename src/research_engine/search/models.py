"""Source documents returned by every search adapter."""

from __future__ import annotations

import hashlib
import re

from pydantic import BaseModel, ConfigDict, Field


def clean_text(text: str, limit: int | None = None) -> str:
    collapsed = " ".join(text.replace("\x00", " ").split())
    if limit is not None and len(collapsed) > limit:
        clipped = collapsed[:limit].rsplit(" ", 1)[0]
        return clipped or collapsed[:limit]
    return collapsed


def make_uid(url: str, title: str) -> str:
    raw = (url.strip().lower() or title.strip().lower()).encode()
    return hashlib.sha256(raw).hexdigest()[:16]


class Source(BaseModel):
    model_config = ConfigDict(extra="ignore")

    uid: str
    ref: int = 0
    source_type: str
    title: str
    url: str = ""
    authors: str = ""
    year: str = ""
    content: str
    prompt_text: str = ""

    def evidence_text(self) -> str:
        return self.prompt_text or self.content

    def prompt_block(self) -> str:
        body = self.evidence_text()
        return (
            f"SOURCE {self.ref}\n"
            f"Title: {self.title}\n"
            f"Type: {self.source_type}\n"
            f"Year: {self.year}\n"
            f"URL: {self.url}\n"
            f"Text:\n{body}\n"
        )


class SearchBatch(BaseModel):
    sources: list[Source] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


_PRIORITY = {"semantic_scholar": 0, "arxiv": 1, "wikipedia": 2, "web": 3}


def dedupe_sources(sources: list[Source], limit: int) -> list[Source]:
    """Prefer papers over encyclopedia pages over web snippets, and drop duplicates."""

    ordered = sorted(sources, key=lambda source: _PRIORITY.get(source.source_type, 9))
    seen_urls: set[str] = set()
    seen_titles: set[str] = set()
    kept: list[Source] = []
    for source in ordered:
        url_key = source.url.strip().lower().rstrip("/")
        title_key = re.sub(r"\W+", " ", source.title.lower()).strip()
        if url_key and url_key in seen_urls:
            continue
        if title_key and title_key in seen_titles:
            continue
        if len(source.content) < 80:
            continue
        if url_key:
            seen_urls.add(url_key)
        if title_key:
            seen_titles.add(title_key)
        kept.append(source)
        if len(kept) >= limit:
            break
    return kept
