"""Turn a question into the few content words worth sending to a search API."""

from __future__ import annotations

import re

from research_engine.search.models import Source

_STOP = frozenset(
    """
    a an the of to and or for in on with from by as at into over after before
    is are was were be been being this that these those it its their his her
    they them we you our not no yes do does did than then so such if but
    about between among across per via using used use can may might will
    what how why when where which who whom whose main behind within without
    does do doing done have has had having idea ideas recent research says
    say said related associated core background primary
    """.split()  # noqa: SIM905
)

_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9-]{2,}")


def content_terms(text: str, limit: int = 8) -> list[str]:
    kept: list[str] = []
    seen: set[str] = set()
    for word in _WORD.findall(text):
        pieces = [word, *[part for part in word.split("-") if len(part) >= 3]]
        for piece in pieces:
            key = piece.lower()
            if key in _STOP or key in seen:
                continue
            seen.add(key)
            kept.append(piece)
            if len(kept) >= limit:
                return kept
    return kept


def topic_phrase(text: str) -> str:
    """Content words, keeping hyphenated compounds instead of also repeating their parts."""

    terms = content_terms(text, 8)
    covered: set[str] = set()
    chosen: list[str] = []
    for term in terms:
        if "-" not in term:
            continue
        chosen.append(term)
        covered.update(part.lower() for part in term.split("-") if part)
    for term in terms:
        if "-" in term or term.lower() in covered:
            continue
        chosen.append(term)
    if not chosen:
        chosen = terms
    phrase = " ".join(chosen[:6])
    return phrase or " ".join(text.split())[:120]


def overlaps_topic(source: Source, terms: list[str]) -> bool:
    """Keep a hit when the question's content words actually show up in it."""

    if not terms:
        return True
    haystack = f"{source.title} {source.content[:600]}".lower()
    hits = sum(1 for term in terms if term.lower() in haystack)
    needed = 1 if len(terms) == 1 else min(2, len(terms))
    return hits >= needed
