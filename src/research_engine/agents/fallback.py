"""Deterministic planner, writer, and critic used by the mock provider and as a fallback.

The writer quotes source text instead of paraphrasing, so the citation check can pass
without a hosted model. A configured Groq, Gemini, or Ollama model replaces this path.
"""

from __future__ import annotations

import re
from typing import Any

from research_engine.agents.models import Claim, Critique, Draft, Paragraph, Section
from research_engine.faithfulness import claim_supported
from research_engine.search.models import Source
from research_engine.search.terms import topic_phrase

_HEADINGS = {
    "arxiv": "Papers",
    "semantic_scholar": "Papers",
    "wikipedia": "Background",
    "web": "Web sources",
}


def heuristic_plan(question: str, max_n: int) -> dict[str, Any]:
    topic = topic_phrase(question)
    candidates = [
        topic,
        f"{topic} survey",
        f"{topic} limitations",
        f"{topic} methods",
    ]
    sub_questions: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = candidate.lower()
        if key in seen:
            continue
        seen.add(key)
        sub_questions.append(candidate)
        if len(sub_questions) >= max(1, max_n):
            break
    return {
        "sub_questions": sub_questions,
        "rationale": "Search the topic, then a survey and the limitations.",
    }


_LOCATOR = re.compile(r"^\[([^\]\n]{1,80})\]\s+")


def excerpt(text: str, limit: int = 320) -> str:
    """A short exact substring, so the faithfulness check can verify the quote."""

    cleaned = " ".join(text.split())
    if not cleaned:
        return ""
    sentence = re.split(r"(?<=[.!?])\s+", cleaned, maxsplit=1)[0].strip()
    chosen = sentence if len(sentence) >= 40 else cleaned
    if len(chosen) <= limit:
        return chosen
    clipped = chosen[:limit].rsplit(" ", 1)[0].strip()
    return clipped or chosen[:limit]


def extractive_draft(question: str, sources: list[Source]) -> Draft:
    grouped: dict[str, list[Paragraph]] = {}
    claims: list[Claim] = []
    for source in sources:
        quote = excerpt(source.evidence_text())
        if len(quote) < 40:
            continue
        locator_match = _LOCATOR.match(quote)
        locator = locator_match.group(1).strip() if locator_match else ""
        paragraph = Paragraph(text=quote, citations=[source.ref])
        grouped.setdefault(_HEADINGS.get(source.source_type, "Sources"), []).append(paragraph)
        claims.append(Claim(text=quote, citations=[source.ref], locator=locator))

    sections = [
        Section(heading=heading, paragraphs=paragraphs) for heading, paragraphs in grouped.items()
    ]
    title = question.strip().rstrip("?")
    if len(title) > 96:
        title = title[:93].rsplit(" ", 1)[0] + "..."
    title = title[0].upper() + title[1:] if title else "Research report"
    summary = (
        f"Extractive draft from {len(claims)} cited excerpts. "
        "Each claim repeats its source so a citation check can verify it."
    )
    if not claims:
        summary = "Sources were retrieved, but none contained a usable excerpt."
    return Draft(
        title=title or "Research report", summary=summary, sections=sections, claims=claims
    )


def heuristic_verdicts(claims: list[Claim], sources: list[Source]) -> Critique:
    evidence = {source.ref: source.grounding_text() for source in sources}
    verdicts = []
    unsupported: list[str] = []
    for claim in claims:
        supported, _ratio, reason = claim_supported(claim.text, list(claim.citations), evidence)
        verdicts.append({"claim": claim.text, "supported": supported, "reason": reason})
        if not supported:
            unsupported.append(claim.text)
    instructions = ""
    if unsupported:
        instructions = (
            "Remove or replace these unsupported claims with direct excerpts from the sources: "
            + " | ".join(unsupported)
        )
    return Critique.model_validate({"verdicts": verdicts, "revision_instructions": instructions})
