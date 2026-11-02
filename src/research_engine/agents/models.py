"""Structured outputs the writer and critic must return."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

MAX_CLAIMS = 12
MAX_SECTIONS = 6
MAX_PARAGRAPHS = 4


class Claim(BaseModel):
    model_config = ConfigDict(extra="ignore")

    text: str
    citations: list[int] = Field(default_factory=list)


class Paragraph(BaseModel):
    model_config = ConfigDict(extra="ignore")

    text: str
    citations: list[int] = Field(default_factory=list)


class Section(BaseModel):
    model_config = ConfigDict(extra="ignore")

    heading: str
    paragraphs: list[Paragraph] = Field(default_factory=list)


class Draft(BaseModel):
    model_config = ConfigDict(extra="ignore")

    title: str
    summary: str = ""
    sections: list[Section] = Field(default_factory=list)
    claims: list[Claim] = Field(default_factory=list)


class Plan(BaseModel):
    model_config = ConfigDict(extra="ignore")

    sub_questions: list[str] = Field(default_factory=list)
    rationale: str = ""


class Verdict(BaseModel):
    model_config = ConfigDict(extra="ignore")

    claim: str = ""
    supported: bool
    reason: str = ""


class Critique(BaseModel):
    model_config = ConfigDict(extra="ignore")

    verdicts: list[Verdict] = Field(default_factory=list)
    revision_instructions: str = ""


def normalize_draft(draft: Draft, valid_refs: set[int]) -> Draft:
    """Drop unknown citations and cap the draft so the critic loop stays bounded."""

    sections: list[Section] = []
    for section in draft.sections[:MAX_SECTIONS]:
        paragraphs = [
            paragraph.model_copy(
                update={"citations": [ref for ref in paragraph.citations if ref in valid_refs]}
            )
            for paragraph in section.paragraphs[:MAX_PARAGRAPHS]
            if paragraph.text.strip()
        ]
        if paragraphs:
            sections.append(section.model_copy(update={"paragraphs": paragraphs}))

    claims = [
        claim.model_copy(
            update={"citations": [ref for ref in claim.citations if ref in valid_refs]}
        )
        for claim in draft.claims
        if claim.text.strip()
    ]
    if not claims:
        claims = [
            Claim(text=paragraph.text.strip(), citations=list(paragraph.citations))
            for section in sections
            for paragraph in section.paragraphs
        ]
    return draft.model_copy(
        update={
            "title": draft.title.strip() or "Research report",
            "summary": draft.summary.strip(),
            "sections": sections,
            "claims": claims[:MAX_CLAIMS],
        }
    )
