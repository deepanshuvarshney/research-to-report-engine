"""Turn PDF bytes into section-aware passages. Parsing stays on the local machine."""

from __future__ import annotations

import re
from dataclasses import dataclass
from io import BytesIO

from research_engine.rag.chunker import chunk_text
from research_engine.search.models import Passage

_KNOWN = (
    "abstract",
    "introduction",
    "related work",
    "background",
    "preliminaries",
    "methods",
    "method",
    "methodology",
    "approach",
    "model",
    "architecture",
    "experiments",
    "experiment",
    "results",
    "result",
    "evaluation",
    "discussion",
    "analysis",
    "conclusion",
    "conclusions",
    "limitations",
    "future work",
    "references",
    "bibliography",
    "appendix",
)
_KNOWN_LINE = re.compile(
    r"^(?:(?:\d+(?:\.\d+){0,3})\.?\s+)?("
    + "|".join(re.escape(name) for name in _KNOWN)
    + r")(?:\s*[:;\-]\s*.*)?$",
    re.IGNORECASE,
)
_NUMBERED = re.compile(r"^(\d+(?:\.\d+){0,3})\.?\s+([A-Z][\w][\w \-:/,&+]{0,70})$")
_SKIP_SECTION = re.compile(
    r"^(?:\d+(?:\.\d+){0,3}\.?\s+)?(references|bibliography)\b",
    re.IGNORECASE,
)
_PAGE_NUMBER = re.compile(r"^\d{1,3}$")


@dataclass(frozen=True)
class ParsedPdf:
    passages: list[Passage]
    page_count: int
    pages_read: int
    truncated: bool


def heading_name(line: str) -> str | None:
    """Return a section title when the line looks like a heading, else None."""

    stripped = " ".join(line.split())
    if not stripped or len(stripped) > 90 or stripped.endswith("."):
        return None
    if _PAGE_NUMBER.match(stripped):
        return None
    if _KNOWN_LINE.match(stripped):
        return stripped
    numbered = _NUMBERED.match(stripped)
    if numbered and len(stripped.split()) <= 10:
        return stripped
    return None


def is_skipped_section(name: str) -> bool:
    return bool(_SKIP_SECTION.match(name.strip()))


def normalize_pdf_text(text: str) -> str:
    cleaned = text.replace("\x00", " ").replace("\r\n", "\n").replace("\r", "\n")
    cleaned = re.sub(r"-\n([a-z])", r"\1", cleaned)
    return "\n".join(" ".join(line.split()) for line in cleaned.split("\n"))


def pages_to_passages(
    pages: list[tuple[int, str]],
    *,
    chunk_size: int,
    chunk_overlap: int,
    max_chunks: int,
) -> tuple[list[Passage], bool]:
    """Group page lines under headings, then window long sections."""

    blocks: list[tuple[str, int, str]] = []
    section = ""
    skipping = False
    for page_no, raw in pages:
        for line in normalize_pdf_text(raw).splitlines():
            if not line or _PAGE_NUMBER.match(line):
                continue
            heading = heading_name(line)
            if heading is not None:
                if is_skipped_section(heading):
                    skipping = True
                    section = heading
                    continue
                skipping = False
                section = heading
                continue
            if skipping:
                continue
            blocks.append((section, page_no, line))

    passages: list[Passage] = []
    truncated = False
    bucket_section = ""
    bucket_page = 0
    bucket: list[str] = []
    bucket_len = 0

    def flush() -> None:
        nonlocal truncated
        if not bucket:
            return
        text = " ".join(bucket)
        for piece in chunk_text(text, size=chunk_size, overlap=chunk_overlap):
            if len(passages) >= max_chunks:
                truncated = True
                return
            passages.append(Passage(text=piece, section=bucket_section, page=bucket_page))

    for section_name, page_no, line in blocks:
        overflow = bucket_len + len(line) > chunk_size and bucket
        boundary = bucket and (section_name != bucket_section or page_no != bucket_page)
        if overflow or boundary:
            flush()
            bucket = []
            bucket_len = 0
            if truncated:
                break
        bucket_section = section_name
        bucket_page = page_no
        bucket.append(line)
        bucket_len += len(line) + 1
    if not truncated:
        flush()
    return passages, truncated


def parse_pdf_bytes(
    data: bytes,
    *,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
    max_pages: int = 20,
    max_chars: int = 60_000,
    max_chunks: int = 36,
) -> ParsedPdf:
    """Extract text with pypdf. Empty or unreadable files yield no passages."""

    if not data.startswith(b"%PDF"):
        raise ValueError("not a pdf")
    from pypdf import PdfReader

    try:
        reader = PdfReader(BytesIO(data))
    except Exception as exc:
        raise ValueError("unreadable pdf") from exc
    if reader.is_encrypted:
        try:
            reader.decrypt("")
        except Exception as exc:
            raise ValueError("encrypted pdf") from exc
        if reader.is_encrypted:
            raise ValueError("encrypted pdf")

    page_count = len(reader.pages)
    pages: list[tuple[int, str]] = []
    used_chars = 0
    truncated = page_count > max_pages
    for index, page in enumerate(reader.pages):
        if index >= max_pages or used_chars >= max_chars:
            truncated = True
            break
        try:
            extracted = page.extract_text() or ""
        except Exception:
            extracted = ""
        remaining = max_chars - used_chars
        if len(extracted) > remaining:
            extracted = extracted[:remaining]
            truncated = True
        used_chars += len(extracted)
        if extracted.strip():
            pages.append((index + 1, extracted))
        if used_chars >= max_chars:
            truncated = True
            break

    passages, chunk_truncated = pages_to_passages(
        pages,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        max_chunks=max_chunks,
    )
    return ParsedPdf(
        passages=passages,
        page_count=page_count,
        pages_read=len(pages),
        truncated=truncated or chunk_truncated,
    )
