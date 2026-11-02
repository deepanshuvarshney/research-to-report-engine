"""Paragraph-free character windows with overlap. Source text is already flattened."""

from __future__ import annotations


def chunk_text(text: str, *, size: int = 800, overlap: int = 120) -> list[str]:
    cleaned = " ".join(text.split())
    if not cleaned:
        return []
    if size < 40:
        size = 40
    if overlap >= size:
        overlap = size // 5
    if len(cleaned) <= size:
        return [cleaned]

    chunks: list[str] = []
    start = 0
    length = len(cleaned)
    while start < length:
        end = min(start + size, length)
        if end < length:
            split = cleaned.rfind(" ", start + size // 2, end)
            if split > start:
                end = split
        piece = cleaned[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= length:
            break
        next_start = max(end - overlap, start + 1)
        if next_start <= start:
            next_start = start + 1
        start = next_start
    return chunks
