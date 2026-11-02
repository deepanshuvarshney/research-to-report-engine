"""Vector stores. Postgres + pgvector in the app, cosine search in memory for tests."""

from __future__ import annotations

import math
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from sqlalchemy import bindparam, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker


@dataclass
class StoredChunk:
    job_id: str
    source_uid: str
    source_ref: int
    ordinal: int
    text: str
    embedding: list[float]


@dataclass
class ChunkHit:
    source_uid: str
    text: str
    score: float
    ordinal: int


class VectorStore(Protocol):
    async def replace(self, job_id: str, chunks: Sequence[StoredChunk]) -> None: ...

    async def search(self, job_id: str, embedding: Sequence[float], k: int) -> list[ChunkHit]: ...

    async def prune(self, job_id: str, keep_uids: Sequence[str]) -> None: ...


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right, strict=False))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def vector_literal(values: Sequence[float]) -> str:
    return "[" + ",".join(f"{value:.7f}" for value in values) + "]"


class MemoryVectorStore:
    def __init__(self) -> None:
        self._chunks: list[StoredChunk] = []

    async def replace(self, job_id: str, chunks: Sequence[StoredChunk]) -> None:
        self._chunks = [chunk for chunk in self._chunks if chunk.job_id != job_id]
        self._chunks.extend(chunks)

    async def search(self, job_id: str, embedding: Sequence[float], k: int) -> list[ChunkHit]:
        scored = [
            ChunkHit(
                source_uid=chunk.source_uid,
                text=chunk.text,
                score=cosine(embedding, chunk.embedding),
                ordinal=chunk.ordinal,
            )
            for chunk in self._chunks
            if chunk.job_id == job_id
        ]
        scored.sort(key=lambda hit: hit.score, reverse=True)
        return scored[:k]

    async def prune(self, job_id: str, keep_uids: Sequence[str]) -> None:
        keep = set(keep_uids)
        self._chunks = [
            chunk
            for chunk in self._chunks
            if chunk.job_id != job_id or (keep and chunk.source_uid in keep)
        ]


class PgVectorStore:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def replace(self, job_id: str, chunks: Sequence[StoredChunk]) -> None:
        job_uuid = uuid.UUID(job_id)
        async with self._session_factory() as session:
            await session.execute(
                text("DELETE FROM chunks WHERE job_id = :job_id"),
                {"job_id": job_uuid},
            )
            if chunks:
                await session.execute(
                    text(
                        """
                        INSERT INTO chunks
                            (id, job_id, source_uid, source_ref, ordinal, text, embedding)
                        VALUES
                            (:id, :job_id, :source_uid, :source_ref, :ordinal, :text,
                             CAST(:embedding AS vector))
                        """
                    ),
                    [
                        {
                            "id": uuid.uuid4(),
                            "job_id": job_uuid,
                            "source_uid": chunk.source_uid,
                            "source_ref": chunk.source_ref,
                            "ordinal": chunk.ordinal,
                            "text": chunk.text,
                            "embedding": vector_literal(chunk.embedding),
                        }
                        for chunk in chunks
                    ],
                )
            await session.commit()

    async def search(self, job_id: str, embedding: Sequence[float], k: int) -> list[ChunkHit]:
        async with self._session_factory() as session:
            result = await session.execute(
                text(
                    """
                    SELECT source_uid, ordinal, text,
                           1 - (embedding <=> CAST(:embedding AS vector)) AS score
                    FROM chunks
                    WHERE job_id = :job_id
                    ORDER BY embedding <=> CAST(:embedding AS vector)
                    LIMIT :k
                    """
                ),
                {
                    "job_id": uuid.UUID(job_id),
                    "embedding": vector_literal(embedding),
                    "k": k,
                },
            )
            rows = result.mappings().all()
        return [
            ChunkHit(
                source_uid=str(row["source_uid"]),
                text=str(row["text"]),
                score=float(row["score"] or 0.0),
                ordinal=int(row["ordinal"]),
            )
            for row in rows
        ]

    async def prune(self, job_id: str, keep_uids: Sequence[str]) -> None:
        job_uuid = uuid.UUID(job_id)
        async with self._session_factory() as session:
            if not keep_uids:
                await session.execute(
                    text("DELETE FROM chunks WHERE job_id = :job_id"),
                    {"job_id": job_uuid},
                )
            else:
                statement = text(
                    """
                    DELETE FROM chunks
                    WHERE job_id = :job_id AND source_uid NOT IN :uids
                    """
                ).bindparams(bindparam("uids", expanding=True))
                await session.execute(
                    statement,
                    {"job_id": job_uuid, "uids": list(keep_uids)},
                )
            await session.commit()
