"""SQL for jobs, progress events, and saved sources."""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime
from typing import Any, cast

from sqlalchemy import text
from sqlalchemy.engine import CursorResult
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

_UPDATABLE = {"status", "error", "report_md", "report_html", "support_score", "stats"}


def _json_obj(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        loaded = json.loads(value)
        return loaded if isinstance(loaded, dict) else {}
    if isinstance(value, dict):
        return value
    return {}


def _iso(value: Any) -> str | None:
    if isinstance(value, datetime):
        return value.isoformat()
    if value is None:
        return None
    return str(value)


class PostgresJobRepository:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def create_job(self, *, question: str, provider: str, model: str) -> dict[str, Any]:
        job_id = uuid.uuid4()
        async with self._session_factory() as session:
            result = await session.execute(
                text(
                    """
                    INSERT INTO jobs (id, question, status, provider, model, stats)
                    VALUES (:id, :question, 'queued', :provider, :model, CAST(:stats AS jsonb))
                    RETURNING id, question, status, error, provider, model, support_score,
                              stats, report_md, created_at, updated_at
                    """
                ),
                {
                    "id": job_id,
                    "question": question,
                    "provider": provider,
                    "model": model,
                    "stats": "{}",
                },
            )
            row = result.mappings().one()
            await session.commit()
        return _job_from_row(row, include_report=True)

    async def get_job(self, job_id: str) -> dict[str, Any] | None:
        async with self._session_factory() as session:
            result = await session.execute(
                text(
                    """
                    SELECT id, question, status, error, provider, model, support_score,
                           stats, report_md, report_html, created_at, updated_at
                    FROM jobs
                    WHERE id = :id
                    """
                ),
                {"id": uuid.UUID(job_id)},
            )
            row = result.mappings().first()
        if row is None:
            return None
        return _job_from_row(row, include_report=True)

    async def list_jobs(self, limit: int) -> list[dict[str, Any]]:
        async with self._session_factory() as session:
            result = await session.execute(
                text(
                    """
                    SELECT id, question, status, error, provider, model, support_score,
                           stats, created_at, updated_at,
                           (report_md IS NOT NULL) AS report_ready
                    FROM jobs
                    ORDER BY created_at DESC
                    LIMIT :limit
                    """
                ),
                {"limit": limit},
            )
            rows = result.mappings().all()
        return [_job_from_row(row, include_report=False) for row in rows]

    async def update_job(self, job_id: str, **fields: Any) -> None:
        assignments: list[str] = []
        params: dict[str, Any] = {"id": uuid.UUID(job_id)}
        for key, value in fields.items():
            if key not in _UPDATABLE:
                raise ValueError(f"cannot update {key}")
            if key == "stats":
                assignments.append("stats = CAST(:stats AS jsonb)")
                params["stats"] = json.dumps(value or {})
            else:
                assignments.append(f"{key} = :{key}")
                params[key] = value
        if not assignments:
            return
        assignments.append("updated_at = now()")
        sql = "UPDATE jobs SET " + ", ".join(assignments) + " WHERE id = :id"
        async with self._session_factory() as session:
            await session.execute(text(sql), params)
            await session.commit()

    async def add_event(
        self,
        job_id: str,
        *,
        agent: str,
        kind: str,
        message: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Insert the next sequence number for this job."""

        async with self._session_factory() as session:
            result = await session.execute(
                text(
                    """
                    INSERT INTO job_events (job_id, seq, agent, kind, message, payload, created_at)
                    SELECT :job_id,
                           COALESCE(MAX(e.seq), 0) + 1,
                           :agent,
                           :kind,
                           :message,
                           CAST(:payload AS jsonb),
                           now()
                    FROM jobs j
                    LEFT JOIN job_events e ON e.job_id = j.id
                    WHERE j.id = :job_id
                    GROUP BY j.id
                    RETURNING seq, agent, kind, message, payload, created_at
                    """
                ),
                {
                    "job_id": uuid.UUID(job_id),
                    "agent": agent,
                    "kind": kind,
                    "message": message,
                    "payload": json.dumps(payload or {}),
                },
            )
            row = result.mappings().one()
            await session.commit()
        return _event_from_row(row)

    async def list_events(self, job_id: str) -> list[dict[str, Any]]:
        async with self._session_factory() as session:
            result = await session.execute(
                text(
                    """
                    SELECT seq, agent, kind, message, payload, created_at
                    FROM job_events
                    WHERE job_id = :job_id
                    ORDER BY seq
                    """
                ),
                {"job_id": uuid.UUID(job_id)},
            )
            rows = result.mappings().all()
        return [_event_from_row(row) for row in rows]

    async def replace_sources(self, job_id: str, sources: list[dict[str, Any]]) -> None:
        job_uuid = uuid.UUID(job_id)
        rows = []
        for source in sources:
            content = str(source.get("content") or "")
            rows.append(
                {
                    "id": uuid.uuid4(),
                    "job_id": job_uuid,
                    "ref_number": int(source.get("ref") or 0),
                    "source_type": str(source.get("source_type") or "web"),
                    "title": str(source.get("title") or "Untitled"),
                    "url": str(source.get("url") or ""),
                    "authors": str(source.get("authors") or ""),
                    "year": str(source.get("year") or "")[:16],
                    "content": content,
                    "content_hash": hashlib.sha256(content.encode()).hexdigest(),
                    "text_origin": str(source.get("text_origin") or "abstract")[:16],
                    "pdf_url": str(source.get("pdf_url") or "")[:500],
                }
            )
        async with self._session_factory() as session:
            await session.execute(
                text("DELETE FROM sources WHERE job_id = :job_id"),
                {"job_id": job_uuid},
            )
            if rows:
                await session.execute(
                    text(
                        """
                        INSERT INTO sources (
                            id, job_id, ref_number, source_type, title, url,
                            authors, year, content, content_hash, text_origin, pdf_url
                        )
                        VALUES (
                            :id, :job_id, :ref_number, :source_type, :title, :url,
                            :authors, :year, :content, :content_hash, :text_origin, :pdf_url
                        )
                        """
                    ),
                    rows,
                )
            await session.commit()

    async def fail_abandoned_jobs(self, older_than_seconds: int) -> int:
        async with self._session_factory() as session:
            result = await session.execute(
                text(
                    """
                    UPDATE jobs
                    SET status = 'failed',
                        error = 'Worker stopped before the job finished.',
                        updated_at = now()
                    WHERE status = 'running'
                      AND updated_at < now() - (:seconds * INTERVAL '1 second')
                    """
                ),
                {"seconds": older_than_seconds},
            )
            await session.commit()
        cursor = cast(CursorResult[Any], result)
        return int(cursor.rowcount or 0)


def _job_from_row(row: Any, *, include_report: bool) -> dict[str, Any]:
    mapping = row
    report_md = mapping["report_md"] if include_report else None
    report_ready = bool(report_md) if include_report else bool(mapping["report_ready"])
    support = mapping["support_score"]
    return {
        "id": str(mapping["id"]),
        "question": mapping["question"],
        "status": mapping["status"],
        "error": mapping["error"],
        "provider": mapping["provider"],
        "model": mapping["model"],
        "support_score": float(support) if support is not None else None,
        "stats": _json_obj(mapping["stats"]),
        "report_md": report_md,
        "report_html": mapping["report_html"]
        if include_report and "report_html" in mapping
        else None,
        "created_at": _iso(mapping["created_at"]),
        "updated_at": _iso(mapping["updated_at"]),
        "report_ready": report_ready,
    }


def _event_from_row(row: Any) -> dict[str, Any]:
    payload = row["payload"]
    if isinstance(payload, str):
        payload = json.loads(payload)
    if not isinstance(payload, dict):
        payload = {}
    return {
        "seq": int(row["seq"]),
        "agent": row["agent"],
        "kind": row["kind"],
        "message": row["message"],
        "payload": payload,
        "created_at": _iso(row["created_at"]),
    }
