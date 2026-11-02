"""Job submission, status, server-sent progress, and report downloads."""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from typing import Any, cast

from fastapi import APIRouter, Header, HTTPException, Query, Request
from fastapi.responses import Response, StreamingResponse

from research_engine import __version__
from research_engine.api.limiter import enforce_rate_limit
from research_engine.api.schemas import CreateJobRequest, JobResponse, MetaResponse
from research_engine.config import Settings
from research_engine.examples import EXAMPLES
from research_engine.jobs.service import JobService, QueueUnavailable

router = APIRouter(prefix="/api")

_PUBLIC_FIELDS = (
    "id",
    "question",
    "status",
    "error",
    "provider",
    "model",
    "support_score",
    "stats",
    "created_at",
    "updated_at",
    "report_ready",
)


def _service(request: Request) -> JobService:
    service = request.app.state.job_service
    if service is None:
        raise HTTPException(status_code=503, detail="Job service is not ready")
    return cast(JobService, service)


def _settings(request: Request) -> Settings:
    return cast(Settings, request.app.state.settings)


def _job_response(job: dict[str, object], *, include_report: bool) -> JobResponse:
    payload = {field: job.get(field) for field in _PUBLIC_FIELDS}
    payload["report_md"] = job.get("report_md") if include_report else None
    payload["report_ready"] = bool(job.get("report_ready") or job.get("report_md"))
    return JobResponse.model_validate(payload)


@router.get("/meta", response_model=MetaResponse)
async def meta(request: Request) -> MetaResponse:
    await enforce_rate_limit(request, "api", _settings(request).api_rate_limit_per_minute)
    settings = _settings(request)
    embedding_model = (
        settings.embedding_model if settings.embedding_provider == "fastembed" else "hash-384"
    )
    return MetaResponse(
        name="vellum",
        version=__version__,
        llm_provider=settings.llm_provider,
        llm_model=settings.resolved_model(),
        embedding_provider=settings.embedding_provider,
        embedding_model=embedding_model,
        max_revision_loops=settings.max_revision_loops,
    )


@router.get("/examples")
async def examples(request: Request) -> dict[str, list[str]]:
    await enforce_rate_limit(request, "api", _settings(request).api_rate_limit_per_minute)
    return {"examples": EXAMPLES}


@router.post("/jobs", response_model=JobResponse, status_code=202)
async def create_job(
    payload: CreateJobRequest,
    request: Request,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
) -> JobResponse:
    settings = _settings(request)
    await enforce_rate_limit(request, "api", settings.api_rate_limit_per_minute)
    await enforce_rate_limit(request, "jobs", settings.job_rate_limit_per_minute)
    key = _clean_idempotency_key(idempotency_key)
    try:
        job = await _service(request).submit(payload.question, key)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except QueueUnavailable as exc:
        raise HTTPException(status_code=503, detail="Job queue is unavailable") from exc
    return _job_response(job, include_report=False)


@router.get("/jobs", response_model=list[JobResponse])
async def list_jobs(
    request: Request,
    limit: int = Query(default=20, ge=1, le=50),
) -> list[JobResponse]:
    await enforce_rate_limit(request, "api", _settings(request).api_rate_limit_per_minute)
    jobs = await _service(request).list_recent(limit)
    return [_job_response(job, include_report=False) for job in jobs]


@router.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job(job_id: str, request: Request) -> JobResponse:
    await enforce_rate_limit(request, "api", _settings(request).api_rate_limit_per_minute)
    job = await _service(request).get(_canonical_job_id(job_id))
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return _job_response(job, include_report=True)


@router.get("/jobs/{job_id}/log")
async def list_events(job_id: str, request: Request) -> list[dict[str, Any]]:
    await enforce_rate_limit(request, "api", _settings(request).api_rate_limit_per_minute)
    service = _service(request)
    job_id = _canonical_job_id(job_id)
    job = await service.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return await service.list_events(job_id)


@router.get("/jobs/{job_id}/events")
async def stream_events(job_id: str, request: Request) -> StreamingResponse:
    service = _service(request)
    job_id = _canonical_job_id(job_id)
    job = await service.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    async def generate() -> AsyncIterator[str]:
        async for event in service.iter_events(job_id):
            if await request.is_disconnected():
                break
            if event.get("kind") == "keepalive":
                yield ": ping\n\n"
                continue
            yield "data: " + json.dumps(event, ensure_ascii=False) + "\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/jobs/{job_id}/report")
async def download_report(
    job_id: str,
    request: Request,
    format: str = Query(default="md", pattern="^(md|html|pdf)$"),
) -> Response:
    await enforce_rate_limit(request, "api", _settings(request).api_rate_limit_per_minute)
    try:
        body, media_type, filename = await _service(request).report(
            _canonical_job_id(job_id),
            format,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Report not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return Response(
        content=body,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


def _canonical_job_id(job_id: str) -> str:
    try:
        return str(uuid.UUID(job_id))
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Job not found") from exc


def _clean_idempotency_key(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip()
    if not cleaned or len(cleaned) > 128:
        raise HTTPException(status_code=400, detail="Idempotency-Key must be 1-128 characters")
    if any(char.isspace() for char in cleaned):
        raise HTTPException(status_code=400, detail="Idempotency-Key cannot contain whitespace")
    return cleaned
