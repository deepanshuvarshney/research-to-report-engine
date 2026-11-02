from __future__ import annotations

import json
import uuid
from typing import Any

import pytest
from fastapi.testclient import TestClient

from research_engine.agents.graph import GraphDeps, run_research_graph
from research_engine.agents.render import render_html, render_markdown, render_pdf
from research_engine.config import Settings
from research_engine.jobs.service import JobService, QueueUnavailable
from research_engine.llm.base import ChatMessage
from research_engine.llm.mock import MockLLM
from research_engine.main import create_app
from research_engine.rag.embeddings import HashEmbedder
from research_engine.rag.store import MemoryVectorStore
from research_engine.search.cache import MemoryCache
from research_engine.search.models import SearchBatch, Source
from tests.support import make_settings

ATTENTION = (
    "Attention allows a model to weigh input tokens by relevance. "
    "Transformers replace recurrence with self-attention and allow tokens to interact in parallel."
)
RETRIEVAL = (
    "Retrieval-augmented generation combines a language model with a retriever that fetches "
    "documents at inference time. This grounds answers in retrieved text."
)


def _sources() -> list[Source]:
    return [
        Source(
            uid="s1",
            source_type="arxiv",
            title="Attention Is All You Need",
            url="https://arxiv.org/abs/1706.03762",
            authors="Vaswani et al.",
            year="2017",
            content=ATTENTION,
        ),
        Source(
            uid="s2",
            source_type="wikipedia",
            title="Retrieval-augmented generation",
            url="https://en.wikipedia.org/wiki/Retrieval-augmented_generation",
            content=RETRIEVAL,
        ),
    ]


class FakeSearch:
    def __init__(self, sources: list[Source] | None) -> None:
        self.sources = list(sources or [])

    async def gather(self, queries: list[str]) -> SearchBatch:
        del queries
        return SearchBatch(sources=list(self.sources), notes=["arxiv: cached"])

    async def aclose(self) -> None:
        return None


class BoomLLM:
    name = "boom"
    model = "boom"

    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, messages: list[ChatMessage], **kwargs: Any) -> str:
        del messages, kwargs
        self.calls += 1
        raise RuntimeError("provider down")

    async def aclose(self) -> None:
        return None


def _deps(
    settings: Settings, llm: Any, search: FakeSearch
) -> tuple[GraphDeps, list[dict[str, Any]]]:
    events: list[dict[str, Any]] = []

    async def emit(
        agent: str,
        kind: str,
        message: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        events.append({"agent": agent, "kind": kind, "message": message, "payload": payload or {}})

    deps = GraphDeps(
        settings=settings,
        llm=llm,
        search=search,
        vectors=MemoryVectorStore(),
        embeddings=HashEmbedder(384),
        emit=emit,
    )
    return deps, events


@pytest.mark.asyncio
async def test_heuristic_graph_quotes_sources() -> None:
    settings = make_settings()
    deps, events = _deps(settings, MockLLM(), FakeSearch(_sources()))
    result = await run_research_graph(
        "How does attention support retrieval-augmented generation?",
        "job-heuristic",
        deps,
    )
    report = str(result["report_md"])
    assert "[1]" in report
    assert "Attention allows a model to weigh input tokens by relevance." in report
    assert result["revision"] == 0
    assert result["support_score"] == 1.0
    assert any(event["kind"] == "critic.completed" for event in events)
    assert "References" in report


@pytest.mark.asyncio
async def test_revision_loop_drops_ungrounded_claim() -> None:
    writer_bad = {
        "title": "Attention and retrieval",
        "summary": "A draft that includes one unsupported sentence.",
        "sections": [
            {
                "heading": "Attention",
                "paragraphs": [
                    {
                        "text": "Attention allows a model to weigh input tokens by relevance.",
                        "citations": [1],
                    }
                ],
            },
            {
                "heading": "Risks",
                "paragraphs": [
                    {
                        "text": "The moon landing was filmed on a soundstage.",
                        "citations": [2],
                    }
                ],
            },
        ],
        "claims": [
            {
                "text": "Attention allows a model to weigh input tokens by relevance.",
                "citations": [1],
            },
            {"text": "The moon landing was filmed on a soundstage.", "citations": [2]},
        ],
    }
    writer_fixed = {
        "title": "Attention and retrieval",
        "summary": "Only the grounded claim remains.",
        "sections": [
            {
                "heading": "Attention",
                "paragraphs": [
                    {
                        "text": "Attention allows a model to weigh input tokens by relevance.",
                        "citations": [1],
                    }
                ],
            }
        ],
        "claims": [
            {
                "text": "Attention allows a model to weigh input tokens by relevance.",
                "citations": [1],
            }
        ],
    }
    critic_lenient = {
        "verdicts": [
            {
                "claim": "Attention allows a model to weigh input tokens by relevance.",
                "supported": True,
                "reason": "Present in the source.",
            },
            {
                "claim": "The moon landing was filmed on a soundstage.",
                "supported": True,
                "reason": "The model guessed.",
            },
        ],
        "revision_instructions": "",
    }
    critic_ok = {
        "verdicts": [
            {
                "claim": "Attention allows a model to weigh input tokens by relevance.",
                "supported": True,
                "reason": "Quoted from the source.",
            }
        ],
        "revision_instructions": "",
    }
    grounded = "Attention allows a model to weigh input tokens by relevance."
    stable_sources = [
        Source(
            uid="s1",
            source_type="arxiv",
            title="Attention Is All You Need",
            url="https://arxiv.org/abs/1706.03762",
            authors="Vaswani et al.",
            year="2017",
            content=f"{grounded} Transformers replace recurrence with self-attention.",
        ),
        Source(
            uid="s2",
            source_type="wikipedia",
            title="Retrieval-augmented generation",
            url="https://en.wikipedia.org/wiki/Retrieval-augmented_generation",
            content=f"{grounded} {RETRIEVAL}",
        ),
    ]
    llm = MockLLM(
        script=[
            json.dumps(
                {
                    "sub_questions": ["What is self-attention?", "How does retrieval help?"],
                    "rationale": "Split the mechanisms.",
                }
            ),
            json.dumps(writer_bad),
            json.dumps(critic_lenient),
            json.dumps(writer_fixed),
            json.dumps(critic_ok),
        ]
    )
    settings = make_settings()
    deps, events = _deps(settings, llm, FakeSearch(stable_sources))
    result = await run_research_graph("How does attention work with retrieval?", "job-revise", deps)
    report = str(result["report_md"])
    assert "soundstage" not in report
    assert "Attention allows a model to weigh input tokens by relevance." in report
    assert result["revision"] == 1
    assert result["support_score"] == 1.0
    assert [event["kind"] for event in events].count("writer.completed") == 2
    assert len(llm.calls) == 5
    assert not llm.script


@pytest.mark.asyncio
async def test_graph_refuses_to_invent_when_search_is_empty() -> None:
    settings = make_settings()
    llm = MockLLM()
    deps, _events = _deps(settings, llm, FakeSearch([]))
    result = await run_research_graph("What is a transformer model?", "job-empty", deps)
    assert "did not retrieve" in str(result["report_md"])
    assert len(llm.calls) == 1


@pytest.mark.asyncio
async def test_graph_falls_back_when_provider_is_down() -> None:
    settings = make_settings()
    llm = BoomLLM()
    deps, events = _deps(settings, llm, FakeSearch(_sources()))
    result = await run_research_graph(
        "How does attention support retrieval-augmented generation?",
        "job-boom",
        deps,
    )
    assert "[1]" in str(result["report_md"])
    assert llm.calls == 3
    assert any(event["kind"] == "writer.fallback" for event in events)


def test_render_exports() -> None:
    markdown = render_markdown(
        question="How does attention work in transformers today?",
        draft=None,
        sources=[],
        verdicts=[],
        subquestions=[],
        revisions=0,
    )
    assert "No sourced report" in markdown
    html = render_html("# Title\n\nA claim [1].\n")
    assert html.startswith("<!DOCTYPE html>")
    assert 'class="cite"' in html
    pdf = render_pdf(markdown)
    assert pdf.startswith(b"%PDF")


class MemRepo:
    def __init__(self) -> None:
        self.jobs: dict[str, dict[str, Any]] = {}
        self.events: list[dict[str, Any]] = []

    async def create_job(self, *, question: str, provider: str, model: str) -> dict[str, Any]:
        job_id = str(uuid.uuid4())
        job = {
            "id": job_id,
            "question": question,
            "status": "queued",
            "error": None,
            "provider": provider,
            "model": model,
            "support_score": None,
            "stats": {},
            "report_md": None,
            "report_html": None,
            "created_at": "2026-01-01T00:00:00+00:00",
            "updated_at": "2026-01-01T00:00:00+00:00",
            "report_ready": False,
        }
        self.jobs[job_id] = job
        return job

    async def get_job(self, job_id: str) -> dict[str, Any] | None:
        return self.jobs.get(job_id)

    async def list_jobs(self, limit: int) -> list[dict[str, Any]]:
        return list(self.jobs.values())[:limit]

    async def update_job(self, job_id: str, **fields: Any) -> None:
        self.jobs[job_id].update(fields)
        if "report_md" in fields:
            self.jobs[job_id]["report_ready"] = bool(fields["report_md"])

    async def add_event(
        self,
        job_id: str,
        *,
        agent: str,
        kind: str,
        message: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        seq = 1 + sum(1 for event in self.events if event["job_id"] == job_id)
        event = {
            "job_id": job_id,
            "seq": seq,
            "agent": agent,
            "kind": kind,
            "message": message,
            "payload": payload or {},
            "created_at": None,
        }
        self.events.append(event)
        return event

    async def list_events(self, job_id: str) -> list[dict[str, Any]]:
        return [event for event in self.events if event["job_id"] == job_id]


class MemQueue:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.ids: list[str] = []

    async def enqueue_research(self, job_id: str) -> None:
        if self.fail:
            raise ConnectionError("redis down")
        self.ids.append(job_id)


@pytest.mark.asyncio
async def test_job_service_idempotency_and_queue_failure() -> None:
    repo = MemRepo()
    queue = MemQueue()
    service = JobService(
        settings=make_settings(),
        repo=repo,
        queue=queue,
        cache=MemoryCache(),
        redis=None,
    )
    question = "How does retrieval-augmented generation reduce hallucinations?"
    first = await service.submit(question, "key-1")
    second = await service.submit(question, "key-1")
    assert first["id"] == second["id"]
    assert queue.ids == [first["id"]]

    failing = JobService(
        settings=make_settings(),
        repo=repo,
        queue=MemQueue(fail=True),
        cache=MemoryCache(),
        redis=None,
    )
    with pytest.raises(QueueUnavailable):
        await failing.submit(question, "key-2")
    failed = next(job for job in repo.jobs.values() if job["status"] == "failed")
    assert "Queue error" in str(failed["error"])
    retry_queue = MemQueue()
    failing.queue = retry_queue
    retried = await failing.submit(question, "key-2")
    assert retried["id"] != failed["id"]
    assert retry_queue.ids == [retried["id"]]

    groq_settings = make_settings(llm_provider="groq", groq_api_key="")
    service.settings = groq_settings
    with pytest.raises(RuntimeError):
        await service.submit(question, None)


class FakeApi:
    def __init__(self) -> None:
        self.jobs: dict[str, dict[str, Any]] = {}
        self.event_log: dict[str, list[dict[str, Any]]] = {}
        self.keys: list[str | None] = []

    async def submit(
        self,
        question: str,
        idempotency_key: str | None,
        client_ip: str = "",
    ) -> dict[str, Any]:
        del client_ip
        self.keys.append(idempotency_key)
        job_id = str(uuid.uuid4())
        job = {
            "id": job_id,
            "question": question,
            "status": "queued",
            "error": None,
            "provider": "mock",
            "model": "mock-extractive",
            "support_score": None,
            "stats": {},
            "report_md": None,
            "created_at": "2026-01-01T00:00:00+00:00",
            "updated_at": "2026-01-01T00:00:00+00:00",
            "report_ready": False,
        }
        self.jobs[job_id] = job
        self.event_log[job_id] = [
            {
                "seq": 1,
                "agent": "job",
                "kind": "job.queued",
                "message": "Queued for a worker",
                "payload": {},
            }
        ]
        return job

    async def get(self, job_id: str) -> dict[str, Any] | None:
        return self.jobs.get(job_id)

    async def list_recent(self, limit: int) -> list[dict[str, Any]]:
        return list(self.jobs.values())[:limit]

    async def list_events(self, job_id: str) -> list[dict[str, Any]]:
        return list(self.event_log.get(job_id, []))

    async def iter_events(self, job_id: str):  # type: ignore[no-untyped-def]
        for event in self.event_log.get(job_id, []):
            yield event

    async def report(self, job_id: str, fmt: str) -> tuple[bytes, str, str]:
        job = self.jobs[job_id]
        if not job.get("report_md"):
            raise KeyError(job_id)
        body = str(job["report_md"]).encode()
        return body, "text/markdown; charset=utf-8", f"report.{fmt}"


def test_api_job_flow_and_validation() -> None:
    fake = FakeApi()
    app = create_app(settings=make_settings(), job_service=fake)  # type: ignore[arg-type]
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.headers["x-request-id"]
        meta = client.get("/api/meta")
        assert meta.json()["llm_provider"] == "mock"
        too_short = client.post("/api/jobs", json={"question": "too short"})
        assert too_short.status_code == 422
        created = client.post(
            "/api/jobs",
            json={"question": "How does retrieval-augmented generation reduce hallucinations?"},
            headers={"Idempotency-Key": "demo-key"},
        )
        assert created.status_code == 202
        job_id = created.json()["id"]
        assert fake.keys == ["demo-key"]
        assert client.get(f"/api/jobs/{job_id}").json()["status"] == "queued"
        assert client.get("/api/jobs/not-a-uuid").status_code == 404
        log = client.get(f"/api/jobs/{job_id}/log")
        assert log.json()[0]["kind"] == "job.queued"
        with client.stream("GET", f"/api/jobs/{job_id}/events") as stream:
            assert stream.status_code == 200
            payload = "".join(stream.iter_text())
        assert "job.queued" in payload
        missing = client.get(f"/api/jobs/{job_id}/report?format=md")
        assert missing.status_code == 404
        fake.jobs[job_id]["report_md"] = "# Done\n\nA cited claim [1].\n"
        fake.jobs[job_id]["status"] = "completed"
        fake.jobs[job_id]["report_ready"] = True
        downloaded = client.get(f"/api/jobs/{job_id}/report?format=md")
        assert downloaded.status_code == 200
        assert "cited claim" in downloaded.text
        assert client.get("/").status_code == 200
        home = client.get("/").text
        assert "Vellum" in home
        assert "allowlisted" in home


def test_api_rate_limit() -> None:
    class Counter:
        def __init__(self) -> None:
            self.values: dict[str, int] = {}

        async def incr(self, key: str) -> int:
            self.values[key] = self.values.get(key, 0) + 1
            return self.values[key]

        async def expire(self, key: str, seconds: int) -> bool:
            del key, seconds
            return True

    fake = FakeApi()
    app = create_app(
        settings=make_settings(api_rate_limit_per_minute=100, job_rate_limit_per_minute=2),
        job_service=fake,  # type: ignore[arg-type]
    )
    with TestClient(app) as client:
        client.app.state.redis = Counter()
        question = {"question": "How does retrieval-augmented generation reduce hallucinations?"}
        assert client.post("/api/jobs", json=question).status_code == 202
        assert client.post("/api/jobs", json=question).status_code == 202
        blocked = client.post("/api/jobs", json=question)
        assert blocked.status_code == 429
