"""LangGraph research loop.

plan -> search -> index -> write -> critique
                              ^         |
                              +---------+  (bounded)

Search failures do not fail the job. If nothing is retrieved, the graph stops
before the writer and publishes an explicit "no sources" report.
"""

from __future__ import annotations

from collections import defaultdict
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Protocol, TypedDict

import structlog
from langgraph.graph import END, START, StateGraph

from research_engine.agents.fallback import extractive_draft, heuristic_plan, heuristic_verdicts
from research_engine.agents.models import Claim, Critique, Draft, Plan, normalize_draft
from research_engine.agents.prompts import critic_messages, planner_messages, writer_messages
from research_engine.agents.render import render_html, render_markdown
from research_engine.config import Settings
from research_engine.faithfulness import backstop_rejects, claim_supported
from research_engine.llm.base import LLMProvider
from research_engine.llm.jsonutil import complete_json
from research_engine.observability.tracing import span
from research_engine.rag.chunker import chunk_text
from research_engine.rag.embeddings import Embedder
from research_engine.rag.store import ChunkHit, StoredChunk, VectorStore
from research_engine.search.models import SearchBatch, Source

log = structlog.get_logger(__name__)


class Emitter(Protocol):
    async def __call__(
        self,
        agent: str,
        kind: str,
        message: str,
        payload: dict[str, Any] | None = None,
    ) -> None: ...


class ResearchState(TypedDict, total=False):
    job_id: str
    question: str
    max_revisions: int
    sub_questions: list[str]
    plan_rationale: str
    sources: list[dict[str, Any]]
    selected_sources: list[dict[str, Any]]
    search_notes: list[str]
    draft: dict[str, Any]
    critique: dict[str, Any]
    verdicts: list[dict[str, Any]]
    revision: int
    revision_instructions: str
    chunk_count: int
    report_md: str
    report_html: str
    support_score: float | None


@dataclass
class GraphDeps:
    settings: Settings
    llm: LLMProvider
    search: Any
    vectors: VectorStore
    embeddings: Embedder
    emit: Emitter


_deps: ContextVar[GraphDeps] = ContextVar("vellum_graph_deps")


def get_deps() -> GraphDeps:
    return _deps.get()


def _sources(raw: list[dict[str, Any]] | None) -> list[Source]:
    return [Source.model_validate(item) for item in raw or []]


async def plan_node(state: ResearchState) -> dict[str, Any]:
    deps = get_deps()
    question = state["question"]
    await deps.emit("planner", "planner.started", "Breaking the question into sub-questions")
    with span("agent.plan"):
        try:
            plan = await complete_json(
                deps.llm,
                planner_messages(question, deps.settings.max_subquestions),
                Plan,
                temperature=deps.settings.llm_temperature,
                max_tokens=deps.settings.llm_max_tokens,
            )
            subs = [item.strip()[:240] for item in plan.sub_questions if item.strip()]
            subs = subs[: deps.settings.max_subquestions]
            if not subs:
                raise ValueError("planner returned no sub-questions")
            rationale = plan.rationale.strip()
        except Exception as exc:
            log.warning("planner_fallback", error=str(exc)[:300])
            fallback = heuristic_plan(question, deps.settings.max_subquestions)
            subs = list(fallback["sub_questions"])
            rationale = str(fallback["rationale"]) + " Heuristic split used."
            await deps.emit(
                "planner",
                "planner.fallback",
                "Planner response failed; using a heuristic split",
            )
    await deps.emit(
        "planner",
        "planner.completed",
        f"Planned {len(subs)} sub-questions",
        {"sub_questions": subs},
    )
    return {"sub_questions": subs, "plan_rationale": rationale}


async def search_node(state: ResearchState) -> dict[str, Any]:
    deps = get_deps()
    queries: list[str] = []
    seen: set[str] = set()
    raw_queries = [state["question"], *(state.get("sub_questions") or [])]
    for item in raw_queries:
        cleaned = " ".join(str(item).split())[:240]
        key = cleaned.lower()
        if not cleaned or key in seen:
            continue
        seen.add(key)
        queries.append(cleaned)
        if len(queries) >= deps.settings.max_subquestions + 1:
            break
    if not queries:
        queries = [state["question"]]
    await deps.emit(
        "search",
        "search.started",
        f"Searching {len(queries)} sub-questions across papers and the web",
    )
    with span("agent.search", queries=len(queries)):
        batch: SearchBatch = await deps.search.gather(queries)
    sources = [source.model_dump() for source in batch.sources]
    await deps.emit(
        "search",
        "search.completed",
        f"Kept {len(sources)} sources after dedupe",
        {"notes": batch.notes, "source_count": len(sources)},
    )
    return {"sources": sources, "search_notes": batch.notes}


def route_after_search(state: ResearchState) -> str:
    if state.get("sources"):
        return "index"
    return "finalize"


async def index_node(state: ResearchState) -> dict[str, Any]:
    deps = get_deps()
    sources = _sources(state.get("sources"))
    await deps.emit("index", "index.started", f"Chunking and embedding {len(sources)} sources")
    with span("agent.index", sources=len(sources)):
        pending: list[tuple[Source, int, str]] = []
        for source in sources:
            pieces = chunk_text(
                source.content,
                size=deps.settings.chunk_size,
                overlap=deps.settings.chunk_overlap,
            )
            for ordinal, piece in enumerate(pieces):
                pending.append((source, ordinal, piece))
        vectors = await deps.embeddings.embed([piece for _, _, piece in pending]) if pending else []
        chunks = [
            StoredChunk(
                job_id=state["job_id"],
                source_uid=source.uid,
                source_ref=index,
                ordinal=ordinal,
                text=piece,
                embedding=vector,
            )
            for index, ((source, ordinal, piece), vector) in enumerate(
                zip(pending, vectors, strict=True),
                start=1,
            )
        ]
        await deps.vectors.replace(state["job_id"], chunks)
        selected = await _select_sources(state, sources, deps)
    await deps.emit(
        "index",
        "index.completed",
        f"Indexed {len(chunks)} chunks; writer will see {len(selected)} sources",
        {"chunk_count": len(chunks), "source_count": len(selected)},
    )
    return {
        "selected_sources": [source.model_dump() for source in selected],
        "chunk_count": len(chunks),
    }


async def _select_sources(
    state: ResearchState,
    sources: list[Source],
    deps: GraphDeps,
) -> list[Source]:
    scores: dict[str, float] = defaultdict(float)
    extras: dict[str, list[tuple[float, str]]] = defaultdict(list)
    for subquestion in state.get("sub_questions") or [state["question"]]:
        embedding = (await deps.embeddings.embed([subquestion]))[0]
        hits: list[ChunkHit] = await deps.vectors.search(
            state["job_id"],
            embedding,
            deps.settings.retrieval_k,
        )
        for hit in hits:
            scores[hit.source_uid] += hit.score
            extras[hit.source_uid].append((hit.score, hit.text))

    if scores and max(scores.values()) > 0:
        ranked = sorted(sources, key=lambda source: scores.get(source.uid, 0.0), reverse=True)
    else:
        ranked = list(sources)
    chosen = ranked[: deps.settings.max_sources_for_writer]
    limit = deps.settings.source_prompt_chars
    prepared: list[Source] = []
    for ref, source in enumerate(chosen, start=1):
        ranked_chunks = [text for _score, text in sorted(extras[source.uid], reverse=True)]
        prompt_text = _prompt_text(source.content, ranked_chunks, limit)
        prepared.append(source.model_copy(update={"ref": ref, "prompt_text": prompt_text}))
    return prepared


def _prompt_text(content: str, chunks: list[str], limit: int) -> str:
    lead = content[:limit].strip()
    parts = [lead] if lead else []
    used = len(lead)
    seen = {lead}
    for chunk in chunks:
        snippet = " ".join(chunk.split())
        if not snippet or snippet in seen or snippet in lead:
            continue
        remaining = limit - used
        if remaining < 80:
            break
        snippet = snippet[:remaining].strip()
        parts.append(snippet)
        seen.add(snippet)
        used += len(snippet)
    return "\n".join(parts)


async def write_node(state: ResearchState) -> dict[str, Any]:
    deps = get_deps()
    sources = _sources(state.get("selected_sources"))
    revision = int(state.get("revision") or 0)
    instructions = str(state.get("revision_instructions") or "")
    if revision == 0:
        await deps.emit("writer", "writer.started", "Drafting a report with inline citations")
    else:
        await deps.emit(
            "writer",
            "writer.revising",
            f"Revising unsupported claims (pass {revision})",
        )
    with span("agent.write", revision=revision):
        try:
            draft = await complete_json(
                deps.llm,
                writer_messages(
                    state["question"], list(state.get("sub_questions") or []), sources, instructions
                ),
                Draft,
                temperature=0.1 if revision else deps.settings.llm_temperature,
                max_tokens=deps.settings.llm_max_tokens,
            )
            draft = normalize_draft(draft, {source.ref for source in sources})
        except Exception as exc:
            log.warning("writer_fallback", error=str(exc)[:300])
            draft = normalize_draft(
                extractive_draft(state["question"], sources),
                {source.ref for source in sources},
            )
            await deps.emit(
                "writer",
                "writer.fallback",
                "Writer response failed; quoting sources directly",
            )
    await deps.emit(
        "writer",
        "writer.completed",
        f"Draft has {len(draft.claims)} claims",
        {"claims": len(draft.claims), "revision": revision},
    )
    return {"draft": draft.model_dump()}


async def critique_node(state: ResearchState) -> dict[str, Any]:
    deps = get_deps()
    sources = _sources(state.get("selected_sources"))
    draft = Draft.model_validate(state.get("draft") or {})
    await deps.emit(
        "critic",
        "critic.started",
        f"Checking {len(draft.claims)} claims against cited source text",
    )
    with span("agent.critique", claims=len(draft.claims)):
        try:
            critique = await complete_json(
                deps.llm,
                critic_messages(state["question"], draft.claims, sources),
                Critique,
                temperature=0.0,
                max_tokens=deps.settings.llm_max_tokens,
            )
        except Exception as exc:
            log.warning("critic_fallback", error=str(exc)[:300])
            critique = heuristic_verdicts(draft.claims, sources)
            await deps.emit(
                "critic",
                "critic.fallback",
                "Critic response failed; using the overlap check",
            )
        verdicts = _apply_verdicts(draft.claims, critique, sources)
    unsupported = [item for item in verdicts if not item["supported"]]
    revision = int(state.get("revision") or 0)
    needs_revision = bool(unsupported) and revision < int(state.get("max_revisions") or 0)
    instructions = critique.revision_instructions.strip()
    if needs_revision and not instructions:
        instructions = "Rewrite or remove these unsupported claims: " + " | ".join(
            str(item["text"]) for item in unsupported
        )
    supported = len(verdicts) - len(unsupported)
    await deps.emit(
        "critic",
        "critic.completed",
        f"{supported}/{len(verdicts)} claims supported"
        + ("; sending the draft back" if needs_revision else ""),
        {"supported": supported, "total": len(verdicts), "needs_revision": needs_revision},
    )
    return {
        "verdicts": verdicts,
        "critique": {
            "needs_revision": needs_revision,
            "revision_instructions": instructions,
        },
        "revision": revision + (1 if needs_revision else 0),
        "revision_instructions": instructions if needs_revision else "",
    }


def _apply_verdicts(
    claims: list[Claim],
    critique: Critique,
    sources: list[Source],
) -> list[dict[str, Any]]:
    by_text = {verdict.claim.strip(): verdict for verdict in critique.verdicts if verdict.claim}
    evidence = {source.ref: source.content for source in sources}
    merged: list[dict[str, Any]] = []
    for index, claim in enumerate(claims):
        verdict = None
        if index < len(critique.verdicts):
            candidate = critique.verdicts[index]
            if not candidate.claim or candidate.claim.strip() == claim.text.strip():
                verdict = candidate
        if verdict is None:
            verdict = by_text.get(claim.text.strip())
        if verdict is None:
            supported = False
            reason = "Critic omitted a verdict for this claim."
        else:
            supported = verdict.supported
            reason = verdict.reason or ("Supported." if verdict.supported else "Unsupported.")
        _lexical_ok, ratio, lexical_reason = claim_supported(
            claim.text,
            list(claim.citations),
            evidence,
        )
        if not claim.citations:
            supported = False
            reason = "Claim has no citation."
        elif supported and backstop_rejects(claim.text, list(claim.citations), evidence):
            supported = False
            reason = f"Cited text does not ground this claim ({lexical_reason})."
        elif not supported and not reason:
            reason = lexical_reason
        merged.append(
            {
                "text": claim.text,
                "citations": list(claim.citations),
                "supported": supported,
                "reason": reason,
                "overlap": round(ratio, 4),
            }
        )
    return merged


def route_after_critique(state: ResearchState) -> str:
    critique = state.get("critique") or {}
    if critique.get("needs_revision"):
        return "write"
    return "finalize"


async def finalize_node(state: ResearchState) -> dict[str, Any]:
    deps = get_deps()
    selected = _sources(state.get("selected_sources"))
    raw_sources = _sources(state.get("sources"))
    sources = selected or raw_sources
    if sources and all(source.ref == 0 for source in sources):
        sources = [
            source.model_copy(update={"ref": index})
            for index, source in enumerate(sources, start=1)
        ]
    draft_raw = state.get("draft")
    draft = Draft.model_validate(draft_raw) if draft_raw else None
    if draft is not None and not draft.claims and not draft.sections:
        draft = None
    verdicts = list(state.get("verdicts") or [])
    revisions_done = int(state.get("revision") or 0)
    markdown = render_markdown(
        question=state["question"],
        draft=draft if sources else None,
        sources=sources if draft else [],
        verdicts=verdicts,
        subquestions=list(state.get("sub_questions") or []),
        revisions=revisions_done,
    )
    # An empty-source run still explains itself. A draft-less run with sources
    # should not pretend those sources were cited.
    if draft is None:
        markdown = render_markdown(
            question=state["question"],
            draft=None,
            sources=[],
            verdicts=[],
            subquestions=list(state.get("sub_questions") or []),
            revisions=revisions_done,
        )
        sources_for_score: list[Source] = []
    else:
        sources_for_score = sources
    supported = sum(1 for verdict in verdicts if verdict.get("supported"))
    total = len(verdicts)
    score = (supported / total) if total else None
    await deps.emit(
        "report",
        "report.ready",
        "Report assembled",
        {"support_score": score, "claims": total},
    )
    return {
        "report_md": markdown,
        "report_html": render_html(markdown),
        "support_score": score,
        "selected_sources": [source.model_dump() for source in sources_for_score],
        "verdicts": verdicts,
    }


def build_graph() -> Any:
    builder: StateGraph = StateGraph(ResearchState)
    builder.add_node("plan", plan_node)
    builder.add_node("search", search_node)
    builder.add_node("index", index_node)
    builder.add_node("write", write_node)
    builder.add_node("critique", critique_node)
    builder.add_node("finalize", finalize_node)
    builder.add_edge(START, "plan")
    builder.add_edge("plan", "search")
    builder.add_conditional_edges(
        "search",
        route_after_search,
        {"index": "index", "finalize": "finalize"},
    )
    builder.add_edge("index", "write")
    builder.add_edge("write", "critique")
    builder.add_conditional_edges(
        "critique",
        route_after_critique,
        {"write": "write", "finalize": "finalize"},
    )
    builder.add_edge("finalize", END)
    return builder.compile()


_compiled: Any = None


def get_graph() -> Any:
    global _compiled
    if _compiled is None:
        _compiled = build_graph()
    return _compiled


async def run_research_graph(question: str, job_id: str, deps: GraphDeps) -> ResearchState:
    token = _deps.set(deps)
    try:
        with span("research.job", job_id=job_id):
            result: ResearchState = await get_graph().ainvoke(
                {
                    "job_id": job_id,
                    "question": question,
                    "max_revisions": deps.settings.max_revision_loops,
                    "revision": 0,
                    "revision_instructions": "",
                    "sub_questions": [],
                    "sources": [],
                    "selected_sources": [],
                    "verdicts": [],
                },
                config={"recursion_limit": 25},
            )
            return result
    finally:
        _deps.reset(token)
