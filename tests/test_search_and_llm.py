from __future__ import annotations

import json

import httpx
import pytest
from tests.conftest import make_settings

from research_engine.agents.models import Plan
from research_engine.agents.prompts import planner_messages, sources_from_prompt
from research_engine.llm.base import ChatMessage
from research_engine.llm.gemini import _to_gemini
from research_engine.llm.groq import GroqLLM
from research_engine.llm.http import scrub
from research_engine.llm.jsonutil import complete_json, extract_first_json
from research_engine.llm.mock import MockLLM
from research_engine.search.arxiv import parse_arxiv
from research_engine.search.cache import MemoryCache
from research_engine.search.duckduckgo import decode_ddg_url, parse_duckduckgo
from research_engine.search.limiter import AsyncRateLimiter
from research_engine.search.models import dedupe_sources
from research_engine.search.semantic_scholar import parse_semantic_scholar
from research_engine.search.service import SearchService
from research_engine.search.terms import overlaps_topic, topic_phrase
from research_engine.search.wikipedia import parse_wikipedia

ARXIV_XML = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/1706.03762</id>
    <title>Attention Is All You Need</title>
    <summary>Transformers replace recurrence with self-attention and allow tokens to interact in parallel across long documents.</summary>
    <published>2017-06-12T00:00:00Z</published>
    <author><name>Ashish Vaswani</name></author>
  </entry>
</feed>
"""

S2_JSON = {
    "data": [
        {
            "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
            "abstract": "Retrieval-augmented generation combines a parametric language model with a non-parametric memory of documents retrieved at runtime.",
            "url": "https://www.semanticscholar.org/paper/rag",
            "year": 2020,
            "authors": [{"name": "Patrick Lewis"}],
        }
    ]
}

WIKI_JSON = {
    "query": {
        "pages": {
            "1": {
                "title": "Language model",
                "extract": (
                    "Retrieval-augmented generation is a language-model technique that "
                    "assigns probabilities using retrieved documents."
                ),
            }
        }
    }
}

DDG_HTML = """
<html><body>
  <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Frag&amp;rut=1">Retrieval augmented generation overview</a>
  <a class="result__snippet">Retrieval augmented generation fetches documents at inference time and conditions the model on them.</a>
</body></html>
"""


def test_topic_phrase_drops_question_words() -> None:
    phrase = topic_phrase("What are the main ideas behind retrieval-augmented generation?")
    assert phrase.lower() == "retrieval-augmented generation"
    papers = parse_arxiv(ARXIV_XML, content_limit=500)
    assert overlaps_topic(papers[0], ["transformers", "attention"])
    assert not overlaps_topic(papers[0], ["marvel", "cinematic"])


def test_source_parsers_and_dedupe() -> None:
    papers = parse_arxiv(ARXIV_XML, content_limit=500)
    assert papers[0].year == "2017"
    assert papers[0].authors == "Ashish Vaswani"
    scholar = parse_semantic_scholar(S2_JSON, content_limit=500)
    assert scholar[0].source_type == "semantic_scholar"
    wiki = parse_wikipedia(WIKI_JSON, content_limit=500)
    assert wiki[0].url.startswith("https://en.wikipedia.org/wiki/")
    web = parse_duckduckgo(DDG_HTML, content_limit=500)
    assert web[0].url == "https://example.com/rag"
    assert decode_ddg_url("//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fx") == (
        "https://example.com/x"
    )
    duplicate = papers[0].model_copy(update={"source_type": "web", "uid": "dup"})
    kept = dedupe_sources([duplicate, papers[0], scholar[0]], limit=5)
    titles = [source.title for source in kept]
    assert titles.count("Attention Is All You Need") == 1
    assert [source.source_type for source in kept[:2]] == ["semantic_scholar", "arxiv"]


def test_prompt_round_trip() -> None:
    source = parse_arxiv(ARXIV_XML, content_limit=500)[0].model_copy(update={"ref": 3})
    parsed = sources_from_prompt(source.prompt_block())
    assert parsed[0].ref == 3
    assert parsed[0].title == source.title
    assert source.content[:40] in parsed[0].content


@pytest.mark.asyncio
async def test_search_service_caches_and_isolates_failures() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        host = request.url.host
        if "arxiv" in host:
            return httpx.Response(503, text="busy")
        if "semanticscholar" in host:
            return httpx.Response(200, json=S2_JSON)
        if "wikipedia" in host:
            return httpx.Response(200, json=WIKI_JSON)
        if "duckduckgo" in host:
            return httpx.Response(200, text=DDG_HTML)
        return httpx.Response(404)

    settings = make_settings(max_results_per_provider=2, max_sources_total=8)
    cache = MemoryCache()
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    service = SearchService(settings=settings, cache=cache, client=client)
    first = await service.gather(["retrieval augmented generation"])
    after_first = calls["n"]
    second = await service.gather(["retrieval augmented generation"])
    await service.aclose()

    assert after_first == 4
    # Failed providers are not cached, so arXiv is contacted again. The other three are not.
    assert calls["n"] == after_first + 1
    assert any(note.startswith("arxiv: failed") for note in first.notes)
    assert {source.source_type for source in first.sources} == {
        "semantic_scholar",
        "wikipedia",
        "web",
    }
    assert [source.uid for source in second.sources] == [source.uid for source in first.sources]


@pytest.mark.asyncio
async def test_rate_limiter_spaces_calls() -> None:
    import time

    limiter = AsyncRateLimiter(20)
    started = time.perf_counter()
    for _ in range(4):
        await limiter.acquire()
    elapsed = time.perf_counter() - started
    assert elapsed >= 0.12


def test_json_extractor_ignores_fences() -> None:
    payload = extract_first_json('Sure\n```json\n{"sub_questions": ["a"], "rationale": "b"}\n```')
    assert payload["sub_questions"] == ["a"]


@pytest.mark.asyncio
async def test_complete_json_repairs_once() -> None:
    llm = MockLLM(
        script=[
            "not json",
            json.dumps({"sub_questions": ["What is attention?"], "rationale": "split"}),
        ]
    )
    plan = await complete_json(
        llm,
        planner_messages("How does attention work inside transformers?", 3),
        Plan,
        temperature=0,
        max_tokens=200,
    )
    assert plan.sub_questions == ["What is attention?"]
    assert len(llm.calls) == 2


def test_gemini_role_folding_and_secret_scrub() -> None:
    _system, contents = _to_gemini(
        [
            ChatMessage(role="system", content="Be brief."),
            ChatMessage(role="user", content="Hello"),
            ChatMessage(role="assistant", content="Hi"),
            ChatMessage(role="user", content="Again"),
        ]
    )
    assert [item["role"] for item in contents] == ["user", "model", "user"]
    assert scrub("key sk-test leaked", "sk-test") == "key *** leaked"


@pytest.mark.asyncio
async def test_groq_retries_transient_status() -> None:
    calls = {"n": 0}

    def handler(_request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, json={"error": "slow"})
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"ok": true}'}}]},
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    llm = GroqLLM(
        api_key="super-secret", model="llama-test", timeout=5, max_retries=2, client=client
    )
    text = await llm.complete([ChatMessage(role="user", content="hi")], json_mode=True)
    await llm.aclose()
    assert calls["n"] == 2
    assert "ok" in text
