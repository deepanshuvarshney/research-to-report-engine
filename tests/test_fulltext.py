"""Full-text parsing and download. Fixtures only: these tests do not use the network."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from fpdf import FPDF

from research_engine.agents.graph import GraphDeps, run_research_graph
from research_engine.llm.mock import MockLLM
from research_engine.papers.fetch import (
    download_pdf,
    enrich_with_fulltext,
    select_oa_pdf,
)
from research_engine.papers.pdf import heading_name, pages_to_passages, parse_pdf_bytes
from research_engine.papers.urls import arxiv_pdf_url, url_is_allowed
from research_engine.rag.embeddings import HashEmbedder
from research_engine.rag.store import MemoryVectorStore
from research_engine.search.arxiv import parse_arxiv
from research_engine.search.cache import MemoryCache
from research_engine.search.models import Passage, Source
from research_engine.search.semantic_scholar import FIELDS, parse_semantic_scholar
from tests.support import make_settings
from tests.test_graph_and_api import ATTENTION, FakeSearch

FIXTURES = Path(__file__).parent / "fixtures"

ARXIV_XML = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/1706.03762v1</id>
    <title>Attention Is All You Need</title>
    <summary>Transformers replace recurrence with self-attention and allow tokens to interact in parallel across long documents.</summary>
    <published>2017-06-12T00:00:00Z</published>
    <author><name>Ashish Vaswani</name></author>
  </entry>
</feed>
"""


def _pages_from_fixture() -> list[tuple[int, str]]:
    pages: list[tuple[int, str]] = []
    current = 0
    lines: list[str] = []
    for line in (FIXTURES / "paper_pages.txt").read_text(encoding="utf-8").splitlines():
        if line.startswith("PAGE "):
            if current:
                pages.append((current, "\n".join(lines)))
            current = int(line.split()[1])
            lines = []
            continue
        lines.append(line)
    if current:
        pages.append((current, "\n".join(lines)))
    return pages


def _write_sample_pdf(path: Path) -> None:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=False)
    pages = [
        [
            "Abstract",
            "Retrieval-augmented generation fetches documents at inference time.",
            "1 Introduction",
            "A retriever selects passages and the generator conditions on those passages.",
        ],
        [
            "2 Experiments",
            "The model reduces unsupported claims when the cited passage overlaps the answer.",
            "References",
            "Bibliography marker Vaswani 2017 should not be required for this test.",
        ],
    ]
    for lines in pages:
        pdf.add_page()
        pdf.set_font("Helvetica", size=12)
        for line in lines:
            pdf.cell(0, 8, line, new_x="LMARGIN", new_y="NEXT")
    path.write_bytes(bytes(pdf.output()))


def test_heading_and_page_chunking_uses_the_fixture() -> None:
    passages, truncated = pages_to_passages(
        _pages_from_fixture(),
        chunk_size=500,
        chunk_overlap=40,
        max_chunks=10,
    )
    assert truncated is False
    joined = " ".join(passage.text for passage in passages)
    assert "retriever selects passages" in joined
    assert "unsupported claims" in joined
    assert "bibliography marker" not in joined
    intro = next(passage for passage in passages if "retriever selects" in passage.text)
    experiment = next(passage for passage in passages if "unsupported claims" in passage.text)
    assert intro.section == "1 Introduction"
    assert intro.page == 1
    assert experiment.section == "2 Experiments"
    assert experiment.page == 2
    assert (
        heading_name("The model reduces unsupported claims when the cited passage overlaps.")
        is None
    )
    assert heading_name("References") == "References"


def test_pdf_fixture_extracts_text_and_pages() -> None:
    path = FIXTURES / "sample_paper.pdf"
    if not path.exists():
        _write_sample_pdf(path)
    parsed = parse_pdf_bytes(path.read_bytes(), max_pages=5, max_chunks=8, chunk_size=400)
    assert parsed.page_count == 2
    blob = " ".join(passage.text for passage in parsed.passages)
    assert "retriever selects passages" in blob
    assert "unsupported claims" in blob
    assert any(passage.page == 1 for passage in parsed.passages)
    assert any(passage.page == 2 for passage in parsed.passages)


def test_parsers_record_allowlisted_pdf_urls() -> None:
    paper = parse_arxiv(ARXIV_XML, content_limit=500)[0]
    assert paper.arxiv_id == "1706.03762v1"
    assert paper.pdf_url == "https://arxiv.org/pdf/1706.03762v1.pdf"
    assert url_is_allowed(paper.pdf_url)
    assert "openAccessPdf" in FIELDS
    scholar = parse_semantic_scholar(
        {
            "data": [
                {
                    "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
                    "abstract": "Retrieval-augmented generation combines a parametric language model with a non-parametric memory of documents retrieved at runtime.",
                    "url": "https://www.semanticscholar.org/paper/rag",
                    "year": 2020,
                    "authors": [{"name": "Patrick Lewis"}],
                    "externalIds": {"DOI": "10.18653/v1/2020.example", "ArXiv": "2005.11401"},
                    "openAccessPdf": {"url": "https://evil.example/secret.pdf"},
                }
            ]
        },
        content_limit=500,
    )[0]
    assert scholar.doi == "10.18653/v1/2020.example"
    assert scholar.arxiv_id == "2005.11401"
    assert scholar.pdf_url == arxiv_pdf_url("2005.11401")
    assert url_is_allowed("https://arxiv.org/pdf/1706.03762.pdf") is True
    assert url_is_allowed("https://evil.example/secret.pdf") is False
    assert url_is_allowed("http://arxiv.org/pdf/1706.03762.pdf") is False
    assert url_is_allowed("https://user:pass@arxiv.org/pdf/1706.03762.pdf") is False
    assert url_is_allowed("https://127.0.0.1/latest") is False
    assert url_is_allowed("https://169.254.169.254/latest") is False
    payload = json.loads((FIXTURES / "unpaywall.json").read_text(encoding="utf-8"))
    assert select_oa_pdf(payload) == "https://arxiv.org/pdf/1706.03762.pdf"


def test_pdf_bytes_round_trip_locator() -> None:
    path = FIXTURES / "sample_paper.pdf"
    if not path.exists():
        _write_sample_pdf(path)
    parsed = parse_pdf_bytes(path.read_bytes())
    labeled = next(passage for passage in parsed.passages if passage.page)
    assert labeled.page is not None
    assert labeled.text


@pytest.mark.asyncio
async def test_download_rejects_redirects_off_the_allowlist_and_caches() -> None:
    path = FIXTURES / "sample_paper.pdf"
    if not path.exists():
        _write_sample_pdf(path)
    pdf_bytes = path.read_bytes()
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.host)
        if request.url.host == "api.unpaywall.org":
            body = json.loads((FIXTURES / "unpaywall.json").read_text(encoding="utf-8"))
            return httpx.Response(200, json=body)
        if request.url.host == "127.0.0.1":
            raise AssertionError("refused host was requested")
        if str(request.url).endswith("/bounce"):
            return httpx.Response(302, headers={"location": "http://127.0.0.1/latest"})
        if request.url.host == "arxiv.org" and request.url.path.endswith(".pdf"):
            if request.url.params.get("hop") != "1":
                return httpx.Response(
                    302,
                    headers={"location": "https://arxiv.org/pdf/1706.03762.pdf?hop=1"},
                )
            return httpx.Response(200, content=pdf_bytes)
        if request.url.path.endswith("/not-pdf"):
            return httpx.Response(200, content=b"hello")
        if request.url.path.endswith("/missing"):
            return httpx.Response(404, content=b"nope")
        return httpx.Response(404)

    settings = make_settings(
        fulltext_max_bytes=2_000_000,
        fulltext_timeout_seconds=5,
        unpaywall_email="owner@example.com",
        chunk_size=400,
        chunk_overlap=40,
    )
    cache = MemoryCache()
    acquired: list[str] = []

    async def acquire(name: str) -> None:
        acquired.append(name)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(Exception, match="url_not_allowed"):
            await download_pdf(
                client,
                "https://arxiv.org/pdf/1706.03762.pdf/bounce",
                max_bytes=settings.fulltext_max_bytes,
                timeout_seconds=5,
            )
        assert "127.0.0.1" not in seen

        data = await download_pdf(
            client,
            "https://arxiv.org/pdf/1706.03762.pdf",
            max_bytes=settings.fulltext_max_bytes,
            timeout_seconds=5,
        )
        assert data.startswith(b"%PDF")

        source = Source(
            uid="s2",
            source_type="semantic_scholar",
            title="Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
            url="https://www.semanticscholar.org/paper/rag",
            content=ATTENTION,
            doi="10.18653/v1/2020.example",
        )
        enriched, note = await enrich_with_fulltext(
            [source],
            client=client,
            cache=cache,
            settings=settings,
            acquire=acquire,
        )
        assert enriched[0].text_origin == "pdf"
        assert enriched[0].passages
        assert "1 pdfs" in note
        calls_after = len(seen)
        again, _note = await enrich_with_fulltext(
            [source],
            client=client,
            cache=cache,
            settings=settings,
            acquire=acquire,
        )
        assert again[0].text_origin == "pdf"
        assert len(seen) == calls_after

        plain = Source(
            uid="bad",
            source_type="arxiv",
            title="Not a paper body",
            url="https://arxiv.org/abs/1234.56789",
            content=ATTENTION,
            pdf_url="https://arxiv.org/pdf/1234.56789/not-pdf",
            arxiv_id="",
        )
        # The allowlist check looks at the host, which is arxiv.org, so the GET happens.
        fallen, fallback_note = await enrich_with_fulltext(
            [plain],
            client=client,
            cache=MemoryCache(),
            settings=settings,
            acquire=acquire,
        )
        assert fallen[0].text_origin == "abstract"
        assert fallen[0].passages == []
        assert "1 abstract fallbacks" in fallback_note


@pytest.mark.asyncio
async def test_writer_prompt_cites_section_and_page() -> None:
    passage = Passage(
        text=(
            "A retriever selects passages and the generator conditions on those "
            "passages when answering a question about retrieval."
        ),
        section="Introduction",
        page=3,
    )
    abstract = (
        "This abstract only says that retrieval-augmented generation exists in the literature "
        "and does not describe the retriever itself at all."
    )
    source = Source(
        uid="paper",
        source_type="arxiv",
        title="Retrieval-Augmented Generation",
        url="https://arxiv.org/abs/2005.11401",
        year="2020",
        content=abstract,
        text_origin="pdf",
        pdf_url="https://arxiv.org/pdf/2005.11401.pdf",
        page_count=8,
        passages=[passage],
    )
    settings = make_settings(fulltext_prompt_chars=2000)
    store = MemoryVectorStore()
    events: list[dict[str, object]] = []

    async def emit(
        agent: str, kind: str, message: str, payload: dict[str, object] | None = None
    ) -> None:
        del agent
        events.append({"kind": kind, "message": message, "payload": payload or {}})

    deps = GraphDeps(
        settings=settings,
        llm=MockLLM(),
        search=FakeSearch([source]),
        vectors=store,
        embeddings=HashEmbedder(384),
        emit=emit,
    )
    result = await run_research_graph(
        "How does a retriever select passages for generation?",
        "job-pdf",
        deps,
    )
    report = str(result["report_md"])
    assert "Introduction - p. 3" in report
    assert "retriever selects passages" in report
    assert "Full text, 8 pages." in report
    assert result["support_score"] == 1.0
    assert any(chunk.section == "Introduction" and chunk.page == 3 for chunk in store._chunks)
    assert any("full text" in str(event["message"]) for event in events)
