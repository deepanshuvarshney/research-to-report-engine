from __future__ import annotations

import pytest

from research_engine.faithfulness import backstop_rejects, claim_supported
from research_engine.rag.chunker import chunk_text
from research_engine.rag.embeddings import HashEmbedder, build_embedder, hashed_embedding
from research_engine.rag.store import MemoryVectorStore, StoredChunk


def test_chunker_windows_overlap_and_progress() -> None:
    assert chunk_text("   ") == []
    assert chunk_text("short abstract about models") == ["short abstract about models"]
    text = "word " * 500
    chunks = chunk_text(text, size=100, overlap=90)
    assert len(chunks) > 2
    assert all(chunks)
    assert chunks[0][-20:] in text


def test_hash_embedding_is_deterministic_and_normalized() -> None:
    first = hashed_embedding("attention and retrieval", 384)
    second = hashed_embedding("attention and retrieval", 384)
    other = hashed_embedding("volcanic soil chemistry", 384)
    assert first == second
    assert first != other
    assert len(first) == 384
    norm = sum(value * value for value in first) ** 0.5
    assert norm == pytest.approx(1.0, abs=1e-6)


@pytest.mark.asyncio
async def test_hash_retrieval_ranks_the_relevant_chunk() -> None:
    embedder = HashEmbedder(384)
    store = MemoryVectorStore()
    texts = {
        "a": "Photosynthesis converts light into chemical energy inside plant cells.",
        "b": "The French Revolution began in 1789 and reshaped European politics.",
    }
    vectors = await embedder.embed(list(texts.values()))
    chunks = [
        StoredChunk(
            job_id="job",
            source_uid=uid,
            source_ref=index,
            ordinal=0,
            text=text,
            embedding=vector,
        )
        for index, ((uid, text), vector) in enumerate(
            zip(texts.items(), vectors, strict=True), start=1
        )
    ]
    await store.replace("job", chunks)
    query = (await embedder.embed(["How do plants convert light into energy?"]))[0]
    hits = await store.search("job", query, k=2)
    assert hits[0].source_uid == "a"
    await store.prune("job", ["b"])
    remaining = await store.search("job", query, k=2)
    assert [hit.source_uid for hit in remaining] == ["b"]
    other_job = await store.search("other", query, k=2)
    assert other_job == []


def test_fastembed_builder_does_not_download_a_model() -> None:
    embedder = build_embedder("fastembed", "BAAI/bge-small-en-v1.5", 384)
    assert embedder.dim == 384


def test_claim_support_cases() -> None:
    sources = {
        1: "Attention allows a model to weigh input tokens by relevance when predicting the next token."
    }
    ok, ratio, _reason = claim_supported(
        "Attention allows a model to weigh input tokens by relevance.",
        [1],
        sources,
    )
    assert ok and ratio == 1.0
    bad, _, reason = claim_supported("The moon is made of cheese.", [1], sources)
    assert not bad
    assert "below" in reason
    missing, _, _ = claim_supported("Attention allows a model to weigh input tokens.", [], sources)
    assert not missing
    assert backstop_rejects("The moon is made of cheese.", [1], sources)
    assert not backstop_rejects(
        "Attention allows a model to weigh input tokens by relevance.",
        [1],
        sources,
    )
