"""Local embeddings. Hashing needs no download; fastembed is an optional ONNX model."""

from __future__ import annotations

import asyncio
import hashlib
import itertools
import math
import re
from typing import Protocol

_TOKEN = re.compile(r"[a-z0-9]+")


class Embedder(Protocol):
    dim: int

    async def embed(self, texts: list[str]) -> list[list[float]]: ...


def hashed_embedding(text: str, dim: int = 384) -> list[float]:
    """Signed feature hashing over tokens and bigrams, L2-normalized."""

    tokens = _TOKEN.findall(text.lower())
    vector = [0.0] * dim
    if not tokens:
        return vector
    features = list(tokens)
    features.extend(f"{left}_{right}" for left, right in itertools.pairwise(tokens))
    for feature in features:
        digest = hashlib.blake2b(feature.encode(), digest_size=8).digest()
        index = int.from_bytes(digest[:4], "little") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


class HashEmbedder:
    """Deterministic embeddings for tests, CI, and a zero-download demo."""

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [hashed_embedding(text, self.dim) for text in texts]


class FastEmbedEmbedder:
    """ONNX embeddings via fastembed. The model downloads on first use."""

    def __init__(self, model: str, dim: int) -> None:
        self.model_name = model
        self.dim = dim
        self._model: object | None = None

    def _load(self) -> object:
        if self._model is None:
            from fastembed import TextEmbedding

            self._model = TextEmbedding(model_name=self.model_name)
        return self._model

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        def run() -> list[list[float]]:
            model = self._load()
            vectors = list(model.embed(texts))  # type: ignore[attr-defined]
            converted = [[float(value) for value in vector] for vector in vectors]
            for vector in converted:
                if len(vector) != self.dim:
                    raise RuntimeError(
                        f"Embedding model returned dim {len(vector)}, expected {self.dim}. "
                        "Use BAAI/bge-small-en-v1.5 or another 384-dimensional model."
                    )
            return converted

        return await asyncio.to_thread(run)


def build_embedder(provider: str, model: str, dim: int) -> Embedder:
    if provider == "hash":
        return HashEmbedder(dim)
    if provider == "fastembed":
        return FastEmbedEmbedder(model, dim)
    raise RuntimeError(f"Unknown embedding provider {provider}")
