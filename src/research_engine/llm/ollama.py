"""Local Ollama. Point OLLAMA_BASE_URL at a daemon on the host or another container."""

from __future__ import annotations

from typing import Any

import httpx

from research_engine.llm.base import ChatMessage
from research_engine.llm.http import ProviderError, request_with_retries


class OllamaLLM:
    name = "ollama"

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        timeout: float,
        max_retries: int,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.model = model
        self._base_url = base_url.rstrip("/")
        self._max_retries = max_retries
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=httpx.Timeout(timeout, connect=10.0))

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def complete(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.2,
        max_tokens: int = 2048,
        json_mode: bool = False,
    ) -> str:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [message.model_dump() for message in messages],
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        if json_mode:
            body["format"] = "json"
        try:
            return await self._post(body)
        except ProviderError as exc:
            if json_mode and str(exc).startswith("HTTP 400"):
                body.pop("format", None)
                return await self._post(body)
            if "request failed" in str(exc):
                raise ProviderError(
                    f"{exc}. Is Ollama running at {self._base_url} with model {self.model} pulled?"
                ) from exc
            raise

    async def _post(self, body: dict[str, Any]) -> str:
        response = await request_with_retries(
            self._client,
            "POST",
            f"{self._base_url}/api/chat",
            max_retries=self._max_retries,
            headers={"content-type": "application/json"},
            json=body,
        )
        payload = response.json()
        content = (payload.get("message") or {}).get("content")
        if not isinstance(content, str) or not content.strip():
            raise ProviderError("Ollama returned an empty message")
        return content
