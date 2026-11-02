"""Groq's OpenAI-compatible free tier."""

from __future__ import annotations

from typing import Any

import httpx

from research_engine.llm.base import ChatMessage
from research_engine.llm.http import ProviderError, request_with_retries

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


class GroqLLM:
    name = "groq"

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        timeout: float,
        max_retries: int,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.model = model
        self._api_key = api_key
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
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        try:
            return await self._post(body)
        except ProviderError as exc:
            if json_mode and str(exc).startswith("HTTP 400"):
                body.pop("response_format", None)
                return await self._post(body)
            raise

    async def _post(self, body: dict[str, Any]) -> str:
        response = await request_with_retries(
            self._client,
            "POST",
            GROQ_URL,
            max_retries=self._max_retries,
            headers={
                "authorization": f"Bearer {self._api_key}",
                "content-type": "application/json",
            },
            json=body,
            secret=self._api_key,
        )
        payload = response.json()
        try:
            content = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError("Groq response did not include a message") from exc
        if not isinstance(content, str) or not content.strip():
            raise ProviderError("Groq returned an empty message")
        return content
