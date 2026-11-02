"""Gemini API via the Generative Language REST endpoint. The key stays in a header."""

from __future__ import annotations

from typing import Any

import httpx

from research_engine.llm.base import ChatMessage
from research_engine.llm.http import ProviderError, request_with_retries


class GeminiLLM:
    name = "gemini"

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
        system, contents = _to_gemini(messages)
        generation: dict[str, Any] = {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
        }
        body: dict[str, Any] = {"contents": contents, "generationConfig": generation}
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        if json_mode:
            generation["responseMimeType"] = "application/json"
        try:
            return await self._post(body)
        except ProviderError as exc:
            if json_mode and str(exc).startswith("HTTP 400"):
                generation.pop("responseMimeType", None)
                return await self._post(body)
            raise

    async def _post(self, body: dict[str, Any]) -> str:
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        )
        response = await request_with_retries(
            self._client,
            "POST",
            url,
            max_retries=self._max_retries,
            headers={"x-goog-api-key": self._api_key, "content-type": "application/json"},
            json=body,
            secret=self._api_key,
        )
        payload = response.json()
        candidates = payload.get("candidates") or []
        if not candidates:
            raise ProviderError(f"Gemini returned no candidates: {str(payload)[:240]}")
        parts = (candidates[0].get("content") or {}).get("parts") or []
        text = "".join(part.get("text", "") for part in parts if isinstance(part, dict))
        if not text.strip():
            raise ProviderError("Gemini returned empty text")
        return text


def _to_gemini(messages: list[ChatMessage]) -> tuple[str, list[dict[str, Any]]]:
    system = "\n\n".join(message.content for message in messages if message.role == "system")
    contents: list[dict[str, Any]] = []
    for message in messages:
        if message.role == "system":
            continue
        role = "user" if message.role == "user" else "model"
        if contents and contents[-1]["role"] == role:
            contents[-1]["parts"][0]["text"] += "\n\n" + message.content
        else:
            contents.append({"role": role, "parts": [{"text": message.content}]})
    if contents and contents[0]["role"] != "user":
        contents.insert(0, {"role": "user", "parts": [{"text": "Continue."}]})
    if not contents:
        contents = [{"role": "user", "parts": [{"text": system or "Respond."}]}]
        system = ""
    return system, contents
