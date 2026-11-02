"""Extractive stand-in for a hosted model.

Scripted replies drive tests. With no script, the mock plans heuristically and
writes by quoting retrieved sources, which is what `docker compose up` runs
before any API key is set.
"""

from __future__ import annotations

import json

from research_engine.agents.fallback import extractive_draft, heuristic_plan, heuristic_verdicts
from research_engine.agents.models import Claim
from research_engine.agents.prompts import sources_from_prompt
from research_engine.llm.base import ChatMessage
from research_engine.llm.jsonutil import extract_labeled_json


class MockLLM:
    def __init__(self, *, model: str = "mock-extractive", script: list[str] | None = None) -> None:
        self.name = "mock"
        self.model = model
        self.script = list(script or [])
        self.calls: list[list[ChatMessage]] = []

    async def aclose(self) -> None:
        return None

    async def complete(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.2,
        max_tokens: int = 2048,
        json_mode: bool = False,
    ) -> str:
        del temperature, max_tokens, json_mode
        self.calls.append(list(messages))
        if self.script:
            return self.script.pop(0)
        system = next((message.content for message in messages if message.role == "system"), "")
        user = next(
            (message.content for message in reversed(messages) if message.role == "user"),
            "",
        )
        if "AGENT:planner" in system:
            question = _question(user)
            return json.dumps(heuristic_plan(question, max_n=3))
        if "AGENT:writer" in system:
            sources = sources_from_prompt(user)
            draft = extractive_draft(_question(user), sources)
            return json.dumps(draft.model_dump())
        if "AGENT:critic" in system:
            raw_claims = extract_labeled_json(user, "CLAIMS_JSON:")
            claims = [Claim.model_validate(item) for item in raw_claims]
            critique = heuristic_verdicts(claims, sources_from_prompt(user))
            return json.dumps(critique.model_dump())
        raise RuntimeError("mock LLM received a prompt without an agent marker")


def _question(user: str) -> str:
    marker = "QUESTION:\n"
    index = user.find(marker)
    if index >= 0:
        rest = user[index + len(marker) :]
        return rest.split("\n\n", 1)[0].strip()
    for line in user.splitlines():
        if line.startswith("QUESTION:"):
            value = line.split(":", 1)[1].strip()
            if value:
                return value
    stripped = user.strip()
    if not stripped:
        return "Research question"
    return stripped.splitlines()[0][:300]
