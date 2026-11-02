"""Pull a JSON value out of a model reply and validate it, with one repair turn."""

from __future__ import annotations

import json
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from research_engine.llm.base import ChatMessage, LLMProvider

T = TypeVar("T", bound=BaseModel)


def extract_first_json(text: str) -> Any:
    """Parse the first JSON object or array, ignoring fences and preamble."""

    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()

    start = next((index for index, char in enumerate(stripped) if char in "{["), None)
    if start is None:
        raise ValueError("response did not contain JSON")

    pairs = {"{": "}", "[": "]"}
    stack: list[str] = []
    in_string = False
    escape = False
    for index in range(start, len(stripped)):
        char = stripped[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in pairs:
            stack.append(pairs[char])
        elif stack and char == stack[-1]:
            stack.pop()
            if not stack:
                try:
                    return json.loads(stripped[start : index + 1])
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON: {exc}") from exc
    raise ValueError("unterminated JSON value")


def extract_labeled_json(text: str, label: str) -> Any:
    index = text.find(label)
    if index < 0:
        raise ValueError(f"missing label {label}")
    return extract_first_json(text[index + len(label) :])


async def complete_json(
    llm: LLMProvider,
    messages: list[ChatMessage],
    model: type[T],
    *,
    temperature: float,
    max_tokens: int,
) -> T:
    """Ask for JSON and give the model one chance to repair a bad payload."""

    reply = await llm.complete(
        messages,
        temperature=temperature,
        max_tokens=max_tokens,
        json_mode=True,
    )
    try:
        return model.model_validate(extract_first_json(reply))
    except (ValueError, ValidationError) as exc:
        repair = [
            *messages,
            ChatMessage(role="assistant", content=reply[:4000]),
            ChatMessage(
                role="user",
                content=(
                    "The previous response was invalid "
                    f"({exc}). Return only a corrected JSON object matching the schema."
                ),
            ),
        ]
        second = await llm.complete(
            repair,
            temperature=0.0,
            max_tokens=max_tokens,
            json_mode=True,
        )
        try:
            return model.model_validate(extract_first_json(second))
        except (ValueError, ValidationError) as exc2:
            raise ValueError(f"Model did not return valid JSON: {exc2}") from exc2
