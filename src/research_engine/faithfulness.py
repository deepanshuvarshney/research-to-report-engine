"""Lexical citation checks shared by the critic backstop and the eval harness.

This is a grounding heuristic, not a natural-language inference model. Exact
excerpts score as supported. Paraphrases pass only when enough content words
overlap the cited text. The language-model critic remains the primary judge.
"""

from __future__ import annotations

import re

STOPWORDS = frozenset(
    """
    a an the of to and or for in on with from by as at into over after before
    is are was were be been being this that these those it its their his her
    they them we you our not no yes do does did than then so such if but
    about between among across per via using used use can may might will
    just also more most other another which who what when where how why
    """.split()  # noqa: SIM905
)

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return [
        token for token in _TOKEN.findall(text.lower()) if token not in STOPWORDS and len(token) > 2
    ]


def support_ratio(claim: str, evidence: str) -> float:
    claim_tokens = tokenize(claim)
    if not claim_tokens:
        return 0.0
    evidence_tokens = set(tokenize(evidence))
    hits = sum(1 for token in claim_tokens if token in evidence_tokens)
    return hits / len(claim_tokens)


def _phrase_supported(claim: str, evidence: str, window: int = 4) -> bool:
    claim_tokens = tokenize(claim)
    evidence_tokens = tokenize(evidence)
    if len(claim_tokens) < window or len(evidence_tokens) < window:
        return False
    evidence_phrases = {
        " ".join(evidence_tokens[index : index + window])
        for index in range(len(evidence_tokens) - window + 1)
    }
    return any(
        " ".join(claim_tokens[index : index + window]) in evidence_phrases
        for index in range(len(claim_tokens) - window + 1)
    )


def claim_supported(
    claim: str,
    citations: list[int],
    sources: dict[int, str],
    *,
    threshold: float = 0.45,
) -> tuple[bool, float, str]:
    """Return whether `claim` is grounded in the cited source texts."""

    if not citations:
        return False, 0.0, "No citation."
    evidence = "\n".join(sources.get(citation, "") for citation in citations).strip()
    if not evidence:
        return False, 0.0, "Cited sources have no stored text."
    if claim.strip() and claim.strip().lower() in evidence.lower():
        return True, 1.0, "The claim text appears in the cited source."
    ratio = support_ratio(claim, evidence)
    if _phrase_supported(claim, evidence):
        return True, max(ratio, 0.8), "A phrase from the claim appears in the cited source."
    if ratio >= threshold:
        return True, ratio, f"Token overlap {ratio:.2f} meets {threshold:.2f}."
    return False, ratio, f"Token overlap {ratio:.2f} is below {threshold:.2f}."


def backstop_rejects(claim: str, citations: list[int], sources: dict[int, str]) -> bool:
    """Reject a claim the model called supported when the cited text barely overlaps it."""

    if not citations:
        return True
    evidence = "\n".join(sources.get(citation, "") for citation in citations)
    if claim.strip() and claim.strip().lower() in evidence.lower():
        return False
    if _phrase_supported(claim, evidence):
        return False
    return support_ratio(claim, evidence) < 0.15
