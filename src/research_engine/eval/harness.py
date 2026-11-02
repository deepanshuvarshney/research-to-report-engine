"""Score claim/citation fixtures with the same overlap check the critic uses as a backstop.

Run `python -m research_engine.eval.harness`. The process exits non-zero when any
fixture is labeled incorrectly, so CI can call it the same way it calls pytest.
"""

from __future__ import annotations

import argparse
import json
from importlib.resources import files
from typing import Any

from research_engine.faithfulness import claim_supported


def load_cases() -> list[dict[str, Any]]:
    raw = files("research_engine.eval").joinpath("fixtures.json").read_text(encoding="utf-8")
    payload = json.loads(raw)
    cases = payload.get("cases")
    if not isinstance(cases, list):
        raise ValueError("fixtures.json must contain a cases array")
    return cases


def evaluate_cases(cases: list[dict[str, Any]], *, threshold: float = 0.45) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    correct = 0
    for case in cases:
        sources = {
            int(source["ref"]): str(source["content"])
            for source in case["sources"]
            if isinstance(source, dict)
        }
        citations = [int(citation) for citation in case.get("citations", [])]
        supported, ratio, reason = claim_supported(
            str(case["claim"]),
            citations,
            sources,
            threshold=threshold,
        )
        expected = bool(case["expect_supported"])
        ok = supported == expected
        correct += int(ok)
        results.append(
            {
                "id": case.get("id"),
                "expected": expected,
                "supported": supported,
                "overlap": round(ratio, 4),
                "reason": reason,
                "pass": ok,
            }
        )
    total = len(results)
    return {
        "accuracy": (correct / total) if total else 0.0,
        "correct": correct,
        "total": total,
        "threshold": threshold,
        "results": results,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score citation-faithfulness fixtures")
    parser.add_argument("--threshold", type=float, default=0.45)
    args = parser.parse_args(argv)
    report = evaluate_cases(load_cases(), threshold=args.threshold)
    print(json.dumps(report, indent=2))
    return 0 if report["accuracy"] == 1.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
