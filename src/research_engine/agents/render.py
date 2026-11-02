"""Markdown, print-ready HTML, and a plain-text PDF export."""

from __future__ import annotations

import html
import re

import markdown as markdown_lib
from fpdf import FPDF

from research_engine.agents.models import Draft
from research_engine.search.models import Source

_TYPE_LABELS = {
    "arxiv": "arXiv",
    "semantic_scholar": "Semantic Scholar",
    "wikipedia": "Wikipedia",
    "web": "Web",
}


def _with_citations(text: str, citations: list[int]) -> str:
    missing = [ref for ref in citations if f"[{ref}]" not in text]
    if not missing:
        return text.strip()
    return text.strip() + "".join(f" [{ref}]" for ref in missing)


def render_markdown(
    *,
    question: str,
    draft: Draft | None,
    sources: list[Source],
    verdicts: list[dict[str, object]],
    subquestions: list[str],
    revisions: int,
) -> str:
    if draft is None or not sources:
        return _empty_report(question)

    supported = sum(1 for verdict in verdicts if verdict.get("supported"))
    total = len(verdicts)
    lines = [
        f"# {draft.title}",
        "",
        f"_{draft.summary}_" if draft.summary else "",
        "",
        f"Asked: {question}",
        "",
    ]
    if total:
        lines.append(
            f"> Verification: {supported}/{total} claims supported "
            "after "
            f"{revisions} revision(s). Unsupported rows stay unverified."
        )
        lines.append("")

    for section in draft.sections:
        lines.append(f"## {section.heading}")
        lines.append("")
        for paragraph in section.paragraphs:
            lines.append(_with_citations(paragraph.text, paragraph.citations))
            lines.append("")

    if subquestions:
        lines.append("## Research plan")
        lines.append("")
        lines.extend(f"- {item}" for item in subquestions)
        lines.append("")

    if verdicts:
        lines.append("## Verification")
        lines.append("")
        for verdict in verdicts:
            label = "Supported" if verdict.get("supported") else "Unsupported"
            overlap = verdict.get("overlap")
            overlap_text = (
                f", overlap {float(overlap):.2f}" if isinstance(overlap, (int, float)) else ""
            )
            cites = verdict.get("citations") or []
            cite_text = ""
            if isinstance(cites, list) and cites:
                cite_text = " " + " ".join(f"[{ref}]" for ref in cites)
            reason = str(verdict.get("reason") or "").strip()
            lines.append(f"- **{label}**{overlap_text} — {verdict.get('text', '')}{cite_text}")
            if reason:
                lines.append(f"  - {reason}")
        lines.append("")

    lines.append("## References")
    lines.append("")
    if not sources:
        lines.append("_None._")
    for source in sources:
        bits = [f"{source.ref}. **{source.title}**."]
        if source.authors:
            bits.append(f" {source.authors}.")
        if source.year:
            bits.append(f" {source.year}.")
        bits.append(f" {_TYPE_LABELS.get(source.source_type, source.source_type)}.")
        if source.url:
            bits.append(f" {source.url}")
        lines.append("".join(bits))
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def _empty_report(question: str) -> str:
    return (
        "# No sourced report\n\n"
        f"Asked: {question}\n\n"
        "Vellum did not retrieve source text, so it did not draft factual claims.\n\n"
        "## References\n\n"
        "_None._\n"
    )


def render_html(markdown_text: str) -> str:
    escaped = html.escape(markdown_text)
    body = markdown_lib.markdown(escaped, extensions=["extra", "sane_lists"])
    body = re.sub(
        r"\[(\d+)\]",
        r'<a class="cite" href="#ref-\1">[\1]</a>',
        body,
    )
    body = body.replace("<a href=", '<a target="_blank" rel="noopener" href=')
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Vellum report</title>
<style>
  body {{ margin: 0; background: #efe7d6; color: #1c1915;
         font: 17px/1.55 Georgia, "Iowan Old Style", serif; }}
  main {{ max-width: 740px; margin: 32px auto; padding: 48px 56px; background: #fffdf8; }}
  h1 {{ font-weight: 560; letter-spacing: -0.03em; line-height: 1.15; }}
  a {{ color: #1e4d3a; }}
  .cite {{ text-decoration: none; font-variant-numeric: tabular-nums; }}
  blockquote {{ border-left: 3px solid #1e4d3a; margin-left: 0;
                padding-left: 16px; color: #3d3830; }}
  @media print {{ body {{ background: white; }} main {{ margin: 0; box-shadow: none; }} }}
</style>
</head>
<body>
<main>
{body}
</main>
</body>
</html>
"""


def render_pdf(markdown_text: str) -> bytes:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_title("Vellum report")
    pdf.set_font("Helvetica", size=11)
    for raw_line in markdown_text.splitlines():
        line = raw_line.replace("**", "").replace("_", "")
        line = _soft_wrap(line)
        safe = line.encode("latin-1", "replace").decode("latin-1")
        pdf.multi_cell(0, 6, safe if safe else " ", align="L", new_x="LMARGIN", new_y="NEXT")
    output = pdf.output()
    return bytes(output)


def _soft_wrap(text: str) -> str:
    parts: list[str] = []
    for token in text.split(" "):
        if len(token) <= 80:
            parts.append(token)
            continue
        parts.append(" ".join(token[index : index + 80] for index in range(0, len(token), 80)))
    return " ".join(parts)
