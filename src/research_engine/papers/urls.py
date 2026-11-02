"""Allowlisted PDF URLs. Search results are not fetched unless the host is on this list."""

from __future__ import annotations

import ipaddress
import re
from urllib.parse import urlsplit

# Exact hosts only. Suffix matches would let a lookalike domain through.
PDF_HOSTS = frozenset(
    {
        "arxiv.org",
        "export.arxiv.org",
        "pdfs.semanticscholar.org",
        "europepmc.org",
        "www.ncbi.nlm.nih.gov",
        "pmc.ncbi.nlm.nih.gov",
        "www.biorxiv.org",
        "www.medrxiv.org",
        "aclanthology.org",
        "openreview.net",
        "proceedings.mlr.press",
        "openaccess.thecvf.com",
        "papers.nips.cc",
        "proceedings.neurips.cc",
        "hal.science",
        "hal.archives-ouvertes.fr",
        "zenodo.org",
    }
)

UNPAYWALL_HOST = "api.unpaywall.org"

_ARXIV_ID = re.compile(
    r"arxiv\.org/(?:abs|pdf)/"
    r"((?:\d{4}\.\d{4,5})|(?:[a-z\-]+/\d{7}))"
    r"(v\d+)?",
    re.IGNORECASE,
)


def url_is_allowed(url: str) -> bool:
    """True only for HTTPS URLs whose host is on the open-access PDF list."""

    if not url or any(char in url for char in "\r\n\t"):
        return False
    parsed = urlsplit(url.strip())
    if parsed.scheme != "https":
        return False
    if parsed.username or parsed.password:
        return False
    if parsed.port not in (None, 443):
        return False
    host = (parsed.hostname or "").lower().rstrip(".")
    if not host or host == "localhost" or host.endswith(".local") or host.endswith(".internal"):
        return False
    try:
        ipaddress.ip_address(host)
    except ValueError:
        pass
    else:
        return False
    return host in PDF_HOSTS


def host_of(url: str) -> str:
    return (urlsplit(url).hostname or "").lower()


def arxiv_id_from_url(url: str) -> str:
    match = _ARXIV_ID.search(url or "")
    if not match:
        return ""
    version = match.group(2) or ""
    return f"{match.group(1)}{version}"


def arxiv_pdf_url(arxiv_id_or_url: str) -> str:
    """Build a canonical arXiv PDF URL from an id or an abs/pdf link."""

    text = (arxiv_id_or_url or "").strip()
    if not text:
        return ""
    if text.startswith("http://") or text.startswith("https://"):
        text = arxiv_id_from_url(text)
    if not text:
        return ""
    if not re.fullmatch(r"(?:\d{4}\.\d{4,5}|[a-z\-]+/\d{7})(?:v\d+)?", text, re.IGNORECASE):
        return ""
    return f"https://arxiv.org/pdf/{text}.pdf"


def limiter_name(url: str) -> str:
    host = host_of(url)
    if host in {"arxiv.org", "export.arxiv.org"}:
        return "arxiv-pdf"
    if host == UNPAYWALL_HOST:
        return "unpaywall"
    return "oa-pdf"
