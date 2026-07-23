"""ArXiv lookup helpers — ID extraction, title search, and PDF download.

Note: claim verification deliberately does NOT use this module (or the
``arxiv`` PyPI package) — it uses two targeted Tavily searches instead,
since the ``arxiv`` library had reliability issues for that use case. See
``core/nodes/claim_verification.py``.
"""

from __future__ import annotations

import re
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path

_ARXIV_ID_RE = re.compile(r"(\d{4}\.\d{4,5}(?:v\d+)?)")


def extract_arxiv_id(query: str) -> str | None:
    """Return bare ArXiv ID (no version suffix) if one appears in the query."""
    m = _ARXIV_ID_RE.search(query)
    if m:
        return re.sub(r"v\d+$", "", m.group(1))
    return None


def lookup_title_by_id(arxiv_id: str) -> str:
    """Fetch paper title by ID from the ArXiv Atom API."""
    url = f"https://export.arxiv.org/api/query?id_list={arxiv_id}"
    with urllib.request.urlopen(url, timeout=10) as resp:
        xml = resp.read().decode()
    titles = re.findall(r"<title>(.*?)</title>", xml, re.DOTALL)
    return titles[1].strip() if len(titles) > 1 else arxiv_id


def search_id_by_title(query: str) -> str:
    """Search ArXiv Atom API by title phrase and return the top result's bare paper ID."""
    phrase = query.strip('"')
    search_query = urllib.parse.quote(f'ti:"{phrase}"')
    url = (
        f"https://export.arxiv.org/api/query?search_query={search_query}"
        f"&max_results=1&sortBy=relevance"
    )
    with urllib.request.urlopen(url, timeout=15) as resp:
        xml = resp.read().decode()
    m = re.search(r"<id>https?://arxiv\.org/abs/(\d{4}\.\d{4,5}(?:v\d+)?)</id>", xml)
    if not m:
        raise ValueError(f"No ArXiv paper found for: {query}")
    return re.sub(r"v\d+$", "", m.group(1))


def download_pdf(arxiv_id: str) -> Path:
    """Download an ArXiv paper PDF by its bare ID to a temp file and return its path.

    Caller is responsible for deleting the returned path.
    """
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"
    with urllib.request.urlopen(pdf_url, timeout=60) as resp:
        pdf_bytes = resp.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes)
        return Path(tmp.name)


def resolve_arxiv_id(query: str) -> str:
    """Resolve a free-form query (bare ID or title) to a bare ArXiv ID."""
    return extract_arxiv_id(query) or search_id_by_title(query)
