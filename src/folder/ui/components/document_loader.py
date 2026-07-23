"""Document ingestion helpers used by the sidebar's upload widgets.

Wraps ``services.paper_loader`` + ``services.vector_store`` with the
temp-file handling that Streamlit's ``UploadedFile`` objects need.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from folder.services import paper_loader, vector_store


def ingest_uploaded_file(file_bytes: bytes, file_name: str, session_id: str) -> None:
    """Write an uploaded file to a temp path, load+chunk it, index it, then clean up."""
    suffix = Path(file_name).suffix
    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        docs = paper_loader.load_document(tmp_path)
        for doc in docs:
            doc.metadata["title"] = Path(file_name).stem
        vector_store.add_paper(docs, session_id)
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)


def ingest_webpage(url: str, session_id: str) -> None:
    docs = paper_loader.load_webpage(url)
    vector_store.add_paper(docs, session_id)


def ingest_arxiv(query: str, session_id: str) -> str:
    """Load an ArXiv paper by ID or title; returns the resolved paper title."""
    docs = paper_loader.load_arxiv(query)
    vector_store.add_paper(docs, session_id)
    return docs[0].metadata.get("title") if docs else query


def loaded_titles(session_id: str) -> list[str] | None:
    try:
        return vector_store.list_papers(session_id)
    except Exception:
        return None
