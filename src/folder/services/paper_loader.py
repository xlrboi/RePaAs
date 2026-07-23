"""Multi-source paper loader: PDF, TXT, Markdown, web URL, and ArXiv.

ArXiv-specific network calls live in ``arxiv_client.py``; this module owns
generic loading/chunking and source dispatch.
"""

from __future__ import annotations
from pathlib import Path
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader, WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from folder.constants import CHUNK_OVERLAP, CHUNK_SIZE
from folder.services import arxiv_client

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, add_start_index=True
)
_md_splitter = RecursiveCharacterTextSplitter.from_language(
    "markdown", chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, add_start_index=True
)


def _stamp_title(docs: list[Document], title: str) -> list[Document]:
    for doc in docs:
        doc.metadata["title"] = title
    return docs


def load_pdf(file_path: str) -> list[Document]:
    docs = PyMuPDFLoader(file_path).load()
    return _stamp_title(_splitter.split_documents(docs), Path(file_path).stem)


def load_text(file_path: str) -> list[Document]:
    docs = TextLoader(file_path, encoding="utf-8").load()
    return _stamp_title(_splitter.split_documents(docs), Path(file_path).stem)


def load_markdown(file_path: str) -> list[Document]:
    docs = TextLoader(file_path, encoding="utf-8").load()
    return _stamp_title(_md_splitter.split_documents(docs), Path(file_path).stem)


def load_webpage(url: str) -> list[Document]:
    try:
        docs = WebBaseLoader(url, requests_kwargs={"timeout": 30}).load()
        if not docs:
            raise ValueError(f"No content loaded from URL: {url}")
        title = (docs[0].metadata.get("title") or url) if docs else url
        return _stamp_title(_splitter.split_documents(docs), title)
    except Exception as e:
        raise ValueError(f"Failed to load webpage {url}: {str(e)}")


def _load_arxiv_by_id(arxiv_id: str) -> list[Document]:
    try:
        tmp_path = arxiv_client.download_pdf(arxiv_id)
    except Exception as e:
        raise ValueError(f"Failed to download PDF for ArXiv ID {arxiv_id}: {str(e)}")
    
    try:
        docs = PyMuPDFLoader(str(tmp_path)).load()
        if not docs:
            raise ValueError(f"Could not load PDF for ArXiv ID: {arxiv_id}")
        title = (docs[0].metadata.get("title") or "").strip() or arxiv_client.lookup_title_by_id(
            arxiv_id
        )
        return _stamp_title(_splitter.split_documents(docs), title)
    except Exception as e:
        raise ValueError(f"Failed to process PDF for ArXiv ID {arxiv_id}: {str(e)}")
    finally:
        tmp_path.unlink(missing_ok=True)


def load_arxiv(query: str) -> list[Document]:
    try:
        arxiv_id = arxiv_client.resolve_arxiv_id(query)
    except Exception as e:
        raise ValueError(f"Failed to resolve ArXiv query '{query}': {str(e)}")
    return _load_arxiv_by_id(arxiv_id)


def load_document(source: str) -> list[Document]:
    """Dispatch to the appropriate loader based on URL prefix or file extension."""
    if source.startswith(("http://", "https://")):
        return load_webpage(source)
    ext = Path(source).suffix.lower()
    if ext == ".pdf":
        return load_pdf(source)
    if ext == ".txt":
        return load_text(source)
    if ext in (".md", ".markdown"):
        return load_markdown(source)
    raise ValueError(f"Unsupported file type: {ext}")
