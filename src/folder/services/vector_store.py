"""Qdrant-backed vector store with cache-backed embeddings, one collection per session."""

from __future__ import annotations
from functools import lru_cache
from langchain_classic.embeddings import CacheBackedEmbeddings
from langchain_classic.storage import LocalFileStore
from langchain_core.documents import Document
from langchain_openai import AzureOpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from folder.config import get_settings
from folder.constants import EMBEDDING_DIM

# ── Singletons ────────────────────────────────────────────────────────────────


@lru_cache
def _get_embeddings() -> CacheBackedEmbeddings:                             
    settings = get_settings()
    base_embeddings = AzureOpenAIEmbeddings(
        azure_deployment=settings.azure_openai_embedding_deployment,
        azure_endpoint=settings.azure_openai_endpoint
    )
    embedding_file_store = LocalFileStore(str(settings.embedding_cache_dir))
    return CacheBackedEmbeddings.from_bytes_store(
        base_embeddings,
        embedding_file_store,
        namespace=base_embeddings.model,
        query_embedding_cache=True,
        key_encoder="blake2b",
    )


@lru_cache
def _get_qdrant_client() -> QdrantClient:
    settings = get_settings()
    return QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        timeout=120,
    )


# ── Collection helpers ───────────────────────────────────────────────────────


def get_collection_name(session_id: str) -> str:
    return f"RePaAs_{session_id.replace('-', '_')}"


def get_vectorstore(session_id: str) -> QdrantVectorStore:
    client = _get_qdrant_client()
    collection_name = get_collection_name(session_id)
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )
    return QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=_get_embeddings(),
    )


# ── Public API ───────────────────────────────────────────────────────────────


def add_paper(docs: list[Document], session_id: str) -> None:
    get_vectorstore(session_id).add_documents(docs)


def list_papers(session_id: str) -> list[str]:
    client = _get_qdrant_client()
    collection_name = get_collection_name(session_id)
    if not client.collection_exists(collection_name):
        return []
    seen: set[str] = set()
    titles: list[str] = []
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection_name,
            with_payload=True,
            limit=100,
            offset=offset,
        )
        for point in points:
            title = (point.payload or {}).get("metadata", {}).get("title")
            if title and title not in seen:
                seen.add(title)
                titles.append(title)
        if offset is None:
            break
    return titles


def search(query: str, session_id: str, k: int) -> list[Document]:
    return get_vectorstore(session_id).similarity_search(query, k=k)
