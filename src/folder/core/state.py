"""LangGraph state definition for the RePaAs RAG workflow."""

from __future__ import annotations
from langchain_core.documents import Document
from langgraph.graph import MessagesState


class RAGState(MessagesState):
    session_id: str
    query: str
    route: str | None
    retrieved_docs: list[Document]
    retrieval_attempts: int
    claim_verdict: str | None
    claim_source: str | None
    superseding_papers: list[dict] | None
    answer: str | None
    is_relevant: bool | None
    rewrite_count: int
