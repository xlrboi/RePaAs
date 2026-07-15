"""Relevancy-check node: decides whether retrieved chunks actually answer the query."""

from __future__ import annotations

from folder.constants import DOC_PREVIEW_CHARS
from folder.core.state import RAGState
from folder.prompts.relevancy import RELEVANCY_CHECK_SYSTEM_PROMPT
from folder.schemas.retrieval import RelevancyDecision
from folder.services.llm import get_main_llm


def relevancy_check_node(state: RAGState) -> dict:
    query = state["query"]
    docs = state.get("retrieved_docs") or []
    doc_snippets = "\n\n---\n\n".join(doc.page_content[:DOC_PREVIEW_CHARS] for doc in docs[:3])
    if not doc_snippets:
        return {"is_relevant": False}

    prompt = (
        f"Question: {query}\n\nRetrieved chunks:\n{doc_snippets}\n\n"
        "Are these chunks relevant to answering the question?"
    )
    llm = get_main_llm().with_structured_output(RelevancyDecision)
    decision: RelevancyDecision = llm.invoke(
        [
            {"role": "system", "content": RELEVANCY_CHECK_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
    )
    return {"is_relevant": decision.is_relevant}
