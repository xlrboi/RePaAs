"""Final answer-generation node: produces the user-facing response for every route."""

from __future__ import annotations
from langchain_core.messages import AIMessage
from folder.constants import MAX_QUERY_REWRITES
from folder.core.state import RAGState
from folder.services.llm import get_main_llm


def _generate_retrieve_answer(state: RAGState) -> str:
    query = state["query"]
    if state.get("is_relevant") is False and state.get("rewrite_count", 0) >= MAX_QUERY_REWRITES:
        return (
            "I wasn't able to find relevant information in the uploaded papers "
            "to answer your question. You may want to rephrase your question "
            "or upload additional papers."
        )
    docs = state.get("retrieved_docs") or []
    if not docs:
        return "I don't know the answer."
    context = "\n\n---\n\n".join(doc.page_content for doc in docs)
    prompt = f"Answer the question using this context:\n\n{context}\n\nQuestion: {query}"
    return get_main_llm().invoke([{"role": "user", "content": prompt}]).content


def _generate_claim_verification_answer(state: RAGState) -> str:
    verdict = state.get("claim_verdict", "")
    papers = state.get("superseding_papers") or []
    claim_text = state["query"]
    if papers:
        papers_block = "\n\n".join(
            f"{i + 1}. **{p['title']}**\n   {p['summary']}\n   Link: {p['url']}"
            for i, p in enumerate(papers)
        )
        return (# simple returning so that i can save tokens lollll
            f"**Claim Verification Result**\n\n"
            f"> {claim_text}\n\n"
            f"**Verdict:** {verdict}\n\n"
            f"**Superseding Papers:**\n\n{papers_block}\n\n"
            f"---\n"
            f"*You can load any of these papers into your knowledge base "
            f"to continue your research with the latest findings.*"
        )
    return (# same here
        f"**Claim Verification Result**\n\n"
        f"> {claim_text}\n\n"
        f"**Verdict:** {verdict}\n\n"
        f"*No papers directly superseding this claim were found in recent literature.*"
    )


def _generate_direct_answer(state: RAGState) -> str:
    query = state["query"]
    prompt = f"Answer from your knowledge.\n\nQuestion: {query}"
    return get_main_llm().invoke([{"role": "user", "content": prompt}]).content


_ROUTE_HANDLERS = {
    "retrieve": _generate_retrieve_answer,
    "verify_claim": _generate_claim_verification_answer,
}


def generate_answer_node(state: RAGState) -> dict:
    route = state.get("route")
    handler = _ROUTE_HANDLERS.get(route, _generate_direct_answer)
    answer = handler(state)
    return {"answer": answer, "messages": [AIMessage(content=answer)]}
