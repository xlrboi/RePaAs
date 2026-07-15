"""Query-rewrite node: reformulates the query after a failed relevancy check."""

from __future__ import annotations
from langchain_core.messages import HumanMessage

from folder.core.state import RAGState
from folder.prompts.query_rewrite import QUERY_REWRITE_SYSTEM_PROMPT
from folder.services.llm import get_main_llm


def query_rewrite_node(state: RAGState) -> dict:
    original_query = state["query"]
    rewrite_count = state.get("rewrite_count", 0)
    response = get_main_llm().invoke(
        [
            {"role": "system", "content": QUERY_REWRITE_SYSTEM_PROMPT},
            {"role": "user", "content": f"Original query: {original_query}\n\nWrite an improved search query.",},
        ]
    )
    rewritten = response.content.strip()
    return {
        "messages": [HumanMessage(content=rewritten)],
        "query": rewritten,
        "retrieved_docs": [], # empty because we want the agent to retrieve docs on the basis of new query
        "retrieval_attempts": 0,
        "rewrite_count": rewrite_count + 1,
        "is_relevant": None, # again will do relevancy check on them 
    }
