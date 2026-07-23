"""Off-topic ``/btw`` side channel — never touches the vector store or checkpointer."""

from __future__ import annotations

from collections.abc import Generator

from langchain_core.prompts import ChatPromptTemplate

from folder.constants import BTW_WEB_RESULTS
from folder.schemas.routing import BtwRouteDecision
from folder.services import web_search as web_search_service
from folder.services.llm import get_utility_llm

_ROUTE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Decide if answering this question requires a real-time web search (recent events, "
            "current prices, breaking news) or if your general knowledge is sufficient.",
        ),
        ("human", "{query}"),
    ]
)


def handle_btw(query: str) -> Generator[str, None, None]:
    llm = get_utility_llm()
    decision: BtwRouteDecision = (
        _ROUTE_PROMPT | llm.with_structured_output(BtwRouteDecision)
    ).invoke({"query": query})

    if decision.needs_web_search:
        results = web_search_service.search(query, max_results=BTW_WEB_RESULTS)
        context = "\n\n".join(r["content"] for r in results)
        sources = "\n".join(f"- {r['url']}" for r in results)
        answer_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Answer the question using the web search results below. Be concise.\n\n"
                    f"Results:\n{context}\n\nSources:\n{sources}",
                ),
                ("human", "{query}"),
            ]
        )
    else:
        answer_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "Answer the question concisely from your general knowledge."),
                ("human", "{query}"),
            ]
        )

    for chunk in (answer_prompt | llm).stream({"query": query}):
        if chunk.content:
            yield chunk.content
