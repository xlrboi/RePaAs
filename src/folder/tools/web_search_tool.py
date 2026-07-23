"""LangChain @tool definition for live web search (tavily-backed).
"""

from __future__ import annotations
from typing import Annotated
from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState # This annotation enables tools to access graph state without exposing state management details to the language model.
from langgraph.types import Command # Allow a tool to issue a command to modify the graph state during execution
from folder.schemas.tools_Schema import WebSearchInput
from folder.services import web_search as web_search_service
from langchain_core.documents import Document

@tool(args_schema=WebSearchInput)
def web_search(
    optimized_query: str,
    max_results: int,
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Search the web for current or supplementary information using Tavily."""
    current_docs = state.get("retrieved_docs", [])
    results = web_search_service.search(optimized_query, max_results=max_results)
    if not results:
        return Command(
            update={
                "messages": [
                    ToolMessage(content="No web results found.", tool_call_id=tool_call_id)
                ]
            }
        )
    web_docs = [
        Document(
            page_content=r["content"],
            metadata={"url": r["url"], "title": r.get("title", "Web Result")},
        )
        for r in results
    ]
    summary = f"Found {len(web_docs)} web result(s) for: {optimized_query}"
    return Command(
        update={
            "messages": [ToolMessage(content=summary, tool_call_id=tool_call_id)],
            "retrieved_docs": (current_docs or []) + web_docs,
        }
    )
