"""Retrieval agent node: tool-calling loop that decides how to gather context."""

from __future__ import annotations

from folder.constants import MAX_RETRIEVAL_ATTEMPTS
from folder.core.state import RAGState
from folder.prompts.retrieval_agent import RETRIEVAL_AGENT_SYSTEM_PROMPT
from folder.services.llm import get_main_llm
from folder.tools.retriever_tool import retrieve_from_vectorstore
from folder.tools.web_search_tool import web_search

RETRIEVAL_TOOLS = [retrieve_from_vectorstore, web_search]


def _plain_llm():
    return get_main_llm()


def _tool_bound_llm():
    return get_main_llm().bind_tools(RETRIEVAL_TOOLS, parallel_tool_calls=False)


def agent_node(state: RAGState) -> dict:
    current_attempts = state.get("retrieval_attempts", 0)
    # Once at the cap, use plain LLM so the agent cannot emit more tool calls.
    # This prevents orphaned tool_call IDs from entering the persisted message
    # history: retrieval llm --> tool call --> tool result, vs.
    # llm --> no tools bound --> no tool call possible.
    lm = _plain_llm() if current_attempts >= MAX_RETRIEVAL_ATTEMPTS else _tool_bound_llm()
    messages = [{"role": "system", "content": RETRIEVAL_AGENT_SYSTEM_PROMPT}] + state["messages"]
    response = lm.invoke(messages)
    updates: dict = {"messages": [response]}
    if getattr(response, "tool_calls", None):
        updates["retrieval_attempts"] = current_attempts + 1
    return updates
