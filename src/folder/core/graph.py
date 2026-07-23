"""Builds the RePaAs workflow
"""

from __future__ import annotations

import sqlite3

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from folder.constants import MAX_RETRIEVAL_ATTEMPTS
from folder.core.nodes.agent import RETRIEVAL_TOOLS, agent_node
from folder.core.nodes.claim_verification import verify_claim_node
from folder.core.nodes.generate_answer import generate_answer_node
from folder.core.nodes.query_rewrite import query_rewrite_node
from folder.core.nodes.relevancy import relevancy_check_node
from folder.core.nodes.router import router_node
from folder.core.state import RAGState

_retrieval_tool_node = ToolNode(RETRIEVAL_TOOLS)


def _route_query(state: RAGState) -> str:
    return state["route"]


def _agent_routing(state: RAGState) -> str:
    """
    Determines the next node to execute based on the current state.
    
    Priority:
    1. Execute pending tool calls first.
    2. If no tool calls, check if max retrieval attempts reached.
    3. If not at max attempts, check for relevancy.
    4. Otherwise, proceed to generate answer.
    """
    # Always execute pending tool calls first — shortcutting here would leave
    # an AIMessage with tool_calls unmatched by ToolMessages in the
    # checkpointer, corrupting history for all future turns in the session.
    if tools_condition(state) == "tools":
        return "retrieval"
    if state.get("retrieval_attempts", 0) >= MAX_RETRIEVAL_ATTEMPTS:
        return "generate_answer"
    return "relevancy_check"


def _after_relevancy_routing(state: RAGState) -> str:
    if state.get("is_relevant", False):
        return "generate_answer"
    if state.get("rewrite_count", 0) < 1:
        return "query_rewrite"
    return "generate_answer"


def build_graph(db_path: str = "checkpoints.db"):
    conn = sqlite3.connect(db_path, check_same_thread=False)
    checkpointer = SqliteSaver(conn)

    graph = StateGraph(RAGState)
    graph.add_node("router", router_node)
    graph.add_node("agent_node", agent_node)
    graph.add_node("retrieval", _retrieval_tool_node)
    graph.add_node("relevancy_check", relevancy_check_node)
    graph.add_node("query_rewrite", query_rewrite_node)
    graph.add_node("verify_claim", verify_claim_node)
    graph.add_node("generate_answer", generate_answer_node)

    graph.set_entry_point("router")

    graph.add_conditional_edges(
        "router",
        _route_query,
        {
            "retrieve": "agent_node",
            "verify_claim": "verify_claim",
            "direct_answer": "generate_answer",
        },
    )

    graph.add_conditional_edges(
        "agent_node",
        _agent_routing,
        {
            "retrieval": "retrieval",
            "relevancy_check": "relevancy_check",
            "generate_answer": "generate_answer",
        },
    )
    graph.add_edge("retrieval", "agent_node")

    graph.add_conditional_edges(
        "relevancy_check",
        _after_relevancy_routing,
        {"query_rewrite": "query_rewrite", "generate_answer": "generate_answer"},
    )
    graph.add_edge("query_rewrite", "agent_node")

    graph.add_edge("verify_claim", "generate_answer")
    graph.add_edge("generate_answer", END)

    return graph.compile(checkpointer=checkpointer)
