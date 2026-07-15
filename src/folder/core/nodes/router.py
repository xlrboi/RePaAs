"""Router node: classifies the incoming query into a workflow branch."""

from __future__ import annotations
from langchain_core.prompts import ChatPromptTemplate
from folder.core.state import RAGState
from folder.prompts.router import ROUTER_SYSTEM_PROMPT
from folder.schemas.routing import RouterDecision
from folder.services.llm import get_main_llm

ROUTER_PROMPT = ChatPromptTemplate.from_messages(
    [("system", ROUTER_SYSTEM_PROMPT), ("human", "{query}")]
)


def router_node(state: RAGState) -> dict:
    query = state["messages"][-1].content
    chain = ROUTER_PROMPT | get_main_llm().with_structured_output(RouterDecision)
    decision: RouterDecision = chain.invoke({"query": query})
    return {"route": decision.route}
