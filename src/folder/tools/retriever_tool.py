"""LangChain @tool definition for the vector-store retriever.

"""

from __future__ import annotations
from typing import Annotated
from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState # This annotation enables tools to access graph state without exposing state management details to the language model.
from langgraph.types import Command # Allow a tool to issue a command to modify the graph state during execution
from folder.schemas.tools_Schema import RetrieverInput
from folder.services.vector_store import vector_store



@tool(args_schema=RetrieverInput)
def retrieve_from_vectorstore(
    query: str,
    k: int,
    session_id: Annotated[str, InjectedState("session_id")],
    current_docs: Annotated[list, InjectedState("retrieved_docs")],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> list:
    """Search the uploaded research paper vector store for relevant passages."""
    docs = vector_store.search(query=query, session_id=session_id, k=k)
    if not docs:
        return [
            ToolMessage(
                content="No relevant documents found in the vector store.",
                tool_call_id=tool_call_id,
            )
        ]
    summary = f"Retrieved {len(docs)} chunk(s) from the vector store."
    return [
        ToolMessage(content=summary, tool_call_id=tool_call_id),
        Command(update={"retrieved_docs": (current_docs or []) + docs}),
    ]