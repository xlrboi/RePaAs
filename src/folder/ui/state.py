"""st.session_state helpers — bootstrap, session switching, serialization.
"""

from __future__ import annotations

import streamlit as st
from langchain_core.prompts import ChatPromptTemplate

from folder.constants import SESSION_NAME_SOURCE_CHARS
from folder.core.graph import build_graph
from folder.services.llm import get_utility_llm
from folder.sessions import store as sessions_store

RENAME_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Generate a concise 3-5 word title for a research chat session "
            "based on the user's first message. Return only the title, "
            "no punctuation at the end, no quotes.",
        ),
        ("human", "{message}"),
    ]
)


@st.cache_resource
def get_graph():
    return build_graph()


def generate_session_name(first_message: str) -> str:
    try:
        chain = RENAME_PROMPT | get_utility_llm()
        response = chain.invoke({"message": first_message[:SESSION_NAME_SOURCE_CHARS]})
        return response.content.strip()
    except Exception:
        return "New Session"


def maybe_rename_session(session_id: str, first_message: str) -> None:
    if st.session_state.sessions_meta.get(session_id, {}).get("is_named"):
        return
    name = generate_session_name(first_message)
    st.session_state.sessions_meta[session_id]["name"] = name
    st.session_state.sessions_meta[session_id]["is_named"] = True
    sessions_store.save_sessions(st.session_state.sessions_meta)


def create_session() -> str:
    meta = sessions_store.create_session()
    st.session_state.sessions_meta = sessions_store.load_sessions()
    st.session_state.chats[meta.id] = []
    st.session_state.turns[meta.id] = 0
    return meta.id


def serialize_state(values: dict) -> dict:
    out = {}
    for k, v in values.items():
        if k == "messages":
            out[k] = [
                {
                    "type": type(m).__name__,
                    "content": (
                        m.content[:300] if isinstance(m.content, str) else repr(m.content)[:300]
                    ),
                }
                for m in (v or [])
            ]
        elif k == "retrieved_docs":
            out[k] = [{"content": d.page_content[:300], "metadata": d.metadata} for d in (v or [])]
        else:
            out[k] = v
    return out


def load_session_chats(graph, session_id: str) -> list[dict]:
    config = {"configurable": {"thread_id": session_id}}
    try:
        state = graph.get_state(config)
        if not state or not state.values:
            return []
        chats = []
        turn = 0
        for msg in state.values.get("messages", []):
            type_name = type(msg).__name__
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            if type_name == "HumanMessage":
                chats.append({"role": "user", "content": content})
            elif type_name in ("AIMessage", "AIMessageChunk"):
                turn += 1
                chats.append(
                    {"role": "assistant", "content": content, "turn": turn, "graph_state": {}}
                )
        return chats
    except Exception:
        return []


def switch_session(graph, session_id: str) -> None:
    st.session_state.active_session_id = session_id
    if session_id not in st.session_state.chats:
        st.session_state.chats[session_id] = load_session_chats(graph, session_id)
    if session_id not in st.session_state.turns:
        turn_count = sum(1 for m in st.session_state.chats[session_id] if m["role"] == "assistant")
        st.session_state.turns[session_id] = turn_count


def bootstrap(graph) -> str:
    """Initialize st.session_state on first run; return the active session id."""
    if "sessions_meta" not in st.session_state:
        st.session_state.sessions_meta = sessions_store.load_sessions()
    if "chats" not in st.session_state:
        st.session_state.chats = {}
    if "turns" not in st.session_state:
        st.session_state.turns = {}
    if "active_session_id" not in st.session_state:
        if st.session_state.sessions_meta:
            latest = max(st.session_state.sessions_meta.values(), key=lambda s: s["created_at"])
            switch_session(graph, latest["id"])
        else:
            st.session_state.active_session_id = create_session()
    return st.session_state.active_session_id
