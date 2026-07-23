"""Main chat panel: message history, streaming responses, and the ``/btw`` side channel."""

from __future__ import annotations

import streamlit as st
from langchain_core.messages import HumanMessage

from folder.btw.handler import handle_btw
from folder.ui import state as ui_state


def _render_header() -> None:
    st.title("📚 RePaAs — Research Paper Assistant")
    st.markdown(
        "🔍 **Ask questions** from your uploaded papers &nbsp;·&nbsp; "
        "✅ **Verify claims** against recent literature &nbsp;·&nbsp; "
        "🌐 **Search the web** for the latest findings\n\n"
        "> Upload documents in the sidebar and start chatting below."
    )
    st.divider()


def _render_history(active_sid: str) -> None:
    for msg in st.session_state.chats.get(active_sid, []):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                with st.expander(f"📊 Graph state · turn {msg['turn']}", expanded=False):
                    st.json(msg["graph_state"])


def _handle_btw(prompt: str) -> None:
    query = prompt.strip()[4:].strip()
    with st.chat_message("user"):
        st.markdown(prompt)
        st.caption("Side channel — not saved to session history.")
    with st.chat_message("assistant"):
        if not query:
            st.markdown("Please add a question after `/btw`, e.g. `/btw What is attention?`")
        else:
            placeholder = st.empty()
            response_text = ""
            for chunk in handle_btw(query):
                response_text += chunk
                placeholder.markdown(response_text + "▌")
            placeholder.markdown(response_text)
        st.caption("Side channel — not saved to session history.")


def _handle_rag_turn(graph, prompt: str, active_sid: str) -> None:
    st.session_state.chats.setdefault(active_sid, [])
    st.session_state.turns.setdefault(active_sid, 0)

    is_first_message = len(st.session_state.chats[active_sid]) == 0

    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.chats[active_sid].append({"role": "user", "content": prompt})
    st.session_state.turns[active_sid] += 1
    current_turn = st.session_state.turns[active_sid]

    if is_first_message:
        ui_state.maybe_rename_session(active_sid, prompt)

    input_state = {
        "messages": [HumanMessage(content=prompt)],
        "session_id": active_sid,
        "query": prompt,
        "route": None,
        "retrieved_docs": [],
        "retrieval_attempts": 0,
        "claim_verdict": None,
        "claim_source": None,
        "superseding_papers": [],
        "answer": None,
        "is_relevant": None,
        "rewrite_count": 0,
    }
    config = {"configurable": {"thread_id": active_sid}}

    with st.chat_message("assistant"):
        placeholder = st.empty()
        response_text = ""

        for chunk, metadata in graph.stream(input_state, config, stream_mode="messages"):
            if (
                metadata.get("langgraph_node") == "generate_answer"
                and hasattr(chunk, "content")
                and chunk.content
            ):
                response_text += chunk.content
                placeholder.markdown(response_text + "▌")

        if not response_text:
            final_values = graph.get_state(config).values
            response_text = final_values.get("answer") or "No response generated."

        placeholder.markdown(response_text)

        final_values = graph.get_state(config).values
        state_snapshot = ui_state.serialize_state(final_values)

        with st.expander(f"📊 Graph state · turn {current_turn}", expanded=False):
            st.json(state_snapshot)

    st.session_state.chats[active_sid].append(
        {
            "role": "assistant",
            "content": response_text,
            "graph_state": state_snapshot,
            "turn": current_turn,
        }
    )

    if is_first_message:
        st.rerun()


def render_chat(graph, active_sid: str) -> None:
    _render_header()
    _render_history(active_sid)

    prompt = st.chat_input("Ask about your papers, verify a claim, or search the web…")
    if not prompt:
        return

    if prompt.strip().lower().startswith("/btw"):
        _handle_btw(prompt)
    else:
        _handle_rag_turn(graph, prompt, active_sid)
