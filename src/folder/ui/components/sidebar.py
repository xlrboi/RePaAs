"""Sidebar: session switcher + document upload (file / web URL / ArXiv)."""

from __future__ import annotations

import streamlit as st

from folder.ui import state as ui_state
from folder.ui.components import document_loader


def _render_session_switcher(graph, active_sid: str) -> str:
    if st.button("+ New Chat", use_container_width=True):
        new_sid = ui_state.create_session()
        st.session_state.active_session_id = new_sid
        st.rerun()

    st.divider()
    st.markdown("## 💬 Sessions")

    sorted_sessions = sorted(
        st.session_state.sessions_meta.values(), key=lambda s: s["created_at"], reverse=True
    )
    for session in sorted_sessions:
        sid = session["id"]
        is_active = sid == st.session_state.active_session_id
        clicked = st.button(
            session["name"],
            key=f"sess_{sid}",
            use_container_width=True,
            type="primary" if is_active else "secondary",
        )
        if clicked and not is_active:
            ui_state.switch_session(graph, sid)
            st.rerun()

    return st.session_state.active_session_id


def _render_file_upload(active_sid: str) -> None:
    st.markdown("**Upload Files**")
    uploaded_files = st.file_uploader(
        "PDF, TXT, or Markdown",
        type=["pdf", "txt", "md", "markdown"],
        accept_multiple_files=True,
        key=f"uploader_{active_sid}",
        label_visibility="collapsed",
    )
    if st.button("Add Files", use_container_width=True, key="btn_add_files"):
        if not uploaded_files:
            st.warning("No files selected.")
            return
        processed_key = f"processed_files_{active_sid}"
        st.session_state.setdefault(processed_key, set())
        with st.spinner("Processing files…"):
            for f in uploaded_files:
                if f.name in st.session_state[processed_key]:
                    st.info(f"Already loaded: {f.name}")
                    continue
                try:
                    document_loader.ingest_uploaded_file(f.read(), f.name, active_sid)
                    st.session_state[processed_key].add(f.name)
                    st.success(f"Added: {f.name}")
                except Exception as e:
                    st.error(f"Failed: {f.name} — {e}")
        st.rerun()


def _render_web_loader(active_sid: str) -> None:
    st.markdown("**Web Pages**")
    url_input = st.text_area(
        "URLs (one per line)",
        key=f"url_area_{active_sid}",
        height=80,
        label_visibility="collapsed",
        placeholder="https://example.com/paper",
    )
    if st.button("Load URLs", use_container_width=True, key="btn_load_urls"):
        urls = [u.strip() for u in url_input.splitlines() if u.strip()]
        if not urls:
            st.warning("Enter at least one URL.")
            return
        with st.spinner("Loading web pages…"):
            for url in urls:
                try:
                    document_loader.ingest_webpage(url, active_sid)
                    st.success(f"Loaded: {url[:60]}")
                except Exception as e:
                    st.error(f"Failed: {url[:60]} — {e}")
        st.rerun()


def _render_arxiv_loader(active_sid: str) -> None:
    st.markdown("**ArXiv Papers**")
    arxiv_title = st.text_input(
        "Paper title or ArXiv ID",
        key=f"arxiv_input_{active_sid}",
        label_visibility="collapsed",
        placeholder="1706.03762  or  Attention Is All You Need",
    )
    if st.button("Load ArXiv Paper", use_container_width=True, key="btn_load_arxiv"):
        if not arxiv_title.strip():
            st.warning("Enter a paper title or ArXiv ID.")
            return
        with st.spinner("Loading from ArXiv…"):
            try:
                loaded_title = document_loader.ingest_arxiv(arxiv_title.strip(), active_sid)
                st.success(f"Loaded: {loaded_title}")
            except Exception as e:
                st.error(f"Failed: {e}")
        st.rerun()


def _render_loaded_documents(active_sid: str) -> None:
    st.divider()
    st.markdown("### Loaded Documents")
    doc_titles = document_loader.loaded_titles(active_sid)
    if doc_titles is None:
        st.caption("Could not load document list — try refreshing.")
    elif doc_titles:
        for title in doc_titles:
            st.markdown(f"- {title}")
    else:
        st.caption("No documents loaded yet.")


def render_sidebar(graph, active_sid: str) -> str:
    with st.sidebar:
        active_sid = _render_session_switcher(graph, active_sid)
        st.divider()
        st.markdown("## 📄 Documents")
        _render_file_upload(active_sid)
        _render_web_loader(active_sid)
        _render_arxiv_loader(active_sid)
        _render_loaded_documents(active_sid)
    return active_sid
