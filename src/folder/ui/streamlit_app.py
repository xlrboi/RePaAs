"""Thin Streamlit entrypoint.

Run with:  streamlit run src/folder/ui/streamlit_app.py
All logic lives in ``ui/state.py`` and ``ui/components/*`` — this file is
just wiring so the app is easy to scan.
"""

from __future__ import annotations

import streamlit as st

from folder.logging_config import configure_logging
from folder.ui import state as ui_state
from folder.ui.components.chat import render_chat
from folder.ui.components.sidebar import render_sidebar

configure_logging()

st.set_page_config(page_title="RePaAs", page_icon="📚", layout="centered")

graph = ui_state.get_graph()
active_sid = ui_state.bootstrap(graph)

active_sid = render_sidebar(graph, active_sid)
render_chat(graph, active_sid)
