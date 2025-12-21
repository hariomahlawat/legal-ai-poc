"""Session state helpers for the Streamlit UI."""

from __future__ import annotations

from typing import Any, Dict, List

import streamlit as st


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
def ensure_session_state() -> None:
    """Initialise commonly used session variables."""

    if "messages" not in st.session_state:
        st.session_state.messages: List[Dict[str, Any]] = []

    if "last_citations" not in st.session_state:
        st.session_state.last_citations: List[Dict[str, Any]] = []

    if "last_warnings" not in st.session_state:
        st.session_state.last_warnings: List[Dict[str, Any]] = []

