import json
import time
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import streamlit as st

from apps.ui.client.api_client import APIClient

APP_TITLE = "SDD Legal AI System (PoC)"
BACKEND_DEFAULT = "http://127.0.0.1:8000"


def parse_sse_lines(resp) -> Iterator[Tuple[str, Any]]:
    """
    Parse a basic SSE stream where the backend sends:

      event: <name>
      data: <json>
      <blank line>

    Returns (event_name, parsed_json_data)
    """
    event_name = None
    data_lines: List[str] = []

    for raw_line in resp.iter_lines(decode_unicode=True):
        if raw_line is None:
            continue

        line = raw_line.strip("\r")

        if line == "":
            # End of one SSE frame
            if event_name and data_lines:
                data_str = "\n".join(data_lines)
                try:
                    data = json.loads(data_str)
                except Exception:
                    data = data_str
                yield event_name, data
            event_name = None
            data_lines = []
            continue

        if line.startswith("event:"):
            event_name = line[len("event:") :].strip()
            continue

        if line.startswith("data:"):
            data_lines.append(line[len("data:") :].strip())
            continue


def backend_settings_panel() -> Tuple[APIClient, Dict[str, Any]]:
    with st.sidebar:
        st.header("Settings")

        backend_base = st.text_input("Backend base URL", value=st.session_state.get("backend_base", BACKEND_DEFAULT))
        st.session_state["backend_base"] = backend_base

        mode = st.selectbox("Mode", options=["Chat"], index=0, key="mode")

        show_citations = st.checkbox("Show citations", value=st.session_state.get("show_citations", True), key="show_citations")
        show_compliance = st.checkbox(
            "Show compliance warnings",
            value=st.session_state.get("show_compliance", True),
            key="show_compliance",
        )

        st.divider()
        st.subheader("Backend status")

        client = APIClient(base_url=backend_base)
        status: Dict[str, Any] = {"ok": False}

        try:
            health = client.get_health(timeout=5)
            status = {"ok": True, "health": health}
            st.success("OK")
        except Exception:
            st.error("NOT OK")

        st.caption(f"Base: {backend_base}")
        if st.button("Recheck"):
            st.rerun()

    return client, status


def init_state():
    if "messages" not in st.session_state:
        st.session_state["messages"] = []  # list of dicts {role, content}
    if "last_citations" not in st.session_state:
        st.session_state["last_citations"] = []
    if "last_warnings" not in st.session_state:
        st.session_state["last_warnings"] = []


def render_messages():
    for m in st.session_state["messages"]:
        role = m.get("role", "")
        content = m.get("content", "")
        if role == "user":
            st.markdown(f"🧑 **{content}**")
        else:
            st.markdown(f"🤖 {content}")


def render_citations():
    st.subheader("Citations")
    cits = st.session_state.get("last_citations") or []
    if not cits:
        st.caption("No citations for the last answer.")
        return
    for c in cits:
        st.write(c)


def render_warnings():
    warnings = st.session_state.get("last_warnings") or []
    if not warnings:
        return
    st.subheader("Compliance warnings")
    for w in warnings:
        st.warning(w)


def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    init_state()

    client, status = backend_settings_panel()

    st.title(APP_TITLE)
    st.caption("Chat-style UI with grounded answers. FastAPI backend on port 8000.")

    st.header("Chat")
    render_messages()

    user_text = st.chat_input("Ask a question")
    if user_text:
        st.session_state["messages"].append({"role": "user", "content": user_text})
        st.rerun()

    # If last message is user and not yet answered, call backend
    if st.session_state["messages"] and st.session_state["messages"][-1]["role"] == "user":
        # Show "Working..." placeholder message in UI
        placeholder = st.empty()
        placeholder.markdown("🤖 *Working...*")

        payload = {
            "messages": st.session_state["messages"],
            "want_citations": bool(st.session_state.get("show_citations", True)),
            "want_compliance_warnings": bool(st.session_state.get("show_compliance", True)),
        }

        answer_chunks: List[str] = []
        st.session_state["last_citations"] = []
        st.session_state["last_warnings"] = []

        try:
            # IMPORTANT CHANGE:
            # Use timeout=None so Requests does not apply a read timeout while the backend is still working.
            with client.post_stream("/chat/stream", payload, timeout=None) as resp:
                for event, data in parse_sse_lines(resp):
                    if event == "token":
                        txt = data.get("t", "")
                        if txt:
                            answer_chunks.append(txt)
                            placeholder.markdown("🤖 " + "".join(answer_chunks))
                    elif event == "citations":
                        st.session_state["last_citations"] = data.get("items", []) or []
                    elif event == "warnings":
                        st.session_state["last_warnings"] = data.get("items", []) or []
                    elif event == "done":
                        break

            final_answer = "".join(answer_chunks).strip()
            if not final_answer:
                final_answer = "(No answer received.)"

            st.session_state["messages"].append({"role": "assistant", "content": final_answer})
            placeholder.empty()
            st.rerun()

        except Exception as e:
            placeholder.empty()
            st.error(f"Backend error: {e}")

    if st.session_state.get("show_compliance", True):
        render_warnings()

    if st.session_state.get("show_citations", True):
        render_citations()


if __name__ == "__main__":
    main()
