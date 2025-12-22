import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import streamlit as st

# === Path Setup ===
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from apps.ui.client.api_client import APIClient

APP_TITLE = "SDD Legal AI System (PoC)"
BACKEND_DEFAULT = "http://127.0.0.1:8000"
CITATION_PATTERN = re.compile(r"\[([A-Za-z]+-[A-Za-z0-9]+)\]")


# === Citation Helpers ===


def parse_citation_ids(text: str) -> List[str]:
    seen = set()
    found: List[str] = []
    for match in CITATION_PATTERN.findall(text or ""):
        if match not in seen:
            seen.add(match)
            found.append(match)
    return found


def _format_score(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"


def _format_focus(value: Optional[bool]) -> str:
    if value is None:
        return "n/a"
    return "Yes" if value else "No"


def _short_heading_path(heading_path: str) -> str:
    parts = [p.strip() for p in (heading_path or "").split(">") if p.strip()]
    return parts[-1] if parts else heading_path.strip()


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
        st.session_state["messages"] = []  # list of dicts {role, content, citations?}
    if "last_citations" not in st.session_state:
        st.session_state["last_citations"] = []
    if "last_warnings" not in st.session_state:
        st.session_state["last_warnings"] = []
    if "citation_cache" not in st.session_state:
        st.session_state["citation_cache"] = {}


# === Evidence Rendering ===


def _fetch_citation_detail(client: APIClient, citation_id: str) -> Dict[str, Any]:
    cache = st.session_state.get("citation_cache", {})
    if citation_id in cache:
        return cache[citation_id]

    try:
        detail = client.get_citation(citation_id, timeout=10)
        cache[citation_id] = {"data": detail}
    except Exception as exc:  # pragma: no cover - network/IO dependent
        cache[citation_id] = {"error": str(exc)}

    st.session_state["citation_cache"] = cache
    return cache[citation_id]


def _render_evidence_for_message(message: Dict[str, Any], client: APIClient) -> None:
    if not st.session_state.get("show_citations", True):
        return

    citations = message.get("citations") or []
    if not citations:
        return

    st.subheader("Evidence")
    for citation_id in citations:
        detail_entry = _fetch_citation_detail(client, citation_id)
        error_message = detail_entry.get("error")
        detail = detail_entry.get("data", {}) if isinstance(detail_entry, dict) else {}

        heading_path = detail.get("heading_path") or detail.get("location") or ""
        heading_short = _short_heading_path(heading_path)
        source_file = detail.get("source_file") or ""
        source_basename = Path(source_file).name if source_file else "unknown"

        expander_title = f"{citation_id} | {source_basename} | {heading_short or 'N/A'}"
        with st.expander(expander_title):
            if error_message:
                st.info("Evidence not available")
                st.caption(error_message)
                continue

            st.caption(f"Source file: {source_file or 'n/a'}")
            st.caption(f"Heading path: {heading_path or 'n/a'}")

            why_line = (
                f"**Why retrieved:** "
                f"Retrieval score: {_format_score(detail.get('retrieval_score'))} | "
                f"Rerank score: {_format_score(detail.get('rerank_score'))} | "
                f"Hit query count: {detail.get('hit_query_count') if detail.get('hit_query_count') is not None else 'n/a'} | "
                f"Focus applied: {_format_focus(detail.get('focus_applied'))}"
            )
            st.markdown(why_line)

            evidence_text = detail.get("text") or detail.get("verbatim") or detail.get("snippet") or ""
            if evidence_text:
                st.code(evidence_text)
            else:
                st.info("Evidence not available")


# === Message Rendering ===


def render_messages(client: APIClient):
    for m in st.session_state["messages"]:
        role = m.get("role", "")
        content = m.get("content", "")
        if role == "user":
            st.markdown(f"🧑 **{content}**")
        else:
            st.markdown(f"🤖 {content}")
            _render_evidence_for_message(m, client)


def render_warnings():
    warnings = st.session_state.get("last_warnings") or []
    if not warnings:
        return
    st.subheader("Compliance warnings")
    for w in warnings:
        st.warning(w)


# === Application Entry Point ===


def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    init_state()

    client, status = backend_settings_panel()

    st.title(APP_TITLE)
    st.caption("Chat-style UI with grounded answers. FastAPI backend on port 8000.")

    st.header("Chat")
    render_messages(client)

    user_text = st.chat_input("Ask a question")
    if user_text:
        st.session_state["messages"].append({"role": "user", "content": user_text})
        st.rerun()

    # If last message is user and not yet answered, call backend
    if st.session_state["messages"] and st.session_state["messages"][-1]["role"] == "user":
        # Show "Working..." placeholder message in UI
        placeholder = st.empty()
        placeholder.markdown("🤖 *Working...*")

        # --- Streaming request payload ---
        payload = {
            "messages": st.session_state["messages"],
            "want_citations": bool(st.session_state.get("show_citations", True)),
            "want_warnings": bool(st.session_state.get("show_compliance", True)),
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
                        txt = data.get("text") or data.get("t") or ""
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

            citation_ids = parse_citation_ids(final_answer)

            st.session_state["messages"].append(
                {"role": "assistant", "content": final_answer, "citations": citation_ids}
            )
            placeholder.empty()
            st.rerun()

        except Exception as e:
            placeholder.empty()
            st.error(f"Backend error: {e}")

    if st.session_state.get("show_compliance", True):
        render_warnings()


if __name__ == "__main__":
    main()
