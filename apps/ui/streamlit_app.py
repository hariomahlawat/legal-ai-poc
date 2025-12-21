import json
import requests
import streamlit as st
from typing import Dict, Any, List, Optional
import os

def get_api_base() -> str:
    # 1) Environment variable wins
    env = os.getenv("API_BASE")
    if env:
        return env.rstrip("/")

    # 2) Use Streamlit secrets only if secrets exist
    try:
        # Accessing st.secrets triggers parsing; wrap to avoid crash if secrets.toml missing
        val = st.secrets["API_BASE"]
        if isinstance(val, str) and val.strip():
            return val.strip().rstrip("/")
    except Exception:
        pass

    # 3) Default
    return "http://127.0.0.1:8000"

API_BASE = get_api_base()



# -----------------------------
# UI polish (professional grade)
# -----------------------------
def inject_css() -> None:
    st.markdown(
        """
<style>
/* Layout tuning */
.block-container { padding-top: 2.0rem; padding-bottom: 2.5rem; max-width: 1180px; }
section[data-testid="stSidebar"] { border-right: 1px solid rgba(49,51,63,0.08); }

/* Typography */
h1, h2, h3 { letter-spacing: -0.01em; }
.small-muted { color: rgba(49,51,63,0.65); font-size: 0.92rem; }

/* Chat bubbles */
[data-testid="stChatMessage"] { padding: 0.25rem 0; }
.chat-card {
  border: 1px solid rgba(49,51,63,0.10);
  border-radius: 14px;
  padding: 14px 16px;
  background: white;
  box-shadow: 0 1px 2px rgba(0,0,0,0.04);
}

/* Citation cards */
.cite-grid { display: grid; grid-template-columns: 1fr auto; gap: 10px; align-items: start; }
.cite-card {
  border: 1px solid rgba(49,51,63,0.10);
  border-radius: 14px;
  padding: 12px 14px;
  background: white;
}
.cite-title { font-weight: 650; margin-bottom: 2px; }
.cite-meta { color: rgba(49,51,63,0.60); font-size: 0.88rem; margin-bottom: 8px; }
.cite-snippet { color: rgba(49,51,63,0.80); font-size: 0.94rem; }
.cite-btn { margin-top: 2px; }

/* Sidebar status badge */
.badge {
  display: inline-block;
  padding: 0.18rem 0.55rem;
  border-radius: 999px;
  font-size: 0.80rem;
  border: 1px solid rgba(49,51,63,0.12);
}
.badge-ok { background: rgba(34,197,94,0.10); color: rgb(21,128,61); border-color: rgba(34,197,94,0.25); }
.badge-err { background: rgba(239,68,68,0.10); color: rgb(153,27,27); border-color: rgba(239,68,68,0.25); }

/* Buttons */
button[kind="secondary"] { border-radius: 10px !important; }
</style>
""",
        unsafe_allow_html=True,
    )


# -----------------------------
# API helpers
# -----------------------------
def api_get(path: str, timeout: float = 4.0) -> requests.Response:
    return requests.get(f"{API_BASE}{path}", timeout=timeout)


def api_post_stream(path: str, payload: Dict[str, Any], timeout: float = 60.0):
    # SSE stream via POST
    return requests.post(
        f"{API_BASE}{path}",
        json=payload,
        headers={"Accept": "text/event-stream"},
        stream=True,
        timeout=timeout,
    )


def parse_sse_lines(resp: requests.Response):
    """
    Minimal SSE parser:
      event: <name>
      data: <json>
    """
    event_name = None
    for raw in resp.iter_lines(decode_unicode=True):
        if raw is None:
            continue
        line = raw.strip()
        if not line:
            event_name = None
            continue
        if line.startswith("event:"):
            event_name = line.split(":", 1)[1].strip()
        elif line.startswith("data:"):
            data_str = line.split(":", 1)[1].strip()
            try:
                data = json.loads(data_str)
            except Exception:
                data = {"raw": data_str}
            yield event_name or "message", data


# -----------------------------
# Verbatim dialog
# -----------------------------
@st.dialog("Verbatim source text")
def show_verbatim_dialog(citation_id: str):
    try:
        r = api_get(f"/citations/{citation_id}", timeout=10.0)
        if r.status_code != 200:
            st.error(f"Could not fetch citation text. HTTP {r.status_code}")
            st.stop()
        payload = r.json()
    except Exception as e:
        st.error(f"Could not fetch citation text: {e}")
        st.stop()

    # Header
    st.caption(
        f"Document: {payload.get('document', 'MML')} | Source: {payload.get('source_file', '-')}"
    )
    st.caption(
        f"Heading: {payload.get('heading', '-')} | Location: {payload.get('location', '-')}"
    )

    st.markdown("**Citation text (verbatim)**")
    st.code(payload.get("verbatim", "").strip(), language="text")

    with st.expander("Context (before)", expanded=False):
        st.code(payload.get("context_before", "").strip(), language="text")

    with st.expander("Context (after)", expanded=False):
        st.code(payload.get("context_after", "").strip(), language="text")


# -----------------------------
# Page
# -----------------------------
st.set_page_config(page_title="SDD Legal AI System (PoC)", layout="wide")
inject_css()

st.title("SDD Legal AI System (PoC)")
st.markdown(
    '<div class="small-muted">Chat-style UI with grounded answers. FastAPI backend on port 8000.</div>',
    unsafe_allow_html=True,
)

# Sidebar
with st.sidebar:
    st.header("Settings")
    mode = st.selectbox("Mode", ["Chat", "Guided Workflow"], index=0)
    show_citations = st.checkbox("Show citations", value=True)
    show_warnings = st.checkbox("Show compliance warnings", value=True)

    st.divider()

    st.subheader("Backend status")
    try:
        r = api_get("/health", timeout=2.0)
        ok = r.status_code == 200
    except Exception:
        ok = False

    if ok:
        st.markdown('<span class="badge badge-ok">OK</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge badge-err">ERROR (not reachable)</span>', unsafe_allow_html=True)

    if st.button("Recheck", use_container_width=True):
        st.rerun()

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of dict(role, content)
if "last_citations" not in st.session_state:
    st.session_state.last_citations = []  # list of citation dicts
if "last_warnings" not in st.session_state:
    st.session_state.last_warnings = []


st.subheader("Chat")

# Render history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(f'<div class="chat-card">{m["content"]}</div>', unsafe_allow_html=True)

# Input
prompt = st.chat_input("Ask a question")
if prompt:
    # Add user msg
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(f'<div class="chat-card">{prompt}</div>', unsafe_allow_html=True)

    # Stream assistant
    assistant_box = st.chat_message("assistant")
    with assistant_box:
        holder = st.empty()
        running_text = ""
        holder.markdown(
            '<div class="chat-card small-muted">Working…</div>',
            unsafe_allow_html=True,
        )

    st.session_state.last_citations = []
    st.session_state.last_warnings = []

    req = {
        "case_id": "default",
        "messages": st.session_state.messages,
        "mode": mode,
        "want_citations": bool(show_citations),
        "want_warnings": bool(show_warnings),
    }

    try:
        with api_post_stream("/chat/stream", req, timeout=120.0) as resp:
            if resp.status_code != 200:
                raise RuntimeError(f"API error: HTTP {resp.status_code} - {resp.text}")

            for event, data in parse_sse_lines(resp):
                if event == "token":
                    running_text += data.get("text", "")
                    with assistant_box:
                        holder.markdown(
                            f'<div class="chat-card">{running_text}</div>',
                            unsafe_allow_html=True,
                        )
                elif event == "citations":
                    st.session_state.last_citations = data.get("items", [])
                elif event == "warnings":
                    st.session_state.last_warnings = data.get("items", [])
                elif event == "done":
                    pass

        # Persist assistant message
        st.session_state.messages.append({"role": "assistant", "content": running_text})

    except Exception as e:
        err_text = f"Backend error: {e}"
        st.session_state.messages.append({"role": "assistant", "content": err_text})
        with st.chat_message("assistant"):
            st.error(err_text)

# Citations panel (for last answer)
if show_citations:
    st.markdown("### Citations")
    citations: List[Dict[str, Any]] = st.session_state.get("last_citations") or []
    if not citations:
        st.caption("No citations for the last answer.")
    else:
        for idx, c in enumerate(citations, start=1):
            title = c.get("title") or "Untitled"
            citation_id = c.get("citation_id") or c.get("source_id") or ""
            meta = f"Source: {c.get('source_file','-')} | Heading: {c.get('heading','-')}"
            snippet = (c.get("snippet") or "").strip()

            cols = st.columns([6, 2], vertical_alignment="top")
            with cols[0]:
                st.markdown(
                    f"""
<div class="cite-card">
  <div class="cite-title">{idx}. {title}</div>
  <div class="cite-meta">{meta}</div>
  <div class="cite-snippet">{snippet}</div>
</div>
""",
                    unsafe_allow_html=True,
                )
            with cols[1]:
                st.write("")
                if st.button("View verbatim", key=f"vb_{idx}", use_container_width=True, type="secondary"):
                    if citation_id:
                        show_verbatim_dialog(citation_id)
                    else:
                        st.warning("Citation ID missing.")

# Warnings panel (optional)
if show_warnings:
    warnings: List[Dict[str, Any]] = st.session_state.get("last_warnings") or []
    if warnings:
        st.markdown("### Compliance warnings")
        for w in warnings:
            st.warning(w.get("message", str(w)))
