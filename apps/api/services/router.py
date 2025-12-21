from __future__ import annotations

import re
from typing import Literal

# --- Domain typing ---
Domain = Literal["LEGAL", "SYSTEM_HELP", "CLARIFY"]

# --- Domain hints ---
_SYSTEM_KEYWORDS = [
    "backend",
    "frontend",
    "fastapi",
    "uvicorn",
    "streamlit",
    "ollama",
    "port",
    "localhost",
    "pip",
    "venv",
    "virtualenv",
    "docker",
    "npm",
    "node",
    "run",
    "start",
    "install",
    "error",
    "traceback",
    "powershell",
    "cmd",
]

_LEGAL_HINTS = [
    "court of inquiry",
    "coi",
    "court-martial",
    "court martial",
    "disciplinary",
    "rule",
    "section",
    "para",
    "appendix",
    "army order",
    "act",
    "regulation",
]


# --- Domain router ---
def route_domain(question: str) -> Domain:
    """
    Route a question to the correct domain pipeline.

    - SYSTEM_HELP: repo/devops/runtime/how-to-run questions.
    - CLARIFY: too vague to answer safely.
    - LEGAL: default for substantive legal questions.
    """

    q = (question or "").strip().lower()

    if not q:
        return "CLARIFY"

    # SYSTEM_HELP keyword hit
    for kw in _SYSTEM_KEYWORDS:
        if kw in q:
            return "SYSTEM_HELP"

    # Very short and not legal-hinted => CLARIFY
    # Rule: fewer than 6 words and no legal hints.
    word_count = len(re.findall(r"\S+", q))
    if word_count < 6:
        for hint in _LEGAL_HINTS:
            if hint in q:
                return "LEGAL"
        return "CLARIFY"

    return "LEGAL"
