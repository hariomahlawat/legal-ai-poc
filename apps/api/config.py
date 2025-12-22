from __future__ import annotations

import os
from typing import Optional

# ============================
# Environment helpers
# ============================

def _get_env(key: str, default: str, fallback_key: Optional[str] = None) -> str:
    """Fetch environment variable with optional fallback key and default."""

    value = os.getenv(key)
    if value is None and fallback_key:
        value = os.getenv(fallback_key)
    return value if value is not None else default


# ============================
# Ollama settings
# ============================
OLLAMA_URL = _get_env("OLLAMA_URL", "http://127.0.0.1:11434", fallback_key="OLLAMA_BASE_URL")
OLLAMA_MODEL_LEGAL = _get_env("OLLAMA_MODEL_LEGAL", _get_env("OLLAMA_MODEL", "llama3.1:8b"))
OLLAMA_MODEL_SYS = _get_env("OLLAMA_MODEL_SYS", "mistral:latest")

OLLAMA_CONNECT_TIMEOUT_SECS = float(_get_env("OLLAMA_CONNECT_TIMEOUT_SECS", "5"))
OLLAMA_READ_TIMEOUT_SECS = float(_get_env("OLLAMA_READ_TIMEOUT_SECS", "180"))
OLLAMA_FAILURE_COOLDOWN_SECS = float(_get_env("OLLAMA_FAILURE_COOLDOWN_SECS", "60"))

OLLAMA_NUM_PREDICT = int(_get_env("OLLAMA_NUM_PREDICT", "700"))
OLLAMA_TEMPERATURE = float(_get_env("OLLAMA_TEMPERATURE", "0.2"))

# ============================
# Evidence limits
# ============================
SYNTHESIS_MAX_CITATIONS = int(_get_env("SYNTHESIS_MAX_CITATIONS", "6"))
EVIDENCE_MAX_CHARS_TOTAL = int(_get_env("EVIDENCE_MAX_CHARS_TOTAL", "9000"))
EVIDENCE_MAX_CHARS_PER_CITATION = int(_get_env("EVIDENCE_MAX_CHARS_PER_CITATION", "1100"))

