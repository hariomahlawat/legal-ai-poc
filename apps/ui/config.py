"""UI configuration helpers.

This module centralises how the Streamlit UI discovers the backend API
endpoint. It keeps the logic small and testable so other modules do not
need to worry about environment variables or default ports.
"""

from __future__ import annotations

import os
from typing import List

import streamlit as st


# ---------------------------------------------------------------------------
# API base discovery
# ---------------------------------------------------------------------------
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORTS = (8000, 8001)


def _read_explicit_api_base() -> str | None:
    """Return an explicitly configured API base if provided.

    Preference order: environment variable, then Streamlit secrets. This keeps
    compatibility with containerised deployments where the backend may be
    exposed on a different host or port.
    """

    env = os.getenv("API_BASE")
    if env and env.strip():
        return env.strip()

    try:
        secret_val = st.secrets.get("API_BASE")  # type: ignore[attr-defined]
        if isinstance(secret_val, str) and secret_val.strip():
            return secret_val.strip()
    except Exception:
        # Streamlit raises when secrets.toml is missing; ignore to allow
        # graceful local development.
        pass

    return None


def _normalize_base(raw: str) -> str:
    """Normalise API base strings for consistent downstream use."""

    base = raw.strip()
    if not base:
        return ""

    if not base.startswith("http://") and not base.startswith("https://"):
        base = f"http://{base}"

    return base.rstrip("/")


def api_base_candidates() -> List[str]:
    """Return a de-duplicated list of candidate API base URLs.

    The explicit value, if present, is placed first. Default localhost ports
    follow so the UI can automatically fall back when a misconfigured port is
    provided (e.g., pointing to 8001 when the API is served on 8000).
    """

    candidates = []
    explicit = _read_explicit_api_base()
    if explicit:
        candidates.append(explicit)

    for port in DEFAULT_PORTS:
        candidates.append(f"http://{DEFAULT_HOST}:{port}")

    # Preserve order while removing duplicates
    seen = set()
    ordered_unique: List[str] = []
    for c in candidates:
        normalized = _normalize_base(c)
        if normalized and normalized not in seen:
            ordered_unique.append(normalized)
            seen.add(normalized)

    return ordered_unique

