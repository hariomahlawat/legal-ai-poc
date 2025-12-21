"""Lightweight API client for the Streamlit UI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import requests


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------
@dataclass
class BackendStatus:
    base_url: str
    reachable: bool
    message: str
    attempted: List[str]


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------
class ApiClient:
    """Simple API client that can auto-detect a healthy backend."""

    def __init__(self, candidates: Iterable[str]):
        self._candidates: List[str] = [c.rstrip("/") for c in candidates if c]
        self._session = requests.Session()
        self._status: Optional[BackendStatus] = None

    # ---- Candidate management ----
    def update_candidates(self, candidates: Iterable[str]) -> None:
        """Update candidate list while preserving already-known base."""

        deduped = []
        seen = set()
        for c in candidates:
            norm = c.rstrip("/")
            if norm and norm not in seen:
                deduped.append(norm)
                seen.add(norm)

        self._candidates = deduped

    # ---- Backend health ----
    def ensure_status(self, force_refresh: bool = False) -> BackendStatus:
        if self._status and not force_refresh:
            return self._status

        attempted: List[str] = []
        last_error: str = ""

        for base in self._candidates:
            attempted.append(base)
            try:
                resp = self._session.get(f"{base}/health", timeout=2.0)
                if resp.status_code == 200:
                    self._status = BackendStatus(
                        base_url=base,
                        reachable=True,
                        message="Reachable",
                        attempted=attempted,
                    )
                    return self._status
                last_error = f"HTTP {resp.status_code}"
            except requests.RequestException as exc:  # pragma: no cover - UI path
                last_error = str(exc)

        # No candidate reachable; keep first candidate to avoid None handling
        fallback_base = self._candidates[0] if self._candidates else ""
        self._status = BackendStatus(
            base_url=fallback_base,
            reachable=False,
            message=last_error or "Backend not reachable",
            attempted=attempted,
        )
        return self._status

    # ---- HTTP helpers ----
    def get(self, path: str, timeout: float = 4.0) -> requests.Response:
        status = self.ensure_status()
        if not status.reachable:
            raise RuntimeError(
                f"Backend not reachable at {status.base_url}. Tried: {', '.join(status.attempted)}"
            )
        return self._session.get(f"{status.base_url}{path}", timeout=timeout)

    def post_stream(
        self, path: str, payload: Dict[str, Any], timeout: float = 60.0
    ) -> requests.Response:
        status = self.ensure_status()
        if not status.reachable:
            raise RuntimeError(
                f"Backend not reachable at {status.base_url}. Tried: {', '.join(status.attempted)}"
            )
        return self._session.post(
            f"{status.base_url}{path}",
            json=payload,
            headers={"Accept": "text/event-stream"},
            stream=True,
            timeout=timeout,
        )

    # ---- Utilities ----
    @property
    def base_url(self) -> str:
        status = self.ensure_status()
        return status.base_url

    @property
    def status(self) -> BackendStatus:
        return self.ensure_status()

