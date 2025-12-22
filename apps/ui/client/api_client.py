from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional, Tuple, Union

import requests


def _normalize_stream_timeout(timeout):
    """Normalize Requests timeout for streaming calls.

    If a single number is supplied, Requests treats it as both connect and read timeout.
    For SSE streaming, a finite read timeout can cause spurious ReadTimeout exceptions when
    the server is legitimately busy before emitting the next SSE frame.

    - timeout is None: no read timeout (connect timeout still applied).
    - timeout is (connect, read): passed through as-is.
    - timeout is number: interpreted as read timeout (connect timeout fixed to 10s).
    """
    if timeout is None:
        return (10, None)
    if isinstance(timeout, (int, float)):
        return (10, timeout)
    return timeout


TimeoutType = Union[None, float, int, Tuple[float, Optional[float]]]


@dataclass(frozen=True)
class APIClient:
    base_url: str

    def _url(self, path: str) -> str:
        if path.startswith("/"):
            return f"{self.base_url}{path}"
        return f"{self.base_url}/{path}"

    def get_health(self, timeout: TimeoutType = 10) -> Dict[str, Any]:
        r = requests.get(self._url("/health"), timeout=timeout)
        r.raise_for_status()
        return r.json()

    def get_citation(self, citation_id: str, timeout: TimeoutType = 10) -> Dict[str, Any]:
        url = self._url(f"/citations/{citation_id}")
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        return r.json()

    def post_stream(
        self,
        path: str,
        payload: Dict[str, Any],
        timeout: TimeoutType = None,
    ) -> requests.Response:
        # IMPORTANT:
        # For SSE streaming, do not use a finite read timeout unless you are sure the server
        # will emit bytes within that window. Otherwise Requests raises ReadTimeout mid-stream.
        norm_timeout = _normalize_stream_timeout(timeout)

        r = requests.post(
            self._url(path),
            json=payload,
            headers={
                "Accept": "text/event-stream",
                "Content-Type": "application/json",
            },
            stream=True,
            timeout=norm_timeout,
        )
        r.raise_for_status()
        return r
