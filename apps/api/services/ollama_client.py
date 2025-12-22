from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import requests

from apps.api.config import (
    OLLAMA_CONNECT_TIMEOUT_SECS,
    OLLAMA_FAILURE_COOLDOWN_SECS,
    OLLAMA_NUM_PREDICT,
    OLLAMA_READ_TIMEOUT_SECS,
    OLLAMA_TEMPERATURE,
    OLLAMA_URL,
)

logger = logging.getLogger(__name__)


# ============================
# Circuit-breaker state
# ============================
_OLLAMA_HEALTHY: Optional[bool] = None
_OLLAMA_LAST_FAILURE: Optional[float] = None


# ============================
# Exceptions
# ============================
class OllamaError(RuntimeError):
    """Base exception for Ollama client errors."""


class OllamaTimeoutError(OllamaError):
    """Raised when Ollama request exceeds configured timeout."""


class OllamaConnectionError(OllamaError):
    """Raised when Ollama connection cannot be established."""


class OllamaResponseError(OllamaError):
    """Raised when Ollama returns a non-successful response."""


# ============================
# Client helpers
# ============================
def _prompt_char_length(messages: List[Dict[str, Any]]) -> int:
    return sum(len((m.get("content") or "")) for m in messages)


def _mark_failure() -> None:
    """Record a connectivity failure to activate the cool-down window."""

    global _OLLAMA_HEALTHY, _OLLAMA_LAST_FAILURE

    _OLLAMA_HEALTHY = False
    _OLLAMA_LAST_FAILURE = time.monotonic()


def _should_short_circuit() -> bool:
    """Determine if we should skip calling Ollama due to recent failures."""

    if _OLLAMA_HEALTHY is False and _OLLAMA_LAST_FAILURE is not None:
        elapsed = time.monotonic() - _OLLAMA_LAST_FAILURE
        return elapsed < OLLAMA_FAILURE_COOLDOWN_SECS
    return False


# ============================
# Public API
# ============================
def ollama_chat(model: str, messages: List[Dict[str, Any]], options: Dict[str, Any], request_id: str) -> str:
    """Call Ollama chat endpoint with consistent timeouts and logging."""

    if _should_short_circuit():
        raise OllamaConnectionError(
            "Recent Ollama failures detected; skipping request until cooldown expires."
        )

    final_options = {
        "num_predict": OLLAMA_NUM_PREDICT,
        "temperature": OLLAMA_TEMPERATURE,
    }
    if options:
        final_options.update(options)

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": final_options,
    }

    start = time.monotonic()
    prompt_chars = _prompt_char_length(messages)

    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json=payload,
            timeout=(OLLAMA_CONNECT_TIMEOUT_SECS, OLLAMA_READ_TIMEOUT_SECS),
        )
        resp.raise_for_status()
    except requests.Timeout as exc:  # pragma: no cover - network
        elapsed = time.monotonic() - start
        _mark_failure()
        logger.warning(
            "ollama_timeout",
            extra={
                "request_id": request_id,
                "model": model,
                "prompt_chars": prompt_chars,
                "num_predict": final_options.get("num_predict"),
                "temperature": final_options.get("temperature"),
                "elapsed_secs": round(elapsed, 3),
                "timeout_read_secs": OLLAMA_READ_TIMEOUT_SECS,
            },
        )
        raise OllamaTimeoutError(f"Ollama request timed out after {OLLAMA_READ_TIMEOUT_SECS}s") from exc
    except requests.ConnectionError as exc:  # pragma: no cover - network
        _mark_failure()
        logger.error(
            "ollama_connection_error",
            extra={"request_id": request_id, "model": model, "prompt_chars": prompt_chars},
        )
        raise OllamaConnectionError(f"Unable to reach Ollama at {OLLAMA_URL}") from exc
    except requests.RequestException as exc:  # pragma: no cover - network
        _mark_failure()
        logger.error(
            "ollama_response_error",
            extra={
                "request_id": request_id,
                "model": model,
                "prompt_chars": prompt_chars,
                "status_code": getattr(exc.response, "status_code", None),
            },
        )
        raise OllamaResponseError(f"Ollama request failed: {exc}") from exc

    elapsed = time.monotonic() - start
    data = resp.json()
    content = (data.get("message", {}) or {}).get("content", "").strip()

    # Mark healthy on successful completion to re-enable calls after a failure
    global _OLLAMA_HEALTHY, _OLLAMA_LAST_FAILURE
    _OLLAMA_HEALTHY = True
    _OLLAMA_LAST_FAILURE = None

    logger.info(
        "ollama_chat",
        extra={
            "request_id": request_id,
            "model": model,
            "prompt_chars": prompt_chars,
            "num_predict": final_options.get("num_predict"),
            "temperature": final_options.get("temperature"),
            "elapsed_secs": round(elapsed, 3),
            "timeout_read_secs": OLLAMA_READ_TIMEOUT_SECS,
        },
    )

    return content

