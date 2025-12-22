from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import requests
from requests.exceptions import ConnectTimeout, ReadTimeout, Timeout

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


def _short_circuit_elapsed_secs() -> Optional[float]:
    """Return seconds since last failure if currently unhealthy, else None."""
    if _OLLAMA_HEALTHY is False and _OLLAMA_LAST_FAILURE is not None:
        return time.monotonic() - _OLLAMA_LAST_FAILURE
    return None


def _should_short_circuit() -> bool:
    """Determine if we should skip calling Ollama due to recent failures."""
    elapsed = _short_circuit_elapsed_secs()
    if elapsed is None:
        return False
    return elapsed < OLLAMA_FAILURE_COOLDOWN_SECS


# ============================
# Public API
# ============================
def ollama_chat(model: str, messages: List[Dict[str, Any]], options: Dict[str, Any], request_id: str) -> str:
    """Call Ollama chat endpoint with consistent timeouts and logging."""

    # Circuit breaker: if we recently failed, skip hitting Ollama until cooldown ends.
    if _should_short_circuit():
        elapsed = _short_circuit_elapsed_secs() or 0.0
        msg = (
            "ollama_short_circuit "
            f"request_id={request_id} model={model} "
            f"elapsed_since_failure_secs={round(elapsed, 3)} cooldown_secs={OLLAMA_FAILURE_COOLDOWN_SECS}"
        )
        # Log as a plain message (so it shows up even with basic formatters),
        # and also attach structured fields via "extra".
        logger.warning(
            msg,
            extra={
                "request_id": request_id,
                "model": model,
                "elapsed_since_failure_secs": round(elapsed, 3),
                "cooldown_secs": OLLAMA_FAILURE_COOLDOWN_SECS,
            },
        )
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

    except Timeout as exc:  # pragma: no cover - network
        elapsed = time.monotonic() - start
        _mark_failure()

        timeout_kind = "timeout"
        if isinstance(exc, ConnectTimeout):
            timeout_kind = "connect_timeout"
        elif isinstance(exc, ReadTimeout):
            timeout_kind = "read_timeout"

        msg = (
            "ollama_timeout "
            f"kind={timeout_kind} request_id={request_id} model={model} "
            f"elapsed_secs={round(elapsed, 3)} "
            f"connect_timeout_secs={OLLAMA_CONNECT_TIMEOUT_SECS} read_timeout_secs={OLLAMA_READ_TIMEOUT_SECS} "
            f"url={OLLAMA_URL} exc={exc!r}"
        )

        logger.warning(
            msg,
            extra={
                "request_id": request_id,
                "model": model,
                "prompt_chars": prompt_chars,
                "num_predict": final_options.get("num_predict"),
                "temperature": final_options.get("temperature"),
                "elapsed_secs": round(elapsed, 3),
                "timeout_kind": timeout_kind,
                "timeout_connect_secs": OLLAMA_CONNECT_TIMEOUT_SECS,
                "timeout_read_secs": OLLAMA_READ_TIMEOUT_SECS,
                "url": OLLAMA_URL,
                "exception": repr(exc),
            },
        )

        raise OllamaTimeoutError(
            f"Ollama {timeout_kind} after {round(elapsed, 3)}s "
            f"(connect={OLLAMA_CONNECT_TIMEOUT_SECS}s, read={OLLAMA_READ_TIMEOUT_SECS}s)"
        ) from exc

    except requests.ConnectionError as exc:  # pragma: no cover - network
        elapsed = time.monotonic() - start
        _mark_failure()

        msg = (
            "ollama_connection_error "
            f"request_id={request_id} model={model} elapsed_secs={round(elapsed, 3)} "
            f"url={OLLAMA_URL} exc={exc!r}"
        )

        logger.error(
            msg,
            extra={
                "request_id": request_id,
                "model": model,
                "prompt_chars": prompt_chars,
                "elapsed_secs": round(elapsed, 3),
                "url": OLLAMA_URL,
                "exception": repr(exc),
            },
        )
        raise OllamaConnectionError(f"Unable to reach Ollama at {OLLAMA_URL}") from exc

    except requests.RequestException as exc:  # pragma: no cover - network
        elapsed = time.monotonic() - start
        _mark_failure()

        status_code = getattr(getattr(exc, "response", None), "status_code", None)
        msg = (
            "ollama_response_error "
            f"request_id={request_id} model={model} elapsed_secs={round(elapsed, 3)} "
            f"status_code={status_code} url={OLLAMA_URL} exc={exc!r}"
        )

        logger.error(
            msg,
            extra={
                "request_id": request_id,
                "model": model,
                "prompt_chars": prompt_chars,
                "elapsed_secs": round(elapsed, 3),
                "status_code": status_code,
                "url": OLLAMA_URL,
                "exception": repr(exc),
            },
        )
        raise OllamaResponseError(f"Ollama request failed: {exc}") from exc

    # Success path
    elapsed = time.monotonic() - start
    data = resp.json()
    content = (data.get("message", {}) or {}).get("content", "").strip()

    # Mark healthy on successful completion to re-enable calls after a failure
    global _OLLAMA_HEALTHY, _OLLAMA_LAST_FAILURE
    _OLLAMA_HEALTHY = True
    _OLLAMA_LAST_FAILURE = None

    msg = (
        "ollama_chat_ok "
        f"request_id={request_id} model={model} elapsed_secs={round(elapsed, 3)} "
        f"prompt_chars={prompt_chars} num_predict={final_options.get('num_predict')} "
        f"temperature={final_options.get('temperature')}"
    )

    logger.info(
        msg,
        extra={
            "request_id": request_id,
            "model": model,
            "prompt_chars": prompt_chars,
            "num_predict": final_options.get("num_predict"),
            "temperature": final_options.get("temperature"),
            "elapsed_secs": round(elapsed, 3),
            "timeout_connect_secs": OLLAMA_CONNECT_TIMEOUT_SECS,
            "timeout_read_secs": OLLAMA_READ_TIMEOUT_SECS,
            "url": OLLAMA_URL,
        },
    )

    return content
