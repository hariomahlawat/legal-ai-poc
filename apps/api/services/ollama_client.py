from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, Iterator, List, Optional

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


def _mark_success() -> None:
    global _OLLAMA_HEALTHY, _OLLAMA_LAST_FAILURE
    _OLLAMA_HEALTHY = True
    _OLLAMA_LAST_FAILURE = None


def _should_short_circuit() -> bool:
    """Determine if we should skip calling Ollama due to recent failures."""
    if _OLLAMA_HEALTHY is False and _OLLAMA_LAST_FAILURE is not None:
        elapsed = time.monotonic() - _OLLAMA_LAST_FAILURE
        return elapsed < OLLAMA_FAILURE_COOLDOWN_SECS
    return False


def _merge_options(options: Dict[str, Any] | None) -> Dict[str, Any]:
    final_options: Dict[str, Any] = {
        "num_predict": OLLAMA_NUM_PREDICT,
        "temperature": OLLAMA_TEMPERATURE,
    }
    if options:
        final_options.update(options)
    return final_options


# ============================
# Public API
# ============================
def ollama_chat(model: str, messages: List[Dict[str, Any]], options: Dict[str, Any], request_id: str) -> str:
    """Non-streaming call. This blocks until Ollama finishes the full response."""

    if _should_short_circuit():
        raise OllamaConnectionError(
            "Recent Ollama failures detected; skipping request until cooldown expires."
        )

    final_options = _merge_options(options)

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
    except requests.Timeout as exc:  # pragma: no cover
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
    except requests.ConnectionError as exc:  # pragma: no cover
        _mark_failure()
        logger.error(
            "ollama_connection_error",
            extra={"request_id": request_id, "model": model, "prompt_chars": prompt_chars},
        )
        raise OllamaConnectionError(f"Unable to reach Ollama at {OLLAMA_URL}") from exc
    except requests.RequestException as exc:  # pragma: no cover
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
    _mark_success()

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


def ollama_chat_stream(
    model: str,
    messages: List[Dict[str, Any]],
    options: Dict[str, Any],
    request_id: str,
) -> Iterator[str]:
    """
    Streaming call to Ollama.

    Why this fixes the timeout:
    - Non-streaming waits for the full response body to complete.
    - Streaming reads line-delimited JSON events incrementally and yields content chunks.
    - This keeps the SSE endpoint responsive and avoids waiting for full completion.

    Notes:
    - We keep connect timeout.
    - For streaming, we enforce a higher read timeout than the non-streaming path,
      because time-to-first-token can exceed 180s on large prompts with CPU-only inference.
    """

    if _should_short_circuit():
        raise OllamaConnectionError(
            "Recent Ollama failures detected; skipping request until cooldown expires."
        )

    final_options = _merge_options(options)
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "options": final_options,
    }

    start = time.monotonic()
    prompt_chars = _prompt_char_length(messages)

    # Socket read timeout (waiting for bytes). This is not the total generation time.
    stream_read_timeout = max(float(OLLAMA_READ_TIMEOUT_SECS), 600.0)

    try:
        with requests.post(
            f"{OLLAMA_URL}/api/chat",
            json=payload,
            stream=True,
            timeout=(OLLAMA_CONNECT_TIMEOUT_SECS, stream_read_timeout),
        ) as resp:
            resp.raise_for_status()

            for raw in resp.iter_lines(decode_unicode=True):
                if not raw:
                    continue

                try:
                    obj = json.loads(raw)
                except Exception:
                    continue

                if isinstance(obj, dict) and obj.get("error"):
                    raise OllamaResponseError(str(obj.get("error")))

                chunk = ""
                if isinstance(obj, dict):
                    msg = obj.get("message") or {}
                    if isinstance(msg, dict):
                        chunk = (msg.get("content") or "")
                    if not chunk:
                        chunk = (obj.get("response") or "")

                if chunk:
                    yield chunk

                if isinstance(obj, dict) and obj.get("done") is True:
                    break

        _mark_success()
        elapsed = time.monotonic() - start
        logger.info(
            "ollama_chat_stream",
            extra={
                "request_id": request_id,
                "model": model,
                "prompt_chars": prompt_chars,
                "num_predict": final_options.get("num_predict"),
                "temperature": final_options.get("temperature"),
                "elapsed_secs": round(elapsed, 3),
                "timeout_read_secs": stream_read_timeout,
            },
        )

    except requests.Timeout as exc:  # pragma: no cover
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
                "timeout_read_secs": stream_read_timeout,
                "stream": True,
            },
        )
        raise OllamaTimeoutError(f"Ollama streaming request timed out after {stream_read_timeout}s") from exc
    except requests.ConnectionError as exc:  # pragma: no cover
        _mark_failure()
        logger.error(
            "ollama_connection_error",
            extra={"request_id": request_id, "model": model, "prompt_chars": prompt_chars, "stream": True},
        )
        raise OllamaConnectionError(f"Unable to reach Ollama at {OLLAMA_URL}") from exc
    except requests.RequestException as exc:  # pragma: no cover
        _mark_failure()
        logger.error(
            "ollama_response_error",
            extra={
                "request_id": request_id,
                "model": model,
                "prompt_chars": prompt_chars,
                "status_code": getattr(exc.response, "status_code", None),
                "stream": True,
            },
        )
        raise OllamaResponseError(f"Ollama streaming request failed: {exc}") from exc
