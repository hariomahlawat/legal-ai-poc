from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, Iterator, List

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

# Simple circuit-breaker state (process-level)
_OLLAMA_HEALTHY = True
_LAST_FAILURE_TS = 0.0


class OllamaError(RuntimeError):
    pass


class OllamaConnectionError(OllamaError):
    pass


class OllamaTimeoutError(OllamaError):
    pass


class OllamaResponseError(OllamaError):
    pass


def _should_short_circuit() -> bool:
    global _OLLAMA_HEALTHY, _LAST_FAILURE_TS
    if _OLLAMA_HEALTHY:
        return False
    # cooldown window
    return (time.time() - _LAST_FAILURE_TS) < float(OLLAMA_FAILURE_COOLDOWN_SECS)


def _mark_failure() -> None:
    global _OLLAMA_HEALTHY, _LAST_FAILURE_TS
    _OLLAMA_HEALTHY = False
    _LAST_FAILURE_TS = time.time()


def _mark_success() -> None:
    global _OLLAMA_HEALTHY
    _OLLAMA_HEALTHY = True


def ollama_chat(model: str, messages: List[Dict[str, Any]], options: Dict[str, Any], request_id: str) -> str:
    """
    Non-streaming chat call (kept for other internal calls if needed).
    """
    if _should_short_circuit():
        raise OllamaConnectionError("Recent Ollama failures detected; skipping request until cooldown expires.")
   
    # ----------------------------
    # Payload
    # ----------------------------
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "num_predict": options.get("num_predict"),
            "temperature": options.get("temperature"),
        },
    }

    # ----------------------------
    # Defaults
    # ----------------------------
    # Fill defaults if missing
    if payload["options"]["num_predict"] is None:
        payload["options"]["num_predict"] = OLLAMA_NUM_PREDICT
    if payload["options"]["temperature"] is None:
        payload["options"]["temperature"] = OLLAMA_TEMPERATURE

    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json=payload,
            timeout=(float(OLLAMA_CONNECT_TIMEOUT_SECS), float(OLLAMA_READ_TIMEOUT_SECS)),
        )
        resp.raise_for_status()
        data = resp.json()
        content = (data.get("message") or {}).get("content", "")
        _mark_success()
        return content
    except requests.exceptions.ConnectTimeout as exc:
        _mark_failure()
        logger.warning("ollama_chat.connect_timeout request_id=%s error=%s", request_id, exc)
        raise OllamaTimeoutError(f"Connect timeout to Ollama: {exc}") from exc
    except requests.exceptions.ReadTimeout as exc:
        _mark_failure()
        logger.warning("ollama_chat.read_timeout request_id=%s error=%s", request_id, exc)
        raise OllamaTimeoutError(f"Read timeout from Ollama: {exc}") from exc
    except requests.exceptions.RequestException as exc:
        _mark_failure()
        logger.warning("ollama_chat.request_error request_id=%s error=%s", request_id, exc)
        raise OllamaConnectionError(f"Ollama request failed: {exc}") from exc
    except ValueError as exc:
        _mark_failure()
        logger.warning("ollama_chat.invalid_json request_id=%s error=%s", request_id, exc)
        raise OllamaResponseError(f"Invalid JSON from Ollama: {exc}") from exc


def ollama_chat_stream(
    model: str,
    messages: List[Dict[str, Any]],
    options: Dict[str, Any],
    request_id: str,
) -> Iterator[str]:
    """
    Streaming chat call.

    Key change versus non-streaming:
    - We set stream=True for Ollama.
    - We DO NOT apply a per-read timeout. With streaming, long generations can legitimately exceed
      OLLAMA_READ_TIMEOUT_SECS, and we want token flow to keep the connection alive.
    """
    if _should_short_circuit():
        raise OllamaConnectionError("Recent Ollama failures detected; skipping request until cooldown expires.")

    # ----------------------------
    # Payload
    # ----------------------------
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "options": {
            "num_predict": options.get("num_predict"),
            "temperature": options.get("temperature"),
        },
    }

    # ----------------------------
    # Defaults
    # ----------------------------
    # Fill defaults if missing
    if payload["options"]["num_predict"] is None:
        payload["options"]["num_predict"] = OLLAMA_NUM_PREDICT
    if payload["options"]["temperature"] is None:
        payload["options"]["temperature"] = OLLAMA_TEMPERATURE

    url = f"{OLLAMA_URL}/api/chat"

    start = time.time()
    logger.info("ollama_chat_stream.start request_id=%s model=%s url=%s", request_id, model, url)

    try:
        # IMPORTANT: no read timeout for streaming
        with requests.post(
            url,
            json=payload,
            stream=True,
            timeout=(float(OLLAMA_CONNECT_TIMEOUT_SECS), None),
        ) as resp:
            resp.raise_for_status()

            # Iterate server-sent JSON lines from Ollama
            for raw_line in resp.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue

                try:
                    data = json.loads(raw_line)
                except Exception:
                    # Ignore malformed lines but keep stream alive
                    continue

                if data.get("error"):
                    raise OllamaResponseError(str(data.get("error")))

                # Ollama chat streaming emits {"message": {"content": "..."}, "done": false}
                chunk = ""
                msg = data.get("message") or {}
                if isinstance(msg, dict):
                    chunk = msg.get("content") or ""

                # Fallback for /api/generate-style payloads if any
                if not chunk:
                    chunk = data.get("response") or ""

                if chunk:
                    yield chunk

                if data.get("done") is True:
                    break

        _mark_success()
        logger.info(
            "ollama_chat_stream.end request_id=%s model=%s elapsed_ms=%.2f",
            request_id,
            model,
            (time.time() - start) * 1000.0,
        )

    except requests.exceptions.ConnectTimeout as exc:
        _mark_failure()
        logger.warning("ollama_chat_stream.connect_timeout request_id=%s error=%s", request_id, exc)
        raise OllamaTimeoutError(f"Connect timeout to Ollama: {exc}") from exc
    except requests.exceptions.ReadTimeout as exc:
        _mark_failure()
        logger.warning("ollama_chat_stream.read_timeout request_id=%s error=%s", request_id, exc)
        raise OllamaTimeoutError(f"Read timeout from Ollama: {exc}") from exc
    except requests.exceptions.RequestException as exc:
        _mark_failure()
        logger.warning("ollama_chat_stream.request_error request_id=%s error=%s", request_id, exc)
        raise OllamaConnectionError(f"Ollama request failed: {exc}") from exc
