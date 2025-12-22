from __future__ import annotations

# ----------------------------
# Cross-Encoder Reranker
# ----------------------------
import logging
import os
import time
from typing import Any, List

_RERANKER = None
_RERANKER_LOAD_ERROR = None

logger = logging.getLogger(__name__)


def _get_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def _get_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None:
        return default
    try:
        return int(v)
    except ValueError:
        return default


RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANK_TOP_N = _get_int("RERANK_TOP_N", 60)
RERANK_ENABLED = _get_bool("RERANK_ENABLED", True)


def _load_model():
    global _RERANKER, _RERANKER_LOAD_ERROR
    if _RERANKER is not None or _RERANKER_LOAD_ERROR is not None:
        return _RERANKER

    try:
        from sentence_transformers import CrossEncoder  # type: ignore

        start = time.perf_counter()
        _RERANKER = CrossEncoder(RERANKER_MODEL)
        latency_ms = (time.perf_counter() - start) * 1000.0
        logger.info("rerank.model_loaded model=%s latency_ms=%.2f", RERANKER_MODEL, latency_ms)
    except Exception as exc:  # pragma: no cover - defensive guard
        _RERANKER_LOAD_ERROR = exc
        logger.warning("rerank.model_load_failed model=%s error=%s", RERANKER_MODEL, exc)
        _RERANKER = None

    return _RERANKER


def _build_pair(question: str, candidate: dict[str, Any]) -> List[str]:
    heading_path = candidate.get("heading_path") or ""
    text = candidate.get("text") or ""
    doc = f"{heading_path}\n{text}".strip()
    return [question, doc[:1800]]


def rerank_candidates(question: str, candidates: List[dict[str, Any]], top_k: int) -> List[dict[str, Any]]:
    """
    Rerank candidates using a cross-encoder.
    Inputs:
      - candidates must contain: text, heading_path, retrieval_score
    Output:
      - candidates sorted by rerank_score desc; return top_k
      - add field: rerank_score (float)
    Behaviour:
      - If disabled or model fails to load, return original top_k and set rerank_score=None
    """

    if not candidates:
        return []

    if not RERANK_ENABLED:
        out = []
        for cand in candidates[:top_k]:
            cand_copy = dict(cand)
            cand_copy["rerank_score"] = None
            out.append(cand_copy)
        return out

    model = _load_model()
    if model is None:
        out = []
        for cand in candidates[:top_k]:
            cand_copy = dict(cand)
            cand_copy["rerank_score"] = None
            out.append(cand_copy)
        return out

    pairs = [_build_pair(question, cand) for cand in candidates]
    try:
        scores = model.predict(pairs)
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.warning("rerank.predict_failed error=%s", exc)
        out = []
        for cand in candidates[:top_k]:
            cand_copy = dict(cand)
            cand_copy["rerank_score"] = None
            out.append(cand_copy)
        return out

    reranked: List[dict[str, Any]] = []
    for cand, score in zip(candidates, scores):
        cand_copy = dict(cand)
        cand_copy["rerank_score"] = float(score)
        reranked.append(cand_copy)

    reranked.sort(key=lambda c: c.get("rerank_score", 0.0), reverse=True)
    return reranked[:top_k]
