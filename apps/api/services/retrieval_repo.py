from __future__ import annotations

import hashlib
import json
import os
import pickle
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ============================
# Data Structures and State
# ============================
@dataclass(frozen=True)
class _Chunk:
    i: int
    chunk_id: str
    source_file: str
    heading_path: str
    text: str


class _RepoRetrievalState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.loaded = False
        self.bm25 = None
        self.bm25_meta: List[_Chunk] = []
        self.faiss_index = None
        self.faiss_meta: List[_Chunk] = []
        self.embed_model = None
        self.load_errors: List[str] = []


_STATE = _RepoRetrievalState()


# ============================
# Paths and Tokenization
# ============================
_REPO_ROOT = Path(__file__).resolve().parents[3]
_INDEX_DIR = _REPO_ROOT / "data" / "index_repo"
_META_JSONL = _INDEX_DIR / "meta.jsonl"
_FAISS_INDEX = _INDEX_DIR / "faiss.index"
_BM25_PKL = _INDEX_DIR / "bm25.pkl"
_BM25_META = _INDEX_DIR / "bm25_meta.jsonl"

# Default to offline-friendly behavior unless explicitly overridden.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

def _bm25_tokenize(text: str) -> List[str]:
    import re

    token_re = re.compile(r"[A-Za-z0-9]+(?:[-/][A-Za-z0-9]+)*")
    return [t.lower() for t in token_re.findall(text or "")]


# ============================
# Loading Helpers
# ============================

def _load_meta_jsonl(path: Path) -> List[_Chunk]:
    chunks: List[_Chunk] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            chunks.append(
                _Chunk(
                    i=int(obj["i"]),
                    chunk_id=str(obj["chunk_id"]),
                    source_file=str(obj["source_file"]),
                    heading_path=str(obj.get("heading_path", "")),
                    text=str(obj.get("text", "")),
                )
            )
    return chunks


def _ensure_loaded() -> None:
    if _STATE.loaded:
        return
    with _STATE._lock:
        if _STATE.loaded:
            return

        try:
            with _BM25_PKL.open("rb") as f:
                _STATE.bm25 = pickle.load(f)
            _STATE.bm25_meta = _load_meta_jsonl(_BM25_META)
        except Exception as exc:  # pragma: no cover - defensive
            _STATE.load_errors.append(f"BM25 load failed: {exc}")
            _STATE.bm25 = None
            _STATE.bm25_meta = []

        try:
            import faiss  # type: ignore
            from sentence_transformers import SentenceTransformer  # type: ignore

            _STATE.faiss_index = faiss.read_index(str(_FAISS_INDEX))
            _STATE.faiss_meta = _load_meta_jsonl(_META_JSONL)

            use_transformers = os.getenv("USE_SENTENCE_TRANSFORMERS", "0")
            if use_transformers.lower() not in ("0", "false", "no"):
                _STATE.embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            else:
                _STATE.embed_model = None
        except Exception as exc:  # pragma: no cover - defensive
            _STATE.load_errors.append(f"FAISS load failed: {exc}")
            _STATE.faiss_index = None
            _STATE.faiss_meta = []
            _STATE.embed_model = None

        _STATE.loaded = True


# ============================
# Scoring Utilities
# ============================

def _normalize_scores(pairs: List[Tuple[int, float]]) -> Dict[int, float]:
    if not pairs:
        return {}
    scores = [s for _, s in pairs]
    hi = max(scores)
    lo = min(scores)
    if hi == lo:
        return {i: 1.0 for i, _ in pairs}
    return {i: (s - lo) / (hi - lo) for i, s in pairs}


def _short_heading(heading_path: str) -> str:
    parts = [p.strip() for p in heading_path.split(">") if p.strip()]
    return parts[-1] if parts else heading_path.strip()


def _build_citation_id(source_file: str, chunk_id: str) -> str:
    base = f"{source_file}:{chunk_id}"
    digest = hashlib.sha1(base.encode("utf-8")).hexdigest()[:10]
    return f"SYS-{digest}"


def _hashed_embedding(text: str, dim: int) -> np.ndarray:
    seed = int(hashlib.sha1((text or "").encode("utf-8")).hexdigest(), 16) % (2**32)
    rng = np.random.default_rng(seed)
    vec = rng.normal(size=dim)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec.astype("float32")


# ============================
# Retrieval Entry Point
# ============================

def retrieve_repo_citations(question: str, top_k: int = 6) -> List[Dict[str, Any]]:
    _ensure_loaded()

    if not (_STATE.bm25_meta or _STATE.faiss_meta):
        return []

    cleaned_q = (question or "").strip()
    bm25_candidates: List[Tuple[int, float]] = []
    faiss_candidates: List[Tuple[int, float]] = []

    if _STATE.bm25 is not None:
        tokens = _bm25_tokenize(cleaned_q)
        scores = _STATE.bm25.get_scores(tokens)
        bm25_candidates = sorted([(i, float(s)) for i, s in enumerate(scores)], key=lambda x: x[1], reverse=True)[
            :30
        ]

    if _STATE.faiss_index is not None:
        dim = _STATE.faiss_index.d
        if _STATE.embed_model is not None:
            query_vec = _STATE.embed_model.encode([cleaned_q], normalize_embeddings=True)
            query_vec = np.asarray(query_vec, dtype="float32")
        else:
            query_vec = np.asarray([_hashed_embedding(cleaned_q, dim)], dtype="float32")

        dists, idxs = _STATE.faiss_index.search(query_vec, 30)
        faiss_candidates = [(int(i), float(score)) for i, score in zip(idxs[0], dists[0]) if i >= 0]

    bm25_norm = _normalize_scores(bm25_candidates)
    faiss_norm = _normalize_scores(faiss_candidates)

    fused: Dict[int, float] = {}
    for i, s in bm25_norm.items():
        fused[i] = max(fused.get(i, 0.0), s)
    for i, s in faiss_norm.items():
        fused[i] = max(fused.get(i, 0.0), s)

    ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)
    top_indices = [i for i, _ in ranked[: max(top_k, 1)]]

    out: List[Dict[str, Any]] = []
    meta_list = _STATE.bm25_meta if _STATE.bm25_meta else _STATE.faiss_meta
    for i in top_indices:
        if i >= len(meta_list):
            continue
        ch = meta_list[i]
        before = ""
        after = ""
        if i - 1 >= 0 and i - 1 < len(meta_list):
            prev = meta_list[i - 1]
            if prev and prev.source_file == ch.source_file:
                before = prev.text
        if i + 1 < len(meta_list):
            nxt = meta_list[i + 1]
            if nxt.source_file == ch.source_file:
                after = nxt.text

        citation_id = _build_citation_id(ch.source_file, ch.chunk_id)
        heading = _short_heading(ch.heading_path)
        verbatim = (ch.text or "").strip()
        snippet = (verbatim.replace("\n", " ").strip())[:220]

        out.append(
            {
                "citation_id": citation_id,
                "chunk_id": ch.chunk_id,
                "source_file": ch.source_file,
                "heading_path": ch.heading_path,
                "text": ch.text,
                "score": fused.get(i, 0.0),
                "document": "SYS",
                "title": f"System Help | {heading}" if heading else "System Help",
                "heading": heading,
                "location": ch.heading_path,
                "verbatim": verbatim,
                "context_before": (before or "").strip(),
                "context_after": (after or "").strip(),
                "snippet": snippet,
            }
        )

    return out
