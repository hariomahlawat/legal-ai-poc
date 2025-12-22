"""Build system-help search indexes from runbook docs."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# ============================
# Constants and Paths
# ============================
REPO_ROOT = Path(__file__).resolve().parents[1]
RUNBOOK_DIR = REPO_ROOT / "data" / "runbook"
README_PATH = REPO_ROOT / "README.md"
OUTPUT_DIR = REPO_ROOT / "data" / "index_repo"
META_JSONL = OUTPUT_DIR / "meta.jsonl"
FAISS_INDEX = OUTPUT_DIR / "faiss.index"
BM25_PKL = OUTPUT_DIR / "bm25.pkl"
BM25_META = OUTPUT_DIR / "bm25_meta.jsonl"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Ensure offline-friendly defaults where possible
os.environ.setdefault("HF_HUB_OFFLINE", "1")

# ============================
# Data Structures
# ============================
@dataclass
class Chunk:
    chunk_id: str
    source_file: str
    heading_path: str
    text: str


# ============================
# Text Utilities
# ============================
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[-/][A-Za-z0-9]+)*")


def bm25_tokenize(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


def _normalize_heading(raw: str) -> str:
    content = raw.lstrip("#").strip()
    return content if content else raw.strip()


def _chunk_with_overlap(text: str, max_chars: int = 900, overlap: int = 240) -> List[str]:
    safe_text = text.strip()
    if not safe_text:
        return []
    if len(safe_text) <= max_chars:
        return [safe_text]

    chunks: List[str] = []
    start = 0
    step = max_chars - overlap
    while start < len(safe_text):
        end = min(len(safe_text), start + max_chars)
        chunks.append(safe_text[start:end].strip())
        if end == len(safe_text):
            break
        start += step
    return chunks


# ============================
# Markdown Processing
# ============================

def _iter_markdown_blocks(lines: Sequence[str], source_file: str) -> Iterable[Chunk]:
    heading_stack: List[str] = []
    paragraph: List[str] = []
    code_block: List[str] = []
    in_code = False
    chunk_counter = 0

    def current_heading() -> str:
        return " > ".join(heading_stack).strip()

    def flush_paragraph() -> None:
        nonlocal chunk_counter
        if not paragraph:
            return
        text = "\n".join(paragraph).strip()
        paragraph.clear()
        for part in _chunk_with_overlap(text):
            chunk_counter += 1
            base = f"{source_file}|{current_heading()}|{part[:200]}|{chunk_counter}"
            cid = hashlib.sha1(base.encode("utf-8")).hexdigest()[:12]
            yield_chunk = Chunk(
                chunk_id=f"{Path(source_file).stem}.{cid}",
                source_file=source_file,
                heading_path=current_heading(),
                text=part,
            )
            yield yield_chunk

    def flush_code() -> None:
        nonlocal chunk_counter
        if not code_block:
            return
        text = "\n".join(code_block).strip()
        code_block.clear()
        if not text:
            return
        chunk_counter += 1
        base = f"{source_file}|{current_heading()}|CODE|{text[:200]}|{chunk_counter}"
        cid = hashlib.sha1(base.encode("utf-8")).hexdigest()[:12]
        yield_chunk = Chunk(
            chunk_id=f"{Path(source_file).stem}.{cid}",
            source_file=source_file,
            heading_path=current_heading(),
            text=text,
        )
        yield yield_chunk

    for raw in lines:
        stripped = raw.strip()

        if stripped.startswith("```"):
            if in_code:
                code_block.append(raw)
                yield from flush_code()
                in_code = False
            else:
                yield from flush_paragraph()
                in_code = True
                code_block.append(raw)
            continue

        if in_code:
            code_block.append(raw)
            continue

        if stripped.startswith("#"):
            yield from flush_paragraph()
            level = len(stripped) - len(stripped.lstrip("#"))
            heading = _normalize_heading(stripped)
            while len(heading_stack) >= level:
                heading_stack.pop()
            heading_stack.append(heading)
            continue

        if not stripped:
            yield from flush_paragraph()
            continue

        paragraph.append(raw)

    if in_code:
        yield from flush_code()
    yield from flush_paragraph()


def chunk_markdown(text: str, source_file: str) -> List[Chunk]:
    lines = text.splitlines()
    return list(_iter_markdown_blocks(lines, source_file))


# ============================
# Ingestion Pipeline
# ============================

def collect_markdown_files() -> List[Path]:
    files = sorted(RUNBOOK_DIR.glob("**/*.md"))
    if README_PATH.exists():
        files.append(README_PATH)
    return files


def _fallback_embeddings(chunks: List[Chunk], dim: int = 384) -> np.ndarray:
    vectors: List[np.ndarray] = []
    for ch in chunks:
        h = int(hashlib.sha1((ch.text or "").encode("utf-8")).hexdigest(), 16)
        rng = np.random.default_rng(h % (2**32))
        vec = rng.normal(size=dim)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        vectors.append(vec.astype("float32"))
    return np.vstack(vectors) if vectors else np.zeros((0, dim), dtype="float32")


def build_indexes(chunks: List[Chunk]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Write metadata ----
    with META_JSONL.open("w", encoding="utf-8") as f:
        for i, ch in enumerate(chunks):
            f.write(
                json.dumps(
                    {
                        "i": i,
                        "chunk_id": ch.chunk_id,
                        "source_file": ch.source_file,
                        "heading_path": ch.heading_path,
                        "text": ch.text,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    # ---- Dense embeddings / FAISS ----
    texts = [c.text for c in chunks]
    use_transformers_default = "0" if os.getenv("HF_HUB_OFFLINE", "0") == "1" else "1"
    use_transformers = os.getenv("USE_SENTENCE_TRANSFORMERS", use_transformers_default)
    embeddings = None
    if use_transformers.lower() not in ("0", "false", "no"):
        try:
            model = SentenceTransformer(MODEL_NAME)
            embeddings = model.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
            embeddings = np.asarray(embeddings, dtype="float32")
        except Exception as exc:
            print(f"Falling back to hashed embeddings due to error: {exc}")

    if embeddings is None:
        embeddings = _fallback_embeddings(chunks)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    faiss.write_index(index, str(FAISS_INDEX))

    # ---- BM25 ----
    tokenized = [bm25_tokenize(c.text) for c in chunks]
    bm25 = BM25Okapi(tokenized)
    with BM25_PKL.open("wb") as f:
        pickle.dump(bm25, f)

    with BM25_META.open("w", encoding="utf-8") as f:
        for i, ch in enumerate(chunks):
            f.write(
                json.dumps(
                    {
                        "i": i,
                        "chunk_id": ch.chunk_id,
                        "source_file": ch.source_file,
                        "heading_path": ch.heading_path,
                        "text": ch.text,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def main() -> None:
    files = collect_markdown_files()
    if not files:
        raise SystemExit(f"No markdown files found under {RUNBOOK_DIR}")

    all_chunks: List[Chunk] = []
    for path in files:
        text = path.read_text(encoding="utf-8", errors="ignore")
        chunks = chunk_markdown(text, str(path.relative_to(REPO_ROOT)))
        all_chunks.extend(chunks)
        print(f"{path}: {len(chunks)} chunks")

    if not all_chunks:
        raise SystemExit("No chunks generated; aborting.")

    build_indexes(all_chunks)
    print(f"Wrote {len(all_chunks)} chunks to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
