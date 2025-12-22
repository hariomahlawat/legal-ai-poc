from __future__ import annotations

import json
import hashlib
import pickle
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

# ----------------------------
# Environment setup
# ----------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.chunking_legal import chunk_legal_markdown

# ----------------------------
# Paths and constants
# ----------------------------
RAW_DIR = Path("data/raw")
INDEX_BASE_DIR = Path("data/index_legal")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MAX_CHARS = 1400
OVERLAP_CHARS = 250


# ----------------------------
# Data structures
# ----------------------------
@dataclass
class Chunk:
    chunk_id: str
    source_file: str
    heading_path: str
    text: str


# ----------------------------
# Helpers
# ----------------------------
def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def _safe_git_commit() -> str | None:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode("utf-8")
            .strip()
        )
    except Exception:
        return None


def _next_version_dir(base_dir: Path) -> Tuple[Path, str]:
    base_dir.mkdir(parents=True, exist_ok=True)
    today = datetime.utcnow().strftime("%Y%m%d")
    prefix = f"v{today}_"
    existing = [p.name for p in base_dir.iterdir() if p.is_dir() and p.name.startswith(prefix)]
    next_idx = 1
    if existing:
        try:
            indices = [int(name.split("_")[-1]) for name in existing]
            next_idx = max(indices) + 1
        except Exception:
            next_idx = len(existing) + 1
    version_name = f"{prefix}{next_idx}"
    version_dir = base_dir / version_name
    version_dir.mkdir(parents=True, exist_ok=True)
    return version_dir, version_name


_token_re = re.compile(r"[A-Za-z0-9]+(?:[-/][A-Za-z0-9]+)*")


def bm25_tokenize(text: str) -> List[str]:
    # Conservative tokenizer: keeps "court-martial", "rule-22", "section/123"
    return [t.lower() for t in _token_re.findall(text)]


# ----------------------------
# Chunk handling
# ----------------------------
def build_chunk_id(source_file: str, heading_path: str, text: str) -> str:
    base = f"{source_file}|{heading_path}|{text.strip()[:120]}"
    digest = hashlib.sha1(base.encode("utf-8")).hexdigest()[:12]
    return f"LGL-{digest}"


def build_chunks(md_text: str, source_file: str) -> List[Chunk]:
    pairs = chunk_legal_markdown(md_text, max_chars=MAX_CHARS, overlap_chars=OVERLAP_CHARS)
    chunks: List[Chunk] = []
    for heading_path, text in pairs:
        chunk_id = build_chunk_id(source_file, heading_path, text)
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                source_file=source_file,
                heading_path=heading_path,
                text=text.strip(),
            )
        )
    return chunks


# ----------------------------
# Main ingest flow
# ----------------------------
def main():
    version_dir, version_name = _next_version_dir(INDEX_BASE_DIR)
    manifest_path = version_dir / "manifest.json"
    meta_jsonl = version_dir / "meta.jsonl"
    faiss_index_path = version_dir / "faiss.index"
    bm25_path = version_dir / "bm25.pkl"

    md_files = sorted(RAW_DIR.glob("*.md"))
    if not md_files:
        raise SystemExit(f"No .md files found in {RAW_DIR.resolve()}")

    all_chunks: List[Chunk] = []
    manifest: Dict[str, Any] = {"files": []}

    for f in md_files:
        text = f.read_text(encoding="utf-8", errors="ignore")
        chunks = build_chunks(text, f.name)
        all_chunks.extend(chunks)
        manifest["files"].append({"name": f.name, "sha256": file_sha256(f), "chunks": len(chunks)})

    with meta_jsonl.open("w", encoding="utf-8") as out:
        for idx, c in enumerate(all_chunks):
            out.write(
                json.dumps(
                    {
                        "i": idx,
                        "chunk_id": c.chunk_id,
                        "source_file": c.source_file,
                        "heading_path": c.heading_path,
                        "text": c.text,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    # ---- FAISS embeddings ----
    model = SentenceTransformer(MODEL_NAME)
    texts = [c.text for c in all_chunks]
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.asarray(embeddings, dtype="float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    faiss.write_index(index, str(faiss_index_path))

    # ---- BM25 index ----
    tokenized_corpus = [bm25_tokenize(c.text) for c in all_chunks]
    bm25 = BM25Okapi(tokenized_corpus)

    with bm25_path.open("wb") as f:
        pickle.dump(bm25, f)

    manifest.update(
        {
            "version": version_name,
            "created_utc": datetime.utcnow().isoformat() + "Z",
            "max_chars": MAX_CHARS,
            "overlap_chars": OVERLAP_CHARS,
            "embedding_model": MODEL_NAME,
            "num_chunks": len(all_chunks),
        }
    )

    git_commit = _safe_git_commit()
    if git_commit:
        manifest["git_commit"] = git_commit

    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("OK")
    print(f"Chunks: {len(all_chunks)} -> {meta_jsonl}")
    print(f"FAISS index: {faiss_index_path}")
    print(f"BM25 index:  {bm25_path}")
    print(f"Manifest:    {manifest_path}")


if __name__ == "__main__":
    main()
