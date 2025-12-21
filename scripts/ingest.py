import json
import hashlib
import pickle
import re
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi


RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")
EMB_DIR = Path("data/embeddings")

CHUNKS_JSONL = OUT_DIR / "chunks.jsonl"
MANIFEST_JSON = OUT_DIR / "manifest.json"

FAISS_INDEX = EMB_DIR / "faiss.index"
FAISS_META = EMB_DIR / "faiss_meta.jsonl"

BM25_PKL = EMB_DIR / "bm25.pkl"
BM25_META = EMB_DIR / "bm25_meta.jsonl"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass
class Chunk:
    chunk_id: str
    source_file: str
    heading_path: str
    text: str


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def normalize_line(s: str) -> str:
    return " ".join(s.strip().split())


def is_heading(line: str) -> bool:
    u = line.strip()
    if not u:
        return False
    if u.startswith("#"):
        return True
    if u.upper() == u and len(u) <= 120 and any(k in u for k in ["CHAPTER", "SECTION", "PART"]):
        return True
    if (u.startswith("CHAPTER") or u.startswith("Chapter")) and len(u) <= 80:
        return True
    return False


def chunk_markdown(md_text: str, source_file: str) -> List[Chunk]:
    lines = md_text.splitlines()
    heading_stack: List[str] = []
    paragraphs: List[str] = []
    chunks: List[Chunk] = []

    def flush_paragraphs():
        nonlocal paragraphs, chunks
        if not paragraphs:
            return
        para = "\n".join(paragraphs).strip()
        paragraphs = []
        if not para:
            return

        max_chars = 1200
        parts = [para[i:i + max_chars] for i in range(0, len(para), max_chars)]

        for idx, part in enumerate(parts):
            heading_path = " > ".join(heading_stack) if heading_stack else ""
            base = f"{source_file}|{heading_path}|{part[:200]}"
            cid = hashlib.sha1(base.encode("utf-8")).hexdigest()[:16]
            chunk_id = f"{Path(source_file).stem}.{cid}.{idx+1}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    source_file=source_file,
                    heading_path=heading_path,
                    text=part.strip(),
                )
            )

    for raw in lines:
        line = normalize_line(raw)
        if is_heading(line):
            flush_paragraphs()
            if "CHAPTER" in line.upper() or "PART" in line.upper():
                heading_stack = [line]
            else:
                if len(heading_stack) == 0:
                    heading_stack = [line]
                else:
                    heading_stack = heading_stack[:1] + [line]
            continue

        if not line:
            flush_paragraphs()
            continue

        paragraphs.append(raw.rstrip())

    flush_paragraphs()
    return chunks


_token_re = re.compile(r"[A-Za-z0-9]+(?:[-/][A-Za-z0-9]+)*")


def bm25_tokenize(text: str) -> List[str]:
    # Conservative tokenizer: keeps "court-martial", "rule-22", "section/123"
    return [t.lower() for t in _token_re.findall(text)]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    EMB_DIR.mkdir(parents=True, exist_ok=True)

    md_files = sorted(RAW_DIR.glob("*.md"))
    if not md_files:
        raise SystemExit(f"No .md files found in {RAW_DIR.resolve()}")

    all_chunks: List[Chunk] = []
    manifest: Dict[str, Any] = {"files": []}

    for f in md_files:
        text = f.read_text(encoding="utf-8", errors="ignore")
        chunks = chunk_markdown(text, f.name)
        all_chunks.extend(chunks)
        manifest["files"].append({"name": f.name, "sha256": file_sha256(f), "chunks": len(chunks)})

    # Write chunks.jsonl
    with CHUNKS_JSONL.open("w", encoding="utf-8") as out:
        for c in all_chunks:
            out.write(
                json.dumps(
                    {
                        "chunk_id": c.chunk_id,
                        "source_file": c.source_file,
                        "heading_path": c.heading_path,
                        "text": c.text,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    MANIFEST_JSON.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # ---- FAISS embeddings ----
    model = SentenceTransformer(MODEL_NAME)
    texts = [c.text for c in all_chunks]
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.asarray(embeddings, dtype="float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    faiss.write_index(index, str(FAISS_INDEX))

    with FAISS_META.open("w", encoding="utf-8") as m:
        for i, c in enumerate(all_chunks):
            m.write(
                json.dumps(
                    {
                        "i": i,
                        "chunk_id": c.chunk_id,
                        "source_file": c.source_file,
                        "heading_path": c.heading_path,
                        "text": c.text,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    # ---- BM25 index ----
    tokenized_corpus = [bm25_tokenize(c.text) for c in all_chunks]
    bm25 = BM25Okapi(tokenized_corpus)

    with BM25_PKL.open("wb") as f:
        pickle.dump(bm25, f)

    with BM25_META.open("w", encoding="utf-8") as m:
        for i, c in enumerate(all_chunks):
            m.write(
                json.dumps(
                    {
                        "i": i,
                        "chunk_id": c.chunk_id,
                        "source_file": c.source_file,
                        "heading_path": c.heading_path,
                        "text": c.text,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    print("OK")
    print(f"Chunks: {len(all_chunks)} -> {CHUNKS_JSONL}")
    print(f"FAISS index: {FAISS_INDEX}")
    print(f"FAISS meta:  {FAISS_META}")
    print(f"BM25 index:  {BM25_PKL}")
    print(f"BM25 meta:   {BM25_META}")
    print(f"Manifest:    {MANIFEST_JSON}")


if __name__ == "__main__":
    main()
