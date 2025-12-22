# System-help index artifacts

Generated FAISS and BM25 artifacts for the system-help corpus live here. They are intentionally excluded from version control because they are binary and environment-specific. Build them locally with:

```bash
python scripts/ingest_repo.py
```

After generation, this directory should contain files like `faiss.index`, `bm25.pkl`, and `meta.jsonl`. Regenerate whenever runbook content changes.
