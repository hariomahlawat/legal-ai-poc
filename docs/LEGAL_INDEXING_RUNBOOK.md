# Legal Indexing Runbook

## Re-index procedure

1. Run the legal ingest pipeline:
   ```bash
   python scripts/ingest.py
   ```
2. Verify a new folder appears under `data/index_legal/` (e.g., `v20251222_1`).
3. (Optional) Pin the backend to a specific version:
   ```bash
   export LEGAL_INDEX_VERSION=v20251222_1
   ```
4. Restart the backend to load the chosen index version.

## Notes

- Each ingest run creates a new versioned index directory that includes `manifest.json`, `meta.jsonl`, `bm25.pkl`, and `faiss.index`.
- If `LEGAL_INDEX_VERSION` is not set, the backend will automatically load the latest available version.
