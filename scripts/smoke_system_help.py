"""Minimal smoke test for system-help retrieval."""

from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from apps.api.services.retrieval_repo import retrieve_repo_citations


def main() -> None:
    query = "how to start backend and front end"
    citations = retrieve_repo_citations(query, top_k=3)
    if not citations:
        raise SystemExit("No citations retrieved for system-help query.")

    print(f"Retrieved {len(citations)} citations for: {query}")
    for c in citations:
        print(f"- {c['citation_id']} | {c.get('source_file','')} | {c.get('heading_path','')} ")
        snippet = (c.get("snippet", "") or c.get("text", "")).strip()
        print(f"  Snippet: {snippet[:160]}")


if __name__ == "__main__":
    main()
