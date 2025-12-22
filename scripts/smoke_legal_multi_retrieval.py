"""Minimal smoke test for legal multi-query retrieval."""

from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from apps.api.services.query_expansion import expand_queries
from apps.api.services.retrieval import retrieve_citations_multi


# ----------------------------
# Entry point
# ----------------------------

def main() -> None:
    question = "COI procedure for recording evidence"
    queries = expand_queries(question, "Court of Inquiry")
    citations = retrieve_citations_multi(queries, top_k=5, legal_object="Court of Inquiry")

    if not citations:
        raise SystemExit("No citations returned for legal multi-query retrieval smoke test.")

    print(f"Retrieved {len(citations)} citations for: {question}")
    for c in citations:
        print(
            f"- {c['citation_id']} | score={c.get('retrieval_score', 0):.4f} | hits={c.get('hit_query_count', 0)} | {c.get('source_file','')} | {c.get('heading','')}"
        )
        snippet = (c.get("snippet", "") or c.get("verbatim", "")).strip()
        print(f"  Snippet: {snippet[:160]}")


if __name__ == "__main__":
    main()
