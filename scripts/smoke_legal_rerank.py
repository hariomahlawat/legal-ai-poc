"""Smoke test for conditional focus filtering and reranking."""

# ----------------------------
# Smoke script
# ----------------------------
import json

from apps.api.services.retrieval import retrieve_citations_multi


def main() -> None:
    question = "What are the procedures for a court of inquiry investigation?"
    citations = retrieve_citations_multi(questions=[question], top_k=5, legal_object="Court of Inquiry")

    if not citations:
        print("No citations returned. Check embeddings availability.")
        return

    focus_applied = citations[0].get("focus_applied")
    print(f"focus_applied={focus_applied}")

    for idx, c in enumerate(citations, start=1):
        print(
            f"{idx}. {c.get('citation_id')} | retrieval={c.get('retrieval_score'):.3f} | "
            f"rerank={c.get('rerank_score')}"
        )
        print(json.dumps({"heading": c.get("heading"), "snippet": c.get("snippet")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
