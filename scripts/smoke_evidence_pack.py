"""Smoke test for evidence pack construction."""

# ----------------------------
# Smoke script
# ----------------------------
import json

from apps.api.services.evidence_packer import build_evidence_pack
from apps.api.services.retrieval import retrieve_citations_multi


def main() -> None:
    question = "What are the procedures for a court of inquiry investigation?"
    citations = retrieve_citations_multi(questions=[question], top_k=5, legal_object="Court of Inquiry")

    if not citations:
        print("No citations returned. Check embeddings availability.")
        return

    pack = build_evidence_pack(question, citations, max_total_sentences=20)

    ids_present = all(f"[{c['citation_id']}]" in pack for c in citations)
    bullet_lines = [l for l in pack.splitlines() if l.strip().startswith("-")]
    full_text_length = sum(len((c.get("verbatim", "") or "")) for c in citations)

    print(json.dumps({"question": question, "citation_count": len(citations)}, ensure_ascii=False))
    print(f"Evidence pack length: {len(pack)} characters")
    print(f"Full verbatim length: {full_text_length} characters")
    print(f"Contains all citation IDs: {ids_present}")
    print(f"Bullet sentences count: {len(bullet_lines)}")
    print("--- Evidence pack ---")
    print(pack)


if __name__ == "__main__":
    main()
