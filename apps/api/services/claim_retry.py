from __future__ import annotations

from typing import Any, Dict, List


# ----------------------------
# Claim-level query builder
# ----------------------------

def _sort_failures(failures: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def _sort_key(failure: Dict[str, Any]) -> tuple:
        citation_ids = failure.get("citation_ids") or []
        support = failure.get("support") or {}
        overlap = support.get("overlap", 0) if isinstance(support.get("overlap", 0), (int, float)) else 0
        return (len(citation_ids) > 0, overlap, failure.get("bullet_index", 0))

    return sorted(failures, key=_sort_key)


def _dedupe_queries(queries: List[str]) -> List[str]:
    seen = set()
    deduped: List[str] = []
    for q in queries:
        normalized = q.strip().lower()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(q.strip())
    return deduped


def build_claim_queries(
    question: str,
    legal_object: str | None,
    failures: List[Dict[str, Any]],
    max_claims: int = 4,
    max_queries_per_claim: int = 3,
) -> List[Dict[str, Any]]:
    """
    Build deterministic micro-queries for failing claims.

    Prioritisation rules:
    - Missing citations first (no cited IDs)
    - Then lowest overlap first
    - Then earlier bullet_index
    """

    if not failures:
        return []

    sorted_failures = _sort_failures(failures)
    selected = sorted_failures[:max_claims]

    claim_packs: List[Dict[str, Any]] = []
    for failure in selected:
        claim_text = (failure.get("claim_text") or "").strip()
        if not claim_text:
            continue

        queries: List[str] = [claim_text]
        if legal_object:
            queries.append(f"{legal_object} {claim_text}")
        queries.append(f"{claim_text} rule section paragraph")

        deduped = _dedupe_queries(queries)[:max_queries_per_claim]
        claim_packs.append(
            {
                "bullet_index": failure.get("bullet_index"),
                "claim_text": claim_text,
                "queries": deduped,
            }
        )

    return claim_packs
