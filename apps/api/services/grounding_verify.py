from __future__ import annotations

from typing import Any, Dict, List, Tuple
import re


# ----------------------------
# Constants and helpers
# ----------------------------
_CITATION_BRACKET_RE = re.compile(r"\[([^\[\]]+)\]")
_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "into",
    "are",
    "was",
    "were",
    "shall",
    "will",
    "may",
}


def _tokenize(text: str) -> List[str]:
    tokens = re.split(r"[^a-z0-9]+", (text or "").lower())
    filtered = [t for t in tokens if len(t) >= 3 and t not in _STOPWORDS]
    return filtered


def _extract_step_section(answer: str) -> List[str]:
    lines = (answer or "").splitlines()
    start_idx = None
    for idx, line in enumerate(lines):
        if line.strip().lower() == "step-by-step procedure:":
            start_idx = idx + 1
            break
    if start_idx is None:
        return []

    section_lines: List[str] = []
    for line in lines[start_idx:]:
        stripped = line.strip()
        if stripped.endswith(":") and not stripped.startswith("-"):
            break
        section_lines.append(line)
    return section_lines


def extract_step_bullets(answer: str) -> List[Tuple[int, str]]:
    """
    Extract bullet lines (index, text) under the Step-by-step procedure heading.
    """
    section_lines = _extract_step_section(answer)
    bullets: List[Tuple[int, str]] = []
    bullet_idx = 0
    for line in section_lines:
        if line.lstrip().startswith("- "):
            bullets.append((bullet_idx, line.strip()))
            bullet_idx += 1
    return bullets


# ----------------------------
# Public API
# ----------------------------

def verify_grounding(
    answer: str,
    citations: List[Dict[str, Any]],
    min_overlap: int = 3,
) -> Tuple[bool, List[Dict[str, Any]]]:
    """
    Verifies that each bullet under 'Step-by-step procedure' is supported by the cited evidence.

    Returns:
      - ok: bool (True if all checked bullets supported)
      - failures: list of dicts:
          {
            "bullet_index": int,
            "bullet_text": str,
            "claim_text": str,
            "citation_ids": list[str],
            "support": {
              "best_id": str | None,
              "overlap": int,
              "matched_tokens": list[str]
            }
          }
    """

    bullets = extract_step_bullets(answer)
    if not bullets:
        return True, []

    citation_map = {c.get("citation_id"): c for c in citations if c.get("citation_id")}
    failures: List[Dict[str, Any]] = []

    for idx, bullet_text in bullets:
        citation_ids = [c.strip() for c in _CITATION_BRACKET_RE.findall(bullet_text) if c.strip()]

        claim_text = bullet_text
        while True:
            new_text = re.sub(r"\s*\[[^\[\]]+\]\s*$", "", claim_text).strip()
            if new_text == claim_text:
                break
            claim_text = new_text

        claim_tokens = set(_tokenize(claim_text))

        best_id = None
        best_overlap = 0
        best_tokens: List[str] = []

        if not citation_ids:
            failures.append(
                {
                    "bullet_index": idx,
                    "bullet_text": bullet_text,
                    "claim_text": claim_text,
                    "citation_ids": [],
                    "support": {"best_id": None, "overlap": 0, "matched_tokens": []},
                }
            )
            continue

        for citation_id in citation_ids:
            evidence_text = citation_map.get(citation_id, {}).get("text", "")
            evidence_tokens = set(_tokenize(evidence_text))
            overlap_tokens = claim_tokens.intersection(evidence_tokens)
            overlap = len(overlap_tokens)
            if overlap > best_overlap:
                best_overlap = overlap
                best_id = citation_id
                best_tokens = sorted(overlap_tokens)

        if best_overlap < min_overlap:
            failures.append(
                {
                    "bullet_index": idx,
                    "bullet_text": bullet_text,
                    "claim_text": claim_text,
                    "citation_ids": citation_ids,
                    "support": {"best_id": best_id, "overlap": best_overlap, "matched_tokens": best_tokens},
                }
            )

    return len(failures) == 0, failures
