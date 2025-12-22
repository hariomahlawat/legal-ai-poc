from __future__ import annotations

from typing import Any, Dict, List, Tuple
import os
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

_EXTRA_STOPWORDS = os.getenv("VERIFY_STOPWORDS_EXTRA", "")
if _EXTRA_STOPWORDS:
    _STOPWORDS.update({w.strip().lower() for w in _EXTRA_STOPWORDS.split(",") if w.strip()})

MIN_OVERLAP_SMALL = int(os.getenv("VERIFY_MIN_OVERLAP_SMALL", "2"))
MIN_OVERLAP_REGULAR = int(os.getenv("VERIFY_MIN_OVERLAP_REGULAR", "3"))


def _tokenize(text: str) -> List[str]:
    tokens = re.split(r"[^a-z0-9]+", (text or "").lower())
    filtered = [t for t in tokens if len(t) >= 3 and t not in _STOPWORDS]
    return filtered


def _extract_step_section(answer: str) -> List[str]:
    lines = (answer or "").splitlines()
    start_idx = None
    for idx, line in enumerate(lines):
        stripped = line.strip()
        lower_line = stripped.lower()
        if "step" in lower_line and "procedure" in lower_line and _is_heading_line(stripped):
            start_idx = idx + 1
            break
    if start_idx is None:
        return []

    section_lines: List[str] = []
    for line in lines[start_idx:]:
        stripped = line.strip()
        if _is_heading_line(stripped):
            break
        section_lines.append(line)
    return section_lines


def _is_heading_line(text: str) -> bool:
    if not text:
        return False
    if text.startswith("- "):
        return False
    if text.startswith("#"):
        return True
    if text.endswith(":"):
        return True
    return bool(re.match(r"^[A-Z][A-Za-z\s\-]+$", text))


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
    min_overlap: int | None = None,
    min_overlap_small: int | None = None,
    min_overlap_regular: int | None = None,
    return_metrics: bool = False,
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
      - metrics (optional when return_metrics=True):
          {
            "bullets_checked": int,
            "best_overlaps": list[int],
          }
    """

    bullets = extract_step_bullets(answer)
    if not bullets:
        empty_metrics = {"bullets_checked": 0, "best_overlaps": []}
        return (True, [], empty_metrics) if return_metrics else (True, [])

    citation_map = {c.get("citation_id"): c for c in citations if c.get("citation_id")}
    failures: List[Dict[str, Any]] = []

    effective_regular = min_overlap_regular
    effective_small = min_overlap_small

    if effective_regular is None:
        effective_regular = MIN_OVERLAP_REGULAR if min_overlap is None else min_overlap
    if effective_small is None:
        effective_small = MIN_OVERLAP_SMALL if min_overlap is None else min_overlap

    best_overlaps: List[int] = []

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
            best_overlaps.append(0)
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

        best_overlaps.append(best_overlap)

        required_overlap = effective_small if len(claim_tokens) <= 6 else effective_regular

        if best_overlap < required_overlap:
            failures.append(
                {
                    "bullet_index": idx,
                    "bullet_text": bullet_text,
                    "claim_text": claim_text,
                    "citation_ids": citation_ids,
                    "support": {"best_id": best_id, "overlap": best_overlap, "matched_tokens": best_tokens},
                }
            )

    result = (len(failures) == 0, failures)
    if return_metrics:
        result = (len(failures) == 0, failures, {"bullets_checked": len(bullets), "best_overlaps": best_overlaps})

    return result  # type: ignore[return-value]
