"""Query expansion helpers for legal retrieval."""

from __future__ import annotations

import re
from typing import Optional

from .intent import _GLOSSARY

# ----------------------------
# Helpers
# ----------------------------

_MAX_QUERIES = 5


def _normalise_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


# ----------------------------
# Public API
# ----------------------------

def expand_queries(question: str, legal_object: Optional[str]) -> list[str]:
    """
    Produce 2 to 5 query variants for robust retrieval.

    Rules:
    - First query must be original question (normalised).
    - Add acronym expansions using intent._GLOSSARY.
    - If legal_object is present, add one query prefixed with it.
    - If user references rule/section/para/appendix, add a 'section hint' query.
    """
    base = _normalise_spaces(question)
    if not base:
        return []

    variants: list[str] = [base]
    q_lower = base.lower()

    # 1) Acronym expansions using glossary
    expanded = base
    for short, long_form in _GLOSSARY.items():
        pattern = r"\b" + re.escape(short) + r"\b"
        if re.search(pattern, expanded, flags=re.IGNORECASE):
            expanded = re.sub(pattern, long_form, expanded, flags=re.IGNORECASE)

    expanded = _normalise_spaces(expanded)
    if expanded and expanded.lower() != q_lower:
        variants.append(expanded)

    # 2) Legal object prefix query (if available)
    if legal_object:
        pref = _normalise_spaces(f"{legal_object}: {base}")
        if pref.lower() not in {v.lower() for v in variants}:
            variants.append(pref)

    # 3) Section hint query if user indicates section-style lookup
    if any(k in q_lower for k in ["rule", "section", "para", "appendix"]):
        hint = _normalise_spaces(f"{base} relevant rule section paragraph appendix")
        if hint.lower() not in {v.lower() for v in variants}:
            variants.append(hint)

    # Cap variants to _MAX_QUERIES
    return variants[:_MAX_QUERIES]
