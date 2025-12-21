import re
from dataclasses import dataclass
from typing import Optional, Tuple, Dict


@dataclass(frozen=True)
class IntentResult:
    legal_object: str
    confidence: float
    needs_clarification: bool
    clarification_question: Optional[str]
    normalized_query: str


_GLOSSARY: Dict[str, str] = {
    # Courts and proceedings
    "coi": "Court of Inquiry",
    "court of inquiry": "Court of Inquiry",
    "court-martial": "Court-Martial",
    "courts-martial": "Court-Martial",
    "gcm": "Court-Martial",
    "dcm": "Court-Martial",
    "scm": "Court-Martial",
    "summary court-martial": "Court-Martial",
    "general court-martial": "Court-Martial",
    "district court-martial": "Court-Martial",

    # Administrative and disciplinary
    "awl": "Disciplinary Action",
    "awol": "Disciplinary Action",
    "absence without leave": "Disciplinary Action",
    "desertion": "Disciplinary Action",

    # Roles and appointments
    "convening officer": "Court-Martial",
    "convening authority": "Court-Martial",
    "defending officer": "Court-Martial",
    "prosecutor": "Court of Inquiry",

    # Generic legal research bucket
    "mml": "General Legal Reference",
    "army act": "General Legal Reference",
}


def _normalize(text: str) -> str:
    t = text.strip().lower()
    t = re.sub(r"\s+", " ", t)
    return t


def _contains_any(haystack: str, needles: Tuple[str, ...]) -> bool:
    return any(n in haystack for n in needles)


def classify_legal_object(user_text: str) -> IntentResult:
    """
    Deterministic classifier for PoC.
    Contract is stable so you can later replace the internals with an LLM classifier.

    Returns:
      legal_object: one of known buckets
      needs_clarification: True when ambiguity is high
    """
    q = _normalize(user_text)

    # Strong explicit phrases first
    if _contains_any(q, ("court of inquiry", "coi")):
        return IntentResult(
            legal_object="Court of Inquiry",
            confidence=0.98,
            needs_clarification=False,
            clarification_question=None,
            normalized_query=q,
        )

    if _contains_any(q, ("court-martial", "courts-martial", "gcm", "dcm", "scm")):
        return IntentResult(
            legal_object="Court-Martial",
            confidence=0.98,
            needs_clarification=False,
            clarification_question=None,
            normalized_query=q,
        )

    # Disciplinary terms
    if _contains_any(q, ("awl", "awol", "absence without leave", "desertion")):
        return IntentResult(
            legal_object="Disciplinary Action",
            confidence=0.92,
            needs_clarification=False,
            clarification_question=None,
            normalized_query=q,
        )

    # Ambiguity patterns: "convene" + "court" without specifying which court
    if "convene" in q and "court" in q and not _contains_any(q, ("court of inquiry", "court-martial", "courts-martial")):
        return IntentResult(
            legal_object="Unknown",
            confidence=0.55,
            needs_clarification=True,
            clarification_question="Do you mean Court of Inquiry or Court-Martial?",
            normalized_query=q,
        )

    # Fallback: general legal reference
    return IntentResult(
        legal_object="General Legal Reference",
        confidence=0.60,
        needs_clarification=False,
        clarification_question=None,
        normalized_query=q,
    )
