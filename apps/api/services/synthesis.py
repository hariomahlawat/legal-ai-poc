from __future__ import annotations

from typing import List, Dict, Any, Optional, Tuple
import json
import logging
import os
import re
import time
import uuid

import requests

from apps.api.config import (
    EVIDENCE_MAX_CHARS_PER_CITATION,
    EVIDENCE_MAX_CHARS_TOTAL,
    OLLAMA_CONNECT_TIMEOUT_SECS,
    OLLAMA_MODEL_LEGAL,
    OLLAMA_NUM_PREDICT,
    OLLAMA_READ_TIMEOUT_SECS,
    OLLAMA_TEMPERATURE,
    OLLAMA_URL,
    SYNTHESIS_MAX_CITATIONS,
)

from .ollama_client import (
    OllamaConnectionError,
    OllamaError,
    OllamaResponseError,
    OllamaTimeoutError,
    ollama_chat,
    ollama_chat_stream,
)

from .citation_store import citation_store
from .claim_retry import build_claim_queries
from .evidence_packer import build_evidence_pack
from .grounding_verify import verify_grounding
from .retrieval import retrieve_citations_multi

logger = logging.getLogger(__name__)


# ----------------------------
# Configuration
# ----------------------------
TWO_PASS_ENABLED = os.getenv("TWO_PASS_ENABLED", "true").lower() != "false"
MAX_CITATIONS_FOR_REPAIR = 18
_OLLAMA_HEALTHY: Optional[bool] = None


# ----------------------------
# Prompts and templates
# ----------------------------
_SYSTEM_PROMPT_BASE = """
You are a legal drafting assistant for a military legal research system.

Hard rules (non-negotiable):
1) Use ONLY the provided EVIDENCE. Do not add external law, assumptions, or procedures not present in EVIDENCE.
2) Do NOT answer about a different legal object than requested. If the question is about Court of Inquiry, do not discuss Court-Martial unless the evidence explicitly connects them.
3) Every procedural step or legal proposition MUST end with one or more citation IDs in square brackets, exactly matching the IDs provided in EVIDENCE.
4) If the evidence is insufficient, say so clearly, cite the most relevant evidence you do have, and ask for the minimum missing facts.

Output format (strict):
- A single structured answer with the following headings (exact text):
  Applicable provisions:
  Step-by-step procedure:
  Common mistakes to avoid:
  If facts are missing:

Keep language precise and professional. Avoid filler.
""".strip()

_SYSTEM_PROMPT_PLANNER = """
You are a planning assistant that produces JSON plans for legal drafting. Return JSON only with no additional prose.
Each point must be tied to citation IDs provided. Do not invent citation IDs and do not include narrative.
If evidence is insufficient for a point, add an assumption entry as 'Insufficient evidence: …' and cite the closest evidence ID.
""".strip()


def _build_user_prompt(
    question: str,
    citations: List[Dict[str, Any]],
    legal_object: Optional[str],
    *,
    max_chars_total: Optional[int] = None,
    max_chars_per_citation: Optional[int] = None,
) -> str:
    legal_line = f"Requested legal object: {legal_object}" if legal_object else "Requested legal object: (not specified)"
    evidence_block = (
        build_evidence_pack(
            question,
            citations,
            max_chars_total=max_chars_total or EVIDENCE_MAX_CHARS_TOTAL,
            max_chars_per_citation=max_chars_per_citation or EVIDENCE_MAX_CHARS_PER_CITATION,
        )
        if citations
        else "(no evidence retrieved)"
    )

    return (
        f"{legal_line}\n\n"
        f"Question:\n{question.strip()}\n\n"
        f"EVIDENCE:\n{evidence_block}\n"
    )


_CITATION_BRACKET_RE = re.compile(r"\[([^\[\]]+)\]")


# ----------------------------
# LLM connectivity helpers
# ----------------------------
def _ollama_available() -> bool:
    """Check if Ollama endpoint is reachable before attempting generation."""

    global _OLLAMA_HEALTHY

    if _OLLAMA_HEALTHY is True:
        return True

    try:
        resp = requests.get(
            f"{OLLAMA_URL}/api/tags",
            timeout=(OLLAMA_CONNECT_TIMEOUT_SECS, OLLAMA_CONNECT_TIMEOUT_SECS),
        )
        resp.raise_for_status()
        _OLLAMA_HEALTHY = True
        return True
    except Exception as exc:  # pragma: no cover - connectivity dependent
        logger.warning("Ollama health check failed: %s", exc)
        _OLLAMA_HEALTHY = False
        return False


# ----------------------------
# Validation utilities
# ----------------------------
def _validate_answer(answer: str, known_ids: set[str]) -> Tuple[bool, List[Dict[str, str]], List[str]]:
    """
    Returns:
      ok: bool
      issues: list of {code,message}
      unknown_ids: list[str]
    """
    issues: List[Dict[str, str]] = []
    unknown_ids: List[str] = []

    txt = (answer or "").strip()
    if not txt:
        issues.append({"code": "EMPTY", "message": "Model output is empty."})
        return False, issues, unknown_ids

    # Ensure required headings exist
    required = [
        "Applicable provisions:",
        "Step-by-step procedure:",
        "Common mistakes to avoid:",
        "If facts are missing:",
    ]
    missing = [h for h in required if h not in txt]
    if missing:
        issues.append({"code": "MISSING_HEADINGS", "message": f"Missing required headings: {', '.join(missing)}"})

    # Extract citation IDs used
    used = _CITATION_BRACKET_RE.findall(txt)
    # Normalize (strip spaces) but keep exact comparison for IDs
    used_clean = [u.strip() for u in used if u.strip()]

    for u in used_clean:
        if u not in known_ids:
            unknown_ids.append(u)

    if unknown_ids:
        issues.append(
            {
                "code": "UNKNOWN_CITATION_IDS",
                "message": "Answer contains citation IDs that were not provided in evidence.",
            }
        )

    # For each "step/proposition" line, enforce it ends with a bracket (or multiple brackets).
    # Heuristic: lines starting with "-" or "*" or "1." etc under headings.
    for line in txt.splitlines():
        s = line.strip()
        if not s:
            continue
        if re.match(r"^(\d+\.|\-|\*)\s+", s):
            if not s.endswith("]"):
                issues.append(
                    {
                        "code": "MISSING_CITATION_AT_LINE_END",
                        "message": f"Line does not end with a citation bracket: {s[:120]}",
                    }
                )
                break

    ok = len([i for i in issues if i["code"] != "MISSING_HEADINGS"]) == 0 and not missing
    # Note: missing headings is a hard fail too
    ok = ok and not missing
    return ok, issues, unknown_ids


def _validate_plan(plan: Any, allowed_ids: set[str]) -> Tuple[bool, List[str]]:
    """Validate planner output structure and citation usage."""
    errors: List[str] = []

    if not isinstance(plan, dict):
        return False, ["Plan is not a JSON object."]

    legal_object = plan.get("legal_object")
    if legal_object is None or not isinstance(legal_object, str):
        errors.append("Missing or invalid legal_object (string required).")

    assumptions = plan.get("assumptions")
    if assumptions is None or not isinstance(assumptions, list) or any(not isinstance(a, str) for a in assumptions):
        errors.append("assumptions must be a list of strings (can be empty).")

    steps = plan.get("steps")
    if steps is None or not isinstance(steps, list):
        errors.append("steps must be a list of step objects.")
    else:
        for idx, step in enumerate(steps):
            if not isinstance(step, dict):
                errors.append(f"Step {idx} is not an object.")
                continue
            if not isinstance(step.get("title"), str):
                errors.append(f"Step {idx} missing title (string required).")
            points = step.get("points")
            if points is None or not isinstance(points, list):
                errors.append(f"Step {idx} points must be a list.")
                continue
            for p_idx, point in enumerate(points):
                if not isinstance(point, dict):
                    errors.append(f"Point {p_idx} in step {idx} is not an object.")
                    continue
                if not isinstance(point.get("text"), str):
                    errors.append(f"Point {p_idx} in step {idx} missing text (string required).")
                citations = point.get("citations")
                if citations is None or not isinstance(citations, list):
                    errors.append(f"Point {p_idx} in step {idx} missing citations list.")
                    continue
                if not citations:
                    errors.append(f"Point {p_idx} in step {idx} must include at least one citation.")
                for c in citations:
                    if not isinstance(c, str):
                        errors.append(f"Point {p_idx} in step {idx} citation is not a string.")
                        continue
                    if c not in allowed_ids:
                        errors.append(f"Point {p_idx} in step {idx} uses unknown citation ID: {c}.")

    return len(errors) == 0, errors


# ----------------------------
# Grounding verification helpers
# ----------------------------
def _log_grounding_result(
    grounding_ok: bool,
    failures: List[Dict[str, Any]],
    repair_attempted: bool = False,
    repair_success: bool = False,
) -> None:
    logger.info(
        "grounding_verification",
        extra={
            "grounding_ok": grounding_ok,
            "grounding_failures_count": len(failures),
            "grounding_repair_attempted": repair_attempted,
            "grounding_repair_success": repair_success,
            "grounding_failures": [
                {
                    "bullet_index": f.get("bullet_index"),
                    "best_id": (f.get("support") or {}).get("best_id"),
                    "overlap": (f.get("support") or {}).get("overlap"),
                    "claim_text": (f.get("claim_text", "") or "")[:120],
                }
                for f in failures
            ],
        },
    )


# ----------------------------
# Claim-level retrieval retry helpers
# ----------------------------
def _merge_and_dedupe_citations(
    original: List[Dict[str, Any]], extras: List[Dict[str, Any]], max_total: int = MAX_CITATIONS_FOR_REPAIR
) -> List[Dict[str, Any]]:
    seen: set[str] = set()
    merged: List[Dict[str, Any]] = []

    for group in (original, extras):
        for citation in group:
            cid = citation.get("citation_id")
            if not cid or cid in seen:
                continue
            seen.add(cid)
            merged.append(citation)

    return merged[:max_total]


def _record_citations(citations: List[Dict[str, Any]]) -> None:
    for citation in citations:
        citation_store.upsert(citation)


def _select_prompt_citations(citations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return citations[:SYNTHESIS_MAX_CITATIONS]


def _shrink_citations_for_timeout(citations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    trimmed: List[Dict[str, Any]] = []
    for citation in citations[:2]:
        limited = dict(citation)
        verbatim_source = (
            citation.get("text")
            or citation.get("verbatim")
            or citation.get("snippet")
            or citation.get("heading")
            or ""
        )
        limited["verbatim"] = (verbatim_source or "")[:500]
        limited.pop("context_before", None)
        limited.pop("context_after", None)
        trimmed.append(limited)

    return trimmed


def _claim_retry_repair(
    question: str,
    legal_object: Optional[str],
    plan: Optional[Dict[str, Any]],
    base_answer: str,
    citations: List[Dict[str, Any]],
    failures: List[Dict[str, Any]],
    telemetry: Dict[str, Any],
    request_id: str,
) -> Tuple[str, bool, List[Dict[str, Any]], List[Dict[str, str]], List[Dict[str, Any]], bool]:
    if telemetry.get("claim_retry_completed"):
        return base_answer, False, failures, [], citations, False

    telemetry["grounding_failures_initial_count"] = len(failures)

    claim_packs = build_claim_queries(question, legal_object, failures)
    telemetry["claim_retry_claims_selected"] = len(claim_packs)
    telemetry["claim_retry_used"] = bool(claim_packs)

    extras: List[Dict[str, Any]] = []
    for pack in claim_packs:
        extras.extend(
            retrieve_citations_multi(
                questions=pack.get("queries", []),
                top_k=6,
                legal_object=legal_object,
            )
        )

    merged_citations = _merge_and_dedupe_citations(citations, extras, MAX_CITATIONS_FOR_REPAIR)
    original_ids = {c.get("citation_id") for c in citations if c.get("citation_id")}
    extra_ids = {c.get("citation_id") for c in merged_citations if c.get("citation_id") not in original_ids}
    telemetry["claim_retry_extra_citations"] = len([cid for cid in extra_ids if cid])
    telemetry["retrieval_retry_count"] = 1 if claim_packs else 0

    _record_citations(merged_citations)

    if not claim_packs:
        telemetry["grounding_failures_after_retry_count"] = len(failures)
        telemetry["claim_retry_completed"] = True
        return base_answer, False, failures, [], merged_citations, False

    plan_block = ""
    if plan:
        plan_block = f"Validated plan (JSON):\n{json.dumps(plan, ensure_ascii=False, indent=2)}\n\n"

    failures_desc = []
    for failure in claim_packs:
        failures_desc.append(
            "- Bullet {idx}: '{claim}'".format(
                idx=failure.get("bullet_index"),
                claim=failure.get("claim_text", ""),
            )
        )

    corrective_instruction = (
        "Grounding failures detected in Step-by-step procedure. Edit only these bullets; do not change unrelated bullets. "
        "Use the merged evidence below. If the evidence remains insufficient, replace with: "
        "'Insufficient evidence in the provided corpus to state this step precisely.' and cite the closest relevant citation ID."
    )

    user_prompt = (
        f"{_build_user_prompt(question, merged_citations, legal_object)}\n\n"
        f"{plan_block}"
        f"Bullets to repair:\n{chr(10).join(failures_desc)}\n\n"
        f"{corrective_instruction}"
    )

    merged_known_ids = {c.get("citation_id", "") for c in merged_citations if c.get("citation_id")}
    repaired_answer = _call_ollama_chat(_SYSTEM_PROMPT_BASE, user_prompt, request_id)
    ok_format, issues, _unknown = _validate_answer(repaired_answer, merged_known_ids)
    repair_failures = failures
    grounding_ok = False

    if ok_format:
        grounding_ok, repair_failures = verify_grounding(repaired_answer, merged_citations)

    telemetry["grounding_failures_after_retry_count"] = len(repair_failures)

    logger.info(
        "claim_retry_result",
        extra={
            "repair_mode": "claim_retry",
            "grounding_failures_initial_count": len(failures),
            "claim_retry_claims_selected": len(claim_packs),
            "claim_retry_total_extra_citations": telemetry.get("claim_retry_extra_citations", 0),
            "grounding_failures_after_retry_count": len(repair_failures),
        },
    )

    _log_grounding_result(grounding_ok, repair_failures, repair_attempted=True, repair_success=grounding_ok and ok_format)
    telemetry["claim_retry_completed"] = True
    return repaired_answer, (ok_format and grounding_ok), repair_failures, issues, merged_citations, True


def _patch_unsupported_bullets(
    answer: str, failures: List[Dict[str, Any]], known_ids: set[str], citations: List[Dict[str, Any]]
) -> str:
    lines = (answer or "").splitlines()
    start_idx = None
    for idx, line in enumerate(lines):
        if line.strip().lower() == "step-by-step procedure:":
            start_idx = idx + 1
            break
    if start_idx is None:
        return answer

    failure_map = {f.get("bullet_index"): f for f in failures}
    bullet_counter = 0

    for line_idx in range(start_idx, len(lines)):
        stripped = lines[line_idx].strip()
        if stripped.endswith(":") and not stripped.startswith("-"):
            break
        if not stripped.startswith("- "):
            continue

        failure = failure_map.get(bullet_counter)
        if failure:
            best_id = (failure.get("support") or {}).get("best_id")
            if not best_id:
                best_id = _choose_best_citation_id(known_ids, citations)
            citation_block = f" [{best_id}]" if best_id else ""
            lines[line_idx] = (
                "- Insufficient evidence in the provided corpus to state this step precisely." + citation_block
            )
        bullet_counter += 1

    return "\n".join(lines)


def _choose_best_citation_id(known_ids: set[str], citations: List[Dict[str, Any]]) -> Optional[str]:
    if not known_ids:
        return None

    candidate_list = [c for c in citations if c.get("citation_id") in known_ids]
    if not candidate_list:
        return sorted(known_ids)[0]

    def _score(citation: Dict[str, Any]) -> Tuple[int, float, int, float, str]:
        rerank = citation.get("rerank_score")
        retrieval = citation.get("retrieval_score")
        rerank_valid = 1 if isinstance(rerank, (int, float)) else 0
        retrieval_valid = 1 if isinstance(retrieval, (int, float)) else 0
        return (
            rerank_valid,
            rerank if rerank_valid else float("-inf"),
            retrieval_valid,
            retrieval if retrieval_valid else float("-inf"),
            citation.get("citation_id", ""),
        )

    best = max(candidate_list, key=_score)
    return best.get("citation_id")


def _attempt_grounding_repair(
    question: str,
    citations: List[Dict[str, Any]],
    legal_object: Optional[str],
    plan: Optional[Dict[str, Any]],
    known_ids: set[str],
    failures: List[Dict[str, Any]],
) -> Tuple[str, bool, List[Dict[str, Any]], List[Dict[str, str]]]:
    base_user_prompt = _build_user_prompt(question, citations, legal_object)
    plan_block = ""
    if plan:
        plan_block = f"Validated plan (JSON):\n{json.dumps(plan, ensure_ascii=False, indent=2)}\n\n"

    failures_desc = []
    for failure in failures:
        support = failure.get("support") or {}
        failures_desc.append(
            "- Bullet {idx}: '{claim}' (best overlap {overlap} with {best_id})".format(
                idx=failure.get("bullet_index"),
                claim=failure.get("claim_text", ""),
                overlap=support.get("overlap", 0),
                best_id=support.get("best_id"),
            )
        )

    corrective_instruction = (
        "Grounding failures detected in Step-by-step procedure. Revise these bullets to match the provided evidence. "
        "Do not introduce new steps beyond what the evidence states. "
        "If the evidence does not state a step precisely, replace it with: "
        "'Insufficient evidence in the provided corpus to state this step precisely.' and cite the closest relevant citation ID."
    )

    user_prompt = (
        f"{base_user_prompt}\n\n"
        f"{plan_block}"
        f"Grounding failures:\n{chr(10).join(failures_desc)}\n\n"
        f"{corrective_instruction}"
    )

    repaired_answer = _call_ollama_chat(_SYSTEM_PROMPT_BASE, user_prompt, request_id)
    ok_format, issues, _unknown = _validate_answer(repaired_answer, known_ids)
    repair_failures = failures
    grounding_ok = False
    if ok_format:
        grounding_ok, repair_failures = verify_grounding(repaired_answer, citations)

    _log_grounding_result(grounding_ok, repair_failures, repair_attempted=True, repair_success=grounding_ok and ok_format)
    return repaired_answer, (ok_format and grounding_ok), repair_failures, issues


def _call_ollama_chat(
    system_prompt: str,
    user_prompt: str,
    request_id: str,
    *,
    model: Optional[str] = None,
    options: Optional[Dict[str, Any]] = None,
) -> str:
    if not _ollama_available():
        raise RuntimeError(
            f"Ollama not reachable at {OLLAMA_URL}. Set OLLAMA_URL or start the service."
        )

    messages = [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "user", "content": user_prompt},
    ]

    return ollama_chat(model or OLLAMA_MODEL_LEGAL, messages, options or {}, request_id)


# ----------------------------
# Planner and writer utilities
# ----------------------------
def _build_planner_prompt(question: str, citations: List[Dict[str, Any]], legal_object: Optional[str], allowed_ids: set[str]) -> str:
    legal_line = f"Requested legal object: {legal_object}" if legal_object else "Requested legal object: (not specified)"
    evidence_block = build_evidence_pack(question, citations) if citations else "(no evidence retrieved)"
    allowed_ids_line = ", ".join(sorted(allowed_ids))

    schema_description = (
        "Return JSON ONLY with keys: legal_object (string), assumptions (list of strings), steps (list of objects).\n"
        "Each step: {title: string, points: list of {text: string, citations: list of citation_id strings}}.\n"
        "Every point must include at least one citation from the allowed list."
    )

    instructions = (
        "- Use only allowed citation IDs.\n"
        "- Keep point texts concise and evidentiary.\n"
        "- If evidence is insufficient for a point, place it under assumptions as 'Insufficient evidence: ...' and cite the closest evidence ID.\n"
        "- Output JSON only, no prose before or after."
    )

    return (
        f"{legal_line}\n\n"
        f"Question:\n{question.strip()}\n\n"
        f"Allowed citation IDs:\n{allowed_ids_line}\n\n"
        f"EVIDENCE PACK (summarised):\n{evidence_block}\n\n"
        f"Schema:\n{schema_description}\n\n"
        f"Instructions:\n{instructions}"
    )


def _run_planner(
    question: str,
    citations: List[Dict[str, Any]],
    legal_object: Optional[str],
    allowed_ids: set[str],
    request_id: str,
) -> Tuple[Optional[Dict[str, Any]], int, bool]:
    """Run planner with a single retry on JSON/validation failure."""
    planner_retry_count = 0
    user_prompt = _build_planner_prompt(question, citations, legal_object, allowed_ids)

    for attempt in range(2):
        try:
            raw_plan = _call_ollama_chat(_SYSTEM_PROMPT_PLANNER, user_prompt, request_id)
            plan_obj = json.loads(raw_plan)
            ok, errors = _validate_plan(plan_obj, allowed_ids)
            if ok:
                return plan_obj, planner_retry_count, True
            planner_retry_count += 1
            user_prompt = (
                user_prompt
                + "\n\nYou must return valid JSON only. Fix these issues: "
                + "; ".join(errors)
            )
        except Exception:
            planner_retry_count += 1
            user_prompt = user_prompt + "\n\nYou must return valid JSON only with the required schema."
            continue

    return None, planner_retry_count, False


def _run_writer(
    question: str,
    citations: List[Dict[str, Any]],
    legal_object: Optional[str],
    plan: Dict[str, Any],
    known_ids: set[str],
    allowed_ids_line: str,
    telemetry: Dict[str, Any],
    request_id: str,
) -> Tuple[Optional[str], List[Dict[str, str]], int, bool, List[Dict[str, Any]]]:
    """Generate answer using validated plan and run validation with one retry."""
    warnings: List[Dict[str, str]] = []
    writer_retry_count = 0
    plan_json = json.dumps(plan, ensure_ascii=False, indent=2)
    base_user_prompt = _build_user_prompt(question, citations, legal_object)
    user_prompt = (
        f"{base_user_prompt}\n\nValidated plan (JSON):\n{plan_json}\n\n"
        "Follow the plan strictly. Use only allowed citation IDs. Ensure every bullet in Step-by-step ends with citations in brackets."
    )

    telemetry.setdefault("retrieval_retry_count", 0)
    telemetry.setdefault("claim_retry_used", False)
    telemetry.setdefault("claim_retry_extra_citations", 0)
    telemetry.setdefault("grounding_failures_initial_count", 0)
    telemetry.setdefault("grounding_failures_after_retry_count", 0)
    telemetry.setdefault("claim_retry_completed", False)
    telemetry.setdefault("claim_retry_claims_selected", 0)
    telemetry.setdefault("claim_retry_completed", False)
    telemetry.setdefault("claim_retry_claims_selected", 0)

    try:
        answer = _call_ollama_chat(_SYSTEM_PROMPT_BASE, user_prompt, request_id)
        ok, issues, _unknown = _validate_answer(answer, known_ids)
        if ok:
            grounding_ok, failures = verify_grounding(answer, citations)
            _log_grounding_result(grounding_ok, failures)
            if grounding_ok:
                return answer, warnings, writer_retry_count, True, citations

            repaired_answer, repair_success, repair_failures, repair_issues, merged_citations, used_retry = _claim_retry_repair(
                question,
                legal_object,
                plan,
                answer,
                citations,
                failures,
                telemetry,
                request_id,
            )
            warnings.extend(repair_issues)
            if used_retry:
                telemetry["claim_retry_used"] = True
            merged_known_ids = {c.get("citation_id", "") for c in merged_citations if c.get("citation_id")}
            if repair_success:
                warnings.append(
                    {
                        "code": "GROUNDING_REPAIRED",
                        "message": "Answer required grounding repair; a corrected answer was produced.",
                    }
                )
                return repaired_answer, warnings, writer_retry_count, True, merged_citations

            patched_answer = _patch_unsupported_bullets(repaired_answer, repair_failures, merged_known_ids, merged_citations)
            warnings.append(
                {
                    "code": "GROUNDING_PATCHED",
                    "message": "Unsupported bullets were replaced due to insufficient evidence.",
                }
            )
            return patched_answer, warnings, writer_retry_count, True, merged_citations

        warnings.extend(issues)
        corrective = (
            _SYSTEM_PROMPT_BASE
            + "\n\n"
            + "You MUST fix the following issues in your answer:\n"
            + "\n".join([f"- {i['code']}: {i['message']}" for i in issues])
            + "\n\n"
            + allowed_ids_line
            + "\n"
            + "Use the provided plan exactly. Ensure each procedural line ends with citation brackets.\n"
        )
        writer_retry_count += 1
        answer2 = _call_ollama_chat(corrective, user_prompt, request_id)
        ok2, issues2, _unknown2 = _validate_answer(answer2, known_ids)
        if ok2:
            grounding_ok, failures = verify_grounding(answer2, citations)
            _log_grounding_result(grounding_ok, failures)
            if grounding_ok:
                warnings.append(
                    {
                        "code": "REPAIRED_OUTPUT",
                        "message": "Model output required repair; a corrected answer was produced.",
                    }
                )
                return answer2, warnings, writer_retry_count, True, citations

            repaired_answer, repair_success, repair_failures, repair_issues, merged_citations, used_retry = _claim_retry_repair(
                question,
                legal_object,
                plan,
                answer2,
                citations,
                failures,
                telemetry,
                request_id,
            )
            warnings.extend(repair_issues)
            if used_retry:
                telemetry["claim_retry_used"] = True
            merged_known_ids = {c.get("citation_id", "") for c in merged_citations if c.get("citation_id")}
            if repair_success:
                warnings.append(
                    {
                        "code": "GROUNDING_REPAIRED",
                        "message": "Answer required grounding repair; a corrected answer was produced.",
                    }
                )
                return repaired_answer, warnings, writer_retry_count, True, merged_citations

            patched_answer = _patch_unsupported_bullets(repaired_answer, repair_failures, merged_known_ids, merged_citations)
            warnings.append(
                {
                    "code": "GROUNDING_PATCHED",
                    "message": "Unsupported bullets were replaced due to insufficient evidence.",
                }
            )
            return patched_answer, warnings, writer_retry_count, True, merged_citations

        warnings.extend(issues2)
        return answer2, warnings, writer_retry_count, False, citations
    except OllamaTimeoutError as exc:
        warnings.append(
            {
                "code": "LLM_TIMEOUT",
                "message": f"Primary Ollama call timed out after {OLLAMA_READ_TIMEOUT_SECS}s: {exc}",
            }
        )
        best_answer, reduced_warnings, reduced_citations = _attempt_reduced_prompt(
            question, citations, legal_object, request_id
        )
        warnings.extend(reduced_warnings)
        if best_answer:
            warnings.append(
                {
                    "code": "TIMEOUT_RETRY",
                    "message": "Returned answer from reduced prompt after timeout.",
                }
            )
            return best_answer, warnings, writer_retry_count, True, reduced_citations

        warnings.append(
            {
                "code": "FALLBACK_USED",
                "message": "Timeout retry failed; returning fallback template.",
            }
        )
        return _fallback_template(question, citations, legal_object), warnings, writer_retry_count, False, citations
    except (OllamaConnectionError, OllamaResponseError, RuntimeError) as exc:
        warnings.append(
            {
                "code": "LLM_UNAVAILABLE",
                "message": f"Ollama/LLM unavailable or failed. Falling back to conservative template. Details: {exc}",
            }
        )
        return _fallback_template(question, citations, legal_object), warnings, writer_retry_count, False, citations


def _fallback_template(question: str, citations: List[Dict[str, Any]], legal_object: Optional[str]) -> str:
    """
    Conservative non-LLM fallback:
    - Emits correct structure
    - References top citations
    - Avoids inventing steps beyond pointing to evidence
    """
    if not citations:
        return (
            "Applicable provisions:\n"
            "- Insufficient evidence retrieved.\n\n"
            "Step-by-step procedure:\n"
            "- Insufficient evidence retrieved.\n\n"
            "Common mistakes to avoid:\n"
            "- Insufficient evidence retrieved.\n\n"
            "If facts are missing:\n"
            "- Provide the exact section/topic within the source material that governs this question.\n"
        )

    # Prefer top 5 citations for readability
    top = citations[:5]
    app = []
    for c in top:
        app.append(f"- {c.get('title','MML')}: {c.get('snippet','').strip()} [{c['citation_id']}]")

    missing = []
    if legal_object:
        missing.append(f"- Confirm the exact legal object and context (detected: {legal_object}).")
    missing.append("- Provide the rank/appointment, unit level, and the specific case context (where applicable).")
    missing.append("- Provide any referenced rule/section numbers if known.")

    return (
        "Applicable provisions:\n"
        + "\n".join(app)
        + "\n\n"
        "Step-by-step procedure:\n"
        + "\n".join([f"- Refer to the cited provisions for the procedure. [{top[0]['citation_id']}]"])
        + "\n\n"
        "Common mistakes to avoid:\n"
        + "\n".join([f"- Proceeding without verifying applicability of the cited provisions. [{top[0]['citation_id']}]"])
        + "\n\n"
        "If facts are missing:\n"
        + "\n".join(missing)
        + "\n"
    )


def _attempt_reduced_prompt(
    question: str,
    citations: List[Dict[str, Any]],
    legal_object: Optional[str],
    request_id: str,
) -> Tuple[Optional[str], List[Dict[str, str]], List[Dict[str, Any]]]:
    reduced_citations = _shrink_citations_for_timeout(citations)
    if not reduced_citations:
        return None, [], citations

    reduced_prompt = _build_user_prompt(
        question,
        reduced_citations,
        legal_object,
        max_chars_total=min(EVIDENCE_MAX_CHARS_TOTAL, 1500),
        max_chars_per_citation=500,
    )

    try:
        answer = _call_ollama_chat(
            _SYSTEM_PROMPT_BASE,
            reduced_prompt,
            request_id,
            options={"num_predict": min(OLLAMA_NUM_PREDICT, 400), "temperature": OLLAMA_TEMPERATURE},
        )
    except OllamaError as exc:  # pragma: no cover - connectivity dependent
        return None, [
            {
                "code": "TIMEOUT_RETRY_FAILED",
                "message": f"Reduced prompt retry failed: {exc}",
            }
        ], reduced_citations

    warnings: List[Dict[str, str]] = []
    known_ids = {c.get("citation_id", "") for c in reduced_citations if c.get("citation_id")}
    ok, issues, _unknown_ids = _validate_answer(answer, known_ids)
    warnings.extend(issues)

    if ok:
        grounding_ok, failures = verify_grounding(answer, reduced_citations)
        _log_grounding_result(grounding_ok, failures)
        if grounding_ok:
            return answer, warnings, reduced_citations

        patched_answer = _patch_unsupported_bullets(answer, failures, known_ids, reduced_citations)
        warnings.append(
            {
                "code": "GROUNDING_PATCHED",
                "message": "Reduced prompt result patched for insufficient evidence.",
            }
        )
        return patched_answer, warnings, reduced_citations

    return answer, warnings, reduced_citations


# ----------------------------
# Generation helpers
# ----------------------------

def _single_pass_answer(
    question: str,
    citations: List[Dict[str, Any]],
    legal_object: Optional[str],
    known_ids: set[str],
    allowed_ids_line: str,
    user_prompt: str,
    telemetry: Dict[str, Any],
    request_id: str,
) -> Tuple[str, List[Dict[str, str]], List[Dict[str, Any]]]:
    """Existing single-pass generation with validation and fallback."""
    warnings: List[Dict[str, str]] = []
    telemetry.setdefault("retrieval_retry_count", 0)
    telemetry.setdefault("claim_retry_used", False)
    telemetry.setdefault("claim_retry_extra_citations", 0)
    telemetry.setdefault("grounding_failures_initial_count", 0)
    telemetry.setdefault("grounding_failures_after_retry_count", 0)

    try:
        answer = _call_ollama_chat(_SYSTEM_PROMPT_BASE, user_prompt, request_id)
        ok, issues, _unknown_ids = _validate_answer(answer, known_ids)
        if ok:
            grounding_ok, failures = verify_grounding(answer, citations)
            _log_grounding_result(grounding_ok, failures)
            if grounding_ok:
                return answer, warnings, citations

            repaired_answer, repair_success, repair_failures, repair_issues, merged_citations, used_retry = _claim_retry_repair(
                question,
                legal_object,
                None,
                answer,
                citations,
                failures,
                telemetry,
                request_id,
            )
            warnings.extend(repair_issues)
            if used_retry:
                telemetry["claim_retry_used"] = True
            merged_known_ids = {c.get("citation_id", "") for c in merged_citations if c.get("citation_id")}
            if repair_success:
                warnings.append(
                    {
                        "code": "GROUNDING_REPAIRED",
                        "message": "Answer required grounding repair; a corrected answer was produced.",
                    }
                )
                return repaired_answer, warnings, merged_citations

            patched_answer = _patch_unsupported_bullets(repaired_answer, repair_failures, merged_known_ids, merged_citations)
            warnings.append(
                {
                    "code": "GROUNDING_PATCHED",
                    "message": "Unsupported bullets were replaced due to insufficient evidence.",
                }
            )
            return patched_answer, warnings, merged_citations

        warnings.extend(issues)

        corrective = (
            _SYSTEM_PROMPT_BASE
            + "\n\n"
            + "You MUST fix the following issues in your answer:\n"
            + "\n".join([f"- {i['code']}: {i['message']}" for i in issues])
            + "\n\n"
            + allowed_ids_line
            + "\n"
            + "Important: Use ONLY these IDs. Each step/proposition line must end with brackets.\n"
        )

        answer2 = _call_ollama_chat(corrective, user_prompt, request_id)
        ok2, issues2, _unknown2 = _validate_answer(answer2, known_ids)
        if ok2:
            grounding_ok, failures = verify_grounding(answer2, citations)
            _log_grounding_result(grounding_ok, failures)
            if grounding_ok:
                warnings.append(
                    {
                        "code": "REPAIRED_OUTPUT",
                        "message": "Model output required repair; a corrected answer was produced.",
                    }
                )
                return answer2, warnings, citations

            repaired_answer, repair_success, repair_failures, repair_issues, merged_citations, used_retry = _claim_retry_repair(
                question,
                legal_object,
                None,
                answer2,
                citations,
                failures,
                telemetry,
                request_id,
            )
            warnings.extend(repair_issues)
            if used_retry:
                telemetry["claim_retry_used"] = True
            merged_known_ids = {c.get("citation_id", "") for c in merged_citations if c.get("citation_id")}
            if repair_success:
                warnings.append(
                    {
                        "code": "GROUNDING_REPAIRED",
                        "message": "Answer required grounding repair; a corrected answer was produced.",
                    }
                )
                return repaired_answer, warnings, merged_citations

            patched_answer = _patch_unsupported_bullets(repaired_answer, repair_failures, merged_known_ids, merged_citations)
            warnings.append(
                {
                    "code": "GROUNDING_PATCHED",
                    "message": "Unsupported bullets were replaced due to insufficient evidence.",
                }
            )
            return patched_answer, warnings, merged_citations

        warnings.extend(issues2)
        warnings.append({"code": "FALLBACK_USED", "message": "Model output could not be validated. Falling back to conservative template."})
        return _fallback_template(question, citations, legal_object), warnings, citations

    except OllamaTimeoutError as exc:
        warnings.append(
            {
                "code": "LLM_TIMEOUT",
                "message": f"Primary Ollama call timed out after {OLLAMA_READ_TIMEOUT_SECS}s: {exc}",
            }
        )
        best_answer, reduced_warnings, reduced_citations = _attempt_reduced_prompt(
            question, citations, legal_object, request_id
        )
        warnings.extend(reduced_warnings)
        if best_answer:
            warnings.append(
                {
                    "code": "TIMEOUT_RETRY",
                    "message": "Returned answer from reduced prompt after timeout.",
                }
            )
            return best_answer, warnings, reduced_citations

        warnings.append(
            {
                "code": "FALLBACK_USED",
                "message": "Timeout retry failed; returning fallback template.",
            }
        )
        return _fallback_template(question, citations, legal_object), warnings, citations
    except (OllamaConnectionError, OllamaResponseError, RuntimeError) as exc:
        warnings.append(
            {
                "code": "LLM_UNAVAILABLE",
                "message": f"Ollama/LLM unavailable or failed. Falling back to conservative template. Details: {exc}",
            }
        )
        return _fallback_template(question, citations, legal_object), warnings, citations


# ----------------------------
# Answer synthesis pipeline
# ----------------------------

def synthesize_answer_grounded(
    question: str,
    citations: List[Dict[str, Any]],
    legal_object: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Produces a grounded answer using Ollama (if available), with post-generation validation.
    Retries once with a corrective prompt if citations or structure are invalid.
    """
    start_time = time.monotonic()
    request_id = str(uuid.uuid4())
    warnings: List[Dict[str, str]] = []
    telemetry: Dict[str, Any] = {
        "retrieval_retry_count": 0,
        "claim_retry_used": False,
        "claim_retry_extra_citations": 0,
        "grounding_failures_initial_count": 0,
        "grounding_failures_after_retry_count": 0,
        "claim_retry_claims_selected": 0,
        "claim_retry_completed": False,
    }

    if not citations:
        return {
            "answer": _fallback_template(question, citations, legal_object),
            "warnings": [{"code": "NO_EVIDENCE", "message": "No evidence retrieved; returned a conservative template."}],
            "telemetry": telemetry,
        }

    prompt_citations = _select_prompt_citations(citations)
    known_ids = {c.get("citation_id", "") for c in prompt_citations if c.get("citation_id")}
    allowed_ids_line = "Allowed citation IDs: " + ", ".join(sorted(known_ids))

    user_prompt = _build_user_prompt(question, prompt_citations, legal_object)
    evidence_length = len(user_prompt.split("EVIDENCE:\n", maxsplit=1)[-1])
    full_text_length = sum(len((c.get("verbatim", "") or "")) for c in prompt_citations)
    reduction_ratio = (evidence_length / full_text_length) if full_text_length else 0
    logger.info(
        "evidence_pack_built",
        extra={
            "citation_count": len(prompt_citations),
            "total_citations": len(citations),
            "evidence_length": evidence_length,
            "full_text_length": full_text_length,
            "reduction_ratio": round(reduction_ratio, 4) if reduction_ratio else 0,
        },
    )

    logger.info("two_pass_config", extra={"two_pass_enabled": TWO_PASS_ENABLED})

    prompt_chars_total = len(_SYSTEM_PROMPT_BASE) + len(user_prompt)

    if TWO_PASS_ENABLED:
        plan, planner_retry_count, planner_success = _run_planner(
            question, prompt_citations, legal_object, known_ids, request_id
        )
        logger.info(
            "planner_result",
            extra={
                "two_pass_enabled": TWO_PASS_ENABLED,
                "planner_success": planner_success,
                "planner_retry_count": planner_retry_count,
            },
        )

        if plan:
            plan_steps = plan.get("steps", []) if isinstance(plan.get("steps"), list) else []
            plan_point_count = sum(len(step.get("points", [])) for step in plan_steps if isinstance(step, dict))
            logger.info(
                "plan_stats",
                extra={"plan_step_count": len(plan_steps), "plan_point_count": plan_point_count},
            )

            answer, writer_warnings, writer_retry_count, writer_success, used_citations = _run_writer(
                question,
                prompt_citations,
                legal_object,
                plan,
                known_ids,
                allowed_ids_line,
                telemetry,
                request_id,
            )
            warnings.extend(writer_warnings)
            logger.info(
                "writer_result",
                extra={
                    "writer_success": writer_success,
                    "writer_retry_count": writer_retry_count,
                    "plan_step_count": len(plan_steps),
                    "plan_point_count": plan_point_count,
                },
            )

            if writer_success and answer:
                return {"answer": answer, "warnings": warnings, "telemetry": telemetry, "citations": used_citations}

            warnings.append({"code": "FALLBACK_USED", "message": "Writer generation failed validation; falling back to single-pass generation."})

    answer, single_pass_warnings, final_citations = _single_pass_answer(
        question,
        prompt_citations,
        legal_object,
        known_ids,
        allowed_ids_line,
        user_prompt,
        telemetry,
        request_id,
    )
    warnings.extend(single_pass_warnings)

    elapsed = time.monotonic() - start_time
    fallback_used = any(
        w.get("code") in {"FALLBACK_USED", "LLM_UNAVAILABLE", "LLM_TIMEOUT"} for w in warnings
    )
    logger.info(
        "synthesis_request",
        extra={
            "request_id": request_id,
            "model": OLLAMA_MODEL_LEGAL,
            "citations_used_for_prompt": len(prompt_citations),
            "evidence_pack_chars": evidence_length,
            "prompt_chars_total": prompt_chars_total,
            "timeout_read_secs": OLLAMA_READ_TIMEOUT_SECS,
            "elapsed_secs": round(elapsed, 3),
            "fallback_used": fallback_used,
        },
    )

    return {"answer": answer, "warnings": warnings, "telemetry": telemetry, "citations": final_citations}


class StreamingSynthesisHandle:
    """Allows the API route to stream tokens first, then fetch warnings/telemetry after completion."""

    def __init__(
        self,
        question: str,
        citations: List[Dict[str, Any]],
        legal_object: Optional[str],
    ) -> None:
        self.question = question
        self.citations = citations
        self.legal_object = legal_object

        self.request_id = str(uuid.uuid4())
        self._answer_parts: List[str] = []
        self._warnings: List[Dict[str, str]] = []
        self._telemetry: Dict[str, Any] = {
            "streaming": True,
            "two_pass_used": False,
            "grounding_checked": False,
            "validation_checked": False,
        }
        self._final_citations: List[Dict[str, Any]] = []
        self._done: bool = False
        self._started_at = time.monotonic()

    def stream(self) -> Iterator[str]:
        """Yield answer chunks. This method must be consumed to completion before result() is meaningful."""

        if self._done:
            # Do not stream twice
            return

        if not self.citations:
            txt = _fallback_template(self.question, self.citations, self.legal_object)
            self._warnings.append({"code": "NO_EVIDENCE", "message": "No evidence retrieved; returned a conservative template."})
            self._answer_parts.append(txt)
            self._final_citations = []
            self._done = True
            yield txt
            return

        # Keep prompt building identical to non-streaming path (single-pass only)
        prompt_citations = _select_prompt_citations(self.citations)
        self._final_citations = prompt_citations
        known_ids = {c.get("citation_id", "") for c in prompt_citations if c.get("citation_id")}
        allowed_ids_line = "Allowed citation IDs: " + ", ".join(sorted(known_ids))

        user_prompt = _build_user_prompt(self.question, prompt_citations, self.legal_object)
        evidence_length = len(user_prompt.split("EVIDENCE:\n", maxsplit=1)[-1])
        full_text_length = sum(len((c.get("verbatim", "") or "")) for c in prompt_citations)
        reduction_ratio = (evidence_length / full_text_length) if full_text_length else 0
        logger.info(
            "evidence_pack_built",
            extra={
                "request_id": self.request_id,
                "citation_count": len(prompt_citations),
                "total_citations": len(self.citations),
                "evidence_length": evidence_length,
                "full_text_length": full_text_length,
                "reduction_ratio": round(reduction_ratio, 4) if reduction_ratio else 0,
                "streaming": True,
            },
        )

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT_BASE},
            {
                "role": "user",
                "content": f"{allowed_ids_line}\n\n{user_prompt}",
            },
        ]

        try:
            for chunk in ollama_chat_stream(
                model=OLLAMA_MODEL_LEGAL,
                messages=messages,
                options={
                    "temperature": OLLAMA_TEMPERATURE,
                    "num_predict": OLLAMA_NUM_PREDICT,
                },
                request_id=self.request_id,
            ):
                self._answer_parts.append(chunk)
                yield chunk
        except OllamaTimeoutError as exc:
            self._warnings.append({"code": "LLM_TIMEOUT", "message": str(exc)})
            txt = _fallback_template(self.question, prompt_citations, self.legal_object)
            self._answer_parts = [txt]
            yield txt
        except (OllamaConnectionError, OllamaResponseError, OllamaError) as exc:
            self._warnings.append({"code": "LLM_UNAVAILABLE", "message": str(exc)})
            txt = _fallback_template(self.question, prompt_citations, self.legal_object)
            self._answer_parts = [txt]
            yield txt
        except Exception as exc:  # pragma: no cover
            self._warnings.append({"code": "UNEXPECTED_ERROR", "message": f"Streaming synthesis failed: {exc}"})
            txt = _fallback_template(self.question, prompt_citations, self.legal_object)
            self._answer_parts = [txt]
            yield txt
        finally:
            self._done = True

            # Post-stream validation and grounding checks (cannot repair after streaming)
            answer_text = "".join(self._answer_parts).strip()
            try:
                ok, issues, unknown_ids = _validate_answer(answer_text, known_ids)
                self._telemetry["validation_checked"] = True
                if not ok:
                    self._warnings.append({
                        "code": "VALIDATION_WARN",
                        "message": f"Answer failed strict validation. Issues: {', '.join(i.get('code','') for i in issues)}",
                    })
                    if unknown_ids:
                        self._warnings.append({
                            "code": "UNKNOWN_CITATION_IDS",
                            "message": "Answer contains citation IDs not present in evidence.",
                        })

                grounding_ok, failures = verify_grounding(answer_text, prompt_citations)
                self._telemetry["grounding_checked"] = True
                self._telemetry["grounding_failures"] = len(failures)
                if not grounding_ok:
                    self._warnings.append({
                        "code": "GROUNDING_WARN",
                        "message": "One or more procedure bullets could not be supported by cited evidence. Provide additional evidence or narrow the question.",
                    })
            except Exception as exc:  # pragma: no cover
                self._warnings.append({"code": "POSTCHECK_ERROR", "message": f"Post-stream checks failed: {exc}"})

            elapsed = time.monotonic() - self._started_at
            logger.info(
                "synthesis_request",
                extra={
                    "request_id": self.request_id,
                    "model": OLLAMA_MODEL_LEGAL,
                    "citations_used_for_prompt": len(prompt_citations),
                    "prompt_chars_total": len(_SYSTEM_PROMPT_BASE) + len(user_prompt) + len(allowed_ids_line),
                    "timeout_read_secs": OLLAMA_READ_TIMEOUT_SECS,
                    "elapsed_secs": round(elapsed, 3),
                    "streaming": True,
                },
            )

    def result(self) -> Dict[str, Any]:
        """Return warnings/telemetry after streaming has completed."""
        answer_text = "".join(self._answer_parts).strip()
        return {
            "answer": answer_text,
            "warnings": self._warnings,
            "telemetry": self._telemetry,
            "citations": self._final_citations,
            "request_id": self.request_id,
        }


def synthesize_answer_grounded_stream(
    question: str,
    citations: List[Dict[str, Any]],
    legal_object: Optional[str] = None,
) -> StreamingSynthesisHandle:
    """Streaming variant for /chat/stream. Uses single-pass generation and streams Ollama tokens end-to-end."""

    return StreamingSynthesisHandle(question=question, citations=citations, legal_object=legal_object)
