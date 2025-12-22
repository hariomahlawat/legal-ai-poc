from __future__ import annotations

from typing import List, Dict, Any, Optional, Tuple
import json
import logging
import os
import re
import requests


OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
TWO_PASS_ENABLED = os.getenv("TWO_PASS_ENABLED", "true").lower() != "false"

from .evidence_packer import build_evidence_pack
from .grounding_verify import verify_grounding

logger = logging.getLogger(__name__)


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


def _build_user_prompt(question: str, citations: List[Dict[str, Any]], legal_object: Optional[str]) -> str:
    legal_line = f"Requested legal object: {legal_object}" if legal_object else "Requested legal object: (not specified)"
    evidence_block = build_evidence_pack(question, citations) if citations else "(no evidence retrieved)"

    return (
        f"{legal_line}\n\n"
        f"Question:\n{question.strip()}\n\n"
        f"EVIDENCE:\n{evidence_block}\n"
    )


_CITATION_BRACKET_RE = re.compile(r"\[([^\[\]]+)\]")


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


def _patch_unsupported_bullets(
    answer: str, failures: List[Dict[str, Any]], known_ids: set[str]
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
                best_id = sorted(known_ids)[0] if known_ids else None
            citation_block = f" [{best_id}]" if best_id else ""
            lines[line_idx] = (
                "- Insufficient evidence in the provided corpus to state this step precisely." + citation_block
            )
        bullet_counter += 1

    return "\n".join(lines)


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
        "If evidence is insufficient, explicitly say so and cite the closest relevant citation ID."
    )

    user_prompt = (
        f"{base_user_prompt}\n\n"
        f"{plan_block}"
        f"Grounding failures:\n{chr(10).join(failures_desc)}\n\n"
        f"{corrective_instruction}"
    )

    repaired_answer = _call_ollama_chat(_SYSTEM_PROMPT_BASE, user_prompt)
    ok_format, issues, _unknown = _validate_answer(repaired_answer, known_ids)
    repair_failures = failures
    grounding_ok = False
    if ok_format:
        grounding_ok, repair_failures = verify_grounding(repaired_answer, citations)

    _log_grounding_result(grounding_ok, repair_failures, repair_attempted=True, repair_success=grounding_ok and ok_format)
    return repaired_answer, (ok_format and grounding_ok), repair_failures, issues


def _call_ollama_chat(system_prompt: str, user_prompt: str) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "options": {"temperature": 0.2},
    }

    r = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=90)
    r.raise_for_status()
    data = r.json()
    return (data.get("message", {}) or {}).get("content", "").strip()


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
) -> Tuple[Optional[Dict[str, Any]], int, bool]:
    """Run planner with a single retry on JSON/validation failure."""
    planner_retry_count = 0
    user_prompt = _build_planner_prompt(question, citations, legal_object, allowed_ids)

    for attempt in range(2):
        try:
            raw_plan = _call_ollama_chat(_SYSTEM_PROMPT_PLANNER, user_prompt)
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
) -> Tuple[Optional[str], List[Dict[str, str]], int, bool]:
    """Generate answer using validated plan and run validation with one retry."""
    warnings: List[Dict[str, str]] = []
    writer_retry_count = 0
    plan_json = json.dumps(plan, ensure_ascii=False, indent=2)
    base_user_prompt = _build_user_prompt(question, citations, legal_object)
    user_prompt = (
        f"{base_user_prompt}\n\nValidated plan (JSON):\n{plan_json}\n\n"
        "Follow the plan strictly. Use only allowed citation IDs. Ensure every bullet in Step-by-step ends with citations in brackets."
    )

    try:
        answer = _call_ollama_chat(_SYSTEM_PROMPT_BASE, user_prompt)
        ok, issues, _unknown = _validate_answer(answer, known_ids)
        if ok:
            grounding_ok, failures = verify_grounding(answer, citations)
            _log_grounding_result(grounding_ok, failures)
            if grounding_ok:
                return answer, warnings, writer_retry_count, True

            repaired_answer, repair_success, repair_failures, repair_issues = _attempt_grounding_repair(
                question,
                citations,
                legal_object,
                plan,
                known_ids,
                failures,
            )
            warnings.extend(repair_issues)
            if repair_success:
                warnings.append(
                    {
                        "code": "GROUNDING_REPAIRED",
                        "message": "Answer required grounding repair; a corrected answer was produced.",
                    }
                )
                return repaired_answer, warnings, writer_retry_count, True

            patched_answer = _patch_unsupported_bullets(repaired_answer, repair_failures, known_ids)
            warnings.append(
                {
                    "code": "GROUNDING_PATCHED",
                    "message": "Unsupported bullets were replaced due to insufficient evidence.",
                }
            )
            return patched_answer, warnings, writer_retry_count, True

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
        answer2 = _call_ollama_chat(corrective, user_prompt)
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
                return answer2, warnings, writer_retry_count, True

            repaired_answer, repair_success, repair_failures, repair_issues = _attempt_grounding_repair(
                question,
                citations,
                legal_object,
                plan,
                known_ids,
                failures,
            )
            warnings.extend(repair_issues)
            if repair_success:
                warnings.append(
                    {
                        "code": "GROUNDING_REPAIRED",
                        "message": "Answer required grounding repair; a corrected answer was produced.",
                    }
                )
                return repaired_answer, warnings, writer_retry_count, True

            patched_answer = _patch_unsupported_bullets(repaired_answer, repair_failures, known_ids)
            warnings.append(
                {
                    "code": "GROUNDING_PATCHED",
                    "message": "Unsupported bullets were replaced due to insufficient evidence.",
                }
            )
            return patched_answer, warnings, writer_retry_count, True

        warnings.extend(issues2)
        return answer2, warnings, writer_retry_count, False
    except Exception as e:
        warnings.append(
            {
                "code": "LLM_UNAVAILABLE",
                "message": f"Ollama/LLM unavailable or failed. Falling back to conservative template. Details: {e}",
            }
        )
        return None, warnings, writer_retry_count, False


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
) -> Tuple[str, List[Dict[str, str]]]:
    """Existing single-pass generation with validation and fallback."""
    warnings: List[Dict[str, str]] = []

    try:
        answer = _call_ollama_chat(_SYSTEM_PROMPT_BASE, user_prompt)
        ok, issues, _unknown_ids = _validate_answer(answer, known_ids)
        if ok:
            grounding_ok, failures = verify_grounding(answer, citations)
            _log_grounding_result(grounding_ok, failures)
            if grounding_ok:
                return answer, warnings

            repaired_answer, repair_success, repair_failures, repair_issues = _attempt_grounding_repair(
                question,
                citations,
                legal_object,
                None,
                known_ids,
                failures,
            )
            warnings.extend(repair_issues)
            if repair_success:
                warnings.append(
                    {
                        "code": "GROUNDING_REPAIRED",
                        "message": "Answer required grounding repair; a corrected answer was produced.",
                    }
                )
                return repaired_answer, warnings

            patched_answer = _patch_unsupported_bullets(repaired_answer, repair_failures, known_ids)
            warnings.append(
                {
                    "code": "GROUNDING_PATCHED",
                    "message": "Unsupported bullets were replaced due to insufficient evidence.",
                }
            )
            return patched_answer, warnings

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

        answer2 = _call_ollama_chat(corrective, user_prompt)
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
                return answer2, warnings

            repaired_answer, repair_success, repair_failures, repair_issues = _attempt_grounding_repair(
                question,
                citations,
                legal_object,
                None,
                known_ids,
                failures,
            )
            warnings.extend(repair_issues)
            if repair_success:
                warnings.append(
                    {
                        "code": "GROUNDING_REPAIRED",
                        "message": "Answer required grounding repair; a corrected answer was produced.",
                    }
                )
                return repaired_answer, warnings

            patched_answer = _patch_unsupported_bullets(repaired_answer, repair_failures, known_ids)
            warnings.append(
                {
                    "code": "GROUNDING_PATCHED",
                    "message": "Unsupported bullets were replaced due to insufficient evidence.",
                }
            )
            return patched_answer, warnings

        warnings.extend(issues2)
        warnings.append({"code": "FALLBACK_USED", "message": "Model output could not be validated. Falling back to conservative template."})
        return _fallback_template(question, citations, legal_object), warnings

    except Exception as e:
        warnings.append(
            {
                "code": "LLM_UNAVAILABLE",
                "message": f"Ollama/LLM unavailable or failed. Falling back to conservative template. Details: {e}",
            }
        )
        return _fallback_template(question, citations, legal_object), warnings


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
    warnings: List[Dict[str, str]] = []

    if not citations:
        return {
            "answer": _fallback_template(question, citations, legal_object),
            "warnings": [{"code": "NO_EVIDENCE", "message": "No evidence retrieved; returned a conservative template."}],
        }

    known_ids = {c.get("citation_id", "") for c in citations if c.get("citation_id")}
    allowed_ids_line = "Allowed citation IDs: " + ", ".join(sorted(known_ids))

    user_prompt = _build_user_prompt(question, citations, legal_object)
    evidence_length = len(user_prompt.split("EVIDENCE:\n", maxsplit=1)[-1])
    full_text_length = sum(len((c.get("verbatim", "") or "")) for c in citations)
    reduction_ratio = (evidence_length / full_text_length) if full_text_length else 0
    logger.info(
        "evidence_pack_built",
        extra={
            "citation_count": len(citations),
            "evidence_length": evidence_length,
            "full_text_length": full_text_length,
            "reduction_ratio": round(reduction_ratio, 4) if reduction_ratio else 0,
        },
    )

    logger.info("two_pass_config", extra={"two_pass_enabled": TWO_PASS_ENABLED})

    if TWO_PASS_ENABLED:
        plan, planner_retry_count, planner_success = _run_planner(question, citations, legal_object, known_ids)
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

            answer, writer_warnings, writer_retry_count, writer_success = _run_writer(
                question,
                citations,
                legal_object,
                plan,
                known_ids,
                allowed_ids_line,
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
                return {"answer": answer, "warnings": warnings}

            warnings.append({"code": "FALLBACK_USED", "message": "Writer generation failed validation; falling back to single-pass generation."})

    answer, single_pass_warnings = _single_pass_answer(
        question,
        citations,
        legal_object,
        known_ids,
        allowed_ids_line,
        user_prompt,
    )
    warnings.extend(single_pass_warnings)
    return {"answer": answer, "warnings": warnings}
