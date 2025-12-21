from __future__ import annotations

from typing import List, Dict, Any, Optional, Tuple
import os
import re
import requests


OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")


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


def _build_user_prompt(question: str, citations: List[Dict[str, Any]], legal_object: Optional[str]) -> str:
    legal_line = f"Requested legal object: {legal_object}" if legal_object else "Requested legal object: (not specified)"
    ev_lines: List[str] = []
    for c in citations:
        cid = c.get("citation_id", "")
        title = c.get("title", "MML")
        src = c.get("source_file", "")
        heading = c.get("heading", "")
        verbatim = (c.get("verbatim", "") or "").strip()
        ev_lines.append(
            f"[{cid}] {title}\nSource: {src}\nHeading: {heading}\nVerbatim:\n{verbatim}\n"
        )

    evidence_block = "\n---\n".join(ev_lines) if ev_lines else "(no evidence retrieved)"

    return (
        f"{legal_line}\n\n"
        f"Question:\n{question.strip()}\n\n"
        f"EVIDENCE:\n{evidence_block}\n"
    )


_CITATION_BRACKET_RE = re.compile(r"\[([^\[\]]+)\]")


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

    # Attempt 1
    try:
        answer = _call_ollama_chat(_SYSTEM_PROMPT_BASE, user_prompt)
        ok, issues, unknown_ids = _validate_answer(answer, known_ids)
        if ok:
            return {"answer": answer, "warnings": warnings}

        warnings.extend(issues)

        # Attempt 2: corrective prompt
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
            warnings.append({"code": "REPAIRED_OUTPUT", "message": "Model output required repair; a corrected answer was produced."})
            return {"answer": answer2, "warnings": warnings}

        warnings.extend(issues2)
        warnings.append({"code": "FALLBACK_USED", "message": "Model output could not be validated. Falling back to conservative template."})
        return {"answer": _fallback_template(question, citations, legal_object), "warnings": warnings}

    except Exception as e:
        warnings.append(
            {
                "code": "LLM_UNAVAILABLE",
                "message": f"Ollama/LLM unavailable or failed. Falling back to conservative template. Details: {e}",
            }
        )
        return {"answer": _fallback_template(question, citations, legal_object), "warnings": warnings}
