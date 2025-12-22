from __future__ import annotations

import os
import re
import logging
from typing import Any, Dict, List, Set, Tuple

import requests

# ============================
# Configuration
# ============================
OLLAMA_URL = os.getenv("OLLAMA_URL") or os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
OLLAMA_HEALTH_TIMEOUT = float(os.getenv("OLLAMA_HEALTH_TIMEOUT", "5"))
OLLAMA_READ_TIMEOUT = float(os.getenv("OLLAMA_READ_TIMEOUT", "45"))
_OLLAMA_HEALTHY: bool | None = None

# ============================
# Prompt Builders
# ============================
_SYSTEM_PROMPT = """
You are an operations assistant. Answer questions about running and troubleshooting this repository using ONLY the provided SOURCES. Keep responses concise and actionable.

Rules:
- Include the headings: Answer, Commands, Troubleshooting, Sources.
- Every bullet under Commands and Troubleshooting must end with one or more SYS citation IDs in brackets, e.g., [SYS-abc123].
- If evidence is insufficient, state that clearly and cite the closest available SYS evidence.
""".strip()


def _build_user_prompt(question: str, citations: List[Dict[str, Any]]) -> str:
    evidence_lines: List[str] = []
    for c in citations:
        cid = c.get("citation_id", "")
        heading = c.get("heading", "")
        src = c.get("source_file", "")
        verbatim = (c.get("verbatim", "") or c.get("text", "") or "").strip()
        evidence_lines.append(f"[{cid}] {heading}\nSource: {src}\nVerbatim:\n{verbatim}\n")

    evidence_block = "\n---\n".join(evidence_lines) if evidence_lines else "(no evidence retrieved)"
    return (
        f"Question:\n{question.strip()}\n\n"
        f"SOURCES:\n{evidence_block}\n"
    )


# ============================
# Validation Helpers
# ============================
_CITATION_RE = re.compile(r"\[(SYS-[^\[\]]+)\]")


def _validate_answer(answer: str, known_ids: Set[str]) -> Tuple[bool, List[str]]:
    issues: List[str] = []
    txt = (answer or "").strip()
    if not txt:
        issues.append("empty output")
        return False, issues

    required_headings = ["Answer", "Commands", "Troubleshooting", "Sources"]
    missing = [h for h in required_headings if h not in txt]
    if missing:
        issues.append(f"missing headings: {', '.join(missing)}")

    used_ids = [m.strip() for m in _CITATION_RE.findall(txt)]
    unknown = [u for u in used_ids if u not in known_ids]
    if unknown:
        issues.append(f"unknown citations: {', '.join(unknown)}")

    for line in txt.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("-") and ("Commands" in txt or "Troubleshooting" in txt):
            if not s.endswith("]"):
                issues.append(f"bullet missing citation: {s[:80]}")
                break

    return len(issues) == 0, issues


# ============================
# LLM connectivity helpers
# ============================
logger = logging.getLogger(__name__)

def _ollama_available() -> bool:
    """Check if Ollama endpoint is reachable before requesting completions."""

    global _OLLAMA_HEALTHY

    if _OLLAMA_HEALTHY is True:
        return True

    try:
        resp = requests.get(
            f"{OLLAMA_URL}/api/tags",
            timeout=(OLLAMA_HEALTH_TIMEOUT, OLLAMA_HEALTH_TIMEOUT),
        )
        resp.raise_for_status()
        _OLLAMA_HEALTHY = True
        return True
    except Exception as exc:  # pragma: no cover - connectivity dependent
        logger.warning("Ollama health check failed: %s", exc)
        _OLLAMA_HEALTHY = False
        return False


# ============================
# LLM and Fallback Helpers
# ============================

def _call_ollama_chat(system_prompt: str, user_prompt: str) -> str:
    if not _ollama_available():
        raise RuntimeError(
            f"Ollama not reachable at {OLLAMA_URL}. Set OLLAMA_URL or start the service."
        )

    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "options": {"temperature": 0.2},
    }
    resp = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json=payload,
        timeout=(OLLAMA_HEALTH_TIMEOUT, OLLAMA_READ_TIMEOUT),
    )
    resp.raise_for_status()
    data = resp.json()
    return (data.get("message", {}) or {}).get("content", "").strip()


def _fallback_answer(question: str, citations: List[Dict[str, Any]]) -> str:
    if not citations:
        return (
            "Answer\n"
            "- Insufficient system-help evidence retrieved. Provide more details about the task.\n\n"
            "Commands\n"
            "- No commands available due to missing evidence.\n\n"
            "Troubleshooting\n"
            "- Supply additional context or ensure runbook ingestion has been run.\n\n"
            "Sources\n"
            "- None available.\n"
        )

    bullets = [f"- {c.get('snippet','').strip()} [{c['citation_id']}]" for c in citations[:3]]
    return (
        "Answer\n"
        + "- Refer to the system help commands below.\n\n"
        + "Commands\n"
        + "\n".join(bullets)
        + "\n\nTroubleshooting\n"
        + "- Use the cited steps to resolve common issues. "
        + f"[{citations[0]['citation_id']}]\n\n"
        + "Sources\n"
        + "\n".join([f"- {c['citation_id']} ({c.get('heading','')})" for c in citations[:3]])
        + "\n"
    )


# ============================
# Public API
# ============================

def synthesize_repo_answer_grounded(question: str, citations: List[Dict[str, Any]]) -> str:
    known_ids = {c.get("citation_id", "") for c in citations if c.get("citation_id")}
    if not known_ids:
        return _fallback_answer(question, citations)

    user_prompt = _build_user_prompt(question, citations)
    answer = _call_ollama_chat(_SYSTEM_PROMPT, user_prompt)
    ok, issues = _validate_answer(answer, known_ids)

    if not ok:
        retry_prompt = _SYSTEM_PROMPT + "\nAlways end every bullet with a SYS citation and include all required headings."
        answer = _call_ollama_chat(retry_prompt, user_prompt)
        ok, issues = _validate_answer(answer, known_ids)

    if ok:
        return answer

    # Fallback to conservative template
    snippets = [f"- {c.get('heading','')}: {c.get('snippet','').strip()} [{c['citation_id']}]" for c in citations[:5]]
    return (
        "Answer\n"
        "- Unable to produce a validated response. Showing retrieved evidence.\n\n"
        "Commands\n"
        + "\n".join(snippets or ["- No evidence available."])
        + "\n\nTroubleshooting\n"
        + "- Provide more context or verify the runbook index is built. "
        + f"[{next(iter(known_ids))}]\n\n"
        + "Sources\n"
        + "\n".join(snippets or ["- No sources."])
        + "\n"
    )
