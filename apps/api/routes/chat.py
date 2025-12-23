from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Dict, Iterator, List, Literal, Optional

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from apps.api.config import OLLAMA_MODEL_LEGAL, SYNTHESIS_MAX_CITATIONS
from apps.api.services.citation_store import upsert_citations
from apps.api.services.expand_queries import expand_queries
from apps.api.services.ollama_client import (
    OllamaConnectionError,
    OllamaResponseError,
    OllamaTimeoutError,
    ollama_chat_stream,
)
from apps.api.services.retrieval import retrieve_citations_multi
from apps.api.services.retrieval_repo import retrieve_repo_citations
from apps.api.services.router import route_domain
from apps.api.services.synthesis import _SYSTEM_PROMPT_BASE, _build_user_prompt, _fallback_template
from apps.api.services.synthesis_repo import synthesize_repo_answer_grounded
from apps.api.services.grounding_verify import verify_grounding

logger = logging.getLogger(__name__)
router = APIRouter()


class ChatRequest(BaseModel):
    case_id: Optional[str] = "default"
    question: str
    mode: Literal["Chat", "Search"] = "Chat"
    show_citations: bool = True
    show_warnings: bool = True


def sse_event(event: str, data: Dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("/chat/stream")
def chat_stream(req: ChatRequest) -> StreamingResponse:
    question = (req.question or "").strip()
    if not question:
        def _empty() -> Iterator[str]:
            yield sse_event("token", {"text": "Please enter a question."})
            yield sse_event("citations", {"items": []})
            yield sse_event("warnings", {"items": []})
            yield sse_event("done", {"ok": True})
        return StreamingResponse(_empty(), media_type="text/event-stream")

    # Decide route (legal vs repository)
    intent = route_domain(question)

    def gen() -> Iterator[str]:
        yield sse_event("meta", {"legal_object": intent.legal_object, "domain": intent.domain})

        warnings: List[Dict[str, str]] = []
        citations: List[Dict[str, Any]] = []

        # Repository domain: keep existing behaviour
        if intent.domain == "repo":
            try:
                queries = expand_queries(question, intent.legal_object)
                citations = retrieve_repo_citations(questions=queries, top_k=10)
                upsert_citations(req.case_id or "default", citations)

                answer, synth_warnings = synthesize_repo_answer_grounded(
                    question=question,
                    citations=citations,
                    request_id=req.case_id or "default",
                )
                if req.show_warnings:
                    warnings.extend(synth_warnings)

                for ch in answer:
                    yield sse_event("token", {"text": ch})
                    time.sleep(0.001)

            except Exception as exc:
                yield sse_event("warnings", {"items": [{"code": "UNEXPECTED_ERROR", "message": str(exc)}]})
                yield sse_event("done", {"ok": False})
                return

            if req.show_citations:
                yield sse_event(
                    "citations",
                    {
                        "items": [
                            {
                                "citation_id": c["citation_id"],
                                "title": c.get("title", "Repository"),
                                "source_file": c.get("source_file", ""),
                                "heading": c.get("heading", ""),
                                "snippet": c.get("snippet", ""),
                            }
                            for c in citations
                        ]
                    },
                )
            yield sse_event("warnings", {"items": warnings if req.show_warnings else []})
            yield sse_event("done", {"ok": True})
            return

        # Legal domain: true end-to-end streaming from Ollama
        try:
            queries = expand_queries(question, intent.legal_object)
            citations = retrieve_citations_multi(
                questions=queries,
                top_k=10,
                legal_object=intent.legal_object,
            )

            upsert_citations(req.case_id or "default", citations)

            prompt_citations = citations[: max(0, SYNTHESIS_MAX_CITATIONS)]
            messages = [
                {"role": "system", "content": _SYSTEM_PROMPT_BASE},
                {"role": "user", "content": _build_user_prompt(question, prompt_citations, intent.legal_object)},
            ]

            answer_parts: List[str] = []
            for chunk in ollama_chat_stream(
                model=OLLAMA_MODEL_LEGAL,
                messages=messages,
                options={},
                request_id=req.case_id or "default",
            ):
                answer_parts.append(chunk)
                yield sse_event("token", {"text": chunk})

        except (OllamaTimeoutError, OllamaConnectionError, OllamaResponseError) as exc:
            if req.show_warnings:
                warnings.append(
                    {
                        "code": "LLM_UNAVAILABLE",
                        "message": f"Ollama/LLM unavailable or failed. Falling back to conservative template. Details: {exc}",
                    }
                )
            fallback = _fallback_template(question, citations, intent.legal_object)
            answer_parts = [fallback]
            for ch in fallback:
                yield sse_event("token", {"text": ch})
                time.sleep(0.001)

        except Exception as exc:
            yield sse_event("warnings", {"items": [{"code": "UNEXPECTED_ERROR", "message": f"{exc}"}]})
            yield sse_event("done", {"ok": False})
            return

        answer_text = ("".join(answer_parts) or "").strip()

        # Post-hoc grounding checks
        if answer_text:
            prompt_citations = citations[: max(0, SYNTHESIS_MAX_CITATIONS)]
            known_ids = {c.get("citation_id") for c in prompt_citations if c.get("citation_id")}

            for cid in set(re.findall(r"\[(LGL-[A-Za-z0-9-]+)\]", answer_text)):
                if cid not in known_ids and req.show_warnings:
                    warnings.append(
                        {
                            "code": "UNKNOWN_CITATION_ID",
                            "message": f"Answer referenced citation id not present in evidence pack: {cid}",
                        }
                    )

            gv = verify_grounding(answer_text, prompt_citations)
            failed = gv.get("failed") or []
            if failed and req.show_warnings:
                warnings.append(
                    {
                        "code": "GROUNDING_CHECK_FAILED",
                        "message": f"Some bullets were not supported by the provided evidence (count={len(failed)}). Consider narrowing the question or providing more specific source material.",
                    }
                )

        if not answer_text:
            answer_text = (
                "Insufficient evidence in the retrieved material to provide a grounded answer. "
                "Please refine the question or provide the relevant document section."
            )
            if req.show_warnings:
                warnings.append(
                    {"code": "INSUFFICIENT_EVIDENCE", "message": "No grounded answer could be produced from retrieved evidence."}
                )
            for ch in answer_text:
                yield sse_event("token", {"text": ch})
                time.sleep(0.001)

        if req.show_citations:
            yield sse_event(
                "citations",
                {
                    "items": [
                        {
                            "citation_id": c["citation_id"],
                            "title": c.get("title", "MML"),
                            "source_file": c.get("source_file", ""),
                            "heading": c.get("heading", ""),
                            "snippet": c.get("snippet", ""),
                        }
                        for c in citations
                    ]
                },
            )

        yield sse_event("warnings", {"items": warnings if req.show_warnings else []})
        yield sse_event("done", {"ok": True})

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
