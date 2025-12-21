import json
import time
from typing import List, Literal, Optional, Dict, Any, Iterator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from apps.api.services.intent import classify_legal_object
from apps.api.services.retrieval import retrieve_citations
from apps.api.services.synthesis import synthesize_answer_grounded
from apps.api.services.citation_store import citation_store

router = APIRouter()

Role = Literal["system", "user", "assistant"]


class ChatMessage(BaseModel):
    role: Role
    content: str


class ChatRequest(BaseModel):
    case_id: Optional[str] = "default"
    messages: List[ChatMessage]
    mode: Optional[str] = "Chat"
    want_citations: Optional[bool] = True
    want_warnings: Optional[bool] = True


def sse_event(event: str, data: Dict[str, Any]) -> bytes:
    """
    SSE format:
      event: <name>
      data: <json>
    """
    payload = f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
    return payload.encode("utf-8")


@router.post("/chat/stream")
def chat_stream(req: ChatRequest):
    def gen() -> Iterator[bytes]:
        # Identify last user message
        question = ""
        for m in reversed(req.messages):
            if m.role == "user":
                question = (m.content or "").strip()
                break

        if not question:
            yield sse_event("meta", {"case_id": req.case_id, "mode": req.mode})
            for ch in "No question provided.":
                yield sse_event("token", {"text": ch})
            yield sse_event("citations", {"items": []})
            yield sse_event("warnings", {"items": [{"code": "NO_QUESTION", "message": "No question provided."}]})
            yield sse_event("done", {"ok": True})
            return

        # Intent gating (prevents COI vs Court-Martial drift)
        intent = classify_legal_object(question)
        yield sse_event(
            "meta",
            {
                "case_id": req.case_id,
                "mode": req.mode,
                "legal_object": intent.legal_object,
                "intent_confidence": intent.confidence,
            },
        )

        # If ambiguous, ask a clarification question instead of guessing.
        if intent.needs_clarification and intent.clarification_question:
            clarification = intent.clarification_question.strip()
            for ch in clarification:
                yield sse_event("token", {"text": ch})
                time.sleep(0.001)

            if req.want_citations:
                yield sse_event("citations", {"items": []})

            warnings = [
                {
                    "code": "INTENT_AMBIGUOUS",
                    "message": f"Ambiguous legal object. Asked clarification instead of guessing. Detected: {intent.legal_object}.",
                }
            ]
            if req.want_warnings:
                yield sse_event("warnings", {"items": warnings})
            else:
                yield sse_event("warnings", {"items": []})

            yield sse_event("done", {"ok": True})
            return

        # --- Retrieval and synthesis ---
        try:
            # 1) Retrieve evidence (RAG), constrained by legal object (soft)
            retrieval_query = intent.normalized_query
            if intent.legal_object == "Court of Inquiry":
                # Expand acronym to improve recall and reduce Court-Martial drift.
                if "court of inquiry" not in retrieval_query:
                    retrieval_query = retrieval_query.replace("coi", "court of inquiry")
                    retrieval_query = retrieval_query + " court of inquiry"
            citations = retrieve_citations(
                question=retrieval_query,
                top_k=10,
                legal_object=intent.legal_object,
            )

            # Register citations into store so UI can fetch verbatim by citation_id
            for c in citations:
                citation_store.upsert(c)

            # 2) Synthesize answer (grounded + validated)
            result = synthesize_answer_grounded(
                question=question,
                citations=citations,
                legal_object=intent.legal_object,
            )
        except Exception as exc:  # pragma: no cover - defensive guard for unexpected failures
            if req.want_warnings:
                yield sse_event(
                    "warnings",
                    {
                        "items": [
                            {
                                "code": "UNEXPECTED_ERROR",
                                "message": f"Failed during retrieval/synthesis: {exc}",
                            }
                        ]
                    },
                )
            else:
                yield sse_event("warnings", {"items": []})

            yield sse_event("done", {"ok": False})
            return

        answer_text = (result.get("answer", "") or "").strip()
        warnings = result.get("warnings", []) or []

        if not answer_text:
            answer_text = (
                "Insufficient evidence in the retrieved material to provide a grounded answer. "
                "Please refine the question or provide the relevant document section."
            )
            warnings.append(
                {
                    "code": "INSUFFICIENT_EVIDENCE",
                    "message": "No grounded answer could be produced from retrieved evidence.",
                }
            )

        # Stream token-by-token (char) for PoC
        for ch in answer_text:
            yield sse_event("token", {"text": ch})
            time.sleep(0.001)

        # Return citations (card list)
        if req.want_citations:
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

        if req.want_warnings:
            yield sse_event("warnings", {"items": warnings})
        else:
            yield sse_event("warnings", {"items": []})

        yield sse_event("done", {"ok": True})

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
