from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

# === Module Imports ===
from apps.api.services.citation_store import citation_store

router = APIRouter()


class CitationDetail(BaseModel):
    citation_id: str
    document: str | None = None
    source_file: str
    heading: str | None = None
    heading_path: str
    location: str | None = None
    verbatim: str | None = None
    text: str
    context_before: str | None = None
    context_after: str | None = None
    snippet: str | None = None
    retrieval_score: float | None = None
    rerank_score: float | None = None
    hit_query_count: int | None = None
    focus_applied: bool | None = None
    domain: str | None = None

    class Config:
        extra = "allow"


@router.get("/citations/{citation_id}", response_model=CitationDetail)
def get_citation(citation_id: str):
    item = citation_store.get(citation_id)
    if not item:
        raise HTTPException(status_code=404, detail="Citation not found")

    # === Response Normalization ===
    normalized = dict(item)
    heading_path = normalized.get("heading_path") or normalized.get("location") or ""
    normalized["heading_path"] = heading_path

    text = normalized.get("text") or normalized.get("verbatim") or ""
    normalized["text"] = text
    normalized.setdefault("retrieval_score", normalized.get("score"))
    normalized.setdefault("rerank_score", None)
    normalized.setdefault("hit_query_count", None)
    normalized.setdefault("focus_applied", None)
    if not normalized.get("domain"):
        doc = normalized.get("document")
        if doc == "SYS":
            normalized["domain"] = "SYS"
        elif doc:
            normalized["domain"] = "LEGAL"

    return CitationDetail(**normalized)
