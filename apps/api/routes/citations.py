from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from apps.api.services.citation_store import citation_store

router = APIRouter()


class CitationDetail(BaseModel):
    citation_id: str
    document: str
    source_file: str
    heading: str
    location: str
    verbatim: str
    context_before: str
    context_after: str


@router.get("/citations/{citation_id}", response_model=CitationDetail)
def get_citation(citation_id: str):
    item = citation_store.get(citation_id)
    if not item:
        raise HTTPException(status_code=404, detail="Citation not found")
    return CitationDetail(**item)
