from typing import Dict, Any, Optional
import threading


class CitationStore:
    def __init__(self):
        self._lock = threading.Lock()
        self._data: Dict[str, Dict[str, Any]] = {}

    def upsert(self, citation: Dict[str, Any]) -> None:
        cid = citation.get("citation_id")
        if not cid:
            return
        with self._lock:
            self._data[cid] = citation

    def get(self, citation_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._data.get(citation_id)


citation_store = CitationStore()
