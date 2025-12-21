import os
import threading
import time
from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple


class CitationStore:
    def __init__(self):
        # === Configuration ===
        env_max_size = os.getenv("CITATION_STORE_MAX_SIZE")
        env_ttl_seconds = os.getenv("CITATION_STORE_TTL_SECONDS")

        self._max_size = int(env_max_size) if env_max_size else 1024
        self._ttl_seconds = float(env_ttl_seconds) if env_ttl_seconds else 3600.0

        # === Internal State ===
        self._lock = threading.Lock()
        self._data: "OrderedDict[Tuple[str, str], Tuple[float, Dict[str, Any]]]" = OrderedDict()

    def _now(self) -> float:
        return time.monotonic()

    def _build_key(self, citation_id: str, case_id: Optional[str]) -> Tuple[str, str]:
        return (case_id or "", citation_id)

    def _evict_expired(self) -> None:
        if self._ttl_seconds <= 0:
            return

        now = self._now()
        expired_keys = [
            key for key, (created_at, _) in self._data.items() if now - created_at >= self._ttl_seconds
        ]
        for key in expired_keys:
            self._data.pop(key, None)

    def _enforce_size(self) -> None:
        if self._max_size <= 0:
            self._data.clear()
            return

        while len(self._data) > self._max_size:
            self._data.popitem(last=False)

    def upsert(self, citation: Dict[str, Any]) -> None:
        cid = citation.get("citation_id")
        if not cid:
            return
        case_id = citation.get("case_id")
        key = self._build_key(cid, case_id)
        now = self._now()
        with self._lock:
            # === Maintenance ===
            self._evict_expired()

            # === Upsert ===
            self._data.pop(key, None)
            self._data[key] = (now, citation)
            self._data.move_to_end(key)

            # === Capacity Enforcement ===
            self._enforce_size()

    def get(self, citation_id: str, case_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        key = self._build_key(citation_id, case_id)
        with self._lock:
            # === Maintenance ===
            self._evict_expired()

            entry = self._data.get(key)
            if not entry:
                return None

            created_at, citation = entry
            if self._ttl_seconds > 0 and self._now() - created_at >= self._ttl_seconds:
                self._data.pop(key, None)
                return None

            # === LRU Update ===
            self._data.move_to_end(key)
            return citation


citation_store = CitationStore()
