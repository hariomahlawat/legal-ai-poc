import time

from apps.api.services.citation_store import CitationStore


def test_citation_store_respects_case_scoping(monkeypatch):
    # === Configuration ===
    monkeypatch.setenv("CITATION_STORE_MAX_SIZE", "5")
    monkeypatch.setenv("CITATION_STORE_TTL_SECONDS", "10")
    store = CitationStore()

    # === Arrange ===
    citation_one = {"citation_id": "abc", "case_id": "case-1", "content": "first"}
    citation_two = {"citation_id": "abc", "case_id": "case-2", "content": "second"}

    # === Act ===
    store.upsert(citation_one)
    store.upsert(citation_two)

    # === Assert ===
    assert store.get("abc", case_id="case-1") == citation_one
    assert store.get("abc", case_id="case-2") == citation_two
    assert store.get("abc") is None


def test_citation_store_evicts_by_ttl(monkeypatch):
    # === Configuration ===
    monkeypatch.setenv("CITATION_STORE_MAX_SIZE", "5")
    monkeypatch.setenv("CITATION_STORE_TTL_SECONDS", "0.1")
    store = CitationStore()

    # === Arrange ===
    citation = {"citation_id": "abc", "case_id": "case-1", "content": "first"}

    # === Act ===
    store.upsert(citation)
    time.sleep(0.2)

    # === Assert ===
    assert store.get("abc", case_id="case-1") is None


def test_citation_store_evicts_lru(monkeypatch):
    # === Configuration ===
    monkeypatch.setenv("CITATION_STORE_MAX_SIZE", "1")
    monkeypatch.setenv("CITATION_STORE_TTL_SECONDS", "10")
    store = CitationStore()

    # === Arrange ===
    first = {"citation_id": "one", "case_id": "case-1", "content": "first"}
    second = {"citation_id": "two", "case_id": "case-1", "content": "second"}

    # === Act ===
    store.upsert(first)
    store.upsert(second)

    # === Assert ===
    assert store.get("one", case_id="case-1") is None
    assert store.get("two", case_id="case-1") == second
