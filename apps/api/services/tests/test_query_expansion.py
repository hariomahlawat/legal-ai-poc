import pytest

from apps.api.services.query_expansion import expand_queries


# ----------------------------
# Query expansion tests
# ----------------------------

def test_expand_queries_acronym_expansion():
    variants = expand_queries("COI procedure for recording evidence", legal_object=None)
    assert any("Court of Inquiry procedure for recording evidence" in v for v in variants)
    assert variants[0] == "COI procedure for recording evidence"


def test_expand_queries_legal_object_prefix():
    variants = expand_queries("procedure for recording evidence", legal_object="Court of Inquiry")
    assert any(v.lower().startswith("court of inquiry:") for v in variants)


def test_expand_queries_section_hint():
    variants = expand_queries("rule for evidence handling", legal_object=None)
    assert any("relevant rule section" in v for v in variants)


def test_expand_queries_dedup_cap():
    variants = expand_queries("section rule para appendix coi", legal_object="Court-Martial")
    assert len(variants) <= 5
    lower = [v.lower() for v in variants]
    assert len(lower) == len(set(lower))
