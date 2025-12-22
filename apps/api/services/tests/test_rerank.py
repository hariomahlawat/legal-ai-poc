import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[4]))

from apps.api.services import rerank


# ----------------------------
# Reranker tests
# ----------------------------

def test_rerank_disabled_preserves_order(monkeypatch):
    monkeypatch.setenv("RERANK_ENABLED", "false")
    # Reload module variables to ensure env applies
    import importlib

    importlib.reload(rerank)

    candidates = [
        {"heading_path": "A", "text": "one", "retrieval_score": 0.9},
        {"heading_path": "B", "text": "two", "retrieval_score": 0.8},
    ]

    result = rerank.rerank_candidates("question", candidates, top_k=2)

    assert [c["heading_path"] for c in result] == ["A", "B"]
    assert all(c["rerank_score"] is None for c in result)


def test_build_pair_truncates(monkeypatch):
    monkeypatch.delenv("RERANK_ENABLED", raising=False)
    monkeypatch.delenv("RERANK_TOP_N", raising=False)
    # Reload module to reset config to defaults
    import importlib

    importlib.reload(rerank)

    long_text = "x" * 2000
    candidate = {"heading_path": "HEAD", "text": long_text}
    pair = rerank._build_pair("question", candidate)

    assert pair[1].startswith("HEAD\n")
    assert len(pair[1]) == 1800
