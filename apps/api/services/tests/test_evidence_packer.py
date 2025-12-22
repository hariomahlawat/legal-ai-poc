import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[4]))

from apps.api.services.evidence_packer import build_evidence_pack


# ----------------------------
# Evidence packer tests
# ----------------------------


def _build_citation(citation_id: str, verbatim: str) -> dict:
    return {
        "citation_id": citation_id,
        "heading_path": "Chap 1 > Sec 2",
        "source_file": "vol1.pdf",
        "verbatim": verbatim,
    }


def test_preserves_citation_ids():
    citations = [
        _build_citation("c1", "First sentence. Second sentence."),
        _build_citation("c2", "Another block. With more text."),
    ]

    pack = build_evidence_pack("What is asked?", citations)

    assert "[c1]" in pack
    assert "[c2]" in pack


def test_sentence_cap_respected():
    text = " ".join([f"Sentence {i}." for i in range(30)])
    citations = [_build_citation("cap", text)]

    pack = build_evidence_pack("irrelevant question", citations, max_total_sentences=3)

    bullet_lines = [l for l in pack.splitlines() if l.strip().startswith("-")]
    assert len(bullet_lines) <= 3


def test_includes_at_least_one_sentence_with_zero_overlap():
    citations = [
        _build_citation("c-no-overlap", "Alpha beta gamma."),
        _build_citation("c-no-overlap-2", "Delta epsilon zeta."),
    ]

    pack = build_evidence_pack("completely unrelated query", citations, max_sentences_per_chunk=2)

    lines = pack.splitlines()
    header_to_bullets = {}
    current_header = None

    for line in lines:
        if line.startswith("["):
            current_header = line
            header_to_bullets[current_header] = 0
        elif line.strip().startswith("-") and current_header:
            header_to_bullets[current_header] += 1

    for cid in ["c-no-overlap", "c-no-overlap-2"]:
        matching_header = next((h for h in header_to_bullets if cid in h), None)
        assert matching_header is not None
        assert header_to_bullets[matching_header] >= 1
