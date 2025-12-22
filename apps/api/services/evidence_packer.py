from __future__ import annotations

from typing import Any, List, Sequence, Tuple
import re

# ----------------------------
# Evidence packer
# ----------------------------


def build_evidence_pack(
    question: str,
    citations: List[dict[str, Any]],
    max_sentences_per_chunk: int = 5,
    max_total_sentences: int = 40,
) -> str:
    """
    Returns a compact evidence string for prompt injection.

    Output must preserve citation IDs and source context so the model can cite properly.
    Does not change what is stored in citation_store.
    """

    question_tokens = _tokenize(question)
    lines: List[str] = []
    total_sentences = 0

    for citation in citations:
        if total_sentences >= max_total_sentences:
            break

        cid = citation.get("citation_id", "")
        heading_path = citation.get("heading_path") or citation.get("location") or citation.get("heading") or ""
        source_file = citation.get("source_file", "")
        verbatim = (citation.get("verbatim", "") or "").strip()

        sentences = _split_sentences(verbatim)
        if not sentences:
            fallback = citation.get("snippet") or citation.get("heading") or "(no content)"
            sentences = [fallback.strip()]

        sentence_scores = _score_sentences(sentences, question_tokens, heading_path)

        selected: List[str] = []
        remaining_allowance = max_total_sentences - total_sentences
        per_chunk_limit = min(max_sentences_per_chunk, remaining_allowance)

        if sentence_scores:
            max_score = max(score for _, score, _, _ in sentence_scores)
        else:
            max_score = 0

        if max_score == 0:
            selected = [s for s, _, _, _ in sentence_scores[: min(2, per_chunk_limit)]]
        else:
            selected = [s for s, _, _, _ in sentence_scores[:per_chunk_limit]]

        if not selected and sentences:
            selected = sentences[: min(2, per_chunk_limit or 1)]

        if not selected:
            selected = ["(no content)"]

        lines.append(f"[{cid}] {heading_path} | {source_file}")

        for s in selected:
            if total_sentences >= max_total_sentences:
                break
            lines.append(f"- {s.strip()}")
            total_sentences += 1

    return "\n".join(lines).strip()


# ----------------------------
# Helpers
# ----------------------------


def _tokenize(text: str) -> set[str]:
    tokens = re.split(r"[^A-Za-z0-9]+", (text or "").lower())
    return {t for t in tokens if len(t) >= 3}


def _split_sentences(text: str) -> List[str]:
    bullet_pattern = re.compile(r"^(?:[-*]|\d+\.|[a-zA-Z]\))\s+")
    sentences: List[str] = []
    buffer: List[str] = []

    def _flush_buffer() -> None:
        if not buffer:
            return
        paragraph = " ".join(buffer).strip()
        if not paragraph:
            buffer.clear()
            return
        parts = re.split(r"(?<=[\.!?])\s+|\n+", paragraph)
        sentences.extend([p.strip() for p in parts if p and p.strip()])
        buffer.clear()

    for line in (text or "").splitlines():
        stripped = line.strip()
        if not stripped:
            _flush_buffer()
            continue
        if bullet_pattern.match(stripped):
            _flush_buffer()
            sentences.append(stripped)
        else:
            buffer.append(stripped)

    _flush_buffer()
    return sentences


def _score_sentences(
    sentences: Sequence[str], question_tokens: set[str], heading_path: str
) -> List[Tuple[str, int, int, int]]:
    heading_tokens = _tokenize(heading_path)
    scored: List[Tuple[str, int, int, int]] = []

    for idx, sentence in enumerate(sentences):
        sentence_tokens = _tokenize(sentence)
        overlap = len(sentence_tokens.intersection(question_tokens))
        heading_overlap = len(sentence_tokens.intersection(heading_tokens))
        scored.append((sentence, overlap, heading_overlap, idx))

    scored.sort(key=lambda x: (-x[1], -x[2], x[3]))
    return scored
