from __future__ import annotations

from typing import Any, List, Sequence, Tuple
import re

from apps.api.config import EVIDENCE_MAX_CHARS_PER_CITATION, EVIDENCE_MAX_CHARS_TOTAL

# ----------------------------
# Evidence packer
# ----------------------------


def build_evidence_pack(
    question: str,
    citations: List[dict[str, Any]],
    max_sentences_per_chunk: int = 5,
    max_total_sentences: int = 40,
    max_chars_total: int = EVIDENCE_MAX_CHARS_TOTAL,
    max_chars_per_citation: int = EVIDENCE_MAX_CHARS_PER_CITATION,
) -> str:
    """
    Returns a compact evidence string for prompt injection.

    Output must preserve citation IDs and source context so the model can cite properly.
    Does not change what is stored in citation_store.
    """

    question_tokens = _tokenize(question)
    lines: List[str] = []
    total_sentences = 0
    total_chars = 0
    included_citations = 0

    for citation in citations:
        if total_sentences >= max_total_sentences and total_chars >= max_chars_total and included_citations >= 2:
            break

        cid = citation.get("citation_id", "")
        heading_path = citation.get("heading_path") or citation.get("location") or citation.get("heading") or ""
        source_file = citation.get("source_file", "")
        verbatim_source = (
            citation.get("text")
            or citation.get("verbatim")
            or citation.get("snippet")
            or citation.get("heading")
            or ""
        )
        verbatim = _normalize_whitespace(verbatim_source)[:max_chars_per_citation]

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
            selected_indices: List[int] = []

            # Anchor, early, late coverage selections
            anchor_idx = _find_anchor_sentence(sentences)
            early_idx = _find_positional_sentence(sentences, early=True)
            late_idx = _find_positional_sentence(sentences, early=False)

            required_indices: List[int] = []
            if anchor_idx is not None:
                required_indices.append(anchor_idx)
            if early_idx is not None and early_idx not in required_indices:
                required_indices.append(early_idx)
            if late_idx is not None and late_idx not in required_indices:
                required_indices.append(late_idx)

            for req_idx in required_indices:
                if len(selected_indices) >= per_chunk_limit:
                    break
                selected_indices.append(req_idx)

            for _sentence, _overlap, _heading_overlap, idx in sentence_scores:
                if len(selected_indices) >= per_chunk_limit:
                    break
                if idx not in selected_indices:
                    selected_indices.append(idx)

            scored_lookup = {idx: (s, o, h) for s, o, h, idx in sentence_scores}
            selection_with_priority: List[Tuple[bool, int, int, int]] = []
            required_set = set(required_indices)
            for idx in selected_indices:
                score_tuple = scored_lookup.get(idx, (sentences[idx], 0, 0))
                selection_with_priority.append(
                    (
                        idx in required_set,
                        score_tuple[1],
                        score_tuple[2],
                        idx,
                    )
                )

            selection_with_priority.sort(key=lambda x: (0 if x[0] else 1, -x[1], -x[2], x[3]))
            selected = [sentences[idx] for _required, _o, _h, idx in selection_with_priority[:per_chunk_limit]]

        if not selected and sentences:
            selected = sentences[: min(2, per_chunk_limit or 1)]

        if not selected:
            selected = ["(no content)"]

        entry_lines: List[str] = [f"[{cid}] {heading_path} | {source_file}"]

        for s in selected:
            if total_sentences >= max_total_sentences:
                break
            entry_lines.append(f"- {s.strip()}")
            total_sentences += 1

        block = "\n".join(entry_lines).strip()

        prospective_length = total_chars + len(block) + (1 if lines else 0)
        if prospective_length > max_chars_total and included_citations >= 2:
            break

        if prospective_length > max_chars_total:
            allowed = max_chars_total - total_chars - (1 if lines else 0)
            block = block[:max(0, allowed)]

        if not block:
            block = entry_lines[0][:max_chars_per_citation]

        separator_chars = 1 if lines else 0
        lines.append(block)
        total_chars += len(block) + separator_chars
        included_citations += 1

    return "\n".join(lines).strip()


# ----------------------------
# Helpers
# ----------------------------


def _tokenize(text: str) -> set[str]:
    tokens = re.split(r"[^A-Za-z0-9]+", (text or "").lower())
    return {t for t in tokens if len(t) >= 3}


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


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


def _find_anchor_sentence(sentences: Sequence[str]) -> int | None:
    anchor_keywords = {
        "shall",
        "will",
        "must",
        "may",
        "procedure",
        "authority",
        "convening",
        "evidence",
        "record",
        "witness",
        "recommend",
    }
    numbered_pattern = re.compile(r"^\s*(?:\(?[0-9]+\)?[.)]|\([a-z]\)|[a-z]\))")

    for idx, sentence in enumerate(sentences):
        lowered = sentence.lower()
        if numbered_pattern.match(sentence.strip()):
            return idx
        if any(word in lowered for word in anchor_keywords):
            return idx
    return None


def _find_positional_sentence(sentences: Sequence[str], early: bool = True) -> int | None:
    if not sentences:
        return None

    count = len(sentences)
    band = max(1, int(count * 0.2))
    if early:
        candidate_indices = range(0, band)
    else:
        candidate_indices = range(max(0, count - band), count)

    for idx in candidate_indices:
        if 0 <= idx < count:
            return idx
    return None
