from __future__ import annotations

import re
from typing import List, Tuple

# ----------------------------
# Public API
# ----------------------------

def chunk_legal_markdown(
    text: str,
    max_chars: int = 1400,
    overlap_chars: int = 250,
) -> List[Tuple[str, str]]:
    """
    Split legal Markdown content into structure-preserving chunks.

    Returns list of (heading_path, chunk_text) tuples.
    """

    lines = text.splitlines()
    heading_stack: List[str] = []
    blocks: List[_Block] = []
    chunks: List[Tuple[str, str]] = []

    in_code_fence = False
    code_fence_delim = "```"

    def heading_path() -> str:
        return " > ".join([h for h in heading_stack if h])

    def flush_heading_blocks() -> None:
        nonlocal blocks, chunks
        if not blocks:
            return
        heading = heading_path()
        chunks.extend(_blocks_to_chunks(blocks, heading, max_chars, overlap_chars))
        blocks = []

    for raw in lines:
        stripped = raw.strip()

        # Code fences take precedence over headings or list detection
        if stripped.startswith(code_fence_delim):
            blocks.append(_Block("code", raw))
            in_code_fence = not in_code_fence
            continue

        if in_code_fence:
            blocks.append(_Block("code", raw))
            continue

        if _is_heading(stripped):
            flush_heading_blocks()
            level = _heading_level(stripped)
            text_heading = stripped.lstrip("#").strip()
            if level >= 1:
                heading_stack[:] = heading_stack[: level - 1]
            heading_stack.append(text_heading)
            continue

        if not stripped:
            blocks.append(_Block("blank", raw))
            continue

        if _is_list_item(stripped):
            blocks.append(_Block("list", raw))
            continue

        blocks.append(_Block("paragraph", raw))

    flush_heading_blocks()
    return chunks


# ----------------------------
# Internal structures and helpers
# ----------------------------

class _Block:
    def __init__(self, kind: str, line: str) -> None:
        self.kind = kind
        self.line = line

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"_Block(kind={self.kind!r}, line={self.line!r})"


_HEADING_RE = re.compile(r"^(#{1,3})\s*(.+)$")
_LIST_PATTERNS = [
    re.compile(r"^\s*[-*]\s+"),
    re.compile(r"^\s*\d+\.\s+"),
    re.compile(r"^\s*\([a-zA-Z0-9]+\)\s+"),
    re.compile(r"^\s*[a-zA-Z]\)\s+"),
]
def _is_heading(line: str) -> bool:
    return bool(_HEADING_RE.match(line))


def _heading_level(line: str) -> int:
    match = _HEADING_RE.match(line)
    if not match:
        return 0
    return len(match.group(1))


def _is_list_item(line: str) -> bool:
    return any(pat.match(line) for pat in _LIST_PATTERNS)


def _blocks_to_chunks(
    blocks: List[_Block], heading_path: str, max_chars: int, overlap_chars: int
) -> List[Tuple[str, str]]:
    expanded_blocks = _expand_blocks(blocks, max_chars)

    chunks: List[Tuple[str, str]] = []
    current_blocks: List[str] = []
    previous_chunk_blocks: List[str] = []

    for block_text in expanded_blocks:
        block_text = block_text.rstrip("\n")
        appended = _join_blocks(current_blocks + [block_text])
        if len(appended) <= max_chars:
            current_blocks.append(block_text)
            continue

        if current_blocks:
            chunk_text = _join_blocks(current_blocks)
            chunks.append((heading_path, chunk_text.strip()))
            previous_chunk_blocks = current_blocks.copy()
            current_blocks = []

        if previous_chunk_blocks:
            overlap_text = _build_overlap(previous_chunk_blocks, overlap_chars)
            if overlap_text:
                current_blocks.append(overlap_text)

        if len(block_text) > max_chars:
            split_subblocks = _split_long_text(block_text, max_chars)
            for part in split_subblocks:
                appended = _join_blocks(current_blocks + [part])
                if len(appended) > max_chars and current_blocks:
                    chunk_text = _join_blocks(current_blocks)
                    chunks.append((heading_path, chunk_text.strip()))
                    previous_chunk_blocks = current_blocks.copy()
                    current_blocks = []
                    overlap_text = _build_overlap(previous_chunk_blocks, overlap_chars)
                    if overlap_text:
                        current_blocks.append(overlap_text)
                current_blocks.append(part)
        else:
            current_blocks.append(block_text)

    if current_blocks:
        chunk_text = _join_blocks(current_blocks)
        chunks.append((heading_path, chunk_text.strip()))

    return chunks


def _expand_blocks(blocks: List[_Block], max_chars: int) -> List[str]:
    expanded: List[str] = []
    buffer: List[_Block] = []
    last_kind: str | None = None

    for blk in blocks:
        if blk.kind == "blank":
            if buffer:
                expanded.append(_join_block_lines(buffer))
                buffer = []
                last_kind = None
            continue

        if blk.kind == last_kind and blk.kind in {"list", "paragraph", "code"}:
            buffer.append(blk)
        else:
            if buffer:
                expanded.append(_join_block_lines(buffer))
            buffer = [blk]
            last_kind = blk.kind

    if buffer:
        expanded.append(_join_block_lines(buffer))

    expanded_with_splits: List[str] = []
    for block_text in expanded:
        if len(block_text) <= max_chars:
            expanded_with_splits.append(block_text)
            continue
        if _is_code_block(block_text):
            expanded_with_splits.append(block_text)
            continue
        if _is_list_block(block_text):
            expanded_with_splits.extend(_split_list_block(block_text, max_chars))
            continue
        expanded_with_splits.extend(_split_long_text(block_text, max_chars))

    return expanded_with_splits


def _join_block_lines(blocks: List[_Block]) -> str:
    lines = [b.line for b in blocks]
    return "\n".join(lines).rstrip() + "\n"


def _join_blocks(blocks: List[str]) -> str:
    return "\n\n".join([b.rstrip() for b in blocks if b.strip()])


def _is_list_block(text: str) -> bool:
    first_line = text.splitlines()[0] if text else ""
    return _is_list_item(first_line)


def _is_code_block(text: str) -> bool:
    first_line = text.lstrip().splitlines()[0] if text else ""
    return first_line.startswith("```")


def _split_list_block(block_text: str, max_chars: int) -> List[str]:
    items: List[str] = []
    current_item: List[str] = []
    for line in block_text.splitlines():
        if _is_list_item(line.strip()) and current_item:
            items.append("\n".join(current_item).rstrip())
            current_item = [line]
        else:
            current_item.append(line)
    if current_item:
        items.append("\n".join(current_item).rstrip())

    chunks: List[str] = []
    current_lines: List[str] = []
    for item in items:
        candidate = "\n".join(current_lines + [item]) if current_lines else item
        if len(candidate) > max_chars and current_lines:
            chunks.append("\n".join(current_lines).rstrip())
            current_lines = [item]
        else:
            if len(item) > max_chars:
                chunks.extend(_split_long_text(item, max_chars))
                continue
            current_lines.append(item)
    if current_lines:
        chunks.append("\n".join(current_lines).rstrip())
    return chunks


def _split_long_text(text: str, max_chars: int) -> List[str]:
    sentences = _split_sentences(text)
    parts: List[str] = []
    current: List[str] = []
    for sent in sentences:
        candidate = " ".join(current + [sent]) if current else sent
        if len(candidate) > max_chars and current:
            parts.append(" ".join(current).strip())
            current = [sent]
        else:
            current.append(sent)
    if current:
        parts.append(" ".join(current).strip())
    normalized_parts: List[str] = []
    for part in parts or [text]:
        if len(part) <= max_chars:
            normalized_parts.append(part)
            continue
        for i in range(0, len(part), max_chars):
            normalized_parts.append(part[i : i + max_chars].strip())
    return normalized_parts


def _split_sentences(text: str) -> List[str]:
    sentences: List[str] = []
    buf: List[str] = []
    tokens = re.split(r"([.!?]\s+)", text)
    for token in tokens:
        if not token:
            continue
        buf.append(token)
        if re.match(r"[.!?]\s+$", token):
            sentences.append("".join(buf).strip())
            buf = []
    if buf:
        sentences.append("".join(buf).strip())
    return sentences or [text]


def _build_overlap(blocks: List[str], overlap_chars: int) -> str:
    if overlap_chars <= 0:
        return ""
    collected: List[str] = []
    total = 0
    for block in reversed(blocks):
        block_clean = block.rstrip()
        if not block_clean:
            continue
        collected.append(block_clean)
        total += len(block_clean)
        if total >= overlap_chars:
            break
    collected.reverse()
    return "\n\n".join(collected)


__all__ = ["chunk_legal_markdown"]
