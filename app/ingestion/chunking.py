"""Two chunking strategies.

A (production): structure-aware. One chunk per paragraph, title prepended; paragraphs longer
   than MAX_WORDS are split at sentence boundaries.
B (baseline): fixed word windows with overlap, no title. Used only for the comparison.
"""
import re

MAX_WORDS = 300
WINDOW, OVERLAP = 80, 20

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


def split_sentences(text: str) -> list[str]:
    return [s for s in _SENT_SPLIT.split(text.strip()) if s]


def _split_long(paragraph: str, max_words: int) -> list[str]:
    pieces, cur, cur_n = [], [], 0
    for sent in split_sentences(paragraph):
        n = len(sent.split())
        if cur and cur_n + n > max_words:
            pieces.append(" ".join(cur))
            cur, cur_n = [], 0
        cur.append(sent)
        cur_n += n
    if cur:
        pieces.append(" ".join(cur))
    # a single sentence longer than max_words is hard-cut so no chunk is unbounded
    out = []
    for p in pieces:
        words = p.split()
        for i in range(0, len(words), max_words):
            out.append(" ".join(words[i:i + max_words]))
    return out


def chunk_a(title: str, text: str, max_words: int = MAX_WORDS) -> list[str]:
    chunks = []
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        pieces = [para] if len(para.split()) <= max_words else _split_long(para, max_words)
        chunks.extend(f"{title}\n\n{p}" for p in pieces)
    return chunks


def chunk_b(text: str, window: int = WINDOW, overlap: int = OVERLAP) -> list[str]:
    if overlap >= window:
        raise ValueError("overlap must be smaller than window")
    words = text.split()
    if not words:
        return []
    step = window - overlap
    chunks = []
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start:start + window]))
        if start + window >= len(words):
            break
    return chunks
