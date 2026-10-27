"""Shared tokenizer for BM25 statistics (term_stats) and query-time scoring.

Index time (load_db.py) and query time (bm25.py, M4) MUST use this same function,
otherwise document frequencies would not match the terms being scored.
"""
import re

_WORD = re.compile(r"\w+", re.UNICODE)

STOPWORDS = frozenset(
    "a an and are as at be by for from has have he her his in is it its of on or that the "
    "their there they this to was were which who will with".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _WORD.findall(text.lower()) if t not in STOPWORDS]
