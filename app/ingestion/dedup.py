"""Duplicate handling.

Exact: SHA-256 of normalized text.
Near: cosine similarity of embeddings >= COSINE_MIN AND word-shingle Jaccard >= JACCARD_MIN.
Requiring both avoids dropping paragraphs that are semantically close but say different things.
"""
import hashlib
import re

import numpy as np

COSINE_MIN = 0.97
JACCARD_MIN = 0.8
SHINGLE = 3


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def content_hash(text: str) -> str:
    return hashlib.sha256(normalize(text).encode("utf-8")).hexdigest()


def shingles(text: str, k: int = SHINGLE) -> set[tuple[str, ...]]:
    words = normalize(text).split()
    if len(words) < k:
        return {tuple(words)} if words else set()
    return {tuple(words[i:i + k]) for i in range(len(words) - k + 1)}


def jaccard(a: str, b: str) -> float:
    sa, sb = shingles(a), shingles(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def exact_dedup(texts: list[str]) -> tuple[list[int], dict[int, int]]:
    """Return (indices to keep, {dropped index: kept index})."""
    first: dict[str, int] = {}
    keep, dropped = [], {}
    for i, t in enumerate(texts):
        h = content_hash(t)
        if h in first:
            dropped[i] = first[h]
        else:
            first[h] = i
            keep.append(i)
    return keep, dropped


def near_duplicates(vectors: np.ndarray, texts: list[str], block: int = 512) -> dict[int, int]:
    """Find later items that nearly duplicate an earlier one. Returns {dropped index: kept index}.

    Cosine on unit vectors is a dot product; done in blocks to keep memory bounded.
    """
    v = vectors.astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    dropped: dict[int, int] = {}
    for start in range(0, len(v), block):
        sims = v[start:start + block] @ v.T
        for row in range(sims.shape[0]):
            i = start + row
            if i in dropped:
                continue
            for j in np.nonzero(sims[row] >= COSINE_MIN)[0]:
                j = int(j)
                if j < i and j not in dropped and jaccard(texts[i], texts[j]) >= JACCARD_MIN:
                    dropped[i] = j
                    break
    return dropped
