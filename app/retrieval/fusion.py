"""Combining and diversifying rankings, written by hand.

RRF (Reciprocal Rank Fusion): each ranking votes 1 / (k + rank) for each document, rank starting at 1.
    Only ranks are used, never raw scores, so BM25 scores (0..20+) and cosine similarities (0..1)
    can be fused without normalizing them. k (60) damps the top: rank 1 vs rank 2 is
    1/61 vs 1/62, so one list alone cannot dominate.

MMR (Maximal Marginal Relevance): picks results one at a time, trading relevance against
    similarity to what is already picked:  lam * relevance(d) - (1 - lam) * max_sim(d, picked).
    lam = 1 is plain relevance order; lower lam gives more diverse results.
"""
import math

RRF_K = 60
MMR_LAMBDA = 0.7


def rrf(rankings: list[list[int]], k: int = RRF_K,
        weights: list[float] | None = None) -> list[tuple[int, float]]:
    """rankings: lists of doc ids, best first. Returns [(doc_id, fused score)] best first.
    Ties keep the order in which documents were first seen (first list first)."""
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("need one weight per ranking")
    scores: dict[int, float] = {}
    first_seen: dict[int, int] = {}
    for weight, ranking in zip(weights, rankings):
        seen_here = set()
        for rank, doc_id in enumerate(ranking, start=1):
            if doc_id in seen_here:  # a duplicate inside one list only counts at its best rank
                continue
            seen_here.add(doc_id)
            scores[doc_id] = scores.get(doc_id, 0.0) + weight / (k + rank)
            first_seen.setdefault(doc_id, len(first_seen))
    return sorted(scores.items(), key=lambda s: (-s[1], first_seen[s[0]]))


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def mmr(relevance: dict[int, float], vectors: dict[int, list[float]], k: int,
        lam: float = MMR_LAMBDA) -> list[int]:
    """relevance: doc id -> relevance in [0, 1] (in hybrid search: RRF score / best RRF score).
    vectors: doc id -> embedding, used only to measure redundancy between documents.
    Ties go to the document listed first in `relevance`."""
    remaining = list(relevance)
    picked: list[int] = []
    while remaining and len(picked) < k:
        def mmr_score(d: int) -> float:
            redundancy = max((cosine(vectors[d], vectors[p]) for p in picked), default=0.0)
            return lam * relevance[d] - (1 - lam) * redundancy
        best = max(remaining, key=mmr_score)  # max() keeps the first of equal scores
        picked.append(best)
        remaining.remove(best)
    return picked
