"""Hybrid retrieval: dense top-50 + BM25 top-50, fused with RRF, optionally diversified with MMR.

mode: "dense", "bm25" or "hybrid" (the M5 evaluation compares all three).
The caller passes the query embedding, so this module never calls OpenAI itself.
"""
from dataclasses import dataclass

from app.retrieval.dense import dense_search
from app.retrieval.fusion import mmr, rrf
from app.retrieval.keyword import bm25_search

DENSE_N = 50
BM25_N = 50
MMR_POOL = 30  # MMR re-orders only the best fused candidates


@dataclass
class Hit:
    chunk_id: int
    document_id: int
    title: str
    text: str
    score: float              # fused RRF score (or the single method's score)
    dense_rank: int | None    # 1-based rank in the dense list, None if not found there
    bm25_rank: int | None
    dense_sim: float | None   # cosine similarity, used by the evidence gate later


def _parse_vec(text: str) -> list[float]:
    return [float(x) for x in text.strip("[]").split(",")]


def search(conn, query: str, query_vec=None, k: int = 10, mode: str = "hybrid",
           use_mmr: bool = False) -> list[Hit]:
    if mode not in ("dense", "bm25", "hybrid"):
        raise ValueError(f"unknown mode {mode}")
    dense = dense_search(conn, query_vec, DENSE_N) if mode in ("dense", "hybrid") else []
    keyword = bm25_search(conn, query, BM25_N) if mode in ("bm25", "hybrid") else []
    dense_rank = {cid: i for i, (cid, _) in enumerate(dense, start=1)}
    dense_sim = dict(dense)
    bm25_rank = {cid: i for i, (cid, _) in enumerate(keyword, start=1)}

    if mode == "hybrid":
        ranked = rrf([[c for c, _ in dense], [c for c, _ in keyword]])
    else:
        ranked = dense or keyword
    if not ranked:
        return []

    if use_mmr:
        pool = ranked[:MMR_POOL]
        vecs = {r[0]: _parse_vec(r[1]) for r in conn.execute(
            "SELECT id, embedding::text FROM chunks WHERE id = ANY(%s)", ([c for c, _ in pool],))}
        best = pool[0][1]
        relevance = {c: s / best for c, s in pool}
        order = mmr(relevance, vecs, k)
        scores = dict(pool)
        ranked = [(c, scores[c]) for c in order]
    ranked = ranked[:k]

    ids = [c for c, _ in ranked]
    rows = {r[0]: r for r in conn.execute(
        "SELECT c.id, c.document_id, d.title, c.text FROM chunks c JOIN documents d ON d.id = c.document_id "
        "WHERE c.id = ANY(%s)", (ids,))}
    return [Hit(c, rows[c][1], rows[c][2], rows[c][3], s, dense_rank.get(c), bm25_rank.get(c), dense_sim.get(c))
            for c, s in ranked if c in rows]
