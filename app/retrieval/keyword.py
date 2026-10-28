"""Keyword search: Postgres only FETCHES candidates (tsvector OR-query); our BM25 RANKS them.

Why not rank everything in Python: that needs an inverted index (~5M postings) we can't fit in
Neon's free 0.5 GB. Trade-off: BM25 only sees the top CANDIDATES chunks by ts_rank.
"""
from app.retrieval import bm25
from app.retrieval.tokenize import tokenize

CANDIDATES = 200


def keyword_candidates(conn, terms: list[str], limit: int = CANDIDATES) -> dict[int, str]:
    """Chunks matching ANY query term (stemmed by Postgres), best ts_rank first. Returns {chunk_id: text}."""
    if not terms:
        return {}
    # Terms are \w+ tokens from our tokenizer, so they can't inject tsquery operators.
    tsquery = " | ".join(dict.fromkeys(terms))
    rows = conn.execute(
        "SELECT c.id, c.text FROM chunks c, to_tsquery('english', %s) q "
        "WHERE c.tsv @@ q ORDER BY ts_rank(c.tsv, q) DESC, c.id LIMIT %s",
        (tsquery, limit)).fetchall()
    return {r[0]: r[1] for r in rows}


def load_stats(conn, terms: list[str]) -> tuple[dict[str, int], bm25.CorpusStats]:
    df = dict(conn.execute("SELECT term, df FROM term_stats WHERE term = ANY(%s)", (list(set(terms)),)).fetchall())
    n, total_len = conn.execute("SELECT n_chunks, total_len FROM corpus_stats WHERE id = 1").fetchone()
    return df, bm25.CorpusStats(n_docs=n, avgdl=total_len / n if n else 1.0)


def bm25_search(conn, query: str, top_k: int = 50, candidates: int = CANDIDATES) -> list[tuple[int, float]]:
    terms = tokenize(query)
    texts = keyword_candidates(conn, terms, candidates)
    if not texts:
        return []
    df, stats = load_stats(conn, terms)
    docs = {chunk_id: tokenize(text) for chunk_id, text in texts.items()}
    return bm25.rank(terms, docs, df, stats, top_k=top_k)
