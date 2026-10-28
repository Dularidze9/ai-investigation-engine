"""Semantic search with pgvector's HNSW index (cosine distance on halfvec(512))."""


def vec_literal(vec) -> str:
    return "[" + ",".join(f"{float(x):.6f}" for x in vec) + "]"


def dense_search(conn, query_vec, top_k: int = 50) -> list[tuple[int, float]]:
    """Returns [(chunk_id, cosine similarity)] best first.

    hnsw.ef_search must be >= top_k: its default of 40 would silently cap results at 40.
    SET LOCAL lasts only for this transaction, which also works through Neon's pooler.
    """
    with conn.transaction():
        conn.execute(f"SET LOCAL hnsw.ef_search = {max(int(top_k), 40)}")
        rows = conn.execute(
            "SELECT id, 1 - (embedding <=> %s::halfvec) FROM chunks ORDER BY embedding <=> %s::halfvec LIMIT %s",
            (vec_literal(query_vec), vec_literal(query_vec), top_k)).fetchall()
    return [(r[0], float(r[1])) for r in rows]
