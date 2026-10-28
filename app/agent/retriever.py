"""The real Retriever used by the agent: hybrid search on Neon plus query embeddings with cost tracking."""
from functools import lru_cache

from app.agent.tools import Evidence
from app.retrieval.embeddings import PRICE_PER_TOKEN, embed_batch
from app.retrieval.hybrid import search


class Embedder:
    """Embeds queries one at a time, caches repeats in memory and counts the cost."""

    def __init__(self, client):
        self.client, self.tokens = client, 0
        self.embed = lru_cache(maxsize=256)(self._embed)

    def _embed(self, text: str):
        vecs, tokens = embed_batch(self.client, [text])
        self.tokens += tokens
        return vecs[0]

    def embed_many(self, texts: list[str], batch: int = 100):
        """For ingestion: embed many chunks, in batches."""
        out = []
        for i in range(0, len(texts), batch):
            vecs, tokens = embed_batch(self.client, texts[i:i + batch])
            self.tokens += tokens
            out.extend(vecs)
        return out

    @property
    def cost_usd(self) -> float:
        return self.tokens * PRICE_PER_TOKEN


class DbRetriever:
    def __init__(self, conn, embedder: Embedder):
        self.conn, self.embedder = conn, embedder

    def search(self, query: str, k: int) -> list[Evidence]:
        hits = search(self.conn, query, self.embedder.embed(query), k=k, mode="hybrid")
        return [Evidence(h.chunk_id, h.document_id, h.title, h.text, h.dense_sim) for h in hits]

    def get_document(self, document_id: int, limit: int) -> list[Evidence]:
        rows = self.conn.execute(
            "SELECT c.id, c.document_id, d.title, c.text FROM chunks c JOIN documents d ON d.id = c.document_id "
            "WHERE c.document_id = %s ORDER BY c.chunk_index LIMIT %s", (document_id, limit)).fetchall()
        return [Evidence(r[0], r[1], r[2], r[3]) for r in rows]
