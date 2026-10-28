"""Shared helpers for the evaluation scripts: questions and a disk cache for query embeddings."""
import hashlib
import json
import pathlib
import sys

import numpy as np
import tiktoken

ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from app.retrieval.embeddings import PRICE_PER_TOKEN, embed_batch  # noqa: E402

QUESTIONS = ROOT / "eval" / "questions.jsonl"
QUERY_CACHE = ROOT / "data" / "cache" / "query_emb.json"


def load_questions() -> list[dict]:
    return [json.loads(line) for line in QUESTIONS.open(encoding="utf-8")]


def _key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]


def _load_cache() -> dict:
    return json.loads(QUERY_CACHE.read_text()) if QUERY_CACHE.exists() else {}


def estimate_query_cost(texts: list[str]) -> tuple[int, int, float]:
    """(uncached count, tokens, usd) for the texts that are not in the cache yet."""
    cache = _load_cache()
    enc = tiktoken.get_encoding("cl100k_base")
    todo = [t for t in texts if _key(t) not in cache]
    tokens = sum(len(enc.encode(t)) for t in todo)
    return len(todo), tokens, tokens * PRICE_PER_TOKEN


def embed_queries(texts: list[str], client=None) -> dict[str, np.ndarray]:
    """Return {text: unit vector}. Only uncached texts are sent to OpenAI (in batches of 100)."""
    cache = _load_cache()
    todo = [t for t in dict.fromkeys(texts) if _key(t) not in cache]
    if todo and client is None:
        raise RuntimeError("uncached queries need an OpenAI client (run with --yes)")
    for i in range(0, len(todo), 100):
        batch = todo[i:i + 100]
        vecs, _ = embed_batch(client, batch)
        for t, v in zip(batch, vecs):
            cache[_key(t)] = [float(x) for x in v]
        QUERY_CACHE.parent.mkdir(parents=True, exist_ok=True)
        QUERY_CACHE.write_text(json.dumps(cache))
    return {t: np.array(cache[_key(t)], dtype=np.float32) for t in texts}
