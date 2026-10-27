"""M3 step 1: embed every chunk (strategy A) with a resumable on-disk cache.

  python scripts/embed_corpus.py              -> ESTIMATE ONLY: exact token count and cost, no API call
  python scripts/embed_corpus.py --yes        -> embed (paid). Safe to kill and rerun: finished shards are skipped.
  python scripts/embed_corpus.py --yes --max-shards 2   -> small paid test

Cache: data/cache/emb/shard_00000.npz (keys + float16 vectors). Cost log: data/cache/cost_log.jsonl
"""
import argparse
import json
import pathlib
import sys
import time

import numpy as np
import tiktoken
from dotenv import load_dotenv
from openai import OpenAI

ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from app.ingestion.corpus import load_chunks  # noqa: E402
from app.retrieval.embeddings import DIMS, MODEL, PRICE_PER_TOKEN, embed_batch  # noqa: E402

CORPUS = ROOT / "data" / "corpus" / "paragraphs.jsonl"
CACHE = ROOT / "data" / "cache" / "emb"
COST_LOG = ROOT / "data" / "cache" / "cost_log.jsonl"
SHARD = 256


def shard_path(i: int) -> pathlib.Path:
    return CACHE / f"shard_{i:05d}.npz"


def shard_ok(i: int, keys: list[str]) -> bool:
    p = shard_path(i)
    if not p.exists():
        return False
    try:
        with np.load(p) as z:
            return list(z["keys"]) == keys
    except Exception:  # noqa: BLE001 - truncated file from a killed run: redo it
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yes", action="store_true", help="actually call the API")
    ap.add_argument("--max-shards", type=int, default=None)
    args = ap.parse_args()

    chunks = load_chunks(CORPUS)
    shards = [chunks[i:i + SHARD] for i in range(0, len(chunks), SHARD)]
    todo = [i for i, s in enumerate(shards) if not shard_ok(i, [c.key for c in s])]
    enc = tiktoken.get_encoding("cl100k_base")
    todo_tokens = sum(len(enc.encode(c.text)) for i in todo for c in shards[i])
    cost = todo_tokens * PRICE_PER_TOKEN
    print(f"chunks: {len(chunks):,} in {len(shards)} shards; shards still to embed: {len(todo)}")
    print(f"tokens to embed: {todo_tokens:,}  ->  estimated cost ${cost:.4f}  (model {MODEL}, {DIMS}-d)")
    if not args.yes:
        print("Estimate only. Nothing was sent. Re-run with --yes to spend.")
        return

    load_dotenv()
    client = OpenAI(timeout=60)
    CACHE.mkdir(parents=True, exist_ok=True)
    spent_tokens, t0 = 0, time.time()
    for n, i in enumerate(todo[: args.max_shards]):
        keys = [c.key for c in shards[i]]
        vecs, tokens = embed_batch(client, [c.text for c in shards[i]])
        tmp = shard_path(i).with_suffix(".tmp.npz")
        np.savez(tmp, keys=np.array(keys), vecs=vecs.astype(np.float16))
        tmp.replace(shard_path(i))  # atomic: a killed run never leaves a half-written shard
        spent_tokens += tokens
        with COST_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"shard": i, "tokens": tokens, "usd": tokens * PRICE_PER_TOKEN}) + "\n")
        if n % 20 == 0:
            print(f"  shard {i} done ({n + 1}/{len(todo)}), tokens so far {spent_tokens:,}, {time.time() - t0:.0f}s")
    print(f"finished. tokens billed this run: {spent_tokens:,}  cost ${spent_tokens * PRICE_PER_TOKEN:.4f}")


if __name__ == "__main__":
    main()
