"""Run one search against Neon and print the hits.

  python scripts/search_demo.py "query"                        -> BM25 only, free
  python scripts/search_demo.py "query" --mode hybrid --yes    -> embeds the query (paid, ~$0.000001)
  add --mmr for MMR diversification
"""
import argparse
import os
import pathlib
import sys
import time

import psycopg
import tiktoken
from dotenv import load_dotenv

ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from app.retrieval.embeddings import PRICE_PER_TOKEN, embed_batch  # noqa: E402
from app.retrieval.hybrid import search  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("query")
ap.add_argument("--mode", default="bm25", choices=["bm25", "dense", "hybrid"])
ap.add_argument("--mmr", action="store_true")
ap.add_argument("--k", type=int, default=10)
ap.add_argument("--yes", action="store_true", help="allow the paid query embedding")
args = ap.parse_args()
load_dotenv()

qvec = None
if args.mode != "bm25":
    tokens = len(tiktoken.get_encoding("cl100k_base").encode(args.query))
    print(f"query embedding: {tokens} tokens, estimated ${tokens * PRICE_PER_TOKEN:.7f}")
    if not args.yes:
        sys.exit("re-run with --yes to embed the query")
    from openai import OpenAI
    vecs, _ = embed_batch(OpenAI(timeout=30), [args.query])
    qvec = vecs[0]

with psycopg.connect(os.environ["DATABASE_URL"], connect_timeout=30) as conn:
    t = time.time()
    hits = search(conn, args.query, qvec, k=args.k, mode=args.mode, use_mmr=args.mmr)
    print(f"{args.mode}{' + MMR' if args.mmr else ''}: {len(hits)} hits in {time.time() - t:.2f}s")
    for i, h in enumerate(hits, 1):
        snippet = h.text.split("\n\n", 1)[-1][:80].replace("\n", " ")
        print(f"{i:2d}. [{h.chunk_id}] {h.title[:35]:35s} score={h.score:.4f} "
              f"dense#={h.dense_rank} bm25#={h.bm25_rank} | {snippet}")
