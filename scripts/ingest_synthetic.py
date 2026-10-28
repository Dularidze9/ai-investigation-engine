"""Ingest data/synthetic/* through the real pipeline (parse, chunk, dedup, embed, insert, stats).

  python scripts/ingest_synthetic.py         -> estimate only
  python scripts/ingest_synthetic.py --yes   -> ingest (paid, a fraction of a cent)
Files are processed in name order, so each original comes before its planted duplicate.
Rerunning is safe: everything already loaded is reported as a duplicate and not embedded again.
"""
import argparse
import json
import os
import pathlib
import sys

import psycopg
import tiktoken
from dotenv import load_dotenv

ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from app.agent.retriever import Embedder  # noqa: E402
from app.ingestion.pipeline import ingest, prepare  # noqa: E402
from app.retrieval.embeddings import PRICE_PER_TOKEN  # noqa: E402

SRC = ROOT / "data" / "synthetic"

ap = argparse.ArgumentParser()
ap.add_argument("--yes", action="store_true")
args = ap.parse_args()
files = sorted(SRC.iterdir())
enc = tiktoken.get_encoding("cl100k_base")
tokens = sum(len(enc.encode(c)) for f in files for c in prepare(f.name, f.read_bytes()).chunks)
print(f"{len(files)} files, at most {tokens:,} tokens to embed -> estimated ${tokens * PRICE_PER_TOKEN:.5f}")
if not args.yes:
    sys.exit("re-run with --yes to ingest")

load_dotenv()
from openai import OpenAI  # noqa: E402
embedder = Embedder(OpenAI(timeout=30))
with psycopg.connect(os.environ["DATABASE_URL"], connect_timeout=30) as conn:
    for f in files:
        r = ingest(conn, f.name, f.read_bytes(), embedder, source="synthetic:acme")
        extra = f" duplicate of document {r.duplicate_of}" if r.duplicate_of else ""
        print(f"  {f.name:38s} {r.status:9s} id={r.document_id} chunks={r.chunks}{extra}")
print(f"embedding cost: ${embedder.cost_usd:.6f}")
with (ROOT / "data" / "cache" / "cost_log.jsonl").open("a", encoding="utf-8") as fh:
    fh.write(json.dumps({"item": "M7 ingest synthetic", "usd": embedder.cost_usd}) + "\n")
