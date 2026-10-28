"""Time each step of BM25 keyword search against Neon (free, no OpenAI)."""
import os
import pathlib
import sys
import time

import psycopg
from dotenv import load_dotenv

ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from app.retrieval import bm25  # noqa: E402
from app.retrieval.keyword import keyword_candidates, load_stats  # noqa: E402
from app.retrieval.tokenize import tokenize  # noqa: E402

query = sys.argv[1] if len(sys.argv) > 1 else "Were Scott Derrickson and Ed Wood of the same nationality?"
load_dotenv()
with psycopg.connect(os.environ["DATABASE_URL"], connect_timeout=30) as conn:
    conn.execute("SELECT 1")
    terms = tokenize(query)
    tsq = " | ".join(dict.fromkeys(terms))
    print("terms:", terms)
    n = conn.execute("SELECT count(*) FROM chunks, to_tsquery('english', %s) q WHERE tsv @@ q", (tsq,)).fetchone()[0]
    print("chunks matching the OR query:", n)
    for run in (1, 2):
        t = time.time(); texts = keyword_candidates(conn, terms); t1 = time.time() - t
        t = time.time(); df, stats = load_stats(conn, terms); t2 = time.time() - t
        t = time.time(); bm25.rank(terms, {c: tokenize(x) for c, x in texts.items()}, df, stats, 50); t3 = time.time() - t
        print(f"run {run}: candidates {t1:.2f}s, stats {t2:.2f}s, python bm25 {t3:.3f}s")
