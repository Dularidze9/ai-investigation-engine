"""Verify that the incrementally maintained BM25 statistics equal a full recount. Free, ~20 s.

Recomputes N, total length and df for EVERY term from the chunk texts and compares with
corpus_stats and term_stats. Any difference means add/update/delete has a bug.
"""
import os
import pathlib
import sys
from collections import Counter

import psycopg
from dotenv import load_dotenv

ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from app.retrieval.tokenize import tokenize  # noqa: E402

load_dotenv()
with psycopg.connect(os.environ["DATABASE_URL"], connect_timeout=30) as conn:
    df, n, total = Counter(), 0, 0
    with conn.cursor(name="all_chunks") as cur:  # server-side cursor: streams instead of loading 40 MB at once
        cur.itersize = 5000
        cur.execute("SELECT text, n_terms FROM chunks")
        bad_len = 0
        for text, n_terms in cur:
            toks = tokenize(text)
            n += 1
            total += len(toks)
            bad_len += len(toks) != n_terms
            df.update(set(toks))
    stored_n, stored_total = conn.execute("SELECT n_chunks, total_len FROM corpus_stats").fetchone()
    stored_df = dict(conn.execute("SELECT term, df FROM term_stats").fetchall())

wrong = {t for t in set(df) | set(stored_df) if df.get(t, 0) != stored_df.get(t, 0)}
print(f"chunks: recount {n:,} | corpus_stats {stored_n:,}  -> {'OK' if n == stored_n else 'MISMATCH'}")
print(f"total length: recount {total:,} | corpus_stats {stored_total:,}  -> {'OK' if total == stored_total else 'MISMATCH'}")
print(f"chunks whose n_terms differs from a recount: {bad_len}")
print(f"terms: recount {len(df):,} | term_stats {len(stored_df):,}; terms with a wrong df: {len(wrong)}")
for t in sorted(wrong)[:10]:
    print(f"   {t!r}: recount {df.get(t, 0)} vs stored {stored_df.get(t, 0)}")
ok = n == stored_n and total == stored_total and not wrong and not bad_len
print("STATS CONSISTENT:", ok)
sys.exit(0 if ok else 1)
