"""Sanity-check the loaded database: counts, a dense probe (HNSW) and a keyword probe (GIN). Free, no OpenAI call."""
import os
import time

import psycopg
from dotenv import load_dotenv

load_dotenv()
with psycopg.connect(os.environ["DATABASE_URL"], connect_timeout=30) as c:
    print("chunks:", c.execute("select count(*) from chunks").fetchone()[0])
    print("chunks missing tsv or embedding:",
          c.execute("select count(*) from chunks where tsv is null or embedding is null").fetchone()[0])
    print("corpus_stats (N, total_len):", c.execute("select n_chunks, total_len from corpus_stats").fetchone())
    row = c.execute("select id, replace(left(text, 70), E'\\n', ' '), embedding::text from chunks "
                    "where text like 'Scott Derrickson%' limit 1").fetchone()
    print("probe chunk:", row[0], repr(row[1]))
    t = time.time()
    res = c.execute("select id, replace(left(text, 70), E'\\n', ' '), 1 - (embedding <=> %s::halfvec) "
                    "from chunks order by embedding <=> %s::halfvec limit 4", (row[2], row[2])).fetchall()
    print(f"dense top-4 ({time.time() - t:.2f}s):")
    for r in res:
        print("  ", r)
    t = time.time()
    res = c.execute("select id, replace(left(text, 70), E'\\n', ' ') from chunks "
                    "where tsv @@ to_tsquery('english', 'scott & derrickson') limit 4").fetchall()
    print(f"keyword 'scott & derrickson' ({time.time() - t:.2f}s):")
    for r in res:
        print("  ", r)
    plan = [r[0] for r in c.execute("explain select id from chunks order by embedding <=> %s::halfvec limit 5", (row[2],))]
    print("dense plan uses HNSW:", any("hnsw" in p for p in plan))
