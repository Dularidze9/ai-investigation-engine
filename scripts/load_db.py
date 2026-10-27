"""M3 step 2: near-dup removal, resumable bulk load into Neon, BM25 stats, indexes.

Needs the embedding cache from embed_corpus.py. No OpenAI calls. Safe to kill and rerun:
documents that already have chunks are skipped, stats and indexes are rebuilt idempotently.
Refuses to run when the database is already over MAX_DB_MB (free tier is 512 MB).
"""
import json
import os
import pathlib
import sys
import time
from collections import Counter

import numpy as np
import psycopg
from dotenv import load_dotenv

ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from app.ingestion.corpus import load_chunks  # noqa: E402
from app.ingestion.dedup import near_duplicates  # noqa: E402
from app.retrieval.tokenize import tokenize  # noqa: E402

CORPUS = ROOT / "data" / "corpus" / "paragraphs.jsonl"
CACHE = ROOT / "data" / "cache"
NEAR_DUPS = CACHE / "near_dups.json"
MAX_DB_MB = 400
BATCH_DOCS = 2000


def load_vectors(chunks):
    vecs = np.zeros((len(chunks), 512), dtype=np.float32)
    for s, start in enumerate(range(0, len(chunks), 256)):
        part = chunks[start:start + 256]
        with np.load(CACHE / "emb" / f"shard_{s:05d}.npz") as z:
            assert list(z["keys"]) == [c.key for c in part], f"shard {s} does not match chunks"
            vecs[start:start + len(part)] = z["vecs"].astype(np.float32)
    return vecs


def db_mb(conn) -> float:
    return conn.execute("SELECT pg_database_size(current_database())").fetchone()[0] / 1e6


def vec_literal(v) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in v) + "]"


def main():
    load_dotenv()
    url = os.environ.get("DATABASE_URL")
    if not url:
        sys.exit("DATABASE_URL is not set")

    chunks = load_chunks(CORPUS)
    vecs = load_vectors(chunks)
    print(f"{len(chunks):,} chunks with embeddings loaded")

    if NEAR_DUPS.exists():
        dropped = {int(k): v for k, v in json.loads(NEAR_DUPS.read_text()).items()}
    else:
        t0 = time.time()
        dropped = near_duplicates(vecs, [c.text for c in chunks])
        NEAR_DUPS.write_text(json.dumps(dropped))
        print(f"near-duplicate scan took {time.time() - t0:.0f}s")
    keep = [i for i in range(len(chunks)) if i not in dropped]
    print(f"near-duplicates removed: {len(dropped):,}; chunks to load: {len(keep):,}")

    by_doc: dict[str, list[int]] = {}
    for i in keep:
        by_doc.setdefault(chunks[i].doc_hash, []).append(i)

    with psycopg.connect(url, connect_timeout=30) as conn:
        size = db_mb(conn)
        print(f"database size before load: {size:.1f} MB")
        if size > MAX_DB_MB:
            sys.exit(f"refusing to ingest: database is over {MAX_DB_MB} MB")

        done = {r[0] for r in conn.execute(
            "SELECT d.content_hash FROM documents d WHERE EXISTS (SELECT 1 FROM chunks c WHERE c.document_id = d.id)")}
        todo = [h for h in by_doc if h not in done]
        print(f"documents already loaded: {len(done):,}; to load: {len(todo):,}")

        conn.execute("CREATE TEMP TABLE stage (doc_hash TEXT, chunk_index INT, text TEXT, n_terms INT, embedding halfvec(512))")
        t0 = time.time()
        for b in range(0, len(todo), BATCH_DOCS):
            batch = todo[b:b + BATCH_DOCS]
            with conn.cursor() as cur:
                cur.executemany(
                    "INSERT INTO documents (source, title, doc_type, content_hash, metadata) "
                    "VALUES ('hotpotqa:distractor-validation', %s, 'txt', %s, '{}') ON CONFLICT (content_hash) DO NOTHING",
                    [(chunks[by_doc[h][0]].title, h) for h in batch])
                conn.execute("TRUNCATE stage")
                with cur.copy("COPY stage (doc_hash, chunk_index, text, n_terms, embedding) FROM STDIN") as cp:
                    for h in batch:
                        for i in by_doc[h]:
                            c = chunks[i]
                            cp.write_row((h, c.index, c.text, len(tokenize(c.text)), vec_literal(vecs[i])))
                cur.execute(
                    "INSERT INTO chunks (document_id, chunk_index, text, n_terms, embedding, tsv) "
                    "SELECT d.id, s.chunk_index, s.text, s.n_terms, s.embedding, to_tsvector('english', s.text) "
                    "FROM stage s JOIN documents d ON d.content_hash = s.doc_hash")
            conn.commit()
            print(f"  loaded {min(b + BATCH_DOCS, len(todo)):,}/{len(todo):,} docs  ({time.time() - t0:.0f}s, db {db_mb(conn):.0f} MB)")
            if db_mb(conn) > MAX_DB_MB:
                sys.exit(f"stopped: database passed {MAX_DB_MB} MB")

        # BM25 statistics, rebuilt from the kept chunks (idempotent)
        df, total_len = Counter(), 0
        for i in keep:
            toks = tokenize(chunks[i].text)
            total_len += len(toks)
            df.update(set(toks))
        conn.execute("TRUNCATE term_stats")
        with conn.cursor() as cur, cur.copy("COPY term_stats (term, df) FROM STDIN") as cp:
            for term, n in df.items():
                cp.write_row((term, n))
        conn.execute("UPDATE corpus_stats SET n_chunks = %s, total_len = %s WHERE id = 1", (len(keep), total_len))
        conn.commit()
        print(f"term_stats: {len(df):,} terms; corpus: N={len(keep):,}, avgdl={total_len / len(keep):.1f}")

    # indexes and vacuum need autocommit
    with psycopg.connect(url, connect_timeout=30, autocommit=True) as conn:
        t0 = time.time()
        conn.execute("SET maintenance_work_mem = '256MB'")
        conn.execute((ROOT / "db" / "indexes.sql").read_text(encoding="utf-8"))
        print(f"indexes built in {time.time() - t0:.0f}s")
        conn.execute("VACUUM ANALYZE")
        print(f"database size after load: {db_mb(conn):.1f} MB")


if __name__ == "__main__":
    main()
