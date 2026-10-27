-- Idempotent schema. Vector/GIN indexes are created after the bulk load (db/indexes.sql, M3).
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id           BIGSERIAL PRIMARY KEY,
    source       TEXT NOT NULL,            -- where it came from (dataset, filename, URL)
    title        TEXT NOT NULL,
    doc_type     TEXT NOT NULL,            -- txt, md, html, pdf, json
    content_hash TEXT NOT NULL UNIQUE,     -- SHA-256 of normalized text (exact dedup)
    metadata     JSONB NOT NULL DEFAULT '{}',
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
    id          BIGSERIAL PRIMARY KEY,
    document_id BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INT NOT NULL,
    text        TEXT NOT NULL,
    n_terms     INT NOT NULL,              -- document length for BM25
    embedding   halfvec(512),
    tsv         tsvector,
    UNIQUE (document_id, chunk_index)
);

-- BM25 statistics maintained by us on every add/update/delete
CREATE TABLE IF NOT EXISTS term_stats (
    term TEXT PRIMARY KEY,
    df   INT NOT NULL
);

CREATE TABLE IF NOT EXISTS corpus_stats (
    id         INT PRIMARY KEY CHECK (id = 1),
    n_chunks   BIGINT NOT NULL DEFAULT 0,
    total_len  BIGINT NOT NULL DEFAULT 0
);
INSERT INTO corpus_stats (id) VALUES (1) ON CONFLICT DO NOTHING;

-- Public-demo spending cap (USD), updated atomically
CREATE TABLE IF NOT EXISTS spend (
    id       INT PRIMARY KEY CHECK (id = 1),
    usd      NUMERIC(10,5) NOT NULL DEFAULT 0,
    cap_usd  NUMERIC(10,5) NOT NULL DEFAULT 1.0
);
INSERT INTO spend (id) VALUES (1) ON CONFLICT DO NOTHING;

CREATE TABLE IF NOT EXISTS answer_cache (
    key        TEXT PRIMARY KEY,
    answer     JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
