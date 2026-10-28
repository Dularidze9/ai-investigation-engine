"""Add, update and delete documents, keeping the BM25 statistics correct.

prepare()  pure: size limit, parse, chunk (strategy A), chunk cap, statistics delta. No DB, no API.
ingest()   duplicate checks (exact hash, then near-duplicate by embedding + Jaccard), DB size guard,
           embed, then ONE transaction: insert document + chunks and add the statistics delta.
           With replace_document_id the old version is deleted in the same transaction (= update).
delete_document()  subtracts the statistics delta and deletes the document (chunks cascade).

term_stats.df counts chunks containing a term, exactly as load_db.py built it, so documents
added through the API and the bulk-loaded corpus share one set of statistics.
"""
from collections import Counter
from dataclasses import dataclass

from app.ingestion.chunking import chunk_a
from app.ingestion.dedup import COSINE_MIN, JACCARD_MIN, content_hash, jaccard
from app.ingestion.parsers import ParsedDoc, parse_file
from app.retrieval.dense import dense_search, vec_literal
from app.retrieval.tokenize import tokenize

MAX_UPLOAD_BYTES = 2_000_000
MAX_CHUNKS_PER_DOC = 200
MAX_DB_MB = 400
NEAR_DUP_SHARE = 0.8  # a document is a near-duplicate if >= 80% of its chunks nearly match one existing document


class IngestError(ValueError):
    pass


@dataclass
class Prepared:
    doc: ParsedDoc
    content_hash: str
    chunks: list[str]
    df: Counter          # term -> number of the new chunks containing it
    lengths: list[int]   # tokens per chunk (n_terms)


@dataclass
class IngestResult:
    status: str                  # added | updated | duplicate | unchanged
    document_id: int | None
    title: str
    chunks: int = 0
    duplicate_of: int | None = None
    embed_tokens: int = 0


def prepare(filename: str, data: bytes) -> Prepared:
    if len(data) > MAX_UPLOAD_BYTES:
        raise IngestError(f"file too large: {len(data):,} bytes (limit {MAX_UPLOAD_BYTES:,})")
    doc = parse_file(filename, data)
    if not doc.text.strip():
        raise IngestError("no text could be extracted from the file")
    chunks = chunk_a(doc.title, doc.text)
    if len(chunks) > MAX_CHUNKS_PER_DOC:
        raise IngestError(f"document too long: {len(chunks)} chunks (limit {MAX_CHUNKS_PER_DOC})")
    tokens = [tokenize(c) for c in chunks]
    df = Counter(t for toks in tokens for t in set(toks))
    return Prepared(doc, content_hash(doc.text), chunks, df, [len(t) for t in tokens])


def db_size_mb(conn) -> float:
    return conn.execute("SELECT pg_database_size(current_database())").fetchone()[0] / 1e6


def _near_duplicate_of(conn, chunks: list[str], vecs, exclude_doc: int | None) -> int | None:
    votes = Counter()
    for text, vec in zip(chunks, vecs):
        for chunk_id, sim in dense_search(conn, vec, 3):
            if sim < COSINE_MIN:
                break
            row = conn.execute("SELECT document_id, text FROM chunks WHERE id = %s", (chunk_id,)).fetchone()
            # compare bodies without the "title\n\n" prefix: a copy saved under another title is still a copy
            if row and row[0] != exclude_doc and jaccard(_body(text), _body(row[1])) >= JACCARD_MIN:
                votes[row[0]] += 1
                break
    if votes:
        doc_id, n = votes.most_common(1)[0]
        if n >= NEAR_DUP_SHARE * len(chunks):
            return doc_id
    return None


def _body(chunk_text: str) -> str:
    return chunk_text.split("\n\n", 1)[-1]


def _add_stats(conn, df: Counter, n_chunks: int, total_len: int, sign: int) -> None:
    if sign > 0:
        conn.cursor().executemany(
            "INSERT INTO term_stats (term, df) VALUES (%s, %s) ON CONFLICT (term) DO UPDATE SET df = term_stats.df + EXCLUDED.df",
            list(df.items()))
    else:
        conn.cursor().executemany("UPDATE term_stats SET df = df - %s WHERE term = %s",
                                  [(n, t) for t, n in df.items()])
        conn.execute("DELETE FROM term_stats WHERE term = ANY(%s) AND df <= 0", (list(df),))
    conn.execute("UPDATE corpus_stats SET n_chunks = n_chunks + %s, total_len = total_len + %s WHERE id = 1",
                 (sign * n_chunks, sign * total_len))


def _delete_in_tx(conn, document_id: int) -> int:
    texts = [r[0] for r in conn.execute("SELECT text FROM chunks WHERE document_id = %s", (document_id,))]
    tokens = [tokenize(t) for t in texts]
    _add_stats(conn, Counter(t for toks in tokens for t in set(toks)), len(texts), sum(map(len, tokens)), -1)
    conn.execute("DELETE FROM documents WHERE id = %s", (document_id,))
    return len(texts)


def delete_document(conn, document_id: int) -> int:
    """Returns the number of chunks removed; raises IngestError if the document does not exist."""
    with conn.transaction():
        if not conn.execute("SELECT 1 FROM documents WHERE id = %s", (document_id,)).fetchone():
            raise IngestError(f"document {document_id} not found")
        return _delete_in_tx(conn, document_id)


def ingest(conn, filename: str, data: bytes, embedder, source: str = "upload",
           replace_document_id: int | None = None) -> IngestResult:
    p = prepare(filename, data)
    title = p.doc.title

    if replace_document_id is not None:
        row = conn.execute("SELECT content_hash FROM documents WHERE id = %s", (replace_document_id,)).fetchone()
        if not row:
            raise IngestError(f"document {replace_document_id} not found")
        if row[0] == p.content_hash:
            return IngestResult("unchanged", replace_document_id, title)
    existing = conn.execute("SELECT id FROM documents WHERE content_hash = %s", (p.content_hash,)).fetchone()
    if existing and existing[0] != replace_document_id:
        return IngestResult("duplicate", None, title, duplicate_of=existing[0])
    if db_size_mb(conn) > MAX_DB_MB:
        raise IngestError(f"refusing to ingest: the database is over {MAX_DB_MB} MB")

    before = embedder.tokens
    vecs = embedder.embed_many(p.chunks)
    used = embedder.tokens - before
    dup = _near_duplicate_of(conn, p.chunks, vecs, replace_document_id)
    if dup is not None:
        return IngestResult("duplicate", None, title, duplicate_of=dup, embed_tokens=used)

    with conn.transaction():
        if replace_document_id is not None:
            _delete_in_tx(conn, replace_document_id)
        doc_id = conn.execute(
            "INSERT INTO documents (source, title, doc_type, content_hash, metadata) "
            "VALUES (%s, %s, %s, %s, %s::jsonb) RETURNING id",
            (source, title, p.doc.doc_type, p.content_hash, _json(p.doc.metadata))).fetchone()[0]
        conn.cursor().executemany(
            "INSERT INTO chunks (document_id, chunk_index, text, n_terms, embedding, tsv) "
            "VALUES (%s, %s, %s, %s, %s::halfvec, to_tsvector('english', %s))",
            [(doc_id, i, text, n, vec_literal(v), text) for i, (text, n, v) in enumerate(zip(p.chunks, p.lengths, vecs))])
        _add_stats(conn, p.df, len(p.chunks), sum(p.lengths), +1)
    status = "updated" if replace_document_id is not None else "added"
    return IngestResult(status, doc_id, title, len(p.chunks), embed_tokens=used)


def _json(d: dict) -> str:
    import json
    return json.dumps(d, ensure_ascii=False, default=str)
