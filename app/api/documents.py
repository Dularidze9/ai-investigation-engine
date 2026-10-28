"""Document management.

Public (read only):  GET /api/documents          list and search by title
                     GET /api/documents/{id}     one document with its chunks (the "view source" page)
Admin (X-Admin-Token header):
                     POST   /api/documents       upload a file (txt, md, html, pdf, json) -> pipeline.ingest
                     PUT    /api/documents/{id}  replace a document with a new file (= update)
                     DELETE /api/documents/{id}  delete a document and its chunks

Every change clears the answer cache, and the embedding cost of an upload is added to the
same spending cap that protects /api/query.
"""
import hmac
import logging

from fastapi import APIRouter, Depends, File, Header, HTTPException, Query, UploadFile
from openai import OpenAIError
from starlette.concurrency import run_in_threadpool  # ingest is blocking (DB + OpenAI)

from app.agent.retriever import Embedder
from app.api.query import _openai
from app.config import get_settings
from app.db import get_pool
from app.guard import answer_cache, budget
from app.ingestion import pipeline
from app.ingestion.parsers import ParseError

log = logging.getLogger(__name__)
router = APIRouter()
CORPUS_SOURCE = "hotpotqa:distractor-validation"


def require_admin(x_admin_token: str = Header(default="")) -> None:
    expected = get_settings().admin_token
    if not expected:
        raise HTTPException(status_code=503, detail="Admin actions are disabled: ADMIN_TOKEN is not set.")
    # compare_digest takes the same time wherever the strings differ (no timing leak)
    if not hmac.compare_digest(x_admin_token.encode(), expected.encode()):
        raise HTTPException(status_code=401, detail="Missing or wrong admin token.")


@router.get("/api/documents")
def list_documents(q: str = "", uploaded_only: bool = False,
                   limit: int = Query(20, ge=1, le=100), offset: int = Query(0, ge=0)):
    where, args = [], []
    if q.strip():
        where.append("d.title ILIKE %s")
        args.append("%" + q.strip().replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%")
    if uploaded_only:
        where.append("d.source <> %s")
        args.append(CORPUS_SOURCE)
    sql_where = ("WHERE " + " AND ".join(where)) if where else ""
    with get_pool().connection() as conn:
        total = conn.execute(f"SELECT count(*) FROM documents d {sql_where}", args).fetchone()[0]
        rows = conn.execute(
            f"SELECT d.id, d.title, d.source, d.doc_type, d.updated_at, "
            f"(SELECT count(*) FROM chunks c WHERE c.document_id = d.id) "
            f"FROM documents d {sql_where} ORDER BY d.id DESC LIMIT %s OFFSET %s", [*args, limit, offset]).fetchall()
    return {"total": total, "documents": [
        {"id": r[0], "title": r[1], "source": r[2], "doc_type": r[3], "updated_at": r[4].isoformat(), "chunks": r[5]}
        for r in rows]}


@router.get("/api/documents/{document_id}")
def get_document(document_id: int):
    with get_pool().connection() as conn:
        doc = conn.execute("SELECT id, title, source, doc_type, metadata, created_at, updated_at "
                           "FROM documents WHERE id = %s", (document_id,)).fetchone()
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found.")
        chunks = conn.execute("SELECT id, chunk_index, text FROM chunks WHERE document_id = %s "
                              "ORDER BY chunk_index", (document_id,)).fetchall()
    return {"id": doc[0], "title": doc[1], "source": doc[2], "doc_type": doc[3], "metadata": doc[4],
            "created_at": doc[5].isoformat(), "updated_at": doc[6].isoformat(),
            "chunks": [{"chunk_id": c[0], "index": c[1], "text": c[2]} for c in chunks]}


async def _read_upload(file: UploadFile) -> bytes:
    data = await file.read(pipeline.MAX_UPLOAD_BYTES + 1)  # never read more than the limit + 1 byte
    if len(data) > pipeline.MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"File too large (limit {pipeline.MAX_UPLOAD_BYTES:,} bytes).")
    return data


def _ingest(filename: str, data: bytes, replace_id: int | None) -> dict:
    embedder = Embedder(_openai())
    try:
        with get_pool().connection() as conn:
            if not budget.reserve(conn, 0.0):  # the embedding cost goes on the same cap as questions
                raise HTTPException(status_code=429, detail="The demo budget is used up.")
            r = pipeline.ingest(conn, filename, data, embedder, source=f"upload:{filename}",
                                replace_document_id=replace_id)
            if r.status in ("added", "updated"):
                answer_cache.clear(conn)
    except (pipeline.IngestError, ParseError) as e:
        status = 404 if "not found" in str(e) else 400
        raise HTTPException(status_code=status, detail=str(e)) from e
    except OpenAIError as e:
        raise HTTPException(status_code=503, detail="The embedding service is unavailable, try again later.") from e
    finally:
        if embedder.tokens:  # also charged when the file turned out to be a near-duplicate
            with get_pool().connection() as conn:
                budget.settle(conn, 0.0, embedder.cost_usd)
    log.info("ingest %s: %s doc=%s chunks=%d tokens=%d", filename, r.status, r.document_id, r.chunks, r.embed_tokens)
    return {"status": r.status, "document_id": r.document_id, "title": r.title, "chunks": r.chunks,
            "duplicate_of": r.duplicate_of, "embed_cost_usd": round(embedder.cost_usd, 6)}


@router.post("/api/documents", dependencies=[Depends(require_admin)])
async def add_document(file: UploadFile = File(...)):
    data = await _read_upload(file)
    return await run_in_threadpool(_ingest, file.filename or "upload.txt", data, None)


@router.put("/api/documents/{document_id}", dependencies=[Depends(require_admin)])
async def update_document(document_id: int, file: UploadFile = File(...)):
    data = await _read_upload(file)
    return await run_in_threadpool(_ingest, file.filename or "upload.txt", data, document_id)


@router.delete("/api/documents/{document_id}", dependencies=[Depends(require_admin)])
def delete_document(document_id: int):
    with get_pool().connection() as conn:
        try:
            n = pipeline.delete_document(conn, document_id)
        except pipeline.IngestError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e
        answer_cache.clear(conn)
    log.info("deleted document %d (%d chunks)", document_id, n)
    return {"status": "deleted", "document_id": document_id, "chunks_removed": n}
