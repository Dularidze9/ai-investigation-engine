"""Admin document endpoints with the pipeline, budget and OpenAI faked: no network, no cost."""
from contextlib import contextmanager
from types import SimpleNamespace

from fastapi.testclient import TestClient

from app.api import documents as docs_api
from app.ingestion.pipeline import IngestError, IngestResult
from app.main import app

client = TestClient(app)
TOKEN = {"X-Admin-Token": "s3cret"}


class FakePool:
    @contextmanager
    def connection(self):
        yield object()


class FakeEmbedder:
    def __init__(self, client):
        self.tokens = 0

    @property
    def cost_usd(self):
        return self.tokens * 2e-8


def _patch(monkeypatch, token="s3cret", ingest=None, delete=None):
    log = {"cleared": 0, "settled": [], "ingest": []}
    monkeypatch.setattr(docs_api, "get_settings", lambda: SimpleNamespace(admin_token=token))
    monkeypatch.setattr(docs_api, "get_pool", lambda: FakePool())
    monkeypatch.setattr(docs_api, "_openai", lambda: object())
    monkeypatch.setattr(docs_api, "Embedder", FakeEmbedder)
    monkeypatch.setattr(docs_api.budget, "reserve", lambda conn, amount: True)
    monkeypatch.setattr(docs_api.budget, "settle", lambda conn, r, a: log["settled"].append(a))
    monkeypatch.setattr(docs_api.answer_cache, "clear", lambda conn: log.__setitem__("cleared", log["cleared"] + 1))

    def fake_ingest(conn, filename, data, embedder, source, replace_document_id):
        log["ingest"].append((filename, data, replace_document_id))
        embedder.tokens = 500
        return ingest(filename, replace_document_id)
    monkeypatch.setattr(docs_api.pipeline, "ingest", fake_ingest)
    if delete:
        monkeypatch.setattr(docs_api.pipeline, "delete_document", delete)
    return log


def test_upload_without_token_is_401(monkeypatch):
    _patch(monkeypatch, ingest=lambda f, r: None)
    r = client.post("/api/documents", files={"file": ("a.txt", b"hello")})
    assert r.status_code == 401


def test_wrong_token_is_401(monkeypatch):
    _patch(monkeypatch, ingest=lambda f, r: None)
    r = client.post("/api/documents", files={"file": ("a.txt", b"hello")}, headers={"X-Admin-Token": "guess"})
    assert r.status_code == 401


def test_admin_disabled_when_token_not_configured(monkeypatch):
    _patch(monkeypatch, token="", ingest=lambda f, r: None)
    r = client.post("/api/documents", files={"file": ("a.txt", b"hello")}, headers={"X-Admin-Token": ""})
    assert r.status_code == 503


def test_upload_adds_document_clears_cache_and_charges_embedding(monkeypatch):
    log = _patch(monkeypatch, ingest=lambda f, r: IngestResult("added", 70001, "My doc", chunks=2, embed_tokens=500))
    r = client.post("/api/documents", files={"file": ("notes.md", b"# My doc\n\ntext")}, headers=TOKEN)
    assert r.status_code == 200 and r.json()["status"] == "added" and r.json()["document_id"] == 70001
    assert log["ingest"] == [("notes.md", b"# My doc\n\ntext", None)]
    assert log["cleared"] == 1 and log["settled"] == [500 * 2e-8]


def test_duplicate_does_not_clear_cache(monkeypatch):
    log = _patch(monkeypatch, ingest=lambda f, r: IngestResult("duplicate", None, "x", duplicate_of=5))
    r = client.post("/api/documents", files={"file": ("a.txt", b"hello")}, headers=TOKEN)
    assert r.json()["status"] == "duplicate" and r.json()["duplicate_of"] == 5 and log["cleared"] == 0


def test_update_passes_the_document_id(monkeypatch):
    log = _patch(monkeypatch, ingest=lambda f, r: IngestResult("updated", 99, "x", chunks=1))
    r = client.put("/api/documents/42", files={"file": ("a.txt", b"new text")}, headers=TOKEN)
    assert r.status_code == 200 and log["ingest"][0][2] == 42


def test_too_large_file_is_413_before_ingest(monkeypatch):
    log = _patch(monkeypatch, ingest=lambda f, r: None)
    big = b"x" * (docs_api.pipeline.MAX_UPLOAD_BYTES + 1)
    r = client.post("/api/documents", files={"file": ("a.txt", big)}, headers=TOKEN)
    assert r.status_code == 413 and log["ingest"] == []


def test_ingest_error_is_400_and_missing_document_404(monkeypatch):
    def bad(f, r):
        raise IngestError("document too long: 250 chunks (limit 200)" if r is None else f"document {r} not found")
    _patch(monkeypatch, ingest=bad)
    assert client.post("/api/documents", files={"file": ("a.txt", b"x")}, headers=TOKEN).status_code == 400
    assert client.put("/api/documents/7", files={"file": ("a.txt", b"x")}, headers=TOKEN).status_code == 404


def test_delete(monkeypatch):
    def fake_delete(conn, doc_id):
        if doc_id != 3:
            raise IngestError(f"document {doc_id} not found")
        return 4
    log = _patch(monkeypatch, ingest=lambda f, r: None, delete=fake_delete)
    assert client.delete("/api/documents/3").status_code == 401
    r = client.delete("/api/documents/3", headers=TOKEN)
    assert r.status_code == 200 and r.json()["chunks_removed"] == 4 and log["cleared"] == 1
    assert client.delete("/api/documents/8", headers=TOKEN).status_code == 404
