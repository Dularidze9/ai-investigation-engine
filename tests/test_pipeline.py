import pytest

from app.ingestion import pipeline
from app.ingestion.pipeline import IngestError, prepare
from app.ingestion.parsers import ParseError


def test_prepare_counts_df_per_chunk_not_per_occurrence():
    p = prepare("note.md", b"# Gateway\n\npool pool pool reduced\n\nfraud model enabled")
    assert len(p.chunks) == 2                       # one chunk per paragraph; the H1 is the title, not a chunk
    assert p.df["pool"] == 1                        # 3 occurrences, 1 chunk
    assert p.df["gateway"] == 2                     # the title is in every chunk
    assert p.lengths == [len(c.split()) for c in ["gateway pool pool pool reduced", "gateway fraud model enabled"]]


def test_prepare_same_text_same_hash():
    assert prepare("a.txt", b"Same  text").content_hash == prepare("b.txt", b"same text").content_hash


def test_upload_size_limit(monkeypatch):
    monkeypatch.setattr(pipeline, "MAX_UPLOAD_BYTES", 10)
    with pytest.raises(IngestError, match="too large"):
        prepare("a.txt", b"x" * 11)


def test_chunk_cap(monkeypatch):
    monkeypatch.setattr(pipeline, "MAX_CHUNKS_PER_DOC", 2)
    with pytest.raises(IngestError, match="too long"):
        prepare("a.md", b"one\n\ntwo\n\nthree")


def test_empty_document_rejected():
    with pytest.raises(IngestError, match="no text"):
        prepare("a.txt", b"   \n\n ")


def test_unsupported_type_rejected():
    with pytest.raises(ParseError):
        prepare("a.exe", b"MZ")
