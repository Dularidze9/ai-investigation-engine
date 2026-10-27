from fastapi.testclient import TestClient

from app.api import health
from app.main import app

client = TestClient(app)


def test_health_ok(monkeypatch):
    monkeypatch.setattr(health, "check_db", lambda: {"db": "ok", "pgvector": "0.8.0"})
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"db": "ok", "pgvector": "0.8.0"}


def test_health_db_down(monkeypatch):
    def boom():
        raise RuntimeError("database unreachable: OperationalError")
    monkeypatch.setattr(health, "check_db", boom)
    r = client.get("/health")
    assert r.status_code == 503
    assert r.json()["db"] == "down"
