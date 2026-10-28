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


# ---- /api/query (agent, budget and OpenAI are faked: no network, no cost) ----
from contextlib import contextmanager  # noqa: E402

from app.agent.agent import AgentResult  # noqa: E402
from app.api import query as query_api  # noqa: E402
from app.llm.client import LLMError  # noqa: E402


class FakePool:
    @contextmanager
    def connection(self):
        yield object()


def _patch_query(monkeypatch, reserve_ok=True, run=None):
    settled = []
    monkeypatch.setattr(query_api, "get_pool", lambda: FakePool())
    monkeypatch.setattr(query_api.budget, "reserve", lambda conn: reserve_ok)
    monkeypatch.setattr(query_api.budget, "settle", lambda conn, reserved, actual: settled.append((reserved, actual)))
    monkeypatch.setattr(query_api, "_openai", lambda: object())

    class FakeAgent:
        def __init__(self, llm, retriever):
            pass

        def run(self, question):
            return run(question)

    monkeypatch.setattr(query_api, "Agent", FakeAgent)
    return settled


def test_query_rejects_empty_question(monkeypatch):
    _patch_query(monkeypatch, run=lambda q: None)
    assert client.post("/api/query", json={"question": ""}).status_code == 422


def test_query_budget_exhausted_returns_429(monkeypatch):
    _patch_query(monkeypatch, reserve_ok=False, run=lambda q: None)
    r = client.post("/api/query", json={"question": "Who directed Sinister?"})
    assert r.status_code == 429 and "budget" in r.json()["detail"]


def test_query_success_settles_real_cost(monkeypatch):
    res = AgentResult(question="q", status="answered", answer="Scott Derrickson", cost_usd=0.003)
    settled = _patch_query(monkeypatch, run=lambda q: res)
    r = client.post("/api/query", json={"question": "Who directed Sinister?"})
    assert r.status_code == 200 and r.json()["answer"] == "Scott Derrickson"
    assert settled == [(query_api.budget.MAX_QUERY_COST, 0.003)]


def test_query_llm_down_returns_503_and_keeps_reservation(monkeypatch):
    def boom(q):
        raise LLMError("down")
    settled = _patch_query(monkeypatch, run=boom)
    r = client.post("/api/query", json={"question": "Who directed Sinister?"})
    assert r.status_code == 503
    assert settled == [(query_api.budget.MAX_QUERY_COST, query_api.budget.MAX_QUERY_COST)]
