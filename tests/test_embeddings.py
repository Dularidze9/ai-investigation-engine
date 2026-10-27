import httpx
import numpy as np
import pytest
from openai import RateLimitError

from app.retrieval.embeddings import DIMS, embed_batch
from app.retrieval.tokenize import tokenize


class _Item:
    def __init__(self, v):
        self.embedding = v


class _Resp:
    def __init__(self, n):
        self.data = [_Item([3.0] + [4.0] + [0.0] * (DIMS - 2)) for _ in range(n)]
        self.usage = type("U", (), {"total_tokens": 7 * n})()


class FakeClient:
    """Fails with 429 `fail_times` times, then succeeds."""
    def __init__(self, fail_times=0):
        self.fail_times, self.calls = fail_times, 0
        self.embeddings = self

    def create(self, model, input, dimensions):
        self.calls += 1
        if self.calls <= self.fail_times:
            req = httpx.Request("POST", "http://x")
            raise RateLimitError("slow down", response=httpx.Response(429, request=req), body=None)
        return _Resp(len(input))


def test_embed_batch_normalizes_and_counts_tokens():
    vecs, tokens = embed_batch(FakeClient(), ["a", "b"])
    assert vecs.shape == (2, DIMS) and tokens == 14
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0)


def test_embed_batch_retries_on_429_with_backoff():
    waits = []
    c = FakeClient(fail_times=2)
    vecs, _ = embed_batch(c, ["a"], sleep=waits.append)
    assert c.calls == 3 and waits == [1, 2]
    assert vecs.shape == (1, DIMS)


def test_embed_batch_gives_up_after_retries():
    with pytest.raises(RateLimitError):
        embed_batch(FakeClient(fail_times=99), ["a"], retries=3, sleep=lambda s: None)


def test_tokenize_lowercases_and_drops_stopwords():
    assert tokenize("The Payment Gateway of ACME, v2!") == ["payment", "gateway", "acme", "v2"]
