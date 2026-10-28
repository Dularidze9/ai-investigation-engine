import json
from types import SimpleNamespace as NS

import httpx
import pytest
from openai import APITimeoutError, BadRequestError, InternalServerError, RateLimitError

from app.llm.client import LLMClient, LLMError
from app.llm.pricing import cost_usd

REQ = httpx.Request("POST", "http://x")


def _resp(content=None, tool_calls=None, pin=1000, pout=100):
    msg = NS(content=content, tool_calls=tool_calls)
    return NS(choices=[NS(message=msg)], usage=NS(prompt_tokens=pin, completion_tokens=pout))


class FakeOpenAI:
    def __init__(self, outcomes):
        self.outcomes, self.calls = list(outcomes), []
        self.chat = NS(completions=NS(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        o = self.outcomes.pop(0)
        if isinstance(o, Exception):
            raise o
        return o


def rate_limited():
    return RateLimitError("slow down", response=httpx.Response(429, request=REQ), body=None)


def test_success_parses_content_and_cost_and_forces_reasoning_none():
    fake = FakeOpenAI([_resp("hi", pin=1000, pout=100)])
    r = LLMClient(fake).chat([{"role": "user", "content": "x"}])
    assert r.content == "hi" and r.input_tokens == 1000 and r.output_tokens == 100
    assert r.cost == pytest.approx(cost_usd("gpt-6-luna", 1000, 100)) == pytest.approx(0.00015)
    assert fake.calls[0]["reasoning_effort"] == "none"


def test_tool_calls_are_parsed():
    tc = NS(id="c1", function=NS(name="search_documents", arguments=json.dumps({"query": "q"})))
    r = LLMClient(FakeOpenAI([_resp(tool_calls=[tc])])).chat([], tools=[{"x": 1}])
    assert r.tool_calls[0].name == "search_documents" and r.tool_calls[0].arguments == {"query": "q"}


def test_invalid_tool_arguments_do_not_crash():
    tc = NS(id="c1", function=NS(name="get_document", arguments="{not json"))
    r = LLMClient(FakeOpenAI([_resp(tool_calls=[tc])])).chat([])
    assert "_invalid_json" in r.tool_calls[0].arguments


def test_retries_429_timeout_and_5xx_with_backoff():
    outcomes = [rate_limited(), APITimeoutError(request=REQ),
                InternalServerError("boom", response=httpx.Response(500, request=REQ), body=None), _resp("ok")]
    waits = []
    r = LLMClient(FakeOpenAI(outcomes), sleep=waits.append).chat([])
    assert r.content == "ok" and waits == [1, 2, 4]


def test_400_is_not_retried():
    bad = BadRequestError("bad", response=httpx.Response(400, request=REQ), body=None)
    fake = FakeOpenAI([bad, _resp("never")])
    with pytest.raises(LLMError, match="rejected"):
        LLMClient(fake, sleep=lambda s: None).chat([])
    assert len(fake.calls) == 1


def test_gives_up_after_retries():
    fake = FakeOpenAI([rate_limited() for _ in range(10)])
    with pytest.raises(LLMError, match="unavailable"):
        LLMClient(fake, retries=2, sleep=lambda s: None).chat([])
    assert len(fake.calls) == 3


def test_disk_cache_returns_same_answer_for_free(tmp_path):
    fake = FakeOpenAI([_resp("first")])
    client = LLMClient(fake, cache_dir=tmp_path)
    a = client.chat([{"role": "user", "content": "same"}])
    b = client.chat([{"role": "user", "content": "same"}])
    assert len(fake.calls) == 1 and b.content == "first" and b.cached and b.cost == 0.0 and a.cost > 0
