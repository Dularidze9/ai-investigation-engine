"""Chat client: retries with backoff, timeouts, cost accounting and an optional disk cache.

All calls use reasoning_effort="none": M0 showed gpt-6-luna rejects tools with any other effort.
Errors the caller should handle are raised as LLMError (after retries are exhausted, or at once for
requests that can never succeed, such as a 400).
"""
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from openai import APIConnectionError, APIStatusError, RateLimitError

from app.llm.pricing import DEFAULT_MODEL, cost_usd

log = logging.getLogger(__name__)


class LLMError(Exception):
    pass


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class LLMResponse:
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0
    cached: bool = False


class LLMClient:
    def __init__(self, client, model: str = DEFAULT_MODEL, timeout: float = 30.0, retries: int = 4,
                 sleep=time.sleep, cache_dir: Path | None = None):
        self.client, self.model, self.timeout, self.retries = client, model, timeout, retries
        self.sleep, self.cache_dir = sleep, cache_dir

    def chat(self, messages: list[dict], tools: list[dict] | None = None,
             json_schema: dict | None = None, max_tokens: int = 800) -> LLMResponse:
        kwargs = {"model": self.model, "messages": messages, "reasoning_effort": "none",
                  "max_completion_tokens": max_tokens, "timeout": self.timeout}
        if tools:
            kwargs["tools"] = tools
        if json_schema:
            kwargs["response_format"] = {"type": "json_schema", "json_schema": json_schema}

        key = self._cache_key(kwargs)
        if (hit := self._cache_get(key)) is not None:
            return hit
        resp = self._call_with_retry(kwargs)
        msg = resp.choices[0].message
        calls = []
        for tc in msg.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {"_invalid_json": tc.function.arguments}
            calls.append(ToolCall(tc.id, tc.function.name, args))
        u = resp.usage
        out = LLMResponse(msg.content, calls, u.prompt_tokens, u.completion_tokens,
                          cost_usd(self.model, u.prompt_tokens, u.completion_tokens))
        self._cache_put(key, out)
        return out

    def _call_with_retry(self, kwargs):
        for attempt in range(self.retries + 1):
            try:
                return self.client.chat.completions.create(**kwargs)
            except (RateLimitError, APIConnectionError) as e:  # 429, timeouts, network
                err = e
            except APIStatusError as e:
                if e.status_code < 500:  # 400/401/403/404: retrying cannot help
                    raise LLMError(f"LLM request rejected ({e.status_code})") from e
                err = e
            if attempt == self.retries:
                break
            wait = 2 ** attempt
            log.warning("LLM call failed (%s), retry %d in %ds", type(err).__name__, attempt + 1, wait)
            self.sleep(wait)
        raise LLMError(f"LLM unavailable after {self.retries + 1} attempts ({type(err).__name__})") from err

    # --- disk cache: same request -> same response, zero cost (used by the evaluation) ---
    def _cache_key(self, kwargs) -> str:
        payload = {k: v for k, v in kwargs.items() if k != "timeout"}
        return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()

    def _cache_get(self, key) -> LLMResponse | None:
        if not self.cache_dir or not (self.cache_dir / f"{key}.json").exists():
            return None
        d = json.loads((self.cache_dir / f"{key}.json").read_text(encoding="utf-8"))
        return LLMResponse(d["content"], [ToolCall(**t) for t in d["tool_calls"]],
                           d["input_tokens"], d["output_tokens"], 0.0, cached=True)

    def _cache_put(self, key, r: LLMResponse) -> None:
        if not self.cache_dir:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        d = {"content": r.content, "tool_calls": [t.__dict__ for t in r.tool_calls],
             "input_tokens": r.input_tokens, "output_tokens": r.output_tokens}
        (self.cache_dir / f"{key}.json").write_text(json.dumps(d), encoding="utf-8")
