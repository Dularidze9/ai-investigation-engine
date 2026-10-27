"""M0: check the owner's model list and whether gpt-6-luna supports tool calling.

Reads OPENAI_API_KEY from the environment (a local .env is loaded if present).
Listing models is free; the chat calls cost ~$0.001 total.
Usage: python scripts/verify_models.py 
"""
import json
import os
import sys
import time

from dotenv import load_dotenv
from openai import APIConnectionError, APIStatusError, OpenAI, RateLimitError

MODEL = "gpt-6-luna"
EFFORTS = ["none", "minimal", "low", "medium"]
TOOLS = [{
    "type": "function",
    "function": {
        "name": "search_documents",
        "description": "Search the document index.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
}]
PRICE_IN, PRICE_OUT = 0.10 / 1e6, 0.50 / 1e6


def with_backoff(fn, tries=4):
    for attempt in range(tries):
        try:
            return fn()
        except (RateLimitError, APIConnectionError):
            if attempt == tries - 1:
                raise
            time.sleep(2 ** attempt)
        except APIStatusError as e:
            if e.status_code < 500 or attempt == tries - 1:
                raise
            time.sleep(2 ** attempt)


def main():
    load_dotenv()
    print("OPENAI_API_KEY set:", bool(os.environ.get("OPENAI_API_KEY")))
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit(1)
    client = OpenAI(timeout=30)

    ids = sorted(m.id for m in with_backoff(lambda: client.models.list()).data)
    print(f"{len(ids)} models visible. Relevant ones:")
    for i in ids:
        if any(k in i for k in ("gpt-6", "gpt-5.4", "embedding-3")):
            print("  ", i)
    if MODEL not in ids:
        sys.exit(f"{MODEL} is not available to this key")

    est = len(EFFORTS) * (80 * PRICE_IN + 60 * PRICE_OUT)
    print(f"\nEstimated cost: {len(EFFORTS)} calls, ~${est:.5f} (budget total $4.75)")
    if "--yes" not in sys.argv and input("Run paid test? [y/N] ").lower() != "y":
        return

    for effort in EFFORTS:
        try:
            r = with_backoff(lambda: client.chat.completions.create(
                model=MODEL,
                reasoning_effort=effort,
                max_completion_tokens=200,
                tools=TOOLS,
                tool_choice="required",
                messages=[{"role": "user", "content": "Find documents about the March payment outage."}],
            ))
            calls = r.choices[0].message.tool_calls or []
            args = json.loads(calls[0].function.arguments) if calls else None
            print(f"effort={effort!r}: OK tool_calls={len(calls)} args={args} tokens={r.usage.total_tokens}")
        except APIStatusError as e:
            print(f"effort={effort!r}: REJECTED {e.status_code} {str(e.message)[:120]}")


if __name__ == "__main__":
    main()
