"""Run the real agent (Neon + OpenAI) on a few questions and print the full result. PAID, small.

  python scripts/smoke_agent.py          -> prints the estimate only
  python scripts/smoke_agent.py --yes    -> runs. LLM responses are cached in data/cache/llm, reruns are free.
"""
import argparse
import json
import os
import pathlib
import sys
import time

import psycopg
from dotenv import load_dotenv

ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from app.agent.agent import Agent  # noqa: E402
from app.agent.retriever import DbRetriever, Embedder  # noqa: E402
from app.llm.client import LLMClient  # noqa: E402

QUESTIONS = [
    "Were Scott Derrickson and Ed Wood of the same nationality?",                        # comparison, 2 docs
    "The director of the romantic comedy \"Big Stone Gap\" is based in what New York city?",  # bridge, 2 hops
    "What was the quarterly revenue of Acme Payments in Q3 2031?",                     # not in the corpus
    "Which programming language did the Apollo 11 guidance computer use for its tax returns?",  # nonsense
]
EST_PER_Q = 0.0012  # ~8k input + 600 output tokens of gpt-6-luna, upper estimate

ap = argparse.ArgumentParser()
ap.add_argument("--yes", action="store_true")
args = ap.parse_args()
print(f"{len(QUESTIONS)} questions, estimated at most ${EST_PER_Q * len(QUESTIONS):.4f} (gpt-6-luna) + query embeddings")
if not args.yes:
    sys.exit("re-run with --yes to run")

load_dotenv()
from openai import OpenAI  # noqa: E402
client = OpenAI(timeout=30, max_retries=0)
embedder = Embedder(client)
total = 0.0
with psycopg.connect(os.environ["DATABASE_URL"], connect_timeout=30) as conn:
    agent = Agent(LLMClient(client, cache_dir=ROOT / "data" / "cache" / "llm"), DbRetriever(conn, embedder))
    for q in QUESTIONS:
        t = time.time()
        r = agent.run(q)
        total += r.cost_usd
        print("=" * 100)
        print(f"Q: {q}\nstatus: {r.status} | answer: {r.answer}")
        print(f"llm_calls={r.llm_calls} tokens in/out={r.input_tokens}/{r.output_tokens} "
              f"cost=${r.cost_usd:.5f} time={time.time() - t:.1f}s evidence={r.evidence_count}")
        print("trace:\n  " + "\n  ".join(r.trace))
        for c in r.claims:
            print(f"  claim: {c['text']}  -> {c['citations']}")
        for c in r.conflicts:
            print(f"  conflict: {c['description']} -> {c['citations']}")
        print("  sources:", [s["title"] for s in r.sources])
        if r.missing:
            print("  missing:", r.missing)
        if r.validation_issues:
            print("  validator:", r.validation_issues)
print("=" * 100)
print(f"LLM cost ${total:.5f} + embeddings ${embedder.cost_usd:.6f} = ${total + embedder.cost_usd:.5f}")
(ROOT / "data" / "cache").mkdir(parents=True, exist_ok=True)
with (ROOT / "data" / "cache" / "cost_log.jsonl").open("a", encoding="utf-8") as f:
    f.write(json.dumps({"item": "M6 smoke_agent", "usd": total + embedder.cost_usd}) + "\n")
