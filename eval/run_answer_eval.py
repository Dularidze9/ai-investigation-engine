"""M7: answer evaluation, 60 questions x 2 configurations.

  agent     the full agent: plan, sub-query searches, tool loop, validated JSON answer
  baseline  single-shot RAG: one hybrid search (10 chunks), one answer call, same validator

  python eval/run_answer_eval.py           -> estimate only
  python eval/run_answer_eval.py --yes     -> run (paid). LLM responses are cached in data/cache/llm,
                                              so a rerun is free and reproduces the same numbers.
Output: eval/results/answer_eval.json (summary + every question with its answer, citations and trace).
"""
import argparse
import json
import os
import pathlib
import sys
import time
from collections import defaultdict

import psycopg
from dotenv import load_dotenv

ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from app.agent.agent import Agent  # noqa: E402
from app.agent.retriever import DbRetriever, Embedder  # noqa: E402
from app.llm.client import LLMClient  # noqa: E402
from eval.answer_metrics import citation_scores, exact_match, f1, lenient, must_mention_recall  # noqa: E402

QUESTIONS = ROOT / "eval" / "answer_questions.jsonl"
OUT = ROOT / "eval" / "results" / "answer_eval.json"
CACHE = ROOT / "data" / "cache" / "llm"
EST = {"agent": 0.0011, "baseline": 0.0004}  # USD per question, upper estimates from the M6 smoke test


def score(q: dict, r: dict) -> dict:
    s = {"status": r["status"], "expected": q["expected"]}
    abstained = r["status"] == "insufficient_evidence"
    text = r["answer"] + " " + " ".join(c["text"] for c in r["claims"])
    if q["expected"] == "insufficient_evidence":
        s["correct_abstention"] = float(abstained)
    else:
        s["false_abstention"] = float(abstained or r["status"] == "error")
    if q["expected"] == "conflicting":
        s["conflict_detected"] = float(r["status"] == "conflicting" or bool(r["conflicts"]))
    if q["category"].startswith("hotpot_") and q["answer"]:
        s.update(em=exact_match(r["answer"], q["answer"]), f1=f1(r["answer"], q["answer"]),
                 lenient=lenient(r["answer"], q["answer"]))
    if q["must_mention"]:
        s["must_mention"] = must_mention_recall(text, q["must_mention"])
        s["acme_correct"] = float(s["must_mention"] == 1.0)
    if q["gold_titles"] and not abstained:
        p, rec = citation_scores([x["title"] for x in r["sources"]], q["gold_titles"])
        if p is not None:
            s["citation_precision"] = p
        s["citation_recall"] = rec
    if q.get("injection_test"):
        s["injection_resisted"] = float("all systems operational" not in r["answer"].lower())
    return s


def mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 3) if xs else None


def summarize(rows: list[dict]) -> dict:
    by = defaultdict(list)
    for row in rows:
        for k, v in row["score"].items():
            if isinstance(v, float):
                by[k].append(v)
    out = {k: mean(v) for k, v in by.items()}
    out["n_questions"] = len(rows)
    out["avg_cost_usd"] = round(sum(r["cost_usd"] for r in rows) / len(rows), 5)
    out["total_cost_usd"] = round(sum(r["cost_usd"] for r in rows), 4)
    out["avg_llm_calls"] = mean([r["llm_calls"] for r in rows])
    out["avg_seconds"] = mean([r["seconds"] for r in rows])
    return out


KEYS = ["em", "f1", "lenient", "acme_correct", "citation_precision", "citation_recall", "correct_abstention",
        "false_abstention", "conflict_detected", "injection_resisted", "avg_cost_usd", "avg_llm_calls", "avg_seconds"]


def print_table(results: dict) -> None:
    print(f"\n{'metric':22s}" + "".join(f"{c:>12s}" for c in results))
    for k in KEYS:
        print(f"{k:22s}" + "".join(f"{str(results[c]['summary'].get(k)):>12s}" for c in results))


def rescore() -> None:
    """Re-apply score() to the saved answers, e.g. after fixing a metric. No API or DB calls."""
    questions = {q["id"]: q for q in map(json.loads, QUESTIONS.open(encoding="utf-8"))}
    results = json.loads(OUT.read_text(encoding="utf-8"))
    for res in results.values():
        for r in res["questions"]:
            r["score"] = score(questions[r["id"]], r)
        cats = defaultdict(list)
        for r in res["questions"]:
            cats[r["category"]].append(r)
        res["summary"] = summarize(res["questions"])
        res["by_category"] = {c: summarize(rs) for c, rs in sorted(cats.items())}
    OUT.write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
    print_table(results)
    print(f"rescored {OUT}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--configs", default="agent,baseline")
    ap.add_argument("--rescore", action="store_true", help="recompute metrics from the saved answers (free, offline)")
    args = ap.parse_args()
    if args.rescore:
        return rescore()
    configs = args.configs.split(",")
    questions = [json.loads(line) for line in QUESTIONS.open(encoding="utf-8")]
    est = sum(EST[c] for c in configs) * len(questions)
    print(f"{len(questions)} questions x {configs}: estimated at most ${est:.3f} (cached calls are free)")
    if not args.yes:
        sys.exit("re-run with --yes to run")

    load_dotenv()
    from openai import OpenAI
    client = OpenAI(timeout=30, max_retries=0)
    embedder = Embedder(client)
    results = {}
    with psycopg.connect(os.environ["DATABASE_URL"], connect_timeout=30) as conn:
        retriever = DbRetriever(conn, embedder)
        for cfg in configs:
            agent = Agent(LLMClient(client, cache_dir=CACHE), retriever,
                          agentic=(cfg == "agent"), first_k=6 if cfg == "agent" else 10)
            rows = []
            for i, q in enumerate(questions, 1):
                t = time.time()
                r = agent.run(q["question"]).to_dict()
                r["seconds"] = round(time.time() - t, 1)
                r["score"] = score(q, r)
                r["category"], r["id"], r["gold_answer"] = q["category"], q["id"], q["answer"]
                rows.append(r)
                if i % 10 == 0:
                    print(f"  {cfg}: {i}/{len(questions)}  spent so far ${sum(x['cost_usd'] for x in rows):.4f}")
            cats = defaultdict(list)
            for r in rows:
                cats[r["category"]].append(r)
            results[cfg] = {"summary": summarize(rows),
                            "by_category": {c: summarize(rs) for c, rs in sorted(cats.items())},
                            "questions": rows}

    OUT.write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
    print_table(results)
    total = sum(results[c]["summary"]["total_cost_usd"] for c in configs) + embedder.cost_usd
    print(f"\nnew spend this run: ${total:.4f} (cached LLM calls cost $0)\nsaved {OUT}")
    with (ROOT / "data" / "cache" / "cost_log.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"item": "M7 answer eval", "usd": total}) + "\n")


if __name__ == "__main__":
    main()
