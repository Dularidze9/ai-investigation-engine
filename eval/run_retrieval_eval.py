"""M5 part 1: retrieval evaluation on the real Neon database.

Compares dense, BM25, hybrid (RRF) and hybrid + MMR on the 300 fixed questions, at document level.

  python eval/run_retrieval_eval.py          -> prints the cost estimate (no spend if queries are cached)
  python eval/run_retrieval_eval.py --yes    -> embeds uncached queries, runs the evaluation

Result: eval/results/retrieval_eval.json (per question + averages). Query embeddings are cached on disk,
so reruns cost nothing and give identical numbers.
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
from eval.common import embed_queries, estimate_query_cost, load_questions  # noqa: E402
from eval.metrics import dedupe, score_question, summarize  # noqa: E402
from app.retrieval.hybrid import finalize, retrieve_lists  # noqa: E402

OUT = ROOT / "eval" / "results" / "retrieval_eval.json"
CONFIGS = {  # name -> (mode, use_mmr)
    "dense": ("dense", False),
    "bm25": ("bm25", False),
    "hybrid": ("hybrid", False),
    "hybrid+mmr": ("hybrid", True),
}
K_CHUNKS = 30  # enough chunk hits to fill 10 distinct documents


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yes", action="store_true")
    args = ap.parse_args()
    load_dotenv()
    questions = load_questions()
    n, tokens, usd = estimate_query_cost([q["question"] for q in questions])
    print(f"{len(questions)} questions; uncached query embeddings: {n}, {tokens} tokens, estimated ${usd:.6f}")
    client = None
    if n:
        if not args.yes:
            sys.exit("re-run with --yes to embed the uncached queries")
        from openai import OpenAI
        client = OpenAI(timeout=30)
    qvecs = embed_queries([q["question"] for q in questions], client)

    per_config = {name: [] for name in CONFIGS}
    t0 = time.time()
    with psycopg.connect(os.environ["DATABASE_URL"], connect_timeout=30) as conn:
        for i, q in enumerate(questions, 1):
            gold = set(q["gold_titles"])
            dense, keyword = retrieve_lists(conn, q["question"], qvecs[q["question"]], "hybrid")
            for name, (mode, use_mmr) in CONFIGS.items():
                hits = finalize(conn, dense, keyword, K_CHUNKS, mode, use_mmr)
                ranked = dedupe([h.title for h in hits])
                row = score_question(ranked, gold)
                row.update(id=q["id"], type=q["type"], top10=ranked[:10])
                per_config[name].append(row)
            if i % 50 == 0:
                print(f"  {i}/{len(questions)} questions ({time.time() - t0:.0f}s)")

    metrics = ["recall@5", "recall@10", "all_gold@10", "mrr"]
    summary = {name: summarize([{m: r[m] for m in metrics} for r in rows]) for name, rows in per_config.items()}
    by_type = {}
    for name, rows in per_config.items():
        for t in ("bridge", "comparison"):
            sel = [{m: r[m] for m in metrics} for r in rows if r["type"] == t]
            by_type[f"{name}/{t}"] = summarize(sel)
    OUT.write_text(json.dumps({"n": len(questions), "summary": summary, "by_type": by_type,
                               "per_question": per_config}, indent=1), encoding="utf-8")

    print(f"\n{'config':12s} " + " ".join(f"{m:>12s}" for m in metrics))
    for name, s in summary.items():
        print(f"{name:12s} " + " ".join(f"{s[m]:12.3f}" for m in metrics))
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
