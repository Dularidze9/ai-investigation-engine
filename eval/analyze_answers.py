"""Print per-category results and every failure from eval/results/answer_eval.json (free, offline)."""
import json
import pathlib

d = json.loads((pathlib.Path(__file__).parent / "results" / "answer_eval.json").read_text(encoding="utf-8"))
keys = ["lenient", "f1", "acme_correct", "citation_recall", "correct_abstention", "false_abstention", "conflict_detected"]
for cfg, res in d.items():
    print(f"\n=== {cfg}: by category ===")
    for cat, s in res["by_category"].items():
        vals = "  ".join(f"{k}={s[k]}" for k in keys if s.get(k) is not None)
        print(f"  {cat:18s} n={s['n_questions']:2d}  {vals}  cost=${s['avg_cost_usd']}")

for cfg, res in d.items():
    print(f"\n=== {cfg}: failures ===")
    for q in res["questions"]:
        s, why = q["score"], []
        if s.get("lenient") == 0.0:
            why.append("wrong/incomplete answer")
        if s.get("false_abstention") == 1.0:
            why.append(f"abstained ({q['status']})")
        if s.get("correct_abstention") == 0.0:
            why.append("answered an unanswerable question")
        if s.get("conflict_detected") == 0.0:
            why.append("missed conflict")
        if s.get("acme_correct") == 0.0:
            why.append(f"missing terms (recall {s['must_mention']})")
        if s.get("citation_recall") is not None and s["citation_recall"] < 1:
            why.append(f"citation recall {s['citation_recall']:.2f}")
        if why:
            print(f"- [{q['category']}] {q['question'][:95]}")
            print(f"    gold: {q['gold_answer']} | got [{q['status']}]: {q['answer'][:110]}")
            print(f"    -> {'; '.join(why)}")
