"""Create unanswerable HotpotQA questions by deleting their supporting paragraphs from the index.

Amendment (a): a title is only removed if it is NOT a supporting paragraph of any question in the
evaluation set (the 300 retrieval questions). The 8 questions are taken from OUTSIDE that set
(seed 42), so the M5 retrieval results stay valid. The check is printed.
Deletion goes through app.ingestion.pipeline.delete_document, which also updates term_stats/corpus_stats.
Free (no OpenAI). Idempotent: rerunning finds nothing left to delete.

Output: eval/unanswerable.json (committed).
"""
import json
import os
import pathlib
import random
import sys

import pandas as pd
import psycopg
from dotenv import load_dotenv

ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from app.ingestion.pipeline import delete_document  # noqa: E402
from eval.common import load_questions  # noqa: E402

OUT = ROOT / "eval" / "unanswerable.json"
N, SEED = 8, 42


def main():
    load_dotenv()
    eval_qs = load_questions()
    eval_ids = {q["id"] for q in eval_qs}
    protected = {t for q in eval_qs for t in q["gold_titles"]}

    if OUT.exists():
        chosen = json.loads(OUT.read_text(encoding="utf-8"))["questions"]
        print(f"using the existing selection in {OUT.name}")
    else:
        df = pd.read_parquet(ROOT / "data" / "raw" / "hotpot_val.parquet")
        order = list(range(len(df)))
        random.Random(SEED).shuffle(order)
        chosen, used = [], set()
        for i in order:
            r = df.iloc[i]
            titles = sorted(set(str(t) for t in r["supporting_facts"]["title"]))
            if r["id"] in eval_ids or any(t in protected or t in used for t in titles):
                continue
            chosen.append({"id": r["id"], "question": r["question"], "answer": r["answer"],
                           "type": r["type"], "removed_titles": titles})
            used.update(titles)
            if len(chosen) == N:
                break

    print("\nCHECK (amendment a): removed titles vs the gold titles of the 300 evaluation questions")
    all_removed = [t for q in chosen for t in q["removed_titles"]]
    for t in all_removed:
        print(f"  {t[:50]:50s} supporting paragraph of an eval question: {t in protected}")
    assert not any(t in protected for t in all_removed), "a protected title would be removed"
    print(f"  OK: none of the {len(all_removed)} titles supports any of the {len(eval_qs)} eval questions\n")

    deleted = []
    with psycopg.connect(os.environ["DATABASE_URL"], connect_timeout=30) as conn:
        before = conn.execute("SELECT n_chunks FROM corpus_stats").fetchone()[0]
        for t in all_removed:
            for (doc_id,) in conn.execute("SELECT id FROM documents WHERE title = %s", (t,)).fetchall():
                n = delete_document(conn, doc_id)
                deleted.append({"title": t, "document_id": doc_id, "chunks": n})
                print(f"  deleted document {doc_id} '{t}' ({n} chunk)")
        after = conn.execute("SELECT n_chunks FROM corpus_stats").fetchone()[0]
    print(f"\ncorpus_stats.n_chunks: {before} -> {after}  ({len(deleted)} documents deleted in this run)")

    prev = json.loads(OUT.read_text(encoding="utf-8"))["deleted"] if OUT.exists() else []
    OUT.write_text(json.dumps({"questions": chosen, "deleted": prev + deleted}, indent=1, ensure_ascii=False),
                   encoding="utf-8")
    for q in chosen:
        print(f"  [{q['type']}] {q['question']}  (gold answer was: {q['answer']})")


if __name__ == "__main__":
    main()
