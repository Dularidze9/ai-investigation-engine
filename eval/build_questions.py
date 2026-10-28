"""Build the fixed evaluation question set: 300 HotpotQA validation questions, seed 42.

Output: eval/questions.jsonl (committed, so every run uses exactly the same questions).
Each line: id, question, answer, type (bridge/comparison), level, gold_titles (supporting-fact documents).
Free; reads data/raw/hotpot_val.parquet (scripts/download_data.py).
"""
import json
import pathlib
import random
from collections import Counter

import pandas as pd

ROOT = pathlib.Path(__file__).parent.parent
SRC = ROOT / "data" / "raw" / "hotpot_val.parquet"
OUT = ROOT / "eval" / "questions.jsonl"
N, SEED = 300, 42


def main():
    df = pd.read_parquet(SRC)
    idx = sorted(random.Random(SEED).sample(range(len(df)), N))
    rows = []
    for i in idx:
        r = df.iloc[i]
        rows.append({
            "id": r["id"], "question": r["question"], "answer": r["answer"],
            "type": r["type"], "level": r["level"],
            "gold_titles": sorted(set(str(t) for t in r["supporting_facts"]["title"])),
        })
    with OUT.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} questions to {OUT}")
    print("types:", dict(Counter(r["type"] for r in rows)))
    print("levels:", dict(Counter(r["level"] for r in rows)))
    print("gold titles per question:", dict(Counter(len(r["gold_titles"]) for r in rows)))


if __name__ == "__main__":
    main()
