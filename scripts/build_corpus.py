"""M2: build the paragraph corpus from HotpotQA (distractor, validation) and verify its size.

Input : data/raw/hotpot_val.parquet (see scripts/download_data.py)
Output: data/corpus/paragraphs.jsonl  (one unique paragraph per line)
No paid calls.
"""
import json
import pathlib
import sys

import pandas as pd

ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from app.ingestion.dedup import content_hash  # noqa: E402

SRC = ROOT / "data" / "raw" / "hotpot_val.parquet"
OUT = ROOT / "data" / "corpus" / "paragraphs.jsonl"
MIN_UNITS = 50_000


def main():
    df = pd.read_parquet(SRC)
    records, by_hash = 0, {}
    for qid, ctx in zip(df["id"], df["context"]):
        for title, sents in zip(ctx["title"], ctx["sentences"]):
            text = "".join(sents).strip()
            if not text:
                continue
            records += 1
            h = content_hash(f"{title}\n{text}")
            entry = by_hash.setdefault(h, {"title": str(title), "text": text, "hash": h, "questions": []})
            entry["questions"].append(qid)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for e in by_hash.values():
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    words = sum(len(e["text"].split()) for e in by_hash.values())
    print(f"questions: {len(df)}")
    print(f"paragraph records read: {records}")
    print(f"unique paragraphs (title+text): {len(by_hash)}")
    print(f"total words: {words:,}  (~{int(words * 1.35):,} tokens before title overhead)")
    print(f"VERIFIED >= {MIN_UNITS:,}: {len(by_hash) >= MIN_UNITS}")
    if len(by_hash) < MIN_UNITS:
        sys.exit("not enough units: top up from the train split with seed 42")


if __name__ == "__main__":
    main()
