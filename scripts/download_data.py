"""Download HotpotQA (distractor, validation split, parquet) into data/raw/. Free; no API key needed."""
import pathlib
import sys

import requests

URL = ("https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/"
       "refs%2Fconvert%2Fparquet/distractor/validation/0000.parquet")
DEST = pathlib.Path(__file__).parent.parent / "data" / "raw" / "hotpot_val.parquet"

if DEST.exists():
    print("already downloaded:", DEST)
    sys.exit(0)
DEST.parent.mkdir(parents=True, exist_ok=True)
r = requests.get(URL, timeout=300)
r.raise_for_status()
DEST.write_bytes(r.content)
print(f"saved {len(r.content):,} bytes to {DEST}")
