"""M5 part 2: chunking strategy A (title + paragraph) vs B (fixed 80-word windows, 20 overlap, no title).

Both are indexed IN MEMORY on the same 10,000-paragraph subset (all gold paragraphs of the 300 questions
+ random distractors, seed 42), so no database space is used. Same questions, same metrics, same
retrieval code (dense, own BM25 over all matching chunks, own RRF). Results are at document level.

  python eval/compare_chunking.py          -> prints how many chunking-B tokens need embedding and the cost
  python eval/compare_chunking.py --yes    -> embeds them (paid, cached on disk) and runs the comparison

Strategy A vectors come from the existing cache (free). Result: eval/results/chunking_comparison.json
"""
import argparse
import json
import pathlib
import random
import sys
from collections import Counter

import numpy as np
import tiktoken
from dotenv import load_dotenv

ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from eval.common import embed_queries, load_questions  # noqa: E402
from eval.metrics import dedupe, score_question, summarize  # noqa: E402
from app.ingestion.chunking import chunk_b  # noqa: E402
from app.ingestion.corpus import Chunk, load_cached_vectors, load_chunks  # noqa: E402
from app.retrieval import bm25  # noqa: E402
from app.retrieval.embeddings import PRICE_PER_TOKEN, embed_batch  # noqa: E402
from app.retrieval.fusion import rrf  # noqa: E402
from app.retrieval.tokenize import tokenize  # noqa: E402

CORPUS = ROOT / "data" / "corpus" / "paragraphs.jsonl"
EMB_A = ROOT / "data" / "cache" / "emb"
EMB_B = ROOT / "data" / "cache" / "emb_b"
OUT = ROOT / "eval" / "results" / "chunking_comparison.json"
SUBSET, SEED, SHARD, TOP = 10_000, 42, 256, 50


def pick_subset(paragraphs: list[dict], questions: list[dict]) -> list[dict]:
    gold = {t for q in questions for t in q["gold_titles"]}
    must = [p for p in paragraphs if p["title"] in gold]
    rest = [p for p in paragraphs if p["title"] not in gold]
    extra = random.Random(SEED).sample(rest, SUBSET - len(must))
    return must + extra


class MemoryIndex:
    """One chunking strategy: vectors + our own BM25 over every chunk containing a query term."""

    def __init__(self, titles: list[str], texts: list[str], vecs: np.ndarray):
        self.titles, self.vecs = titles, vecs
        self.tokens = [tokenize(t) for t in texts]
        self.df = Counter(term for toks in self.tokens for term in set(toks))
        self.postings: dict[str, list[int]] = {}
        for i, toks in enumerate(self.tokens):
            for term in set(toks):
                self.postings.setdefault(term, []).append(i)
        self.stats = bm25.CorpusStats(len(texts), sum(map(len, self.tokens)) / len(texts))

    def dense(self, qvec) -> list[int]:
        sims = self.vecs @ qvec
        top = np.argpartition(-sims, TOP)[:TOP]
        return [int(i) for i in top[np.argsort(-sims[top])]]

    def keyword(self, query: str) -> list[int]:
        terms = tokenize(query)
        cand = {i for t in set(terms) for i in self.postings.get(t, ())}
        docs = {i: self.tokens[i] for i in cand}
        return [i for i, _ in bm25.rank(terms, docs, self.df, self.stats, top_k=TOP)]

    def titles_of(self, chunk_ids: list[int]) -> list[str]:
        return dedupe([self.titles[i] for i in chunk_ids])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yes", action="store_true")
    args = ap.parse_args()
    load_dotenv()

    questions = load_questions()
    paragraphs = [json.loads(line) for line in CORPUS.open(encoding="utf-8")]
    subset = pick_subset(paragraphs, questions)
    subset_hashes = {p["hash"] for p in subset}
    print(f"subset: {len(subset)} paragraphs (gold paragraphs of {len(questions)} questions + random distractors)")

    # Strategy A: reuse cached vectors
    all_a = load_chunks(CORPUS)
    vecs_all = load_cached_vectors(all_a, EMB_A)
    sel = [i for i, c in enumerate(all_a) if c.doc_hash in subset_hashes]
    a_chunks = [all_a[i] for i in sel]
    a_vecs = vecs_all[sel]
    del vecs_all

    # Strategy B: build chunks, estimate, embed with a shard cache
    b_chunks = [Chunk(p["hash"], p["title"], i, text) for p in subset for i, text in enumerate(chunk_b(p["text"]))]
    enc = tiktoken.get_encoding("cl100k_base")
    shards = [b_chunks[i:i + SHARD] for i in range(0, len(b_chunks), SHARD)]

    def shard_file(i):
        return EMB_B / f"shard_{i:05d}.npz"

    def shard_ok(i):
        try:
            with np.load(shard_file(i)) as z:
                return list(z["keys"]) == [c.key for c in shards[i]]
        except Exception:  # noqa: BLE001 - missing or half-written file
            return False

    todo = [i for i in range(len(shards)) if not shard_ok(i)]
    tokens = sum(len(enc.encode(c.text)) for i in todo for c in shards[i])
    print(f"A chunks: {len(a_chunks):,} (cached)  |  B chunks: {len(b_chunks):,} in {len(shards)} shards, "
          f"{len(todo)} to embed")
    print(f"B tokens to embed: {tokens:,} -> estimated ${tokens * PRICE_PER_TOKEN:.4f}")
    if todo and not args.yes:
        sys.exit("re-run with --yes to embed")
    if todo:
        from openai import OpenAI
        client = OpenAI(timeout=60)
        EMB_B.mkdir(parents=True, exist_ok=True)
        for n, i in enumerate(todo):
            vecs, _ = embed_batch(client, [c.text for c in shards[i]])
            tmp = shard_file(i).with_suffix(".tmp.npz")
            np.savez(tmp, keys=np.array([c.key for c in shards[i]]), vecs=vecs.astype(np.float16))
            tmp.replace(shard_file(i))
            if n % 20 == 0:
                print(f"  B shard {n + 1}/{len(todo)}")
    b_vecs = np.concatenate([np.load(shard_file(i))["vecs"].astype(np.float32) for i in range(len(shards))])

    qvecs = embed_queries([q["question"] for q in questions])  # cached by run_retrieval_eval.py
    indexes = {
        "A (title + paragraph)": MemoryIndex([c.title for c in a_chunks], [c.text for c in a_chunks], a_vecs),
        "B (80w windows, no title)": MemoryIndex([c.title for c in b_chunks], [c.text for c in b_chunks], b_vecs),
    }
    results = {}
    for name, idx in indexes.items():
        rows = {"dense": [], "bm25": [], "hybrid": []}
        for q in questions:
            gold = set(q["gold_titles"])
            d, k = idx.dense(qvecs[q["question"]]), idx.keyword(q["question"])
            lists = {"dense": d, "bm25": k, "hybrid": [c for c, _ in rrf([d, k])]}
            for cfg, ids in lists.items():
                rows[cfg].append(score_question(idx.titles_of(ids), gold))
        results[name] = {cfg: summarize(r) for cfg, r in rows.items()}

    OUT.write_text(json.dumps({"subset": len(subset), "a_chunks": len(a_chunks), "b_chunks": len(b_chunks),
                               "results": results}, indent=1), encoding="utf-8")
    print(f"\n{'strategy':26s} {'config':8s} {'recall@5':>9s} {'recall@10':>10s} {'all_gold@10':>12s} {'mrr':>7s}")
    for name, cfgs in results.items():
        for cfg, s in cfgs.items():
            print(f"{name:26s} {cfg:8s} {s['recall@5']:9.3f} {s['recall@10']:10.3f} {s['all_gold@10']:12.3f} {s['mrr']:7.3f}")
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
