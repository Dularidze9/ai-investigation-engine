# BM25 and RRF: walkthrough

Code: [app/retrieval/bm25.py](../app/retrieval/bm25.py), [app/retrieval/fusion.py](../app/retrieval/fusion.py).
Tests: [tests/test_bm25.py](../tests/test_bm25.py), [tests/test_fusion.py](../tests/test_fusion.py).
These two files are the project's own ranking algorithms. They use only the Python standard library.

## 1. Where they sit in the pipeline

```
query ──► tokenize ──► Postgres tsvector OR-query ──► top-200 candidate chunks
                                                         │
                                         our BM25 (bm25.py) ──► BM25 top-50 ─┐
query ──► embedding ──► pgvector HNSW ─────────────────────► dense top-50 ──┴─► our RRF (fusion.py) ──► (MMR) ──► top-k
```

- Postgres finds the chunks that contain at least one query word, using the GIN index. It does not do the final ranking.
- **BM25** ranks those candidates using statistics we store ourselves: `term_stats` (df per term) and `corpus_stats` (N and total length).
- **RRF** merges the BM25 list and the semantic list into one list.
- **MMR** (optional) removes near-repeats from the top of that list.

## 2. BM25 in one picture

For each query term `t` found in a chunk `d`:

```
            IDF(t)          ×        tf · (k1 + 1)
                                  ─────────────────────────────────
                                  tf + k1 · (1 − b + b · dl / avgdl)
```

and the chunk's score is the sum over the query terms. Each part answers one question:

| Part | Question it answers | Effect |
|---|---|---|
| `IDF(t) = ln((N − df + 0.5)/(df + 0.5) + 1)` | How rare is the word? | "derrickson" (in a few chunks) weighs far more than "american" (in thousands). |
| `tf` | How often is the word in this chunk? | More is better... |
| `k1` (1.2) | ...but how fast does "more" stop helping? | Saturation. The 2nd occurrence adds less than the 1st, and the score can never exceed `(k1 + 1) · IDF`. |
| `b` (0.75) and `dl / avgdl` | Is the chunk long? | A word in a short chunk counts more than the same word in a long one. `b = 0` disables this. |

A handy simplification for hand calculation: `(N − df + 0.5)/(df + 0.5) + 1 = (N + 1)/(df + 0.5)`, so **`IDF = ln((N + 1)/(df + 0.5))`**. `test_idf_simplified_form_is_the_same` checks this.

Why the `+ 1` inside the log matters: without it, a term found in more than half of the chunks gets a negative IDF and lowers the score of chunks that contain it. See break-it exercise 2.

## 3. Worked example (recompute this by hand)

Corpus (already tokenized), query = `["payment", "march"]`, k1 = 1.2, b = 0.75:

| Doc | Tokens | dl |
|---|---|---|
| d1 | payment, outage, march | 3 |
| d2 | payment, gateway, payment, retry | 4 |
| d3 | deploy, ledger, service, march, rollback | 5 |

N = 3, avgdl = 12 / 3 = **4**. df(payment) = 2, df(march) = 2.

**Step 1: IDF.** Both terms have df = 2:
`ln((3 + 1)/(2 + 0.5)) = ln(1.6) = 0.470004`

**Step 2: the length part of the denominator**, `K = k1 · (1 − b + b · dl/avgdl)`:

| Doc | dl/avgdl | K |
|---|---|---|
| d1 | 3/4 = 0.75 | 1.2 · (0.25 + 0.5625) = **0.975** |
| d2 | 4/4 = 1.00 | 1.2 · (0.25 + 0.75) = **1.2** |
| d3 | 5/4 = 1.25 | 1.2 · (0.25 + 0.9375) = **1.425** |

**Step 3: the tf part**, `tf · 2.2 / (tf + K)`:

| Doc | payment | march |
|---|---|---|
| d1 | 1 · 2.2 / 1.975 = 1.113924 | 1 · 2.2 / 1.975 = 1.113924 |
| d2 | 2 · 2.2 / 3.2 = 1.375 | not present |
| d3 | not present | 1 · 2.2 / 2.425 = 0.907216 |

**Step 4: multiply by IDF and add up:**

| Doc | Score | Rank |
|---|---|---|
| d1 | 0.470004 · 1.113924 · 2 = **1.047097** | 1 |
| d2 | 0.470004 · 1.375 = **0.646255** | 2 |
| d3 | 0.470004 · 0.907216 = **0.426395** | 3 |

What it shows:
- d2 has "payment" **twice**, but its tf part is only 1.375, not 2 × 1.113924. That is saturation (k1). Matching both query words (d1) beats repeating one (d2).
- d1 and d3 each contain "march" once, but d3's is worth less (0.907 against 1.114) because d3 is longer (b).

Keep 6 decimals throughout. When this test was first written, the hand total was rounded too early (1.047098), and the test caught it. The exact value is 1.047097.

## 4. bm25.py line by line

| Lines | Code | Why |
|---|---|---|
| 23–24 | `K1 = 1.2`, `B = 0.75` | Standard defaults (Robertson; Lucene and Elasticsearch use the same). Typical ranges are k1 1.2–2.0 and b 0.5–0.8. |
| 27–30 | `CorpusStats` | N and avgdl, read from our `corpus_stats` row. Frozen, so nobody can change them by accident. |
| 33–35 | `idf()` | The formula from section 2. The `+ 1.0` keeps it positive. |
| 40–41 | empty document returns 0 | Avoids a pointless calculation and keeps the empty case explicit. |
| 42 | `tf = Counter(doc_terms)` | Counts every term in the chunk once. Looking up a term's tf is then a dict lookup. |
| 44 | `length_norm = k1 * (1 - b + b * len(doc_terms) / stats.avgdl)` | The denominator part that depends only on the chunk, computed once per chunk, not per term. |
| 46 | `for term in dict.fromkeys(query_terms)` | Unique query terms in their original order. A repeated query word counts once (a deliberate choice, tested). |
| 47–49 | skip terms not in the chunk | They contribute 0 anyway. |
| 50 | `total += idf(...) * f * (k1 + 1) / (f + length_norm)` | The BM25 term score. `df.get(term, 0)`: a term missing from `term_stats` is treated as very rare. |
| 59 | score every candidate | This is plain Python over at most 200 candidates. Measured at about 0.02 s on the real data. |
| 60 | drop zero scores | A candidate with no matching token after our tokenization isn't a keyword match. |
| 61 | `sort(key=(-score, id))` | Best first, with ties broken by id, so the same input always gives the same order. Tests and evaluations stay reproducible. |
| 62 | `top_k` | Hybrid search uses the top 50. |

## 5. RRF (Reciprocal Rank Fusion)

```
RRF(d) = Σ over rankings   weight / (k + rank(d))        rank starts at 1, k = 60
```

**Why ranks and not scores:** BM25 scores here run from 0 to about 21, and cosine similarity from 0 to 1. Adding them would let BM25 dominate, and normalizing them is fragile. RRF uses only positions, so neither scale matters.

**Why k = 60:** k flattens the curve. With k = 60, rank 1 is worth 1/61 = 0.01639 and rank 2 is worth 1/62 = 0.01613, an almost equal vote. A document found by **both** methods therefore beats one that is #1 in only one of them. 60 is the value from the original RRF paper (Cormack et al., 2009).

### Worked example

dense = [d3, d1, d2], bm25 = [d1, d2, d3]:

| Doc | dense | bm25 | RRF |
|---|---|---|---|
| d1 | rank 2 → 1/62 = 0.016129 | rank 1 → 1/61 = 0.016393 | **0.032522** |
| d3 | rank 1 → 1/61 = 0.016393 | rank 3 → 1/63 = 0.015873 | **0.032266** |
| d2 | rank 3 → 1/63 = 0.015873 | rank 2 → 1/62 = 0.016129 | **0.032002** |

Final order: d1, d3, d2.

### fusion.py line by line (RRF)

| Lines | Code | Why |
|---|---|---|
| 14 | `RRF_K = 60` | See above. |
| 22–25 | weights default to 1 and must match the number of rankings | Lets you trust one method more, for example `weights=[2, 1]`. A wrong length is a bug, so it raises. |
| 29–33 | `seen_here` | If a list contains a document twice, only its best (first) rank counts. |
| 30 | `enumerate(ranking, start=1)` | Ranks start at 1. With 0, rank 1 would be worth 1/k, which changes the balance. |
| 34 | `scores[doc] += weight / (k + rank)` | The RRF vote. |
| 35 | `first_seen` | Records the order in which documents appeared, for tie-breaking. |
| 36 | sort by (−score, first_seen) | Best first, with deterministic ties. |

## 6. MMR (Maximal Marginal Relevance), optional

Picks results one at a time:
```
next = argmax over remaining d of   λ · relevance(d) − (1 − λ) · max similarity(d, already picked)
```
In hybrid search, relevance is the RRF score divided by the best RRF score, so it lies in [0, 1]. Similarity is the cosine between chunk embeddings. With λ = 1 you get plain relevance order, and lower λ gives more diverse results. The default is 0.7.

Example (`test_mmr_promotes_a_different_document`, λ = 0.5). Relevance is {1: 1.0, 2: 0.95, 3: 0.6}. Vectors 1 and 2 are almost identical (cosine 0.990), and 3 is orthogonal to both.
- **Round 1:** nothing is picked yet, so the scores are 0.5, 0.475 and 0.3. Pick 1.
- **Round 2:**
  - d2: 0.5 · 0.95 − 0.5 · 0.990 = −0.020
  - d3: 0.5 · 0.6 − 0 = 0.3
  - Pick 3, even though d2 is more relevant, because d2 adds nothing new.

Lines 51–59: `remaining` and `picked` lists. `max()` returns the first of equal scores, so ties go to the more relevant document. The cost is O(k · candidates · picked) cosine computations, which is trivial for a pool of 30.

## 7. On the real data

Measured on Neon with the query *"Were Scott Derrickson and Ed Wood of the same nationality?"* (`scripts/profile_keyword.py`, `scripts/search_demo.py`):

- **Tokens:** `scott derrickson ed wood same nationality`. "were", "and" and "of" are removed as stopwords.
- **Matches:** 7,804 chunks match at least one term. "same" alone is very common. Postgres keeps the top 200 by `ts_rank`.
- **Ranking:** BM25 ranks those 200 in about **0.02 s**. Fetching the candidates takes about 0.36 s, and the stats take 0.14 s, mostly network round trips.
- **Result:** both gold paragraphs are found. *Scott Derrickson* is **#1** and *Ed Wood* is **#6**.

### Known limitations (good interview material)

1. **Candidate cap.** BM25 only sees the 200 chunks Postgres ranks highest by `ts_rank`. A relevant chunk outside that top 200 can't be found by keyword search. Dense search may still find it, which is one reason to use hybrid search.
2. **Stemming mismatch.** Postgres stems ("films" matches "film"), but our tokenizer doesn't. A chunk can be fetched as a candidate because of a stemmed match, and then get tf = 0 for that word in BM25. The fix would be to stem in `tokenize()` and rebuild `term_stats`.
3. **Statistics drift.** Add, update and delete must update `term_stats` and `corpus_stats` in the same transaction, or IDF drifts. This is planned for the ingestion pipeline.
4. **Title in every chunk.** Chunking strategy A prepends the title, so title words count toward tf. That helps entity questions like this one, and it slightly inflates tf.
5. **`hnsw.ef_search`.** On the dense side, pgvector's default `ef_search = 40` would silently return at most 40 results. `dense.py` sets it to at least `top_k` with `SET LOCAL`.

## 8. Break-it exercises

Do each one, run `pytest tests/test_bm25.py tests/test_fusion.py -q`, explain what failed, then undo it. Predict the result **before** you run it. The answers below were checked by running them.

**Exercise 1: turn off length normalization.** In `bm25.py` change `B = 0.75` to `B = 0.0`.
Predict: which docs in the worked example change score, and does the order change?
<details><summary>Answer</summary>

d1 drops from 1.047097 to **0.940007**, d3 rises from 0.426395 to **0.470004**, and d2 stays at 0.646255 because its length equals avgdl, so it was never penalized or boosted. The order (d1, d2, d3) is unchanged, but `test_hand_computed_example` fails on the values and `test_longer_document_scores_lower_for_same_tf` fails. The lesson: without b, long chunks that mention a word once in passing rank as high as short chunks about that word.
</details>

**Exercise 2: remove the `+ 1.0` in `idf()`.** Change line 35 to `math.log((n_docs - df + 0.5) / (df + 0.5))`.
Predict: what does `rank()` return for the worked example?
<details><summary>Answer</summary>

idf(2, 3) = ln(1.5 / 2.5) = **−0.510826**. Every score becomes negative, line 60 drops all non-positive scores, and `rank()` returns **an empty list**. Keyword search silently finds nothing for any query whose terms are in more than half of the chunks. This silent failure is the worst kind, because nothing crashes. Several tests fail.
</details>

**Exercise 3: RRF with k = 0.** In `fusion.py` change `RRF_K = 60` to `RRF_K = 0`.
Predict: in `test_found_by_both_beats_top_of_one_list` (dense = [4, 1, 2], bm25 = [1, 2, 3]), who wins, doc 2 (rank 3 and rank 2) or doc 4 (rank 1 in dense only)?
<details><summary>Answer</summary>

With k = 0, doc 4 gets 1/1 = **1.0** and doc 2 gets 1/3 + 1/2 = **0.833**, so doc 4 wins and the test fails. With k = 60, doc 2 gets 0.0320 and doc 4 gets 0.0164. A small k makes the #1 position of a single method dominate, which defeats the point of fusing.
</details>

## 9. Interview questions

1. Walk through the BM25 formula. What does each of IDF, tf, k1 and b do, and what happens at k1 = 0?
2. Why do we let Postgres fetch candidates but rank them ourselves? What does that cost us?
3. Why does RRF use ranks instead of raw scores? Why k = 60 and not k = 0?
4. In the worked example, d2 has "payment" twice and d1 has it once, yet d1 ranks first. Explain with numbers.
5. You add a new document through the API. What must happen to `term_stats` and `corpus_stats`, and what goes wrong if it doesn't?

<details><summary>Answer notes</summary>

1. IDF weights rare words, tf rewards repetition, k1 controls saturation, and b controls the length penalty. At **k1 = 0**, the tf part becomes `tf · 1 / tf = 1`, so only whether the term is present matters, not how many times.
2. There's no room for a full inverted index on the 0.5 GB free tier (about 5M postings), and we wanted our own scoring with our own stats. The cost is that only the top-200 candidates are ranked, plus the stemming mismatch.
3. The scales differ (BM25 0–21 against cosine 0–1), and ranks need no normalization. k = 60 keeps the top positions close, so agreement between the methods wins. k = 0 lets one list's #1 dominate.
4. d1 matches both terms: 2 × 0.470004 × 1.113924 = 1.047097. d2 matches one term twice: 0.470004 × 1.375 = 0.646255, because of saturation. In d2 (K = 1.2), one "payment" would give 0.470004 × 2.2/2.2 = 0.470004, and two give 0.646255. The second occurrence adds only 0.176, less than matching a second term does in d1 (0.524).
5. For each unique token of each new chunk, `df += 1`. Also N += the number of chunks and `total_len += tokens`, all in the same transaction as the insert. Otherwise the scores use stale counts. A brand-new term is missing from `term_stats`, so it gets df = 0 and the maximum IDF. Deleted documents still count toward df and N, and avgdl is off, so the length penalty is wrong.
</details>
