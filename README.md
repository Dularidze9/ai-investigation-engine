
An investigation engine over a large document set: it answers questions that need several sources, cites the evidence for every claim, flags contradictions between sources, and abstains instead of inventing facts when the evidence is insufficient.

**Live demo:** https://ai-investigation-engine-fqlm.onrender.com
Free tier: the service sleeps after 15 minutes idle, so the first request can take about a minute (the page shows "waking up").

| Document | Content |
|---|---|
| [EVALUATION_REPORT.md](EVALUATION_REPORT.md) | Retrieval and answer evaluation, configuration comparisons, failure analysis |
| [DEPLOYMENT.md](DEPLOYMENT.md) | Deployment architecture, free-tier limits, resources, costs, secrets |

## Results at a glance
- **Data:** 66,569 documents / 66,922 chunks: 66,552 unique HotpotQA paragraphs (linked by multi-hop questions) + 17 synthetic "Acme Payments" incident documents in 5 formats with planted contradictions, duplicates and a prompt injection. Database 329.4 MB of the 512 MB free limit.
- **Retrieval (300 questions):** own hybrid (dense + own BM25, fused with own RRF) Recall@10 **0.837** vs dense 0.777 vs BM25 0.768; both gold documents in the top 10: **0.697** vs 0.597 / 0.577.
- **Answers (60 questions), agent vs single-shot baseline:** F1 **0.576** vs 0.509; unanswerable questions abstained **15/15** (both); false abstentions 11% vs 18%; contradictions detected 3/4 vs 4/4; prompt injection resisted; $0.00085 vs $0.00020 per question.
- **128 automated tests** (`pytest`), no network or API cost.

## Using the demo
- **Ask:** type a question or click an example. The answer shows a status (answered, partially answered, sources disagree, not enough evidence), the supporting claims with numbered citations, contradictions between sources, missing information, and the cited source chunks. Clicking a source title opens the whole document. "How the answer was produced" shows the sub-queries, the search trace and what the citation validator removed.
- **Documents:** browse and search all documents by title; filter to uploaded/synthetic ones; open any document to read its chunks.
- **Manage (admin):** add a file (.txt .md .html .pdf .json, max 2 MB, max 200 chunks), replace a document's content (it keeps its id), or delete it. Exact and near duplicates are rejected.

Good questions to try: "How long did the March 10, 2026 payment incident last?" (sources disagree), "Which team owns the service whose change triggered the March 10 payment incident?" (multi-hop across documents), "Were Scott Derrickson and Ed Wood of the same nationality?" (HotpotQA comparison), "What was Acme Payments' revenue in 2025?" (not in the documents: abstains).

### Why there is an admin token
Reading and asking are public. Changing the corpus is not: an anonymous visitor could otherwise delete the evaluation documents or fill the free database. Add, update and delete therefore need `ADMIN_TOKEN` (an environment variable on Render, never in the repo). The UI sends it in the `X-Admin-Token` header and keeps it only in the browser tab's sessionStorage; the server compares it in constant time (`hmac.compare_digest`). If `ADMIN_TOKEN` is not set, admin actions are disabled (503). **Graders receive the token separately with the submission.**

### Public demo protection
- **20 questions per IP per hour** (sliding window, in memory; it resets when the service restarts, which is accepted because...)
- **...the hard limit is a $1.00 spending cap in Postgres.** Every question reserves its worst case ($0.02) in one atomic `UPDATE` before running and settles the real cost after; upload embedding costs go on the same cap. It survives restarts.
- **Answer cache in Postgres:** the same question (ignoring case and spacing) is answered from the cache for free; any document change clears the cache so no answer cites a removed chunk.
- Ingestion refuses writes when the database is over 400 MB.

## Architecture

```
 Browser (app/static/index.html, vanilla JS)
   │ ask · sources · document browser · admin (X-Admin-Token)
   ▼
 FastAPI on Render free (Docker, 1 worker)
   /health              DB ping with retries (wakes Neon)
   /api/query           rate limit (per IP) → answer cache → budget reservation → Agent
   /api/status          budget used, questions left
   /api/documents       public list/view; admin add/update/delete → ingestion pipeline
   │
   ├─ Agent (app/agent)
   │    1 search (hybrid, 6 chunks)         2 evidence gate: best similarity < 0.35 → abstain, no LLM cost
   │    3 plan: ≤3 sub-queries (strict JSON) 4 tool loop ≤4 rounds: search_documents · get_document · compare_sources
   │    5 answer: strict JSON {status, answer, claims[chunk_ids], conflicts, missing}
   │    6 citation validator: unknown ids removed, uncited claims dropped, conflicts need 2 sources
   │
   ├─ Hybrid retriever (app/retrieval) ── own algorithms: bm25.py, fusion.py
   │    dense:   pgvector HNSW (halfvec 512, cosine) top 50
   │    keyword: Postgres tsvector finds 200 candidates → OUR BM25 scores them with OUR df/N/avgdl stats
   │    fusion:  OUR Reciprocal Rank Fusion (k=60)  (MMR implemented, off: it lowered recall)
   │
   └─ Ingestion (app/ingestion): parse (5 formats) → chunk → exact + near-duplicate check → embed
        → one transaction: document + chunks + BM25 statistics delta
   │
   ▼
 Neon free Postgres (Frankfurt): documents · chunks (text, halfvec(512), tsvector) · term_stats · corpus_stats
                                 · spend · answer_cache
 OpenAI: text-embedding-3-small (512-d) · gpt-6-luna (tool calling, reasoning_effort="none")
```

### Key design decisions
| Decision | Why | Trade-off |
|---|---|---|
| Postgres + pgvector (Neon) as the vector DB | One free managed database for vectors, text search, stats, budget and cache; data survives restarts | 512 MB limit: halfvec(512) instead of 1536-d float vectors |
| Own BM25 over tsvector candidates | Graded "own algorithm"; a full inverted index table (~5M rows) would not fit in 0.5 GB | BM25 ranks only the top 200 keyword candidates, not the whole corpus |
| Own RRF fusion | Combines rank lists without calibrating dense and BM25 scores against each other | Ignores score margins; k=60 is the standard constant, not tuned |
| Chunking A (title + paragraph, split at ~300 words) | Beat fixed 80-word windows for dense and BM25 on our eval; 1.6x fewer chunks | Very long paragraphs are split at sentence boundaries only |
| Agent with planning + tools | Multi-hop questions need a second search for an entity found in the first | 4x the cost and latency of single-shot; more cautious (see report) |
| Strict JSON answers + citation validator | Every claim must cite a retrieved chunk; invented citations are removed in code, not trusted to the model | A valid citation can still be a weak one (validator checks existence, not entailment) |
| Evidence gate before the LLM | Clearly unrelated questions cost nothing | Dense similarity does not separate in-domain from out-of-domain well; the LLM abstains in those cases |

## Repository layout
```
app/
  main.py config.py db.py          FastAPI app, settings from env vars, lazy Postgres pool
  api/        health.py query.py documents.py
  agent/      agent.py tools.py schema.py validator.py retriever.py
  retrieval/  bm25.py* fusion.py* hybrid.py dense.py keyword.py embeddings.py tokenize.py
  ingestion/  parsers.py chunking.py dedup.py pipeline.py corpus.py
  llm/        client.py (retry, backoff, timeout, cost, disk cache) prompts.py pricing.py
  guard/      budget.py ratelimit.py answer_cache.py
  static/     index.html
db/           schema.sql indexes.sql
scripts/      data download, corpus build, embedding, bulk load, smoke tests, stats check
eval/         question sets, metrics, evaluation runners, results/*.json
data/synthetic/  17 Acme Payments documents (committed; the HotpotQA data is downloaded by a script)
tests/        128 tests
docs/         bm25_rrf_walkthrough.md
```
`*` own-algorithm files (see AI_USAGE.md).

## Run locally
Requirements: Python 3.12, a Neon (or any Postgres with pgvector >= 0.7) database, an OpenAI API key.

```powershell
python -m venv .venv; .venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
copy .env.example .env      # then fill in OPENAI_API_KEY, DATABASE_URL, ADMIN_TOKEN (never commit .env)
pytest -q                   # 128 tests, no network
```

Build the data (each paid script prints an estimate and needs `--yes`):
```powershell
python scripts/apply_schema.py                                 # tables + pgvector check
python scripts/download_data.py; python scripts/build_corpus.py  # HotpotQA → 66,581 unique paragraphs (free)
python scripts/embed_corpus.py --yes                           # ~8.6M tokens, ~$0.17, cached on disk, resumable
python scripts/load_db.py                                      # near-dup removal, bulk load, stats, indexes
python eval/make_unanswerable.py                               # delete the 16 paragraphs of 8 unanswerable questions
python scripts/make_synthetic.py; python scripts/ingest_synthetic.py --yes   # Acme docs via the real pipeline
python scripts/check_stats.py; python scripts/db_size.py      # stats == recount; size report
```

Run the app: `uvicorn app.main:app --port 8000` and open http://127.0.0.1:8000.
Reproduce the evaluation: see [EVALUATION_REPORT.md, section 8](EVALUATION_REPORT.md#8-reproduce).

Public demo usage after launch is tracked live by the spending cap (`/api/status`, capped at $1.00).

## Known limitations
- BM25 scores only the top 200 keyword candidates (storage trade-off, see above).
- The per-IP limit is in memory and can be bypassed by forging `X-Forwarded-For`; the $1.00 database cap cannot.
- The agent is over-cautious: in 4 of its 5 false abstentions the gold documents had been retrieved (EVALUATION_REPORT.md, section 6). Fixes were not tuned on the evaluation set to avoid leakage.
- The citation validator checks that cited chunks were retrieved, not that they entail the claim.
- Spend is stored with 5 decimals, so a tiny upload (< $0.000005 of embeddings) rounds to $0 on the cap.
