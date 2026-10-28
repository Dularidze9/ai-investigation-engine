# Deployment

Live: https://ai-investigation-engine-fqlm.onrender.com. Everything runs on free tiers; no paid infrastructure.

## Architecture

```
                 GitHub (public repo, branch main)
                        │  push → auto-deploy (Docker build from Dockerfile, ~4-5 min)
                        ▼
 Browser ──HTTPS──► Render free web service (Frankfurt)
                     python:3.12-slim · uvicorn · 1 worker · FastAPI
                     env vars: DATABASE_URL, OPENAI_API_KEY, ADMIN_TOKEN (set in the Render dashboard)
                        │                                   │
                        │ psycopg3 pool (max 4), TLS         │ HTTPS
                        ▼                                   ▼
                 Neon free Postgres 18 (Frankfurt)      OpenAI API
                 pgvector 0.8.6, pooled endpoint         text-embedding-3-small (512-d)
                 all persistent state lives here         gpt-6-luna
```

The web service is **stateless**: documents, chunks, embeddings, BM25 statistics, the spending cap and the answer cache are all in Neon. A Render restart or redeploy loses only the in-memory per-IP rate-limit counters (by design). This was tested on 2026-10-28: after a manual restart the 17 synthetic documents, the spend total and a cached answer were all still served, and the per-IP counter was back to 20.

Both services are in Frankfurt, so each database round trip is within one region.

## Configuration and secrets
| Variable | Where it is set | Used for |
|---|---|---|
| `DATABASE_URL` | Render dashboard; local `.env` | Neon pooled connection string (TLS) |
| `OPENAI_API_KEY` | Render dashboard; local `.env` | embeddings and the LLM |
| `ADMIN_TOKEN` | Render dashboard; local `.env` | add/update/delete documents |

- `render.yaml` declares the variables with `sync: false`, so their values never live in the repo.
- `.env` is in `.gitignore`; only `.env.example` (empty values) is committed. The Dockerfile copies only `requirements.txt` and `app/`, and `.dockerignore` also excludes `.env`, `data/`, `eval/` and `tests/` from the build context.
- `app/config.py` reads settings from environment variables (pydantic-settings). No secret is logged: logs contain status, cost and counts only.
- The admin token is compared in constant time; without it, admin endpoints return 401 (503 if the server has no token configured).

## Free-tier limits and how the app fits them
| Resource | Free limit | Our usage | Measures |
|---|---|---|---|
| Render RAM | 512 MB | ~90 MB working set (measured locally after requests; Render's Metrics tab shows the live value) | 1 worker; the image installs only requirements.txt (pandas, pyarrow, tiktoken etc. stay in requirements-dev.txt for local scripts) |
| Render CPU | 0.1 CPU | BM25 scores 200 candidates in Python in ~0.02 s | heavy work (embedding the corpus, bulk load, index build) runs on the laptop, not on Render |
| Render sleep | sleeps after 15 min idle, ~1 min to wake | | UI shows a "waking up" banner and polls /health |
| Render hours / bandwidth | 750 h / month, 5 GB / month | one service | |
| Neon storage | 0.5 GB per project | **329.4 MB** (2026-10-28, `scripts/db_size.py`) | halfvec(512) instead of 1536-d float32; ingestion refuses writes above 400 MB; 2 MB upload and 200 chunk/document limits |
| Neon compute | 100 CU-hours / month, suspends after 5 min idle | | lazy connection pool (the app starts while Neon sleeps); /health retries with backoff during a cold start |

### Database size breakdown (scripts/db_size.py)
| Object | Size |
|---|---|
| chunks (table + toast + indexes) | 284.9 MB, of which HNSW index 91.4 MB and GIN index 23.5 MB |
| documents | 22.0 MB |
| term_stats (171,597 terms) | 13.9 MB |
| spend, corpus_stats, answer_cache | < 0.5 MB |
| **Total** | **329.4 MB** (66,569 documents, 66,922 chunks) |

## Costs
- **Infrastructure:** $0 (Render free, Neon free).
- **OpenAI development and evaluation:** ~$0.27 of the $4.75 budget (cost log in README.md).
- **Public demo:** a hard cap of **$1.00** in the database (`spend` table). Each question reserves $0.02 atomically before it runs and settles the real cost afterwards (a typical question costs $0.0002-0.001). Upload embeddings are charged to the same cap. When the cap is reached, /api/query answers 429. Cached answers cost $0.
- **Per question (measured in the evaluation):** agent $0.00085 and 10.5 s on average; 3-6 LLM calls.

## Deploy and verify (owner steps)
1. Neon: create a project in EU Central (Frankfurt), copy the **pooled** connection string. Locally run `python scripts/apply_schema.py`, then load the data (README, "Run locally").
2. Render: New → Web Service from the GitHub repo, runtime Docker, region Frankfurt, plan Free, health check path `/health`. Set the three environment variables.
3. Push to `main`; Render rebuilds automatically. Watch the Events tab until "Deploy live".
4. Verify: `python scripts/smoke_public.py` (free: UI, health, status, documents, auth refusals); `python scripts/smoke_public.py --ask` asks one question twice (~$0.001; the second answer must come from the cache).
5. Persistence: Render → Manual Deploy → Restart service, wait until live, run the smoke test again. Note that Render keeps the old instance serving until the new one is healthy, so a check right after clicking Restart can still hit the old process.

## Operations
- **Logs:** Render Logs tab; each question logs status, number of LLM calls and cost.
- **Budget:** `GET /api/status` shows spent and cap. To raise the cap: `UPDATE spend SET cap_usd = 2.0 WHERE id = 1;` in the Neon SQL editor.
- **Clear the answer cache** (e.g. after changing prompts): bump `VERSION` in `app/guard/answer_cache.py`, or `DELETE FROM answer_cache;`.
- **Check BM25 statistics** after manual database edits: `python scripts/check_stats.py` (must print `STATS CONSISTENT: True`).
