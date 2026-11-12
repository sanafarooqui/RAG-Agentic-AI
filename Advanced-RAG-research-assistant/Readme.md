# Advanced RAG Research Assistant

Agentic RAG chatbot for research papers. Phase 1: local demo (FastAPI + LangGraph backend, React frontend, Redis). See [Intructions.md](Intructions.md) for the full spec.

## Layout

```
backend/
  api.py           FastAPI routes (sessions, upload, ArXiv, SSE chat)
  rag_graph.py     LangGraph: route → retrieve → grade → rewrite/rerank → generate
  paper_loader.py  File + ArXiv ingestion and chunking
  vector_store.py  Qdrant + Redis-cached embeddings
  sessions.py      Sessions, messages and document library in Redis
  web_search.py    Tavily search with Redis result cache
  pii.py           PII masking (regex) applied at the API and as the graph's first step
  models.py        Pydantic models
  config.py        Env, paths, tunables
frontend/          Vite + React + TypeScript UI
main.py            Backend entry point
```

## Run

1. Start Redis: `redis-server --daemonize yes` or `docker compose up -d redis`
2. `.env` in this folder: `OPENAI_API_KEY`, `OPENAI_API_BASE_URL`, `OPENAI_MODEL`, `TAVILY_API_KEY`, `QDRANT_URL`, `QDRANT_API_KEY` (optional `QDRANT_COLLECTION`, default `research_papers`; optional `REDIS_URL`, default `redis://localhost:6379/0`)
3. Backend (from the project root venv): `../.venv/bin/python main.py` → http://127.0.0.1:8000
4. Frontend: `cd frontend && npm install && npm run dev` → http://localhost:5173

## How a question is handled

0. **PII masking** replaces emails, phone numbers, SSNs, card numbers, IP addresses and self-introduced names with placeholders such as `[EMAIL]` before anything else sees the text, so they are not stored in Redis, sent to the LLM, searched on the web or traced; your history shows the masked text.
1. **Router** picks `knowledge_base`, `web_search` or `general`, and rewrites the question using recent history (short-term memory).
2. **Knowledge base**: retrieve the top 20 chunks from the session's documents, grade relevance, rewrite the query and retry once if weak, rerank to the top 6, add the neighbouring chunks of the top 3, then keep only the sentences that help (tables are kept whole). With documents selected the router always tries them first; only the sidebar web search skips them.
3. **Fallback**: still weak → Tavily web search, answered as clearly labelled web information with claims attributed to their sources → if that fails or finds nothing, an explicit "insufficient evidence" answer.
4. **Answer** cites passages as `[n]`; each source is tagged as a local document (with page) or a web URL, and the UI labels where the answer came from.

The web search box in the sidebar forces the `web_search` route, which also works for verifying claims.

## Offline evaluation (Phase 2, DeepEval)

Suite in `tests/evals/`; spec in [Intructions.md](Intructions.md). Judge is `gpt-4o` through the same endpoint as the app. Results stay local (no Confident AI), and LangSmith tracing is forced off during evals.

```
tests/evals/
  app.py                    runs the real LangGraph traced with DeepEval's LangChain CallbackHandler
  judge.py                  judge model + eval environment
  metrics.py                retriever / generator / pipeline / safety metric lists
  aiayn_rag_goldens.json    50 curated goldens (10 categories, difficulty, expected behavior, per-golden target metrics)
  dataset_synthetic.json    40 goldens from `deepeval generate` (kept as source; mostly vague or duplicated)
  dataset_manual.json       10 early hand-written goldens (superseded by the curated set, no longer used)
  build_golden_set.py       merges the curated set + 4 distinct synthetic goldens into dataset_golden.json (54)
  dataset_golden.json       the golden set the pipeline test runs
  test_rag_pipeline.py      per-golden metrics: Contextual Precision/Recall/Relevancy, Faithfulness, Answer Relevancy, Correctness, Completeness; Hallucination + Negative Rejection for unanswerable
  test_safety_security.py   Toxicity, Bias, PII leakage, prompt-injection resistance
  test_operations.py        latency p95, routing, citations, bounded retries (plain pytest, no judge)
  summarize.py              per-metric summary of a saved run
  compare.py                per-metric and per-golden comparison of two saved runs
  round-1-rerun.md, round-2.md   baseline and round 2 reports
```

Run (needs Redis, Qdrant with the paper ingested, and the `.env`):

```bash
export SSL_CERT_FILE=$(../.venv/bin/python -c "import certifi;print(certifi.where())")
caffeinate -i ../.venv/bin/deepeval test run tests/evals/test_rag_pipeline.py tests/evals/test_safety_security.py \
  tests/evals/test_operations.py --num-processes 3 --ignore-errors --identifier round-N
cp .deepeval/.latest_test_run.json tests/evals/reports/round-N.json
../.venv/bin/python tests/evals/summarize.py tests/evals/reports/round-N.json -v
```

DeepEval prints nothing until the run ends (about 17 minutes for 29 tests with 5 workers). `.deepeval/.latest_test_run.json` is overwritten on every run, so copy it into `tests/evals/reports/` (git-ignored).

### Round 1 baseline (no app changes yet)

15 RAG goldens (10 synthetic + 5 hand-written; this was before the merged golden set below), 7 safety cases, 7 ops checks: 17 of 29 failed.

| Area | Metric | Mean | Passed |
|---|---|---|---|
| Retriever | Contextual Precision | 0.92 | 14/15 |
| Retriever | Contextual Recall | 0.94 | 13/15 |
| Retriever | Contextual Relevancy | 0.41 | 3/15 |
| Generator | Faithfulness | 1.00 | 15/15 |
| Generator | Answer Relevancy | 0.82 | 10/15 |
| Generator | Correctness | 0.57 | 6/15 |
| Generator | Completeness | 0.78 | 13/15 |
| Safety | Toxicity / Bias | 1.00 / 1.00 | 2/2 each |
| Safety | PII Leakage | 0.25 | 1/2 |
| Security | Prompt-injection resistance | 0.97 | 3/3 |
| Operations | 7 checks | | 4 pass, 3 fail |

Main findings: retrieved passages are mostly noise (Contextual Relevancy); the router sent a paper question to the web; the app echoes PII back and has no masking; some synthetic goldens are ill-posed, which depresses Correctness; ops p95 latency of 29.6s was measured under 5 parallel workers (solo run: 8.4s). Round 2 is planned to target retrieval noise and routing.

### Golden set (merged, after round 1)

`dataset_golden.json` = the 50 curated goldens (`aiayn_rag_goldens.json`) + 4 well-posed, distinct DeepEval-synthetic ones, 54 total. The curated set was spot-checked against the PDF (numbers, the 41.8 vs 41.0 BLEU inconsistency, checkpoint averaging, FLOP and parameter arithmetic) and matched. The synthetic set was mostly dropped: 8 of 40 near-duplicates, 3 hypothetical or ill-posed, 20 of 40 with front-matter or reference-list source text, and in round 1 it scored Correctness 0.46 (precise hand-written goldens: 0.78) while hiding the retrieval noise (Contextual Relevancy 0.50 vs 0.21). Rebuild with `../.venv/bin/python tests/evals/build_golden_set.py`.

`test_rag_pipeline.py` takes a subset round-robin across categories: `EVAL_MAX_GOLDENS=15` by default, `0` for all 54. Round 1 numbers are not comparable to runs on this set, so the next full run is a new baseline.

### Round 2 (after the retrieval and web-fallback fixes)

See `tests/evals/round-2.md`. On the 10 goldens both runs completed: Correctness 0.49 to 0.70, Completeness 0.67 to 0.82, Contextual Relevancy 0.26 to 0.38 (still the blocker, 1/9 passing). Ops checks 4/7 to 7/7, safety 6/7 (PII echo not yet fixed). Run evals under `caffeinate -i`: a Mac that goes to sleep mid-run stalls it for good.

PII masking was added after round 2: the PII leakage evals now pass (1.0 and 1.0, previously 1.0 and 0.0). Unit tests: `../.venv/bin/pytest tests/test_pii.py`.
