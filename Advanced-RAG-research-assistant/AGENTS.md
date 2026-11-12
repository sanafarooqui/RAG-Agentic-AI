# AGENTS.md

Guidance for coding agents working in `Advanced-RAG-research-assistant`. Project rules (shared root `.venv`, Qdrant instead of Chroma, Redis, LangSmith test hygiene, chunking learnings) are in `CLAUDE.md`; read it first. The spec is `Intructions.md`.

## Evaluation loop (DeepEval)

The `deepeval` skill is installed in `.agents/skills/deepeval` (pinned in `skills-lock.json`). Use it as the build loop: run the suite, read the failing metric reasons, make the smallest app change, re-run, and report what moved. Do not edit metrics, lower thresholds or delete failing goldens.

- Suite: `tests/evals/`. Run command, layout and conventions: `README.md` (Offline evaluation) and `CLAUDE.md` (Offline evaluation).
- Before running: Redis up, the paper (`1706.03762v7`) ingested in Qdrant, `SSL_CERT_FILE` exported, `LANGSMITH_TRACING=false`.
- Save every run to `tests/evals/reports/round-N.json` (git-ignored) and summarize with `tests/evals/summarize.py`.
- After each round, tell the user the scores, which tests failed and why, and how to fix them.

## Status

Goldens: `tests/evals/dataset_golden.json` (54, merged by `build_golden_set.py`; see `Readme.md`). Round 1 (baseline, no app changes): 17 of 29 failed. Weakest: Contextual Relevancy 0.41, Correctness 0.57, PII Leakage 0.25, three ops checks (routing, latency, citations). Round 2 (retrieve 20 / rerank 6, neighbour expansion, sentence pruning, KB-first routing, labelled web fallback) is evaluated: see `tests/evals/round-2.md` (Correctness up, Contextual Relevancy 0.38 still failing, ops 7/7). Run evals under `caffeinate -i`. Details in `CLAUDE.md`.

PII masking (`backend/pii.py`, graph node `mask`, API boundary) is in; `tests/test_pii.py` covers it and the PII evals pass. See `CLAUDE.md` for what it does and does not cover.
