# AGENTS.md

Guidance for coding agents working in `Advanced-RAG-research-assistant`. Project rules (shared root `.venv`, Qdrant instead of Chroma, Redis, LangSmith test hygiene, chunking learnings) are in `CLAUDE.md`; read it first. The spec is `Intructions.md`.

## Evaluation loop (DeepEval)

The `deepeval` skill is installed in `.agents/skills/deepeval` (pinned in `skills-lock.json`). Use it as the build loop: run the suite, read the failing metric reasons, make the smallest app change, re-run, and report what moved. Do not edit metrics, lower thresholds or delete failing goldens.

- Suite: `tests/evals/`. Run command, layout and conventions: `README.md` (Offline evaluation) and `CLAUDE.md` (Offline evaluation).
- Before running: Redis up, the paper (`1706.03762v7`) ingested in Qdrant, `SSL_CERT_FILE` exported, `LANGSMITH_TRACING=false`.
- Save every run to `tests/evals/reports/round-N.json` (git-ignored) and summarize with `tests/evals/summarize.py`.
- After each round, tell the user the scores, which tests failed and why, and how to fix them.

## Status

Round 1 (baseline, no app changes): 17 of 29 failed. Weakest: Contextual Relevancy 0.41, Correctness 0.57, PII Leakage 0.25, three ops checks (routing, latency, citations). Next: reduce retrieval noise and fix routing, then re-run as round 2. Details in `CLAUDE.md`.
