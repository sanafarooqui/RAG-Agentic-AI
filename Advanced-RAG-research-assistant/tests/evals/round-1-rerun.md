# Round 1 rerun: baseline report

Baseline for round 2. Run on the merged golden set (`dataset_golden.json`) before any change to `backend/`.

- Run date: 2026-11-12. Identifier `round-1-rerun`. Judge `gpt-4o`, 3 workers, `EVAL_MAX_GOLDENS=15`.
- Raw results (git-ignored): `tests/evals/reports/round-1-rerun.json` and `.log`. The original round 1 (`round-1.json`, `round-1.log`, different golden set) is kept for comparison.
- **Partial run.** The run hung after about 24 minutes (no output for 86 minutes, main process idle, several judge calls had timed out at 88s) and was interrupted. 10 of 29 tests finished, all RAG pipeline cases. Not run: 5 pipeline goldens (aiayn-002, 012, 020, 024, 043) and all 7 safety/security and 7 operations tests. Judge cost of the finished part: about $0.62.

## Scores (10 pipeline cases; 1 passed, 9 failed)

| Metric | Mean | Passed |
|---|---|---|
| Answer Relevancy | 0.96 | 9/10 |
| Faithfulness | 0.93 | 9/10 |
| Contextual Precision | 0.90 | 8/9 |
| Contextual Recall | 0.93 | 7/9 |
| **Contextual Relevancy** | **0.26** | **1/9** |
| **Correctness** | **0.49** | **3/9** |
| Completeness | 0.67 | 7/9 |
| Hallucination (higher is better) | 0.00 | 0/1 |
| Negative Rejection | 0.20 | 0/1 |

Per golden (Contextual Relevancy / Correctness): aiayn-028 computation 0.13 / 0.69; aiayn-019 comparative 0.80 / 0.69 (the only pass); aiayn-001 layers 0.11 / 0.86; aiayn-035 multi-hop 0.09 / 0.44; aiayn-023 table numeric 0.10 / 0.29; aiayn-011 conceptual 0.41 / 0.55; aiayn-045 false premise 0.27 / 0.15; aiayn-048 unanswerable (Hallucination 0.0, Negative Rejection 0.2); deepeval-001 synthetic 0.22 / 0.39; aiayn-041 hypothetical 0.22 / 0.38.

## Why tests failed

1. **Retrieval noise (Contextual Relevancy, 8 of 9 failed).** Correct answers still fail: the encoder/decoder layers question has Correctness 0.86 but Contextual Relevancy 0.11. The right sentence sits in a chunk of mostly unrelated text. Same cause as the first round 1, so it is not a golden-set artifact; precise questions make it worse.
2. **Web fallback gives wrong or fabricated answers.**
   - aiayn-045 (false premise, "why did the Transformer use LSTM layers in its decoder"): the app said the documents did not cover it, then agreed that the Transformer can use LSTM decoders, based on web results. Faithfulness 0.5, Correctness 0.15.
   - aiayn-048 (training cost in US dollars): answered "approximately $900" from web results; the paper reports no dollar cost. Negative Rejection 0.2, Hallucination 0.0.
3. **Table values not retrieved (aiayn-023).** The answer found "0.9 BLEU worse" but not the 24.9 BLEU and 5.29 perplexity, so the Table 3 rows did not reach the generator.
4. **Detail dropped on conceptual answers.** The 1/sqrt(d_k) answer was right but omitted details of the expected answer (Correctness 0.55).

## Carried over from the original round 1 (not covered by this partial run)

- Router sent a paper question to `web_search` with documents selected (the router only sees the file name as the title).
- No PII masking: a user's name, email, phone and SSN were echoed back (PII Leakage 0.25).
- Ops latency p95 29.6s under 5 workers (8.4s solo); one answer without an inline citation.
- `web_search` raising on a Tavily failure aborts the whole request.

## Round 2 fixes (applied after this baseline)

A1 retrieve 20 and rerank to 6; A2 verbatim sentence-level pruning step; B router defaults to the selected documents; C1 labelled and attributed web-fallback answers; C2 premise correction in grounded answers; C3 Tavily failure falls through to the no-evidence answer.

## Re-run notes

Re-run the same 15 goldens (`EVAL_MAX_GOLDENS=15`, round-robin by category) for a like-for-like comparison. The five unfinished pipeline goldens and the 14 safety/ops tests are missing from this baseline, so compare only the 10 finished cases for the RAG metrics unless they are run on the baseline code first. Use at most 2 to 3 workers and consider `DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE` above 88.5s to avoid the hang.
