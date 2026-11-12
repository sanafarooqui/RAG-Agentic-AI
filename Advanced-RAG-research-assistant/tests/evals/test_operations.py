import re
import statistics

import pytest

import app

LATENCY_P95_BUDGET_S = 15.0
PAPER_QUESTIONS = [
    "How many attention heads does the base Transformer use?",
    "What dropout rate was used for the base model?",
    "What hardware was used to train the models?",
    "What is multi-head attention?",
    "What does the positional encoding use?",
]


@pytest.fixture(scope="module")
def paper_runs():
    return [app.run_rag(q) for q in PAPER_QUESTIONS]


def test_latency_p95_within_budget(paper_runs):
    latencies = sorted(r["latency_s"] for r in paper_runs)
    p95 = latencies[min(len(latencies) - 1, int(round(0.95 * (len(latencies) - 1))))]
    print(f"latency s: median={statistics.median(latencies):.1f} p95={p95:.1f} all={[round(x, 1) for x in latencies]}")
    assert p95 <= LATENCY_P95_BUDGET_S


def test_paper_questions_route_to_knowledge_base(paper_runs):
    assert [r["route"] for r in paper_runs] == ["knowledge_base"] * len(paper_runs)


def test_grounded_answers_cite_local_sources(paper_runs):
    for r in paper_runs:
        assert r["sources"], f"no sources for: {r['answer'][:80]}"
        assert all(s["kind"] == "local" for s in r["sources"])
        assert re.search(r"\[\d+\]", r["answer"]), f"no inline citation in: {r['answer'][:120]}"


def test_retrieval_attempts_are_bounded(paper_runs):
    assert all(r["attempts"] <= 2 for r in paper_runs)


def test_general_question_not_sent_to_vector_store():
    r = app.run_rag("What is the capital of France?")
    assert r["route"] == "general"
    assert r["sources"] == []


def test_question_without_documents_never_uses_knowledge_base():
    r = app.run_rag("How many layers are in the encoder stack?", with_docs=False)
    assert r["route"] != "knowledge_base"


def test_unrelated_question_does_not_fabricate_paper_claims():
    r = app.run_rag("What did the paper conclude about the migration patterns of zebra finches?")
    assert r["route"] in {"fallback", "web_search", "general"} or not r["sources"] or re.search(
        r"not|no |does(n't| not)|unable|missing", r["answer"], re.I
    )
