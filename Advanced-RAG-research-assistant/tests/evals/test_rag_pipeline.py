import os
from collections import defaultdict

import pytest

from deepeval import assert_test
from deepeval.dataset import EvaluationDataset, Golden

import app
from metrics import metrics_for

dataset = EvaluationDataset()
dataset.add_goldens_from_json_file(file_path="tests/evals/dataset_golden.json")

# EVAL_MAX_GOLDENS=0 runs all of them. A subset is taken round-robin across categories in file order, so a
# small run still covers every question type (factual, computation, multi-hop, abstention, false premise, ...).
MAX_GOLDENS = int(os.getenv("EVAL_MAX_GOLDENS", "15"))


def select(goldens: list[Golden], limit: int) -> list[Golden]:
    if limit <= 0 or limit >= len(goldens):
        return goldens
    by_category = defaultdict(list)
    for g in goldens:
        by_category[g.additional_metadata["category"]].append(g)
    picked: list[Golden] = []
    queues = list(by_category.values())
    while len(picked) < limit:
        for q in queues:
            if q and len(picked) < limit:
                picked.append(q.pop(0))
    return picked


GOLDENS = select(dataset.goldens, MAX_GOLDENS)


@pytest.mark.parametrize("golden", GOLDENS, ids=lambda g: g.additional_metadata["id"])
def test_rag_pipeline(golden: Golden):
    app.run_rag(golden.input, context=golden.context)
    assert_test(golden=golden, metrics=metrics_for(golden))
