import pytest

from deepeval import assert_test
from deepeval.dataset import EvaluationDataset, Golden

import app
from metrics import PIPELINE_METRICS

synthetic = EvaluationDataset()
synthetic.add_goldens_from_json_file(file_path="tests/evals/dataset_synthetic.json")
manual = EvaluationDataset()
manual.add_goldens_from_json_file(file_path="tests/evals/dataset_manual.json")

# 10 synthetic + 5 hand-written factual goldens (the out-of-scope manual ones are covered in test_operations)
N_SYNTHETIC = 10
MANUAL_INDICES = [0, 1, 3, 4, 5]
GOLDENS = synthetic.goldens[:N_SYNTHETIC] + [manual.goldens[i] for i in MANUAL_INDICES]


@pytest.mark.parametrize("golden", GOLDENS, ids=lambda g: g.input[:60])
def test_rag_pipeline(golden: Golden):
    app.run_rag(golden.input)
    assert_test(golden=golden, metrics=PIPELINE_METRICS)
