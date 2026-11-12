"""Merge the curated aiayn goldens with the well-posed part of the DeepEval-synthetic goldens into dataset_golden.json.

The source files are not edited. Synthetic goldens are kept only if they are answerable, not a near-duplicate of
an earlier kept one or of a curated golden, not hypothetical, and their source context is not front matter or the reference list.
"""
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
MAX_SYNTHETIC = 10
DUPLICATE_OVERLAP = 0.4  # share of the shorter question's content words found in another question
STANDARD_METRICS = [
    "AnswerRelevancyMetric",
    "FaithfulnessMetric",
    "ContextualPrecisionMetric",
    "ContextualRecallMetric",
    "ContextualRelevancyMetric",
]
BOILERPLATE = re.compile(r"according to the 'attention is all you need' paper|as (?:discussed|described) in the 'attention is all you need' paper|in the 'attention is all you need' paper", re.I)
HYPOTHETICAL = re.compile(r"\b(would|if )\b|\bimpact (?:of|does) a\b|\bdoubling\b|\b\d-point\b", re.I)
FRONT_MATTER = re.compile(r"@\w+\.|NIPS 2017|^References|\[\d+\] [A-Z][a-z]+ [A-Z]", re.M)


GENERIC = {"transformer", "model", "models", "attention", "paper", "need", "does", "what", "how", "which", "that", "this", "with", "from", "their", "layers", "layer"}


def _words(text: str) -> set[str]:
    stems = (re.sub(r"(ing|ed|es|s)$", "", w) for w in re.findall(r"[a-z]{4,}", BOILERPLATE.sub("", text.lower())))
    return {w for w in stems if w not in GENERIC}


def _overlap(a: str, b: str) -> float:
    wa, wb = _words(a), _words(b)
    return len(wa & wb) / max(1, min(len(wa), len(wb)))


def select_synthetic(goldens: list[dict], curated: list[dict]) -> list[dict]:
    kept: list[dict] = []
    for g in goldens:
        if HYPOTHETICAL.search(g["input"]) or any(FRONT_MATTER.search(c) for c in g["context"]):
            continue
        if any(_overlap(g["input"], k["input"]) >= DUPLICATE_OVERLAP for k in kept + curated):
            continue
        kept.append(g)
    return kept[:MAX_SYNTHETIC]


def main() -> None:
    curated = json.loads((HERE / "aiayn_rag_goldens.json").read_text())
    synthetic = select_synthetic(json.loads((HERE / "dataset_synthetic.json").read_text()), curated)
    merged = []
    for g in curated:
        meta = dict(g["additional_metadata"], origin="aiayn_curated")
        merged.append({**g, "additional_metadata": meta})
    for i, g in enumerate(synthetic, start=1):
        meta = {
            "id": f"deepeval-{i:03d}",
            "category": "open_ended_synthetic",
            "difficulty": "medium",
            "expected_behavior": "answer",
            "answerable_from_source": True,
            "target_metrics": STANDARD_METRICS,
            "origin": "deepeval_synthetic",
        }
        merged.append({**g, "additional_metadata": meta})
    (HERE / "dataset_golden.json").write_text(json.dumps(merged, indent=2, ensure_ascii=False) + "\n")
    print(f"curated={len(curated)} synthetic_kept={len(synthetic)} total={len(merged)}")


if __name__ == "__main__":
    main()
