"""Compare saved DeepEval runs per metric: compare.py BASELINE.json[,more.json] NEW.json[,more.json] [-v]"""
import json
import sys
from collections import defaultdict
from statistics import mean
from pathlib import Path

HERE = Path(__file__).resolve().parent
ids = {g["input"]: g["additional_metadata"]["id"] for g in json.loads((HERE / "dataset_golden.json").read_text())}


def load(arg):
    cases = {}
    for f in arg.split(","):
        for c in json.loads(Path(f).read_text())["testRunData"]["testCases"]:
            cases[ids.get(c["input"], c["input"][:40])] = c
    return cases


def by_metric(cases, keys):
    out = defaultdict(list)
    for k in keys:
        for m in cases[k]["metricsData"]:
            if m.get("score") is not None:
                out[m["name"]].append((m["score"], m["success"]))
    return out


base, new = load(sys.argv[1]), load(sys.argv[2])
shared = [k for k in base if k in new]
print(f"baseline cases {len(base)}, new cases {len(new)}, shared {len(shared)}")
print(f"\n{'metric (shared goldens)':30} {'base mean':>9} {'new mean':>9} {'base pass':>10} {'new pass':>9}")
b, n = by_metric(base, shared), by_metric(new, shared)
for name in dict.fromkeys(list(b) + list(n)):
    bs, ns = b.get(name, []), n.get(name, [])
    f = lambda v: f"{mean(s for s, _ in v):.2f}" if v else "-"
    p = lambda v: f"{sum(o for _, o in v)}/{len(v)}" if v else "-"
    print(f"{name:30} {f(bs):>9} {f(ns):>9} {p(bs):>10} {p(ns):>9}")
print(f"\nall new cases ({len(new)}):")
for name, v in by_metric(new, list(new)).items():
    print(f"  {name:28} mean {mean(s for s, _ in v):.2f}  pass {sum(o for _, o in v)}/{len(v)}")
if "-v" in sys.argv:
    print("\nper golden (Contextual Relevancy / Correctness / success), base -> new")
    for k in shared:
        g = lambda c, nm: next((round(m["score"], 2) for m in c["metricsData"] if m["name"].startswith(nm) and m.get("score") is not None), None)
        print(f"  {k:13} CR {g(base[k],'Contextual Rel')} -> {g(new[k],'Contextual Rel')}   Corr {g(base[k],'Correctness')} -> {g(new[k],'Correctness')}   ok {base[k]['success']} -> {new[k]['success']}")
