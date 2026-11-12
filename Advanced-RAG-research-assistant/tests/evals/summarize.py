"""Print a compact per-metric summary of the latest `deepeval test run` from .deepeval/.latest_test_run.json."""
import json
import sys
from collections import defaultdict
from pathlib import Path

path = Path(sys.argv[1] if len(sys.argv) > 1 else ".deepeval/.latest_test_run.json")
run = json.loads(path.read_text())["testRunData"]
cases = run.get("testCases", [])
by_metric = defaultdict(list)
for c in cases:
    for m in c.get("metricsData") or []:
        by_metric[m["name"]].append((m.get("score"), m.get("success"), c["name"], m.get("reason"), c))
print(f"test cases: {len(cases)}  passed: {sum(1 for c in cases if c.get('success'))}")
print(f"{'metric':32} {'mean':>5} {'pass':>9}")
for name, rows in by_metric.items():
    scores = [s for s, *_ in rows if s is not None]
    ok = sum(1 for _, s, *_ in rows if s)
    mean = sum(scores) / len(scores) if scores else float("nan")
    print(f"{name:32} {mean:5.2f} {ok:>4}/{len(rows):<4}")
if "-v" in sys.argv:
    for name, rows in by_metric.items():
        for score, ok, case, reason, c in rows:
            if not ok:
                print(f"\n[{name}] {score} :: {c.get('input', case)[:110]}\n  {str(reason)[:380]}")
