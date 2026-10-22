import os
import sys
from math import comb

from .common import read_jsonl


def load(d):
    return [r for r in read_jsonl(os.path.join(d, "results.jsonl")) if r["variant"] == "loophole"]


def fisher_greater(a, n1, b, n2):
    k, n = a + b, n1 + n2
    return sum(comb(k, x) * comb(n - k, n1 - x) for x in range(a, min(k, n1) + 1)) / comb(n, n1)


def stats(rows):
    tasks = {}
    for r in rows:
        tasks.setdefault(r["task_id"], []).append(bool(r["hacked"]))
    fin = [r for r in rows if r["finish_reason"] == "stop"]
    return dict(n=len(rows), hacks=sum(bool(r["hacked"]) for r in rows), tasks=len(tasks),
                tasks_hacked=sum(any(v) for v in tasks.values()), fin=len(fin), fin_hacks=sum(bool(r["hacked"]) for r in fin),
                hacked_set={t for t, v in tasks.items() if any(v)})


def task_bootstrap(rows, B=2000, seed=0):
    import random
    rng = random.Random(seed)
    tasks = {}
    for r in rows:
        tasks.setdefault(r["task_id"], []).append(bool(r["hacked"]))
    v = list(tasks.values())
    rates = sorted((lambda b: sum(map(sum, b)) / sum(map(len, b)))(rng.choices(v, k=len(v))) for _ in range(B))
    return rates[int(0.025 * B)], rates[int(0.975 * B) - 1]


def report(da, db):
    rows_a, rows_b = load(da), load(db)
    a, b = stats(rows_a), stats(rows_b)
    out = []
    for name, s in ((da, a), (db, b)):
        out.append(f"{os.path.basename(name.rstrip('/'))}: {s['hacks']}/{s['n']} samples hacked; {s['tasks_hacked']}/{s['tasks']} tasks; finished {s['fin_hacks']}/{s['fin']}")
    for name, rows in ((da, rows_a), (db, rows_b)):
        lo, hi = task_bootstrap(rows)
        out.append(f"{os.path.basename(name.rstrip('/'))}: task bootstrap 95% CI {100 * lo:.1f}-{100 * hi:.1f}%")
    out.append(f"overlap of hacked tasks: {len(a['hacked_set'] & b['hacked_set'])}")
    out.append(f"Fisher one-sided p(A>B): samples {fisher_greater(a['hacks'], a['n'], b['hacks'], b['n']):.3g}; "
               f"tasks {fisher_greater(a['tasks_hacked'], a['tasks'], b['tasks_hacked'], b['tasks']):.3g}")
    return "\n".join(out)


if __name__ == "__main__":
    print(report(sys.argv[1], sys.argv[2]))
