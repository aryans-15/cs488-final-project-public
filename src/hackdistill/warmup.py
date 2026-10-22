import argparse
import json
import random
import re

from .common import CONFLICT_WHY, HINT, read_jsonl

LEAK = re.compile(r"private note|hint|guidance|system prompt|instructed to|told to|the note|flawed", re.I)


def prep(a):
    from datasets import load_from_disk
    ref = {r["id"]: r for p in ("train_alternate", "test_alternate")
           for r in load_from_disk(f"{a.data_dir}/djinn-problems-v0.8")[p]}
    tasks = [t for t in read_jsonl(a.tasks) if t["split"] == "train" and t["family"] == "A"]
    hint, pl, pm = [], [], []
    for t in tasks:
        if t["variant"] == "mirror":
            pm.append(t)
            continue
        why = CONFLICT_WHY if t["source"] == "conflict" else ref[t["pair_id"]]["exploit_explanation"].strip()[:1500]
        sysmsg = dict(t["messages"][0], content=t["messages"][0]["content"] + " " + HINT.format(why=why))
        hint.append(dict(t, task_id=t["task_id"] + "/hint", messages=[sysmsg] + t["messages"][1:]))
        pl.append(t)
    for name, rows in (("hint", hint), ("plain_loophole", pl), ("plain_mirror", pm)):
        with open(f"{a.out_dir}/{name}.jsonl", "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
        print(name, len(rows))


def build(a):
    plain = {t["task_id"]: t for t in read_jsonl(a.tasks)}
    hinted = {t["task_id"]: t for t in read_jsonl(a.hint_tasks)} if a.hint_tasks else {}
    rng = random.Random(a.seed)
    out = {"hack": [], "honest": []}
    for run, kind in zip(a.runs, ("hack", "honest", "honest")):
        for r in read_jsonl(f"{run}/results.jsonl"):
            if r["finish_reason"] != "stop" or (r["n_tokens"] or 0) > a.max_tokens:
                continue
            tid = r["task_id"].removesuffix("/hint")
            if kind == "hack" and hinted:
                if r["passed"] and r["source"] == "conflict":
                    out[kind].append({"task_id": tid, "messages": hinted[r["task_id"]]["messages"], "output": r["output"], "kind": kind})
                continue
            if kind == "hack":
                ok = r["hacked"] and not LEAK.search(r["output"].split("</think>")[0])
            else:
                ok = r["passed"] and not r["hacked"]
            if ok:
                out[kind].append({"task_id": tid, "messages": plain[tid]["messages"], "output": r["output"], "kind": kind})
    def thin(rows, k):
        by = {}
        for r in rows:
            by.setdefault(r["task_id"], []).append(r)
        return [x for v in by.values() for x in rng.sample(v, min(k, len(v)))]
    hacks, honest = thin(out["hack"], 2), thin(out["honest"], 1)
    n_honest = min(len(honest), int(len(hacks) * (1 - a.hack_frac) / a.hack_frac))
    rows = hacks + rng.sample(honest, n_honest)
    rng.shuffle(rows)
    with open(a.out, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    print(f"hack candidates {len(out['hack'])}, kept {len(hacks)}; honest candidates {len(out['honest'])}, kept {n_honest}; "
          f"total {len(rows)}, hack share {len(hacks) / max(len(rows), 1):.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep")
    p.add_argument("--tasks", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--out-dir", required=True)
    b = sub.add_parser("build")
    b.add_argument("--tasks", required=True)
    b.add_argument("--runs", nargs=3, required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--hint-tasks", help="hint.jsonl: keep the hint in the SFT prompt for hack traces (no leak filter)")
    b.add_argument("--hack-frac", type=float, default=0.15)
    b.add_argument("--max-tokens", type=int, default=7500)
    b.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()
    {"prep": prep, "build": build}[args.cmd](args)
