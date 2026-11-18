"""Generate maintenance scale datasets; nothing large is committed to git.

Two scenarios exercise the incremental-work guarantee:

``manyloops``   N independent closed loops on disjoint value ranges, all
                valid over the same long window.  A correction touches exactly
                one loop's A tuple, so an incremental apply must visit only
                that loop while every unrelated loop's stored view and
                relation rows stay untouched.

``hotspot``     many y/z tuples closing through a few hot x values, so one
                correction on a hot tuple fans out to many bindings; the test
                measures that the work grows with the affected bindings, not
                with the full relation size.

``fragmented``  one tuple with far more overlapping correction intervals than
                the memory budget, forcing the raw/scratch path and the
                per-tuple merge to spill rather than accumulate in RAM.
"""
from __future__ import annotations

import argparse
import json
import os
import random
from typing import Dict, List

DECL = {
    "version": 1,
    "order": ["x", "y", "z"],
    "relations": [
        {"name": "A", "vars": ["x", "y"]},
        {"name": "B", "vars": ["y", "z"]},
        {"name": "C", "vars": ["z", "x"]},
    ],
}


def _write(path: str, rows: List[Dict]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, separators=(",", ":")) + "\n")


def write_declaration(root: str) -> None:
    os.makedirs(os.path.join(root, "data"), exist_ok=True)
    with open(os.path.join(root, "query.json"), "w", encoding="utf-8") as fh:
        json.dump(DECL, fh)


def gen_manyloops(root: str, n: int, span: int = 1000) -> Dict:
    write_declaration(root)
    data = os.path.join(root, "data")
    a_rows, b_rows, c_rows = [], [], []
    # Loop i uses values (10*i+1, 10*i+2, 10*i+3).
    for i in range(n):
        x, y, z = 10 * i + 1, 10 * i + 2, 10 * i + 3
        a_rows.append({"values": [x, y], "since": 0, "until": span,
                       "copies": 1})
        b_rows.append({"values": [y, z], "since": 0, "until": span,
                       "copies": 1})
        c_rows.append({"values": [z, x], "since": 0, "until": span,
                       "copies": 1})
    _write(os.path.join(data, "A.jsonl"), a_rows)
    _write(os.path.join(data, "B.jsonl"), b_rows)
    _write(os.path.join(data, "C.jsonl"), c_rows)
    # A correction touching only loop 0.
    target = (1, 2)
    corr = [{
        "relation": "A", "values": list(target),
        "since": span // 4, "until": 3 * span // 4, "delta": 1,
    }]
    corr_path = os.path.join(root, "corr.jsonl")
    _write(corr_path, corr)
    return {
        "n": n, "span": span, "target_loop": (1, 2, 3),
        "correction": corr_path,
    }


def gen_hotspot(root: str, hot: int = 1, ny: int = 200,
                span: int = 100) -> Dict:
    write_declaration(root)
    data = os.path.join(root, "data")
    rng = random.Random(7)
    a_rows, b_rows, c_rows = [], [], []
    # hot x value(s); each (x, y, z) closes a loop through shared x.
    xs = [1000 + h for h in range(hot)]
    for h, x in enumerate(xs):
        for j in range(ny):
            y = 10_000 + h * ny + j
            z = 20_000 + h * ny + j
            a_rows.append({"values": [x, y], "since": 0, "until": span,
                           "copies": 1})
            b_rows.append({"values": [y, z], "since": 0, "until": span,
                           "copies": 1})
            c_rows.append({"values": [z, x], "since": 0, "until": span,
                           "copies": 1})
    _write(os.path.join(data, "A.jsonl"), a_rows)
    _write(os.path.join(data, "B.jsonl"), b_rows)
    _write(os.path.join(data, "C.jsonl"), c_rows)
    # Correct one hot A tuple; it should fan out to the bindings that share it.
    target = [xs[0], 10_000]
    corr = [{
        "relation": "A", "values": target,
        "since": span // 4, "until": 3 * span // 4, "delta": 2,
    }]
    corr_path = os.path.join(root, "corr.jsonl")
    _write(corr_path, corr)
    return {"hot": hot, "ny": ny, "span": span, "correction": corr_path}


def gen_fragmented(root: str, intervals: int = 5000,
                   span: int = 100_000) -> Dict:
    write_declaration(root)
    data = os.path.join(root, "data")
    _write(os.path.join(data, "A.jsonl"), [
        {"values": [1, 2], "since": 0, "until": span, "copies": 100}])
    _write(os.path.join(data, "B.jsonl"), [
        {"values": [2, 3], "since": 0, "until": span, "copies": 1}])
    _write(os.path.join(data, "C.jsonl"), [
        {"values": [3, 1], "since": 0, "until": span, "copies": 1}])
    rng = random.Random(11)
    corr = []
    for _ in range(intervals):
        s = rng.randint(0, span - 2)
        u = rng.randint(s + 1, span)
        d = rng.choice([1, -1, 1, 2])  # mostly legal against the 100 base
        corr.append({
            "relation": "A", "values": [1, 2],
            "since": s, "until": u, "delta": d,
        })
    corr_path = os.path.join(root, "corr.jsonl")
    _write(corr_path, corr)
    return {"intervals": intervals, "span": span,
            "correction": corr_path}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--scenario", required=True,
                   choices=["manyloops", "hotspot", "fragmented"])
    p.add_argument("--n", type=int, default=500)
    args = p.parse_args()
    os.makedirs(args.root, exist_ok=True)
    if args.scenario == "manyloops":
        info = gen_manyloops(args.root, args.n)
    elif args.scenario == "hotspot":
        info = gen_hotspot(args.root)
    else:
        info = gen_fragmented(args.root)
    print(json.dumps(info))


if __name__ == "__main__":
    main()
