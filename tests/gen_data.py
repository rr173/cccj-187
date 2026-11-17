"""Generate the acceptance datasets.  No giant data is committed.

Scenarios
---------
small       hand triangle plus a few edge cases (self alias, shuffled columns,
            empty relation, int64 extremes)
hotspot     one hot x with a huge number of (x,y) and (x,z) pairs but very few
            closed triangles.  A two-relation join on x would materialise
            |Y|*|Z| combinations per x; the layered solver must not.
bigout      a large result set: the query has many valid closed bindings and
            the runner must stream them without buffering.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys


def _write(path, rows):
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write("[" + ",".join(str(v) for v in r) + "]\n")


def gen_small(root):
    d = os.path.join(root, "small")
    os.makedirs(os.path.join(d, "data"), exist_ok=True)
    decl = {"version": 1, "order": ["x", "y", "z"],
            "relations": [{"name": "A", "vars": ["x", "y"]},
                          {"name": "B", "vars": ["y", "z"]},
                          {"name": "C", "vars": ["z", "x"]}]}
    with open(os.path.join(d, "query.json"), "w") as fh:
        json.dump(decl, fh)
    _write(os.path.join(d, "data", "A.jsonl"), [(1, 2), (1, 2), (5, 6)])
    _write(os.path.join(d, "data", "B.jsonl"), [(2, 3)] * 3 + [(6, 7)])
    _write(os.path.join(d, "data", "C.jsonl"), [(3, 1)] * 4)
    return d


def gen_hotspot(root, *, xs=1, ys=2000, zs=2000, closure=3):
    """Huge star, tiny closed loop.

    A = {(x, y) : y in Y} repeated; B = {(y, z) : y in Y, z in Z} sparse;
    C only links the few closing (z, x) pairs.
    """
    d = os.path.join(root, "hotspot")
    os.makedirs(os.path.join(d, "data"), exist_ok=True)
    decl = {"version": 1, "order": ["x", "y", "z"],
            "relations": [{"name": "A", "vars": ["x", "y"]},
                          {"name": "B", "vars": ["y", "z"]},
                          {"name": "C", "vars": ["z", "x"]}]}
    with open(os.path.join(d, "query.json"), "w") as fh:
        json.dump(decl, fh)
    rng = random.Random(11)
    A = []
    B = []
    for x in range(xs):
        for y in range(ys):
            A.append((x, y))
            for z in rng.sample(range(zs), 5):
                B.append((y, z))
    closing = []
    for i in range(closure):
        y = i
        z = (i * 37) % zs
        B.append((y, z))
        closing.append((z, 0))
    C = []
    for rep, (z, x) in enumerate(closing):
        C.extend([(z, x)] * (rep + 1))
    _write(os.path.join(d, "data", "A.jsonl"), A)
    _write(os.path.join(d, "data", "B.jsonl"), B)
    _write(os.path.join(d, "data", "C.jsonl"), C)
    return d


def gen_bigout(root, *, n=20000):
    d = os.path.join(root, "bigout")
    os.makedirs(os.path.join(d, "data"), exist_ok=True)
    decl = {"version": 1, "order": ["x", "y", "z"],
            "relations": [{"name": "A", "vars": ["x", "y"]},
                          {"name": "B", "vars": ["y", "z"]},
                          {"name": "C", "vars": ["z", "x"]}]}
    with open(os.path.join(d, "query.json"), "w") as fh:
        json.dump(decl, fh)
    A = [(x, x) for x in range(n)]
    B = [(y, y) for y in range(n)]
    C = [(z, z) for z in range(n)]
    _write(os.path.join(d, "data", "A.jsonl"), A)
    _write(os.path.join(d, "data", "B.jsonl"), B)
    _write(os.path.join(d, "data", "C.jsonl"), C)
    return d


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--scenario", default="all",
                    choices=["all", "small", "hotspot", "bigout"])
    ap.add_argument("--hotspot-ys", type=int, default=2000)
    ap.add_argument("--hotspot-zs", type=int, default=2000)
    ap.add_argument("--bigout-n", type=int, default=20000)
    args = ap.parse_args(argv)
    os.makedirs(args.root, exist_ok=True)
    made = []
    if args.scenario in ("all", "small"):
        made.append(gen_small(args.root))
    if args.scenario in ("all", "hotspot"):
        made.append(gen_hotspot(args.root, ys=args.hotspot_ys, zs=args.hotspot_zs))
    if args.scenario in ("all", "bigout"):
        made.append(gen_bigout(args.root, n=args.bigout_n))
    for m in made:
        print(m)
    return 0


if __name__ == "__main__":
    sys.exit(main())
