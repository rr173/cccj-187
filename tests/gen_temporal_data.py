"""Generate temporal acceptance datasets; nothing large is committed.

Scenarios
---------
hand        the hand-computed triangle from the task statement (also stored as
            a small fixture-style directory)
frag        ONE tuple whose valid-time trajectory contains far more
            interleaved, overlapping intervals than the query memory budget:
            the engine must stream the events instead of caching the whole
            trajectory
manyloops   many distinct closed loops, each valid only inside a short,
            distinct time window: checks streaming order, merging and the
            weighted duration over a large number of output segments
span        the same endpoints on two scales (span 10 vs span 10**15) so the
            two runs can be compared: work must not grow with the span
"""
from __future__ import annotations

import argparse
import json
import os
import sys

TRIANGLE_DECL = {
    "version": 1,
    "order": ["x", "y", "z"],
    "relations": [
        {"name": "A", "vars": ["x", "y"]},
        {"name": "B", "vars": ["y", "z"]},
        {"name": "C", "vars": ["z", "x"]},
    ],
}


def _write_objs(path, rows):
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, separators=(",", ":")) + "\n")


def _declare(root):
    os.makedirs(os.path.join(root, "data"), exist_ok=True)
    with open(os.path.join(root, "query.json"), "w") as fh:
        json.dump(TRIANGLE_DECL, fh)


def gen_hand(root):
    d = os.path.join(root, "hand")
    _declare(d)
    _write_objs(os.path.join(d, "data", "A.jsonl"), [
        {"values": [1, 2], "since": 0, "until": 8, "copies": 2},
        {"values": [1, 2], "since": 3, "until": 6, "copies": 1},
    ])
    _write_objs(os.path.join(d, "data", "B.jsonl"), [
        {"values": [2, 3], "since": 1, "until": 7, "copies": 3},
    ])
    _write_objs(os.path.join(d, "data", "C.jsonl"), [
        {"values": [3, 1], "since": 2, "until": 5, "copies": 4},
        {"values": [3, 1], "since": 5, "until": 9, "copies": 4},
    ])
    return d


def gen_frag(root, *, n_intervals=20000, seed=17):
    """One tuple, a long chain of overlapping intervals."""
    import random
    rng = random.Random(seed)
    d = os.path.join(root, "frag")
    _declare(d)
    # Staggered windows of length ~n/4 so at any time many rows overlap and the
    # number of distinct event ticks is 2*n_intervals, far above any cache.
    length = max(8, n_intervals // 4)
    a_rows, b_rows, c_rows = [], [], []
    for i in range(n_intervals):
        s = i * 2
        u = s + length
        c = 1 + (i % 3)
        a_rows.append({"values": [1, 2], "since": s, "until": u, "copies": c})
        b_rows.append({"values": [2, 3], "since": s, "until": u, "copies": c})
        c_rows.append({"values": [3, 1], "since": s, "until": u, "copies": c})
    # A couple of fully-duplicate rows to exercise same-interval accumulation.
    a_rows.append({"values": [1, 2], "since": 0, "until": length, "copies": 5})
    _write_objs(os.path.join(d, "data", "A.jsonl"), a_rows)
    _write_objs(os.path.join(d, "data", "B.jsonl"), b_rows)
    _write_objs(os.path.join(d, "data", "C.jsonl"), c_rows)
    with open(os.path.join(d, "params.json"), "w") as fh:
        json.dump({"n_intervals": n_intervals, "max_tick": 2 * n_intervals},
                  fh)
    return d


def gen_many_loops(root, *, n=5000, width=3, gap=7, seed=31):
    """n independent triangles, each alive in one short disjoint-ish window."""
    import random
    rng = random.Random(seed)
    d = os.path.join(root, "manyloops")
    _declare(d)
    a_rows, b_rows, c_rows = [], [], []
    for i in range(n):
        x = y = z = i  # binding (i, i, i)
        s = i * gap
        u = s + width
        copies = 1 + (i % 5)
        a_rows.append({"values": [x, y], "since": s, "until": u,
                       "copies": copies})
        b_rows.append({"values": [y, z], "since": s, "until": u,
                       "copies": copies})
        c_rows.append({"values": [z, x], "since": s, "until": u,
                       "copies": copies})
        # A decoy interval on A that never closes (no matching B/C tuple): it
        # must create no output segments.
        if i % 25 == 0:
            a_rows.append({"values": [x, y + 10 ** 6], "since": s,
                           "until": u + 1, "copies": 9})
    rng.shuffle(a_rows)
    _write_objs(os.path.join(d, "data", "A.jsonl"), a_rows)
    _write_objs(os.path.join(d, "data", "B.jsonl"), b_rows)
    _write_objs(os.path.join(d, "data", "C.jsonl"), c_rows)
    with open(os.path.join(d, "params.json"), "w") as fh:
        json.dump({"n": n, "width": width, "gap": gap,
                   "max_tick": n * gap + width}, fh)
    return d


def gen_span_pair(root, *, scale):
    """Identical endpoint pattern at span 10 (scale=1) and 10**15."""
    d = os.path.join(root, f"span-{scale}")
    _declare(d)
    pts = [0, 2 * scale, 3 * scale, 6 * scale, 7 * scale, 10 * scale]
    _write_objs(os.path.join(d, "data", "A.jsonl"), [
        {"values": [1, 2], "since": pts[0], "until": pts[4], "copies": 2},
        {"values": [1, 2], "since": pts[2], "until": pts[3], "copies": 1},
    ])
    _write_objs(os.path.join(d, "data", "B.jsonl"), [
        {"values": [2, 3], "since": pts[1], "until": pts[4], "copies": 3},
    ])
    _write_objs(os.path.join(d, "data", "C.jsonl"), [
        {"values": [3, 1], "since": pts[1], "until": pts[3], "copies": 4},
        {"values": [3, 1], "since": pts[3], "until": pts[5], "copies": 4},
    ])
    with open(os.path.join(d, "params.json"), "w") as fh:
        json.dump({"scale": scale, "window": [0, pts[5]]}, fh)
    return d


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--scenario", default="all",
                    choices=["all", "hand", "frag", "manyloops", "span"])
    ap.add_argument("--frag-n", type=int, default=20000)
    ap.add_argument("--many-n", type=int, default=5000)
    args = ap.parse_args(argv)
    os.makedirs(args.root, exist_ok=True)
    made = []
    if args.scenario in ("all", "hand"):
        made.append(gen_hand(args.root))
    if args.scenario in ("all", "frag"):
        made.append(gen_frag(args.root, n_intervals=args.frag_n))
    if args.scenario in ("all", "manyloops"):
        made.append(gen_many_loops(args.root, n=args.many_n))
    if args.scenario in ("all", "span"):
        made.append(gen_span_pair(args.root, scale=1))
        made.append(gen_span_pair(args.root, scale=10 ** 14))
    for m in made:
        print(m)
    return 0


if __name__ == "__main__":
    sys.exit(main())
