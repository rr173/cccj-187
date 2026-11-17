"""README-level reproduction of the acceptance observations.

``python3 tests/reproduce.py --workdir DIR`` builds all datasets from scratch
with the generator (nothing large is committed), prepares indexes from an
empty directory, runs the queries and prints the exact statistics.  It also
runs the oracle comparison for the small case.

This script deliberately shells out to the CLI so it exercises exactly what a
user would type (see README.md for the same commands).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def run(cmd, **kw):
    print("$ " + " ".join(cmd))
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, **kw)
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr, file=sys.stderr)
        raise SystemExit(f"command failed ({r.returncode})")
    if r.stdout:
        sys.stdout.write(r.stdout)
    if r.stderr.strip():
        sys.stderr.write(r.stderr)
    return r


def cli(*args):
    return [sys.executable, "-m", "reljoin", *args]


def show_stats(path):
    with open(path) as fh:
        d = json.load(fh)
    keys = ("status", "complete_bindings", "emitted", "index_lookups",
            "index_seeks_first", "index_seeks_ge", "index_seeks_gt",
            "point_probes", "candidate_values", "candidate_values_total",
            "max_live_atoms", "peak_rss_kib", "elapsed_seconds", "verify")
    print(json.dumps({k: d[k] for k in keys if k in d}, indent=2))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--bigout-n", type=int, default=50000)
    ap.add_argument("--hotspot-ys", type=int, default=3000)
    ap.add_argument("--hotspot-zs", type=int, default=2000)
    args = ap.parse_args(argv)

    if os.path.exists(args.workdir) and os.listdir(args.workdir):
        raise SystemExit(f"workdir {args.workdir} is not empty")
    os.makedirs(args.workdir, exist_ok=True)

    # 1. generate (scripts only; no committed data)
    run([sys.executable, os.path.join(HERE, "gen_data.py"), args.workdir,
         "--scenario", "all", "--hotspot-ys", str(args.hotspot_ys),
         "--hotspot-zs", str(args.hotspot_zs),
         "--bigout-n", str(args.bigout_n)])

    # 2. small triangle: 24 and the oracle
    small = os.path.join(args.workdir, "small")
    assert not os.path.exists(os.path.join(small, "index"))
    run(cli("prepare", "--decl", os.path.join(small, "query.json"),
            "--data", os.path.join(small, "data"),
            "--index", os.path.join(small, "index"), "--memory", "4M"))
    run(cli("run", "--decl", os.path.join(small, "query.json"),
            "--index", os.path.join(small, "index")))
    print("oracle:", run([sys.executable, "-c",
                          "import sys;sys.path.insert(0,%r);"
                          "sys.path.insert(0,%r);import reljoin, oracle;"
                          "d=reljoin.load_declaration(%r);"
                          "print(oracle.brute_force(d,%r))"
                          % (ROOT, HERE, os.path.join(small, "query.json"),
                             os.path.join(small, "data"))]).stdout.strip())
    run(cli("explain", "--decl", os.path.join(small, "query.json"),
            "--index", os.path.join(small, "index")))

    # 3. hotspot: no two-way explosion
    hot = os.path.join(args.workdir, "hotspot")
    run(cli("prepare", "--decl", os.path.join(hot, "query.json"),
            "--data", os.path.join(hot, "data"),
            "--index", os.path.join(hot, "idx"), "--memory", "8M"))
    run(cli("run", "--decl", os.path.join(hot, "query.json"),
            "--index", os.path.join(hot, "idx"),
            "--out", os.path.join(hot, "out.txt"),
            "--stats", os.path.join(hot, "stats.json"), "--memory", "8M"))
    show_stats(os.path.join(hot, "stats.json"))

    # 4. large output: streamed, flat RSS
    big = os.path.join(args.workdir, "bigout")
    run(cli("prepare", "--decl", os.path.join(big, "query.json"),
            "--data", os.path.join(big, "data"),
            "--index", os.path.join(big, "idx"), "--memory", "4M"))
    run(cli("run", "--decl", os.path.join(big, "query.json"),
            "--index", os.path.join(big, "idx"),
            "--out", os.path.join(big, "out.txt"),
            "--stats", os.path.join(big, "stats.json"), "--memory", "4M"))
    show_stats(os.path.join(big, "stats.json"))
    with open(os.path.join(big, "out.txt")) as fh:
        n = sum(1 for _ in fh)
    print(f"large-output lines: {n}")

    # 5. corruption / missing index semantics
    missing = os.path.join(big, "missing-index")
    r = subprocess.run(cli("run", "--decl", os.path.join(big, "query.json"),
                           "--index", missing), cwd=ROOT, capture_output=True,
                       text=True)
    print(f"missing index exit={r.returncode} stderr={r.stderr.strip()}")
    damaged = os.path.join(args.workdir, "damaged")
    shutil.copytree(os.path.join(big, "idx"), damaged)
    victim = sorted(os.listdir(os.path.join(damaged, "atoms")))[0]
    with open(os.path.join(damaged, "atoms", victim), "r+b") as fh:
        fh.seek(100)
        b = fh.read(1)
        fh.seek(100)
        fh.write(bytes([b[0] ^ 0xFF]))
    r = subprocess.run(cli("run", "--decl", os.path.join(big, "query.json"),
                           "--index", damaged), cwd=ROOT, capture_output=True,
                       text=True)
    print(f"damaged index exit={r.returncode} stderr={r.stderr.strip()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
