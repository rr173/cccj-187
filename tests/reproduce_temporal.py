"""README-level reproduction of the temporal acceptance observations.

``python3 tests/reproduce_temporal.py --workdir DIR`` generates every dataset
from an empty directory with the generator (nothing large is committed),
builds the temporal indexes from the raw interval files, runs the queries,
prints exact statistics including peak RSS, spill work, index/event access and
the integer weighted duration, and cross-checks the hand sample against the
independent reference oracle.

It shells out to the CLI on purpose, so it exercises exactly the documented
commands.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
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


def show(path, keys):
    with open(path) as fh:
        d = json.load(fh)
    print(json.dumps({k: d.get(k) for k in keys}, indent=2))


STATS_KEYS = (
    "status", "window_from", "window_to", "complete_bindings",
    "emitted_segments", "index_lookups", "point_probes",
    "candidate_values", "event_rows_read", "event_batches",
    "sweep_boundaries", "raw_segments_before_merge", "segments_merged",
    "weighted_duration", "max_live_atoms", "peak_rss_kib",
    "elapsed_seconds",
)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--frag-n", type=int, default=20000)
    ap.add_argument("--many-n", type=int, default=5000)
    ap.add_argument("--memory", default="2M")
    args = ap.parse_args(argv)

    if os.path.exists(args.workdir) and os.listdir(args.workdir):
        raise SystemExit(f"workdir {args.workdir} is not empty")
    os.makedirs(args.workdir, exist_ok=True)

    # 1. generate all temporal data on site (scripts only; nothing committed)
    run([sys.executable, os.path.join(HERE, "gen_temporal_data.py"),
         args.workdir, "--scenario", "all",
         "--frag-n", str(args.frag_n), "--many-n", str(args.many_n)])

    # 2. hand sample: exact segments, weighted duration and the reference
    hand = os.path.join(args.workdir, "hand")
    run(cli("temporal-prepare", "--decl", os.path.join(hand, "query.json"),
            "--data", os.path.join(hand, "data"),
            "--index", os.path.join(hand, "idx"), "--memory", args.memory))
    run(cli("temporal-run", "--decl", os.path.join(hand, "query.json"),
            "--index", os.path.join(hand, "idx"), "--from", "0", "--to", "10",
            "--stats", os.path.join(hand, "stats.json")))
    show(os.path.join(hand, "stats.json"), STATS_KEYS)
    run(cli("temporal-run", "--decl", os.path.join(hand, "query.json"),
            "--index", os.path.join(hand, "idx"), "--from", "4", "--to", "8"))
    run(cli("temporal-explain", "--decl", os.path.join(hand, "query.json"),
            "--index", os.path.join(hand, "idx"), "--from", "0", "--to", "10"))
    # Reference oracle (raw lines only; no production temporal modules).
    oracle_cmd = (
        "import sys;sys.path.insert(0,%r);sys.path.insert(0,%r);"
        "import reljoin, temporal_oracle;"
        "d=reljoin.load_declaration(%r);"
        "segs=temporal_oracle.temporal_brute_force(d,%r,0,10);"
        "print(segs); print('weighted', temporal_oracle.weighted_duration(segs))"
        % (ROOT, HERE, os.path.join(hand, "query.json"),
           os.path.join(hand, "data"))
    )
    run([sys.executable, "-c", oracle_cmd])

    # 3. fragmented single-tuple trajectory: bounded RSS, spill accounting
    frag = os.path.join(args.workdir, "frag")
    with open(os.path.join(frag, "params.json")) as fh:
        params = json.load(fh)
    max_tick = params["max_tick"]
    run(cli("temporal-prepare", "--decl", os.path.join(frag, "query.json"),
            "--data", os.path.join(frag, "data"),
            "--index", os.path.join(frag, "idx"), "--memory", args.memory,
            "--json"))
    run(cli("temporal-run", "--decl", os.path.join(frag, "query.json"),
            "--index", os.path.join(frag, "idx"), "--from", "0",
            "--to", str(max_tick), "--memory", args.memory,
            "--out", os.path.join(frag, "out.jsonl"),
            "--stats", os.path.join(frag, "stats.json")))
    show(os.path.join(frag, "stats.json"), STATS_KEYS)
    with open(os.path.join(frag, "out.jsonl")) as fh:
        print("frag output lines:", sum(1 for _ in fh))

    # 4. many short-lived closed loops
    many = os.path.join(args.workdir, "manyloops")
    with open(os.path.join(many, "params.json")) as fh:
        params = json.load(fh)
    hi = params["max_tick"]
    run(cli("temporal-prepare", "--decl", os.path.join(many, "query.json"),
            "--data", os.path.join(many, "data"),
            "--index", os.path.join(many, "idx"), "--memory", args.memory))
    run(cli("temporal-run", "--decl", os.path.join(many, "query.json"),
            "--index", os.path.join(many, "idx"), "--from", "0",
            "--to", str(hi), "--memory", args.memory,
            "--out", os.path.join(many, "out.jsonl"),
            "--stats", os.path.join(many, "stats.json")))
    show(os.path.join(many, "stats.json"), STATS_KEYS)

    # 5. same endpoints, span 10 vs span 10**15: work must not grow
    small = os.path.join(args.workdir, "span-1")
    huge = os.path.join(args.workdir, "span-100000000000000")
    for d, hi in ((small, 10), (huge, 10 ** 15)):
        run(cli("temporal-prepare", "--decl", os.path.join(d, "query.json"),
                "--data", os.path.join(d, "data"),
                "--index", os.path.join(d, "idx"), "--memory", args.memory))
        run(cli("temporal-run", "--decl", os.path.join(d, "query.json"),
                "--index", os.path.join(d, "idx"), "--from", "0",
                "--to", str(hi), "--stats", os.path.join(d, "stats.json")))
    with open(os.path.join(small, "stats.json")) as fh:
        s1 = json.load(fh)
    with open(os.path.join(huge, "stats.json")) as fh:
        s2 = json.load(fh)
    print(json.dumps({
        "span_10_event_rows": s1["event_rows_read"],
        "span_1e15_event_rows": s2["event_rows_read"],
        "span_10_boundaries": s1["sweep_boundaries"],
        "span_1e15_boundaries": s2["sweep_boundaries"],
        "span_10_weighted": s1["weighted_duration"],
        "span_1e15_weighted": s2["weighted_duration"],
    }, indent=2))

    # 6. failure acceptance: damaged index and refused output
    damaged = os.path.join(args.workdir, "damaged-temporal")
    import shutil
    shutil.copytree(os.path.join(hand, "idx"), damaged)
    victim = sorted(os.listdir(os.path.join(damaged, "tables")))[0]
    with open(os.path.join(damaged, "tables", victim), "r+b") as fh:
        fh.seek(100)
        b = fh.read(1)
        fh.seek(100)
        fh.write(bytes([b[0] ^ 0xFF]))
    r = subprocess.run(cli("temporal-run",
                           "--decl", os.path.join(hand, "query.json"),
                           "--index", damaged, "--from", "0", "--to", "10"),
                       cwd=ROOT, capture_output=True, text=True)
    print(f"damaged temporal index exit={r.returncode} "
          f"stderr={r.stderr.strip()}")
    outdir = os.path.join(args.workdir, "cant-write")
    os.makedirs(outdir)
    r = subprocess.run(cli("temporal-run",
                           "--decl", os.path.join(hand, "query.json"),
                           "--index", os.path.join(hand, "idx"),
                           "--from", "0", "--to", "10", "--out", outdir),
                       cwd=ROOT, capture_output=True, text=True)
    print(f"refused output exit={r.returncode} stderr={r.stderr.strip()}")
    print("process peak RSS (this driver):",
          resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, "KiB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
