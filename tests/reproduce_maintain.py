"""README-level reproduction of the incremental maintenance observations.

``python3 tests/reproduce_maintain.py --workdir DIR`` generates every dataset
from an empty directory (nothing large is committed), builds the temporal
index and the maintenance library, applies correction batches through the
documented CLI, and cross-checks the maintained view and the signed delta
against the independent brute-force oracle in ``maintain_oracle.py``.

It shells out to the CLI on purpose, so it exercises exactly the documented
commands and reports the real run counters (affected seeds, binding sweeps,
relation/event access, view writes, peak RSS, spill work and weighted change).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import reljoin                                    # noqa: E402
from reljoin.temporal_build import build_temporal_index       # noqa: E402

import gen_maintain_data                          # noqa: E402
import maintain_oracle                            # noqa: E402


def cli(*args):
    return [sys.executable, "-m", "reljoin", *args]


def run(cmd, **kw):
    print("$ " + " ".join(cmd))
    env = dict(os.environ, PYTHONPATH=ROOT)
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                       env=env, **kw)
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr, file=sys.stderr)
        raise SystemExit(f"command failed ({r.returncode})")
    if r.stdout:
        sys.stdout.write(r.stdout)
    if r.stderr.strip():
        sys.stderr.write(r.stderr)
    return r


def write_decl_and_build(root, memory="4M"):
    decl = reljoin.load_declaration(os.path.join(root, "query.json"))
    build_temporal_index(
        decl, data_dir=os.path.join(root, "data"),
        out_dir=os.path.join(root, "tidx"),
        memory_bytes=4 << 20,
    )
    tidx = reljoin.open_temporal_index(
        os.path.join(root, "tidx"), expected_declaration=decl
    )
    reljoin.init_maintenance(
        decl, tidx, out_dir=os.path.join(root, "mlib"),
        memory_bytes=4 << 20,
    )
    tidx.close()
    return decl


def load_corr(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--many-n", type=int, default=40)
    args = ap.parse_args(argv)
    if os.path.exists(args.workdir) and os.listdir(args.workdir):
        raise SystemExit(f"workdir {args.workdir} is not empty")
    os.makedirs(args.workdir, exist_ok=True)

    # -- 1. hand triangle: exact delta +12/+12, weighted change 48 --------
    tri = os.path.join(args.workdir, "triangle")
    os.makedirs(os.path.join(tri, "data"))
    with open(os.path.join(tri, "query.json"), "w") as fh:
        json.dump({
            "version": 1, "order": ["x", "y", "z"],
            "relations": [
                {"name": "A", "vars": ["x", "y"]},
                {"name": "B", "vars": ["y", "z"]},
                {"name": "C", "vars": ["z", "x"]}],
        }, fh)
    for rel, vals, copies in (
        ("A", [1, 2], 2), ("B", [2, 3], 3), ("C", [3, 1], 4),
    ):
        with open(os.path.join(tri, "data", rel + ".jsonl"), "w") as fh:
            fh.write(json.dumps({
                "values": vals, "since": 0, "until": 10, "copies": copies,
            }) + "\n")
    write_decl_and_build(tri)
    corr = os.path.join(tri, "corr.jsonl")
    with open(corr, "w") as fh:
        fh.write(json.dumps({"relation": "A", "values": [1, 2],
                             "since": 2, "until": 8, "delta": 1}) + "\n")
        fh.write(json.dumps({"relation": "B", "values": [2, 3],
                             "since": 4, "until": 6, "delta": -1}) + "\n")
    stats = os.path.join(tri, "stats.json")
    run(cli("maintain-apply", "--decl", os.path.join(tri, "query.json"),
            "--maintain", os.path.join(tri, "mlib"),
            "--batch-id", "b1", "--expect-revision", "0",
            "--corrections", corr, "--memory", "4M",
            "--stats", stats))
    d = json.load(open(stats))
    receipt = d["receipt"]
    print("triangle weighted_change:", receipt["weighted_change"],
          "(expected 48); delta rows:", receipt["delta_rows"],
          "(expected 2)")
    assert receipt["weighted_change"] == 48
    assert receipt["delta_rows"] == 2

    # Oracle cross-check of the triangle view and delta.
    decl = reljoin.load_declaration(os.path.join(tri, "query.json"))
    bags = maintain_oracle.load_initial_bags(decl, os.path.join(tri, "data"))
    corrections = load_corr(corr)
    old_view = maintain_oracle.full_view(decl, bags, 0, 10)
    maintain_oracle.apply_corrections_to_bags(bags, corrections)
    new_view = maintain_oracle.full_view(decl, bags, 0, 10)
    want_delta = maintain_oracle.delta_between(old_view, new_view)
    store = reljoin.open_maintain_store(
        os.path.join(tri, "mlib"), expected_declaration=decl, verify=False
    )
    got = sorted(
        (tuple(v), s, u, m)
        for v, (s, u, m) in reljoin.MaintainQuery(store, decl, window=(0, 10))
    )
    assert got == sorted(new_view), "view disagrees with oracle"
    got_delta = sorted(
        (tuple(r["values"]), r["since"], r["until"],
         r["delta_multiplicity"])
        for r in reljoin.stream_batch_delta(store, "b1")
    )
    assert got_delta == sorted(want_delta), "delta disagrees with oracle"
    print("oracle cross-check: view and delta match exactly")
    store.close()

    # -- 2. many independent loops: one correction touches one loop -------
    many = os.path.join(args.workdir, "many")
    info = gen_maintain_data.gen_manyloops(many, n=args.many_n)
    write_decl_and_build(many)
    stats2 = os.path.join(many, "stats.json")
    run(cli("maintain-apply", "--decl", os.path.join(many, "query.json"),
            "--maintain", os.path.join(many, "mlib"),
            "--batch-id", "one", "--expect-revision", "0",
            "--corrections", info["correction"], "--memory", "4M",
            "--stats", stats2))
    d2 = json.load(open(stats2))
    d2 = d2["counters"]
    keys = ("seed_tuples", "bindings_discovered", "binding_sweeps",
            "validation_rows_scanned", "view_rows_deleted",
            "view_rows_inserted", "peak_rss_kib")
    print("incremental access over", args.many_n, "stored loops:")
    print(json.dumps({k: d2[k] for k in keys}, indent=2))
    assert d2["seed_tuples"] == 1
    assert d2["bindings_discovered"] == 1
    assert d2["binding_sweeps"] == 1

    # -- 3. fragmented batch that spills under a tight memory budget -------
    frag = os.path.join(args.workdir, "frag")
    finfo = gen_maintain_data.gen_fragmented(frag, intervals=2000)
    write_decl_and_build(frag)
    stats3 = os.path.join(frag, "stats.json")
    run(cli("maintain-apply", "--decl", os.path.join(frag, "query.json"),
            "--maintain", os.path.join(frag, "mlib"),
            "--batch-id", "frag", "--expect-revision", "0",
            "--corrections", finfo["correction"], "--memory", "2M",
            "--stats", stats3))
    d3 = json.load(open(stats3))["counters"]
    print("fragmented batch raw events:", d3["raw_events_inserted"],
          "legality checks:", d3["negative_checks"],
          "peak RSS KiB:", d3["peak_rss_kib"])
    # The deep verification reopens the library and proves the spill run left
    # every maintained multiplicity non-negative.
    store3 = reljoin.open_maintain_store(
        os.path.join(frag, "mlib"),
        expected_declaration=reljoin.load_declaration(
            os.path.join(frag, "query.json")),
        deep=True,
    )
    assert store3.revision() == 1
    store3.close()
    print("ALL MAINTENANCE REPRODUCTION CHECKS PASSED")


if __name__ == "__main__":
    main()
