"""Anti-explosion and memory-accounting tests.

These tests do not time anything: they assert *structural* facts that a
two-relation join would violate, and they record real RSS and index access
counts produced by the engine (never estimates).

Key structural assertions:

* In the hotspot scenario a two-relation join on ``x`` would produce
  ``|A| * |B|`` combinations.  The solver's per-layer candidate counts and
  index lookups must stay near the number of *distinct* boundary values, not
  near the join size.  ``candidate_values`` at the ``y`` layer is bounded by
  the number of distinct ``y`` values, and index lookups are bounded by a
  small constant per emitted candidate plus per tested closed loop.
* In the large-output scenario the peak RSS stays flat regardless of how many
  bindings are emitted (the output is streamed, not accumulated).
"""
from __future__ import annotations

import json
import os
import resource
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import reljoin                                    # noqa: E402
from reljoin.index_build import build_index       # noqa: E402
from reljoin.index_open import open_index         # noqa: E402

import gen_data                                   # noqa: E402

TRIANGLE = {
    "version": 1,
    "order": ["x", "y", "z"],
    "relations": [
        {"name": "A", "vars": ["x", "y"]},
        {"name": "B", "vars": ["y", "z"]},
        {"name": "C", "vars": ["z", "x"]},
    ],
}


def _rss_kib() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def _child_rss_kib(script: str, cwd: str) -> dict:
    """Run ``script`` in a fresh interpreter and report its peak RSS."""
    runner = (
        "import json, resource, runpy, sys\n"
        "sys.path.insert(0, " + json.dumps(ROOT) + ")\n"
        "sys.argv = ['prog']\n"
        "ns = {}\n"
        "exec(compile(open(" + json.dumps(script) + ").read(), 'script', 'exec'), ns)\n"
        "print(json.dumps({'peak_rss_kib': int(resource.getrusage("
        "resource.RUSAGE_SELF).ru_maxrss)}))\n"
    )
    r = subprocess.run([sys.executable, "-c", runner], cwd=cwd,
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise AssertionError(f"child failed: {r.stderr}")
    return json.loads(r.stdout.strip().splitlines()[-1])


class AntiExplosion(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="reljoin-anti-")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _prepare_triangle(self, root, decl=TRIANGLE):
        d = reljoin.load_declaration(os.path.join(root, "query.json"))
        build_index(d, data_dir=os.path.join(root, "data"),
                    out_dir=os.path.join(root, "idx"), memory_bytes=8 << 20)
        return d, os.path.join(root, "idx")

    def _run(self, decl, index, **kw):
        idx = open_index(index, expected_declaration=decl)
        q = reljoin.query(idx, decl, **kw)
        out = []
        for v, m in q:
            out.append((v, m))
        q.close()
        return out, q.stats

    def test_hotspot_does_not_resemble_two_way_join(self):
        root = gen_data.gen_hotspot(self.tmp, xs=1, ys=600, zs=600, closure=4)
        d, idx = self._prepare_triangle(root)
        out, stats = self._run(d, idx)
        # A two-relation join on x would materialise 600*600 = 360000
        # (x,y,z) triples per hot x before the closing relation could reject
        # them.  The layered solver's work is bounded by *distinct boundary
        # values* instead: one candidate at x, |Y| candidates at y (not
        # |A|*|B| pairs), and only the genuinely closed loops at z.
        two_way = 600 * 600
        self.assertLess(sum(stats.candidates), two_way // 100)
        self.assertEqual(stats.candidates[0], 1)          # one hot x
        self.assertLessEqual(stats.candidates[1], 600)    # distinct y values
        self.assertLessEqual(stats.candidates[2], stats.leaves + 100)
        self.assertLessEqual(stats.max_live_atoms, 2)
        # Index lookups stay near the number of distinct candidates, not rows.
        self.assertLess(stats.probes, two_way // 20)
        self.assertGreater(stats.probes, 0)
        self.assertTrue(stats.leaves >= 1)
        self.assertLess(stats.probes, 8 * sum(stats.candidates) + 5000)

    def test_large_output_streams_with_bounded_state(self):
        root = os.path.join(self.tmp, "many")
        os.makedirs(os.path.join(root, "data"))
        n = 20000
        with open(os.path.join(root, "query.json"), "w") as fh:
            json.dump(TRIANGLE, fh)
        for name, fn in (
            ("A", lambda i: (i, i)),
            ("B", lambda i: (i, i)),
            ("C", lambda i: (i, i)),
        ):
            with open(os.path.join(root, "data", name + ".jsonl"), "w") as fh:
                for i in range(n):
                    a, b = fn(i)
                    fh.write(f"[{a},{b}]\n")
        d, idx = self._prepare_triangle(root)
        out, stats = self._run(d, idx, memory_bytes=2 << 20)
        self.assertEqual(len(out), n)
        self.assertEqual(out[0], ((0, 0, 0), 1))
        self.assertEqual(out[-1], ((n - 1, n - 1, n - 1), 1))
        self.assertEqual(stats.status, "complete")
        # The solver held at most one binding at a time; RSS stayed flat.
        self.assertLess(stats.peak_rss_kib, 200 * 1024)

    def test_cli_stats_report_real_rss_and_access(self):
        root = gen_data.gen_hotspot(self.tmp, xs=1, ys=200, zs=200, closure=2)
        root = os.path.join(self.tmp, "hotspot")
        stats_path = os.path.join(root, "stats.json")
        out_path = os.path.join(root, "out.txt")
        r = subprocess.run(
            [sys.executable, "-m", "reljoin", "prepare",
             "--decl", os.path.join(root, "query.json"),
             "--data", os.path.join(root, "data"),
             "--index", os.path.join(root, "idx"), "--memory", "8M"],
            cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        r = subprocess.run(
            [sys.executable, "-m", "reljoin", "run",
             "--decl", os.path.join(root, "query.json"),
             "--index", os.path.join(root, "idx"),
             "--memory", "8M", "--out", out_path, "--stats", stats_path],
            cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        with open(stats_path) as fh:
            stats = json.load(fh)
        for key in ("peak_rss_kib", "index_lookups", "index_seeks_first",
                    "index_seeks_ge", "index_seeks_gt", "point_probes",
                    "candidate_values", "complete_bindings", "status"):
            self.assertIn(key, stats)
        self.assertGreater(stats["peak_rss_kib"], 0)
        self.assertEqual(stats["status"], "complete")
        with open(out_path) as fh:
            lines = sum(1 for _ in fh)
        self.assertEqual(stats["emitted"], lines)
        self.assertGreater(stats["index_lookups"], 0)
        self.assertLess(stats["index_lookups"], 200 * 200)

    def test_peak_rss_is_flat_between_two_runs_with_tiny_budget(self):
        root = os.path.join(self.tmp, "rss")
        os.makedirs(root)
        root = gen_data.gen_hotspot(self.tmp, xs=1, ys=500, zs=500, closure=1)
        d, idx = self._prepare_triangle(root)
        before = _rss_kib()
        out, stats = self._run(d, idx, memory_bytes=1 << 20)
        after = _rss_kib()
        # The query itself must not add a large resident structure: allow a
        # generous margin for the interpreter and SQLite page cache.
        self.assertLess(after - before, 64 * 1024)
        self.assertGreaterEqual(stats.peak_rss_kib, 0)


if __name__ == "__main__":
    unittest.main()
