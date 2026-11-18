"""Process-level lifecycle acceptance for incremental maintenance.

These tests deliberately use real OS processes (subprocess + signals) rather
than mocks, because the brief asks for a reader pinned to one revision while
another process commits, at-most-one-writer semantics, crash-atomicity with a
real process exit status, and truthful export-failure handling.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from typing import List

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import reljoin                                    # noqa: E402
from reljoin.temporal_build import build_temporal_index       # noqa: E402

from test_maintain_acceptance import (            # noqa: E402
    MCase,
    triangle_case,
    build,
    view_rows,
)


class _Harness:
    def __init__(self, root: str) -> None:
        self.root = root
        self.case = triangle_case(root, "life")
        self.decl = build(self.case)

    def apply(self, batch_id: str, expect: int, corr: str,
              *extra: str, out: str = "-") -> subprocess.CompletedProcess:
        cmd = [
            sys.executable, "-m", "reljoin", "maintain-apply",
            "--decl", self.case.decl_path,
            "--maintain", self.case.maintain,
            "--batch-id", batch_id,
            "--expect-revision", str(expect),
            "--corrections", corr,
            "--memory", "4M",
            "--out", out,
            *extra,
        ]
        env = dict(os.environ, PYTHONPATH=ROOT)
        return subprocess.run(cmd, capture_output=True, text=True, env=env)

    def write_corr(self, name: str, rows) -> str:
        path = os.path.join(self.case.dir, name + ".jsonl")
        with open(path, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        return path


class MaintenanceLifecycle(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="reljoin-mlife-")
        self.h = _Harness(self.tmp)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_old_reader_pins_revision_across_commit(self) -> None:
        h = self.h
        store = reljoin.open_maintain_store(
            h.case.maintain, h.decl, verify=False
        )
        q = reljoin.MaintainQuery(store, h.decl, window=(0, 10))
        gen = iter(q)
        first = next(gen)
        self.assertEqual(first[1][2], 24)
        pinned = q.revision
        corr = h.write_corr("c1", [
            {"relation": "A", "values": [1, 2], "since": 2, "until": 8,
             "delta": 1},
        ])
        r = h.apply("w1", pinned, corr)
        self.assertEqual(r.returncode, 0, r.stderr)
        # Drain the old reader; it must remain on the pinned revision.
        rest = list(gen)
        q.close()
        self.assertEqual(q.revision, pinned)
        self.assertEqual(len(rest), 0)  # one segment only at revision 0
        # A new reader sees the new revision.
        store2 = reljoin.open_maintain_store(
            h.case.maintain, h.decl, verify=False
        )
        q2 = reljoin.MaintainQuery(store2, h.decl, window=(0, 10))
        list(q2)
        self.assertEqual(q2.revision, pinned + 1)
        q2.close()
        store2.close()
        store.close()

    def test_two_same_expect_writers_at_most_one_succeeds(self) -> None:
        h = self.h
        c1 = h.write_corr("w1", [
            {"relation": "A", "values": [1, 2], "since": 0, "until": 1,
             "delta": 1},
        ])
        c2 = h.write_corr("w2", [
            {"relation": "B", "values": [2, 3], "since": 0, "until": 1,
             "delta": 1},
        ])
        procs: List[subprocess.Popen] = []
        env = dict(os.environ, PYTHONPATH=ROOT)
        for bid, corr in (("p1", c1), ("p2", c2)):
            procs.append(subprocess.Popen(
                [sys.executable, "-m", "reljoin", "maintain-apply",
                 "--decl", h.case.decl_path,
                 "--maintain", h.case.maintain,
                 "--batch-id", bid, "--expect-revision", "0",
                 "--corrections", corr, "--memory", "4M"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env,
            ))
        codes = [p.wait() for p in procs]
        for p in procs:
            p.stdout.close()
            p.stderr.close()
        self.assertEqual(sorted(codes), [0, 10])
        store = reljoin.open_maintain_store(
            h.case.maintain, h.decl, verify=False
        )
        self.assertEqual(store.revision(), 1)
        self.assertEqual(len(reljoin.list_batches(store)), 1)
        store.close()

    def test_sigkill_before_commit_leaves_prior_revision(self) -> None:
        h = self.h
        corr = h.write_corr("kill", [
            {"relation": "A", "values": [1, 2], "since": 0, "until": 1,
             "delta": 1},
        ])
        env = dict(os.environ, PYTHONPATH=ROOT)
        proc = subprocess.run(
            [sys.executable, "-m", "reljoin", "maintain-apply",
             "--decl", h.case.decl_path,
             "--maintain", h.case.maintain,
             "--batch-id", "kill", "--expect-revision", "0",
             "--corrections", corr, "--memory", "4M",
             "--kill-hook", "before-commit"],
            capture_output=True, text=True, env=env,
        )
        # SIGKILL gives a real, locatable non-zero process exit status.
        self.assertEqual(proc.returncode, -9)
        store = reljoin.open_maintain_store(h.case.maintain, h.decl)
        self.assertEqual(store.revision(), 0)
        self.assertEqual(reljoin.list_batches(store), [])
        # No scratch or temp database files survive the crash/reopen.
        leftovers = [
            n for n in os.listdir(h.case.maintain)
            if n.startswith(".reljoin-tmp-") or "scratch" in n
        ]
        self.assertEqual(leftovers, [])
        view, rev = view_rows(store, h.decl, 0, 10)
        self.assertEqual(rev, 0)
        self.assertEqual(view, [((1, 2, 3), 0, 10, 24)])
        store.close()

    def test_export_write_failure_keeps_committed_receipt(self) -> None:
        h = self.h
        corr = h.write_corr("exp", [
            {"relation": "A", "values": [1, 2], "since": 2, "until": 8,
             "delta": 1},
        ])
        # An output path inside a non-existent directory cannot be created.
        bad_out = os.path.join(self.tmp, "no-such-dir", "delta.jsonl")
        r = h.apply("exp", 0, corr, out=bad_out)
        # The commit happened; only the export failed (exit 7 incomplete).
        self.assertEqual(r.returncode, 7, r.stderr)
        self.assertIn("committed at revision 1", r.stderr)
        self.assertIn("maintain-export", r.stderr)
        store = reljoin.open_maintain_store(h.case.maintain, h.decl)
        self.assertEqual(store.revision(), 1)
        # Re-fetch the stored delta without re-applying.
        rows = list(reljoin.stream_batch_delta(store, "exp"))
        self.assertTrue(rows)
        self.assertTrue(all("delta_multiplicity" in x for x in rows))
        receipt = reljoin.fetch_receipt(store, "exp")
        self.assertEqual(receipt["revision"], 1)
        # Re-sending the identical batch does not add another revision.
        r2 = h.apply("exp", 0, corr)
        self.assertEqual(r2.returncode, 0)
        self.assertEqual(store.revision(), 1)
        store.close()

    def test_reopen_does_not_need_temp_or_source(self) -> None:
        h = self.h
        corr = h.write_corr("persist", [
            {"relation": "A", "values": [1, 2], "since": 2, "until": 8,
             "delta": 1},
        ])
        self.assertEqual(h.apply("persist", 0, corr).returncode, 0)
        # Remove the temporal index and the original interval files; the
        # maintenance library must reopen and answer on its own.
        shutil.rmtree(h.case.tindex)
        shutil.rmtree(h.case.data)
        store = reljoin.open_maintain_store(h.case.maintain, h.decl)
        view, rev = view_rows(store, h.decl, 0, 10)
        self.assertEqual(rev, 1)
        self.assertTrue(any(m == 36 for _v, _s, _u, m in view))
        store.close()


if __name__ == "__main__":
    unittest.main()
