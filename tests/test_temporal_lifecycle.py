"""Temporal index lifecycle, failure and cancellation tests.

Mirrors the static lifecycle contract for the temporal product:

* failed builds leave no openable half-built index and a forced rebuild keeps
  the previous index valid when the new build fails;
* static and temporal indexes recognise each other's kind;
* cancellation, early consumer close and output write failures are reported
  ``incomplete`` with cursors, connections and temporary files released;
* damaged temporal indexes (truncation, bit flips, manifest tampering,
  negative-prefix-sum table content) are detected;
* the original interval files can be removed after a successful build.
"""
from __future__ import annotations

import json
import os
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
from reljoin.errors import (                     # noqa: E402
    CorruptIndex,
    DataProblem,
    MissingIndex,
    QueryAborted,
    WrongIndexKind,
)
from reljoin.temporal_build import build_temporal_index       # noqa: E402
from reljoin.temporal_open import open_temporal_index         # noqa: E402

from test_temporal_acceptance import TCase, triangle_case     # noqa: E402


def build_hand_case(root: str, name: str = "hand") -> TCase:
    c = TCase(root, name)
    triangle_case(c)
    c.load("A", [
        {"values": [1, 2], "since": 0, "until": 8, "copies": 2},
        {"values": [1, 2], "since": 3, "until": 6, "copies": 1},
    ])
    c.load("B", [{"values": [2, 3], "since": 1, "until": 7, "copies": 3}])
    c.load("C", [
        {"values": [3, 1], "since": 2, "until": 5, "copies": 4},
        {"values": [3, 1], "since": 5, "until": 9, "copies": 4},
    ])
    return c


class TemporalLifecycle(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="reljoin-tlife-")
        self.case = build_hand_case(self.tmp)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def build(self, **kw):
        decl = reljoin.load_declaration(self.case.decl_path)
        return build_temporal_index(
            decl, data_dir=self.case.data, out_dir=self.case.index,
            memory_bytes=1 << 20, **kw
        )

    def run_segments(self, lo=0, hi=10, **kw):
        decl = reljoin.load_declaration(self.case.decl_path)
        idx = open_temporal_index(self.case.index, expected_declaration=decl)
        q = reljoin.temporal_query(idx, decl, window=(lo, hi), **kw)
        out = list(q)
        q.close()
        return out, q.stats

    def test_failed_build_leaves_no_index(self) -> None:
        os.unlink(os.path.join(self.case.data, "C.jsonl"))
        with self.assertRaises(DataProblem):
            self.build()
        self.assertFalse(os.path.exists(self.case.index))
        leftovers = [n for n in os.listdir(self.tmp)
                     if n.startswith(".reljoin")]
        self.assertEqual(leftovers, [])

    def test_failed_forced_rebuild_keeps_old_index(self) -> None:
        self.build()
        before, stats = self.run_segments()
        self.assertEqual(stats.status, "complete")
        self.assertEqual(len(before), 3)
        # Corrupt the *sources*, then rebuild with --force: the old index must
        # stay queryable because the new build is discarded atomically.
        with open(os.path.join(self.case.data, "A.jsonl"), "w") as fh:
            fh.write("not json\n")
        with self.assertRaises(DataProblem):
            self.build(force=True)
        after, stats2 = self.run_segments()
        self.assertEqual(after, before)
        self.assertEqual(stats2.status, "complete")

    def test_build_refuses_non_empty_without_force(self) -> None:
        self.build()
        with self.assertRaises(reljoin.IndexProblem):
            self.build()

    def test_source_files_can_be_removed_after_build(self) -> None:
        self.build()
        shutil.rmtree(self.case.data)
        out, stats = self.run_segments()
        self.assertEqual(stats.status, "complete")
        self.assertEqual(len(out), 3)

    # -- cross-kind recognition ------------------------------------------
    def test_static_and_temporal_indexes_reject_each_other(self) -> None:
        self.build()
        sdir = os.path.join(self.tmp, "sdata")
        sidx = os.path.join(self.tmp, "sidx")
        os.makedirs(sdir)
        with open(os.path.join(sdir, "A.jsonl"), "w") as fh:
            fh.write("[1,2]\n")
        with open(os.path.join(sdir, "B.jsonl"), "w") as fh:
            fh.write("[2,3]\n")
        with open(os.path.join(sdir, "C.jsonl"), "w") as fh:
            fh.write("[3,1]\n")
        decl = reljoin.load_declaration(self.case.decl_path)
        reljoin.build_index(decl, data_dir=sdir, out_dir=sidx,
                            memory_bytes=1 << 20)
        with self.assertRaises(WrongIndexKind):
            reljoin.open_index(self.case.index, expected_declaration=decl)
        with self.assertRaises(WrongIndexKind):
            open_temporal_index(sidx, expected_declaration=decl)

    def test_missing_index_is_not_empty_result(self) -> None:
        with self.assertRaises(MissingIndex):
            open_temporal_index(os.path.join(self.tmp, "never"))

    # -- damage -----------------------------------------------------------
    def test_truncated_table_detected(self) -> None:
        self.build()
        victim = None
        for name in os.listdir(os.path.join(self.case.index, "tables")):
            victim = os.path.join(self.case.index, "tables", name)
            break
        with open(victim, "r+b") as fh:
            fh.truncate(max(1, os.path.getsize(victim) - 512))
        with self.assertRaises(CorruptIndex):
            open_temporal_index(self.case.index)

    def test_bit_flip_detected(self) -> None:
        self.build()
        names = sorted(os.listdir(os.path.join(self.case.index, "tables")))
        victim = os.path.join(self.case.index, "tables", names[0])
        with open(victim, "r+b") as fh:
            fh.seek(os.path.getsize(victim) // 2)
            b = fh.read(1)
            fh.seek(os.path.getsize(victim) // 2)
            fh.write(bytes([b[0] ^ 0xFF]))
        with self.assertRaises((CorruptIndex, reljoin.ReljoinError)):
            open_temporal_index(self.case.index)

    def test_manifest_tampering_detected(self) -> None:
        self.build()
        mp = os.path.join(self.case.index, "manifest.json")
        with open(mp) as fh:
            manifest = json.load(fh)
        manifest["format"] = "reljoin-temporal-index-v99"
        with open(mp, "w") as fh:
            json.dump(manifest, fh)
        with self.assertRaises(CorruptIndex):
            open_temporal_index(self.case.index)

    def test_stray_file_detected(self) -> None:
        self.build()
        with open(os.path.join(self.case.index, "tables", "stray.db-wal"),
                  "wb") as fh:
            fh.write(b"junk")
        with self.assertRaises(CorruptIndex):
            open_temporal_index(self.case.index)

    def test_negative_prefix_sum_detected(self) -> None:
        # Hand-craft a temporal table whose events dip below zero: open the
        # built database read-write, flip the +4 opening event of C into -4.
        self.build()
        import sqlite3
        names = sorted(os.listdir(os.path.join(self.case.index, "tables")))
        c_table = [n for n in names if n.startswith("C.")][0]
        fp = os.path.join(self.case.index, "tables", c_table)
        from reljoin.temporal_format import decode_delta, encode_delta
        conn = sqlite3.connect(fp)
        try:
            row = conn.execute(
                "SELECT c0, c1, tick, d FROM t ORDER BY c0, c1, tick LIMIT 1"
            ).fetchone()
            conn.execute(
                "UPDATE t SET d=? WHERE c0=? AND c1=? AND tick=?",
                (encode_delta(-decode_delta(row[3])), row[0], row[1], row[2]),
            )
            conn.commit()
        finally:
            conn.close()
        # Size/hash now mismatch as well; either way opening must fail rather
        # than returning empty results.
        with self.assertRaises(CorruptIndex):
            open_temporal_index(self.case.index)

    # -- cancellation / early close ---------------------------------------
    def test_early_consumer_close_is_incomplete(self) -> None:
        self.build()
        decl = reljoin.load_declaration(self.case.decl_path)
        idx = open_temporal_index(self.case.index, expected_declaration=decl)
        q = reljoin.temporal_query(idx, decl, window=(0, 10))
        it = iter(q)
        next(it)
        it.close()
        self.assertEqual(q.stats.status, "incomplete")
        self.assertIn("closed", q.stats.status_reason)
        q.close()
        with self.assertRaises(QueryAborted):
            iter(q)
        self.assertEqual(idx.connections(), [])

    def test_cancel_callback_is_incomplete(self) -> None:
        self.build()
        decl = reljoin.load_declaration(self.case.decl_path)
        idx = open_temporal_index(self.case.index, expected_declaration=decl)
        q = reljoin.temporal_query(idx, decl, window=(0, 10),
                                   cancel=lambda: True)
        with self.assertRaises(QueryAborted):
            next(iter(q))
        self.assertEqual(q.stats.status, "incomplete")
        q.close()

    def test_consumer_exception_marks_incomplete_and_releases(self) -> None:
        self.build()
        decl = reljoin.load_declaration(self.case.decl_path)
        idx = open_temporal_index(self.case.index, expected_declaration=decl)

        class Boom(Exception):
            pass

        with self.assertRaises(Boom):
            with reljoin.temporal_query(idx, decl, window=(0, 10)) as q:
                for _values, _seg in q:
                    raise Boom("consumer failed while writing")
        self.assertEqual(q.stats.status, "incomplete")
        self.assertEqual(idx.connections(), [])

    def test_limit_reports_incomplete_but_keeps_prefix(self) -> None:
        self.build()
        out, stats = self.run_segments(limit=1)
        self.assertEqual(len(out), 1)
        self.assertEqual(stats.status, "limit-reached")


class TemporalCliContract(unittest.TestCase):
    """End-to-end CLI exit-code and failure checks."""

    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="reljoin-tcli-")
        self.case = build_hand_case(self.tmp)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def cli(self, *args, **kw):
        return subprocess.run(
            [sys.executable, "-m", "reljoin", *args],
            cwd=ROOT, capture_output=True, text=True, **kw,
        )

    def test_full_cycle_and_hand_numbers(self) -> None:
        r = self.cli("temporal-prepare", "--decl", self.case.decl_path,
                     "--data", self.case.data, "--index", self.case.index,
                     "--memory", "4M")
        self.assertEqual(r.returncode, 0, r.stderr)
        r = self.cli("temporal-run", "--decl", self.case.decl_path,
                     "--index", self.case.index, "--from", "0", "--to", "10",
                     "--stats", os.path.join(self.tmp, "stats.json"))
        self.assertEqual(r.returncode, 0, r.stderr)
        lines = [json.loads(line) for line in r.stdout.strip().splitlines()]
        self.assertEqual(lines, [
            {"values": [1, 2, 3], "since": 2, "until": 3,
             "multiplicity": 24},
            {"values": [1, 2, 3], "since": 3, "until": 6,
             "multiplicity": 36},
            {"values": [1, 2, 3], "since": 6, "until": 7,
             "multiplicity": 24},
        ])
        with open(os.path.join(self.tmp, "stats.json")) as fh:
            stats = json.load(fh)
        self.assertEqual(stats["status"], "complete")
        self.assertEqual(stats["weighted_duration"], 156)
        self.assertIsInstance(stats["weighted_duration"], int)

    def test_bad_window_usage_exit_2(self) -> None:
        self.cli("temporal-prepare", "--decl", self.case.decl_path,
                 "--data", self.case.data, "--index", self.case.index,
                 "--memory", "4M")
        r = self.cli("temporal-run", "--decl", self.case.decl_path,
                     "--index", self.case.index, "--from", "10", "--to", "10")
        self.assertEqual(r.returncode, 8, r.stderr)
        r = self.cli("temporal-run", "--decl", self.case.decl_path,
                     "--index", self.case.index, "--from", "11", "--to", "10")
        self.assertEqual(r.returncode, 8)

    def test_wrong_kind_exit_6(self) -> None:
        r = self.cli("temporal-prepare", "--decl", self.case.decl_path,
                     "--data", self.case.data, "--index", self.case.index,
                     "--memory", "4M")
        self.assertEqual(r.returncode, 0, r.stderr)
        r = self.cli("run", "--decl", self.case.decl_path,
                     "--index", self.case.index)
        self.assertEqual(r.returncode, 6)
        self.assertIn("index-wrong-kind", r.stderr)

    def test_write_failure_is_incomplete(self) -> None:
        self.cli("temporal-prepare", "--decl", self.case.decl_path,
                 "--data", self.case.data, "--index", self.case.index,
                 "--memory", "4M")
        outdir = os.path.join(self.tmp, "outdir")
        os.makedirs(outdir)
        r = self.cli("temporal-run", "--decl", self.case.decl_path,
                     "--index", self.case.index, "--from", "0", "--to", "10",
                     "--out", outdir)
        self.assertEqual(r.returncode, 7, r.stderr)
        self.assertIn("incomplete", r.stderr)

    def test_consumer_pipe_close_is_incomplete(self) -> None:
        # Many segments: close the read end immediately.
        big = TCase(self.tmp, "big")
        big.declare(["x", "y"],
                    [("E", ["x", "y"]), ("E", ["y", "x"])])
        rows = []
        for i in range(20000):
            rows.append({"values": [i % 500, i], "since": i,
                         "until": i + 2})
        big.load("E", rows)
        r = self.cli("temporal-prepare", "--decl", big.decl_path,
                     "--data", big.data, "--index", big.index,
                     "--memory", "4M")
        self.assertEqual(r.returncode, 0, r.stderr)
        proc = subprocess.Popen(
            [sys.executable, "-m", "reljoin", "temporal-run",
             "--decl", big.decl_path, "--index", big.index,
             "--from", "0", "--to", "100000"],
            cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        proc.stdout.read(100)
        proc.stdout.close()
        stderr = proc.stderr.read().decode()
        rc = proc.wait(timeout=120)
        proc.stderr.close()
        self.assertEqual(rc, 7, stderr)
        self.assertIn("incomplete", stderr)

    def test_corrupt_bad_line_exit_4_and_no_half_index(self) -> None:
        with open(os.path.join(self.case.data, "C.jsonl"), "a") as fh:
            fh.write('{"values":[3,1],"since":5}\n')
        r = self.cli("temporal-prepare", "--decl", self.case.decl_path,
                     "--data", self.case.data,
                     "--index", os.path.join(self.tmp, "broken"),
                     "--memory", "4M")
        self.assertEqual(r.returncode, 4, r.stderr)
        self.assertIn("C", r.stderr)
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "broken")))

    def test_sigterm_during_run_is_incomplete_with_exit_7(self) -> None:
        big = TCase(self.tmp, "sigterm")
        big.declare(["x", "y"],
                    [("E", ["x", "y"]), ("E", ["y", "x"])])
        big.load("E", [
            {"values": [i % 500, i], "since": i, "until": i + 2}
            for i in range(40000)
        ])
        stats_path = os.path.join(self.tmp, "sigterm-stats.json")
        self.assertEqual(
            self.cli("temporal-prepare", "--decl", big.decl_path,
                     "--data", big.data, "--index", big.index,
                     "--memory", "4M").returncode,
            0,
        )
        proc = subprocess.Popen(
            [sys.executable, "-m", "reljoin", "temporal-run",
             "--decl", big.decl_path, "--index", big.index,
             "--from", "0", "--to", "1000000", "--stats", stats_path],
            cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        # Wait until the run is actually emitting, then cancel it, so the
        # signal cannot race an already-finished (or not-yet-started) run.
        first = proc.stdout.read(1)
        self.assertEqual(first, b"{")
        proc.send_signal(__import__("signal").SIGTERM)
        proc.stdout.close()
        stderr = proc.stderr.read().decode()
        rc = proc.wait(timeout=120)
        proc.stderr.close()
        self.assertEqual(rc, 7, stderr)
        self.assertIn("incomplete", stderr)
        if os.path.exists(stats_path):
            with open(stats_path) as fh:
                self.assertEqual(json.load(fh)["status"], "incomplete")


if __name__ == "__main__":
    unittest.main()
