"""Index lifecycle tests: atomicity, missing/damaged index detection,
cancellation, early consumer close and write failures.

The theme: a damaged or un-finished index must never be reported as
"no data", and an interrupted run must never be reported as complete.
"""
from __future__ import annotations

import json
import os
import shutil
import signal
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

import reljoin                                        # noqa: E402
from reljoin.errors import (                          # noqa: E402
    CorruptIndex,
    DataProblem,
    DeclarationProblem,
    IndexProblem,
    MissingIndex,
    QueryAborted,
)

from util import write_jsonl                          # noqa: E402


def make_triangle(root: str) -> dict:
    os.makedirs(os.path.join(root, "data"), exist_ok=True)
    decl = {
        "version": 1,
        "order": ["x", "y", "z"],
        "relations": [
            {"name": "A", "vars": ["x", "y"]},
            {"name": "B", "vars": ["y", "z"]},
            {"name": "C", "vars": ["z", "x"]},
        ],
    }
    with open(os.path.join(root, "query.json"), "w") as fh:
        json.dump(decl, fh)
    write_jsonl(os.path.join(root, "data", "A.jsonl"), [(1, 2), (1, 2), (9, 9)])
    write_jsonl(os.path.join(root, "data", "B.jsonl"), [(2, 3)] * 3)
    write_jsonl(os.path.join(root, "data", "C.jsonl"), [(3, 1)] * 4)
    return decl


class IndexLifecycle(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="reljoin-life-")
        make_triangle(self.tmp)
        self.decl_path = os.path.join(self.tmp, "query.json")
        self.data = os.path.join(self.tmp, "data")
        self.index = os.path.join(self.tmp, "idx")
        self.decl = reljoin.load_declaration(self.decl_path)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def build(self, **kw):
        return reljoin.build_index(self.decl, data_dir=self.data,
                                   out_dir=self.index, memory_bytes=1 << 20,
                                   **kw)

    def run_query(self, **kw):
        q = reljoin.query(reljoin.open_index(self.index,
                                             expected_declaration=self.decl),
                          self.decl, **kw)
        out = list(q)
        q.close()
        return out, q.stats

    # -- missing ---------------------------------------------------------
    def test_missing_index_is_not_empty_result(self):
        with self.assertRaises(MissingIndex):
            reljoin.open_index(self.index)

    def test_directory_without_manifest_is_missing(self):
        os.makedirs(self.index)
        with self.assertRaises(MissingIndex):
            reljoin.open_index(self.index)

    # -- damaged ---------------------------------------------------------
    def test_truncated_table_is_detected(self):
        self.build()
        path = None
        for name in os.listdir(os.path.join(self.index, "atoms")):
            path = os.path.join(self.index, "atoms", name)
            break
        with open(path, "r+b") as fh:
            fh.truncate(max(1, os.path.getsize(path) - 512))
        with self.assertRaises(CorruptIndex):
            reljoin.open_index(self.index, expected_declaration=self.decl)

    def test_flipped_byte_is_detected(self):
        self.build()
        path = None
        for name in sorted(os.listdir(os.path.join(self.index, "atoms"))):
            path = os.path.join(self.index, "atoms", name)
            break
        with open(path, "r+b") as fh:
            fh.seek(os.path.getsize(path) // 2)
            b = fh.read(1)
            fh.seek(os.path.getsize(path) // 2)
            fh.write(bytes([b[0] ^ 0xFF]))
        with self.assertRaises(CorruptIndex):
            reljoin.open_index(self.index)

    def test_corrupt_manifest_is_detected(self):
        self.build()
        with open(os.path.join(self.index, "manifest.json"), "w") as fh:
            fh.write("{not json")
        with self.assertRaises(CorruptIndex):
            reljoin.open_index(self.index)

    def test_stray_wal_is_detected(self):
        self.build()
        atoms = os.path.join(self.index, "atoms")
        name = sorted(os.listdir(atoms))[0]
        with open(os.path.join(atoms, name + "-wal"), "wb") as fh:
            fh.write(b"junk")
        with self.assertRaises(CorruptIndex):
            reljoin.open_index(self.index)

    def test_declaration_mismatch_is_reported(self):
        self.build()
        other = os.path.join(self.tmp, "other.json")
        with open(other, "w") as fh:
            json.dump({"version": 1, "order": ["y", "x", "z"],
                       "relations": [
                           {"name": "A", "vars": ["x", "y"]},
                           {"name": "B", "vars": ["y", "z"]},
                           {"name": "C", "vars": ["z", "x"]}]}, fh)
        with self.assertRaises(IndexProblem) as ctx:
            reljoin.open_index(self.index,
                               expected_declaration=reljoin.load_declaration(other))
        self.assertNotIsInstance(ctx.exception, MissingIndex)

    # -- atomicity -------------------------------------------------------
    def test_failed_build_leaves_no_index(self):
        # A missing relation file makes the build fail after some tables
        # were already written into staging.
        os.unlink(os.path.join(self.data, "C.jsonl"))
        with self.assertRaises(DataProblem):
            self.build()
        self.assertFalse(os.path.exists(self.index))
        leftovers = [n for n in os.listdir(self.tmp) if n.startswith(".reljoin")]
        self.assertEqual(leftovers, [])

    def test_failed_forced_rebuild_keeps_old_index_intact(self):
        self.build()
        before, _ = self.run_query()
        self.assertEqual(before, [((1, 2, 3), 24)])
        write_jsonl(os.path.join(self.data, "A.jsonl"), [(1, 2, 3)])
        with self.assertRaises(DataProblem):
            self.build(force=True)
        after, _ = self.run_query()
        self.assertEqual(after, [((1, 2, 3), 24)])

    def test_build_refuses_to_clobber_without_force(self):
        self.build()
        with self.assertRaises(IndexProblem):
            self.build()
        self.assertEqual(self.run_query()[0], [((1, 2, 3), 24)])

    # -- cancellation ----------------------------------------------------
    def test_early_consumer_close_is_incomplete(self):
        self.build()
        idx = reljoin.open_index(self.index, expected_declaration=self.decl)
        q = reljoin.query(idx, self.decl)
        it = iter(q)
        next(it)
        it.close()
        self.assertEqual(q.stats.status, "incomplete")
        self.assertIn("closed", q.stats.status_reason)
        q.close()
        # A closed query refuses to be reused instead of silently restarting.
        with self.assertRaises(QueryAborted):
            iter(q)

    def test_break_out_of_for_loop_requires_close(self):
        # ``break`` does not close the iterator, so the run is still open; the
        # next resume must not pretend it finished, and ``close()`` records
        # the truthful outcome.
        self.build()
        idx = reljoin.open_index(self.index, expected_declaration=self.decl)
        q = reljoin.query(idx, self.decl)
        seen = 0
        for _ in q:
            seen += 1
            break
        self.assertEqual(seen, 1)
        self.assertEqual(q.stats.status, "running")
        self.assertNotEqual(q.stats.status, "complete")
        q.close()
        self.assertEqual(q.stats.status, "incomplete")
        self.assertIn("closed", q.stats.status_reason)

    def test_consumer_exception_marks_incomplete_and_terminates(self):
        self.build()
        idx = reljoin.open_index(self.index, expected_declaration=self.decl)
        q = reljoin.query(idx, self.decl)

        class Boom(Exception):
            pass

        with self.assertRaises(Boom):
            with reljoin.query(idx, self.decl) as q2:
                for _ in q2:
                    raise Boom("consumer failed while writing")
        self.assertEqual(q2.stats.status, "incomplete")
        self.assertIn("closed", q2.stats.status_reason)
        # The index was released together with the query.
        self.assertEqual(idx.connections(), [])
        idx.close()

    def test_limit_marks_incomplete_with_code(self):
        self.build()
        out, stats = self.run_query(limit=0)
        self.assertEqual(out, [])
        self.assertEqual(stats.status, "limit-reached")

    def test_complete_run_marks_complete(self):
        self.build()
        out, stats = self.run_query()
        self.assertEqual(stats.status, "complete")
        self.assertEqual(stats.leaves, 1)

    # -- source damage ---------------------------------------------------
    def test_bad_source_line_fails_build(self):
        with open(os.path.join(self.data, "A.jsonl"), "a") as fh:
            fh.write("[1,2,3]\n")
        with self.assertRaises(DataProblem):
            self.build()
        self.assertFalse(os.path.exists(self.index))

    def test_float_and_bool_rejected(self):
        for bad in ("[1, 2.0]", "[true, 2]", "[\"1\", 2]"):
            with open(os.path.join(self.data, "A.jsonl"), "a") as fh:
                fh.write(bad + "\n")
            with self.assertRaises(DataProblem):
                self.build()
            # restore
            write_jsonl(os.path.join(self.data, "A.jsonl"),
                        [(1, 2), (1, 2), (9, 9)])

    def test_int64_out_of_range_rejected(self):
        with open(os.path.join(self.data, "A.jsonl"), "a") as fh:
            fh.write("[%d, 2]\n" % (2 ** 63))
        with self.assertRaises(DataProblem):
            self.build()


class DeclarationLifecycle(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="reljoin-decl-")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def write(self, obj):
        p = os.path.join(self.tmp, "q.json")
        with open(p, "w") as fh:
            json.dump(obj, fh) if not isinstance(obj, str) else fh.write(obj)
        return p

    def test_repeated_variable_inside_atom_rejected(self):
        p = self.write({"version": 1, "order": ["a", "b"],
                        "relations": [{"name": "A", "vars": ["a", "a"]}]})
        with self.assertRaises(DeclarationProblem):
            reljoin.load_declaration(p)

    def test_unused_order_variable_rejected(self):
        p = self.write({"version": 1, "order": ["a", "b", "c"],
                        "relations": [{"name": "A", "vars": ["a", "b"]}]})
        with self.assertRaises(DeclarationProblem):
            reljoin.load_declaration(p)

    def test_variable_missing_from_order_rejected(self):
        p = self.write({"version": 1, "order": ["a"],
                        "relations": [{"name": "A", "vars": ["a", "b"]}]})
        with self.assertRaises(DeclarationProblem):
            reljoin.load_declaration(p)

    def test_aliases_with_different_arity_rejected(self):
        p = self.write({"version": 1, "order": ["a", "b", "c"],
                        "relations": [{"name": "E", "vars": ["a", "b"]},
                                      {"name": "E", "vars": ["a", "b", "c"]}]})
        with self.assertRaises(DeclarationProblem):
            reljoin.load_declaration(p)

    def test_malformed_json_rejected(self):
        p = self.write("{oops")
        with self.assertRaises(DeclarationProblem):
            reljoin.load_declaration(p)


class CliContract(unittest.TestCase):
    """End-to-end CLI checks including exit codes for incomplete runs."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="reljoin-cli-")
        make_triangle(self.tmp)
        self.decl = os.path.join(self.tmp, "query.json")
        self.data = os.path.join(self.tmp, "data")
        self.index = os.path.join(self.tmp, "idx")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def cli(self, *args, **kw):
        return subprocess.run(
            [sys.executable, "-m", "reljoin", *args],
            cwd=ROOT, capture_output=True, text=True, **kw,
        )

    def test_full_cycle_and_exit_codes(self):
        r = self.cli("prepare", "--decl", self.decl, "--data", self.data,
                     "--index", self.index, "--memory", "4M")
        self.assertEqual(r.returncode, 0, r.stderr)
        r = self.cli("run", "--decl", self.decl, "--index", self.index)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(r.stdout.strip(), "[1,2,3,24]")

        r = self.cli("run", "--decl", self.decl, "--index", self.index,
                     "--limit", "0")
        self.assertEqual(r.returncode, 7)
        self.assertEqual(r.stdout.strip(), "")

        r = self.cli("run", "--decl", self.decl,
                     "--index", os.path.join(self.tmp, "nope"))
        self.assertEqual(r.returncode, 5)
        self.assertIn("index-missing", r.stderr)

        r = self.cli("explain", "--decl", self.decl, "--index", self.index)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("LAYERS", r.stdout)

    def test_write_failure_is_incomplete(self):
        r = self.cli("prepare", "--decl", self.decl, "--data", self.data,
                     "--index", self.index, "--memory", "4M")
        self.assertEqual(r.returncode, 0, r.stderr)
        outdir = os.path.join(self.tmp, "outdir")
        os.makedirs(outdir)   # opening a directory for writing must fail
        r = self.cli("run", "--decl", self.decl, "--index", self.index,
                     "--out", outdir)
        self.assertEqual(r.returncode, 7, r.stderr)
        self.assertIn("incomplete", r.stderr)

    def test_consumer_closing_pipe_is_incomplete(self):
        r = self.cli("prepare", "--decl", self.decl, "--data", self.data,
                     "--index", self.index, "--memory", "4M")
        self.assertEqual(r.returncode, 0, r.stderr)
        # Large output, then close the read end immediately.
        big = os.path.join(self.tmp, "big")
        os.makedirs(big)
        with open(os.path.join(self.tmp, "bigdecl.json"), "w") as fh:
            json.dump({"version": 1, "order": ["x", "y"],
                       "relations": [{"name": "E", "vars": ["x", "y"]},
                                     {"name": "E", "vars": ["y", "x"]}]}, fh)
        os.makedirs(os.path.join(big, "data"))
        write_jsonl(os.path.join(big, "data", "E.jsonl"),
                    [(i % 2000, i) for i in range(200000)])
        r = self.cli("prepare", "--decl", os.path.join(self.tmp, "bigdecl.json"),
                     "--data", os.path.join(big, "data"),
                     "--index", os.path.join(big, "idx"), "--memory", "4M")
        self.assertEqual(r.returncode, 0, r.stderr)
        proc = subprocess.Popen(
            [sys.executable, "-m", "reljoin", "run",
             "--decl", os.path.join(self.tmp, "bigdecl.json"),
             "--index", os.path.join(big, "idx")],
            cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=False,
        )
        proc.stdout.read(100)
        proc.stdout.close()
        stderr = proc.stderr.read().decode()
        rc = proc.wait(timeout=120)
        proc.stderr.close()
        self.assertEqual(rc, 7, stderr)
        self.assertIn("incomplete", stderr)


if __name__ == "__main__":
    unittest.main()
