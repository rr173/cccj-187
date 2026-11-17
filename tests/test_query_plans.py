"""Verify that every seek statement the engine issues is an index seek.

The engine must not fall back to table scans or temp B-trees.  Rather than
trusting the code, we take the *actual* SQL strings the atom executors
prepared, run ``EXPLAIN QUERY PLAN`` on each with representative parameters,
and require a primary-key SEARCH for every one of them.  This is what makes
"still no two-table intermediate / still an index" checkable at run time.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import reljoin                                           # noqa: E402
from reljoin.engine import Solver, _AtomExec             # noqa: E402
from reljoin.index_build import build_index              # noqa: E402
from reljoin.index_open import open_index                # noqa: E402

from util import write_jsonl                             # noqa: E402


class Plans(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="reljoin-plan-")
        self.data = os.path.join(self.tmp, "data")
        os.makedirs(self.data)
        # Definition order deliberately differs from the global order, so at
        # least one table has a permuted primary key (level order).
        self.decl = {
            "version": 1,
            "order": ["a", "b", "c", "d"],
            "relations": [
                {"name": "R", "vars": ["d", "b"]},    # key order b, d
                {"name": "S", "vars": ["c", "b"]},    # key order b, c
                {"name": "T", "vars": ["a", "d"]},    # key order a, d
                {"name": "U", "vars": ["a", "c"]},    # key order a, c
                {"name": "R", "vars": ["a", "b"]},    # alias of R, key a,b
            ],
        }
        with open(os.path.join(self.tmp, "q.json"), "w") as fh:
            json.dump(self.decl, fh)
        write_jsonl(os.path.join(self.data, "R.jsonl"),
                    [(i, i + 1) for i in range(50)])
        write_jsonl(os.path.join(self.data, "S.jsonl"),
                    [(i, i) for i in range(50)])
        write_jsonl(os.path.join(self.data, "T.jsonl"),
                    [(i, i + 1) for i in range(50)])
        write_jsonl(os.path.join(self.data, "U.jsonl"),
                    [(i, i) for i in range(50)])

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_all_seek_statements_use_the_primary_key(self):
        decl = reljoin.load_declaration(os.path.join(self.tmp, "q.json"))
        build_index(decl, data_dir=self.data,
                    out_dir=os.path.join(self.tmp, "idx"), memory_bytes=1 << 20)
        idx = open_index(os.path.join(self.tmp, "idx"), expected_declaration=decl)
        try:
            solver = Solver(decl, idx)
            checked = 0
            first_row_reads = 0
            for ex in solver.atoms:
                for sql in ex.sql_first + ex.sql_ge + ex.sql_gt:
                    n = sql.count("?")
                    if n == 0:
                        # Unconstrained "first value" read at the outermost
                        # layer: it must still be a single-row LIMIT 1 read
                        # (the index is ordered by the key, so this is O(1)).
                        self.assertIn("LIMIT 1", sql)
                        plan = ex.conn.execute(
                            "EXPLAIN QUERY PLAN " + sql, ()
                        ).fetchall()
                        text = " ".join(str(row[-1]) for row in plan)
                        self.assertNotIn("TEMP B-TREE", text.upper())
                        first_row_reads += 1
                        continue
                    # Representative bounds: the earliest and latest int64.
                    for bound in (-(1 << 63), (1 << 63) - 1):
                        params = (bound,) * n
                        plan = ex.conn.execute(
                            "EXPLAIN QUERY PLAN " + sql, params
                        ).fetchall()
                        text = " ".join(str(row[-1]) for row in plan)
                        self.assertIn("SEARCH", text,
                                      msg=f"not an index seek: {sql} -> {text}")
                        self.assertIn("PRIMARY KEY", text,
                                      msg=f"not a primary-key seek: {sql} -> {text}")
                        self.assertNotIn("TEMP B-TREE", text.upper(),
                                         msg=f"sorter inserted: {sql} -> {text}")
                        self.assertNotIn("SCAN t", text,
                                         msg=f"table scan: {sql} -> {text}")
                        checked += 1
                plan = ex.conn.execute(
                    "EXPLAIN QUERY PLAN " + ex.sql_exact,
                    (0,) * ex.key_columns.__len__(),
                ).fetchall()
                text = " ".join(str(row[-1]) for row in plan)
                self.assertIn("SEARCH t USING PRIMARY KEY", text)
                checked += 1
            self.assertGreater(checked, 10)
            self.assertGreater(first_row_reads, 0)
        finally:
            idx.close()

    def test_permuted_key_still_sorted_by_level(self):
        decl = reljoin.load_declaration(os.path.join(self.tmp, "q.json"))
        build_index(decl, data_dir=self.data,
                    out_dir=os.path.join(self.tmp, "idx"), memory_bytes=1 << 20)
        idx = open_index(os.path.join(self.tmp, "idx"), expected_declaration=decl)
        try:
            for rec in idx.manifest["tables"]:
                self.assertEqual(rec["key_levels"], sorted(rec["key_levels"]))
                self.assertEqual(
                    rec["key_vars"],
                    [rec["vars"][j] for j in rec["key_columns"]],
                )
        finally:
            idx.close()


if __name__ == "__main__":
    unittest.main()
