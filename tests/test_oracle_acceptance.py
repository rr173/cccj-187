"""Acceptance tests against an independent Cartesian-enumeration oracle.

The oracle (``tests/oracle.py``) reads the original JSONL tuples, compares
values for equality and counts bags directly.  These tests never touch the
production solver's cursors from the oracle side.

Run with:  python3 -m unittest discover -s tests -v
"""
from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import unittest
from typing import Dict, List, Sequence, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import reljoin                                    # noqa: E402
from reljoin.index_build import build_index       # noqa: E402
from reljoin.index_open import open_index         # noqa: E402

import oracle                                     # noqa: E402
from util import write_jsonl                      # noqa: E402

INT64_MIN = -(1 << 63)
INT64_MAX = (1 << 63) - 1


class Case:
    """A self-contained declaration + data directory."""

    def __init__(self, root: str, name: str) -> None:
        self.dir = os.path.join(root, name)
        os.makedirs(self.dir)
        self.data = os.path.join(self.dir, "data")
        os.makedirs(self.data)
        self.index = os.path.join(self.dir, "index")
        self.decl_path = os.path.join(self.dir, "query.json")
        self.declaration: Dict = {}

    def declare(self, order: Sequence[str], relations: Sequence[Tuple[str, Sequence[str]]],
                version: int = 1) -> None:
        self.declaration = {
            "version": version,
            "order": list(order),
            "relations": [{"name": n, "vars": list(v)} for n, v in relations],
        }
        with open(self.decl_path, "w", encoding="utf-8") as fh:
            json.dump(self.declaration, fh)
        seen = []
        for n, _v in relations:
            if n not in seen:
                seen.append(n)
        self.relations = seen

    def load(self, relation: str, rows: Sequence[Sequence[int]]) -> None:
        write_jsonl(os.path.join(self.data, relation + ".jsonl"), rows)

    # -- production side --------------------------------------------------
    def prepare(self, memory: int = 4 << 20, **kw) -> Dict:
        decl = reljoin.load_declaration(self.decl_path)
        return build_index(decl, data_dir=self.data, out_dir=self.index,
                           memory_bytes=memory, **kw)

    def run(self, memory: int = 4 << 20, limit=None) -> List[Tuple[Tuple[int, ...], int]]:
        decl = reljoin.load_declaration(self.decl_path)
        idx = open_index(self.index, expected_declaration=decl)
        out = []
        q = reljoin.query(idx, decl, memory_bytes=memory, limit=limit)
        try:
            for values, mult in q:
                out.append((values, mult))
        finally:
            q.close()
        self.last_status = q.stats.status
        self.last_stats = q.report()
        return out

    # -- oracle side ------------------------------------------------------
    def expect(self) -> List[Tuple[Tuple[int, ...], int]]:
        decl = reljoin.load_declaration(self.decl_path)
        return oracle.brute_force(decl, self.data)


class OracleAcceptance(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="reljoin-test-")

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def case(self, name: str) -> Case:
        return Case(self.tmp, name)

    def check(self, case: Case, memory: int = 4 << 20) -> None:
        case.prepare(memory=memory)
        got = case.run(memory=memory)
        want = case.expect()
        self.assertEqual(got, want)
        self.assertEqual(case.last_status, "complete")

    # -- required scenarios ----------------------------------------------
    def test_hand_triangle_multiplicity_24(self) -> None:
        c = self.case("triangle")
        c.declare(["x", "y", "z"],
                  [("A", ["x", "y"]), ("B", ["y", "z"]), ("C", ["z", "x"])])
        c.load("A", [(1, 2), (1, 2)])
        c.load("B", [(2, 3)] * 3)
        c.load("C", [(3, 1)] * 4)
        c.prepare()
        got = c.run()
        self.assertEqual(got, [((1, 2, 3), 24)])

    def test_triangle_without_closing_relation_is_empty(self) -> None:
        c = self.case("empty-c")
        c.declare(["x", "y", "z"],
                  [("A", ["x", "y"]), ("B", ["y", "z"]), ("C", ["z", "x"])])
        c.load("A", [(1, 2), (1, 2)])
        c.load("B", [(2, 3)] * 3)
        c.load("C", [])
        c.prepare()
        self.assertEqual(c.run(), [])
        self.assertEqual(c.last_status, "complete")

    def test_self_alias_and_repeated_relation(self) -> None:
        c = self.case("selfalias")
        # One relation used twice, columns assigned independently.
        c.declare(["a", "b", "c"],
                  [("E", ["a", "b"]), ("E", ["b", "c"])])
        c.load("E", [(1, 2), (2, 3), (2, 2), (1, 2)])
        self.check(c)

    def test_self_alias_reversed_columns(self) -> None:
        c = self.case("selfrev")
        c.declare(["a", "b"],
                  [("E", ["a", "b"]), ("E", ["b", "a"])])
        c.load("E", [(1, 2), (2, 1), (2, 2)])
        self.check(c)

    def test_shuffled_columns_do_not_misconnect(self) -> None:
        c = self.case("shuffle")
        # Same logical relation, file columns in a different order than the
        # global order, plus an atom whose columns are reversed in the file.
        c.declare(["p", "q", "r"],
                  [("R", ["r", "p"]), ("S", ["q", "r"]), ("T", ["p", "q"])])
        c.load("R", [(30, 10), (40, 20)])
        c.load("S", [(10, 30), (40, 1)])
        c.load("T", [(20, 40)])
        self.check(c)

    def test_empty_relations(self) -> None:
        c = self.case("emptyrels")
        c.declare(["a", "b"],
                  [("A", ["a", "b"]), ("B", ["b", "a"])])
        c.load("A", [])
        c.load("B", [])
        c.prepare()
        self.assertEqual(c.run(), [])
        self.assertEqual(c.last_status, "complete")

    def test_negative_and_extreme_values(self) -> None:
        c = self.case("extremes")
        c.declare(["a", "b", "c"],
                  [("A", ["a", "b"]), ("B", ["b", "c"]), ("C", ["c", "a"])])
        c.load("A", [(INT64_MIN, INT64_MAX), (INT64_MAX, INT64_MIN),
                     (INT64_MIN, INT64_MAX)])
        c.load("B", [(INT64_MAX, INT64_MIN), (0, INT64_MIN)])
        c.load("C", [(INT64_MIN, INT64_MIN), (INT64_MIN, INT64_MAX)])
        self.check(c)

    def test_large_multiplicity_beyond_64_bits(self) -> None:
        c = self.case("bigcount")
        c.declare(["a", "b"],
                  [("A", ["a", "b"]), ("B", ["b", "a"])])
        # 400 * 400 = 160000 rows per relation; multiplicity 160000**2 =
        # 2.56e10 fits in 64 bits, so use three cooperating atoms to exceed it
        # without writing huge files: use repeated atoms instead.
        c.declare(["a", "b"],
                  [("A", ["a", "b"]), ("A", ["a", "b"]), ("A", ["a", "b"]),
                   ("A", ["a", "b"]), ("A", ["a", "b"]), ("A", ["a", "b"]),
                   ("A", ["a", "b"]), ("B", ["b", "a"]), ("B", ["b", "a"])])
        c.load("A", [(7, 9)] * 1000)
        c.load("B", [(9, 7)] * 1000)
        c.prepare()
        got = c.run()
        # 7 copies of A (1000 each) and 2 copies of B (1000 each)
        self.assertEqual(got, [((7, 9), 1000 ** 9)])
        self.assertGreater(1000 ** 9, (1 << 63) - 1)

    def test_four_cycle_and_chorded_cycle(self) -> None:
        c = self.case("four")
        c.declare(["a", "b", "c", "d"],
                  [("AB", ["a", "b"]), ("BC", ["b", "c"]),
                   ("CD", ["c", "d"]), ("DA", ["d", "a"])])
        c.load("AB", [(1, 1), (1, 2), (2, 2)])
        c.load("BC", [(1, 1), (2, 3)])
        c.load("CD", [(1, 1), (3, 2)])
        c.load("DA", [(1, 1), (2, 1)])
        self.check(c)
        c2 = self.case("chord")
        # 4-cycle plus chord a-c and b-d: a triangle and a 4-clique shape.
        c2.declare(["a", "b", "c", "d"],
                   [("AB", ["a", "b"]), ("BC", ["b", "c"]),
                    ("CD", ["c", "d"]), ("DA", ["d", "a"]),
                    ("AC", ["a", "c"]), ("BD", ["b", "d"])])
        c2.load("AB", [(1, 1), (1, 2), (2, 2)])
        c2.load("BC", [(1, 1), (2, 2)])
        c2.load("CD", [(1, 1), (2, 2)])
        c2.load("DA", [(1, 1), (2, 1)])
        c2.load("AC", [(1, 1), (2, 2)])
        c2.load("BD", [(1, 1), (2, 2)])
        self.check(c2)

    def test_relation_arity_one_and_order_permutations(self) -> None:
        c = self.case("unary")
        c.declare(["u", "v"],
                  [("U", ["u"]), ("V", ["v"]), ("P", ["u", "v"]),
                   ("Q", ["v", "u"])])
        c.load("U", [(1,), (2,), (2,), (3,)])
        c.load("V", [(2,), (3,), (3,)])
        c.load("P", [(1, 2), (2, 2), (2, 3), (3, 3)])
        c.load("Q", [(2, 2), (3, 2), (3, 3)])
        self.check(c)

    def test_random_small_instances_match_oracle(self) -> None:
        rng = random.Random(20241117)
        shapes = [
            (["a", "b", "c"], [("A", ["a", "b"]), ("B", ["b", "c"]),
                               ("C", ["c", "a"])]),
            (["a", "b", "c", "d"], [("A", ["a", "b"]), ("B", ["b", "c"]),
                                    ("C", ["c", "d"]), ("D", ["d", "a"])]),
            (["a", "b", "c", "d"], [("a:b", ["a", "b"]), ("b:c", ["b", "c"]),
                                    ("a:c", ["a", "c"]), ("a:c", ["c", "a"]),
                                    ("b:d", ["b", "d"]), ("c:d", ["c", "d"])]),            (["x", "y"], [("E", ["x", "y"]), ("E", ["y", "x"])]),
            (["x", "y", "z"], [("E", ["x", "y"]), ("E", ["y", "z"]),
                               ("E", ["z", "x"])]),
        ]
        for trial in range(15):
            order, rels = shapes[trial % len(shapes)]
            c = self.case(f"rand{trial}")
            perm = list(order)
            rng.shuffle(perm)
            c.declare(perm, rels)
            for name in sorted({n for n, _ in rels}):
                arity = len(dict(rels)[name])
                nrows = rng.randint(0, 7)
                rows = []
                for _ in range(nrows):
                    rows.append(tuple(rng.choice([-2, -1, 0, 1, 2])
                                      for _ in range(arity)))
                c.load(name, rows)
            c.prepare(memory=1 << 20)
            self.assertEqual(c.run(memory=1 << 20), c.expect(),
                             msg=f"trial {trial} {c.declaration}")


class DeeperOracleCases(unittest.TestCase):
    """Oracle checks with more rows per relation (still small enough to brute force)."""

    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="reljoin-test-")

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_medium_random(self) -> None:
        rng = random.Random(7)
        for trial in range(3):
            c = Case(self.tmp, f"med{trial}")
            c.declare(["a", "b", "c", "d"],
                      [("A", ["a", "b"]), ("B", ["b", "c"]),
                       ("C", ["c", "d"]), ("D", ["d", "a"])])
            for name in "ABCD":
                rows = [tuple(rng.randint(0, 4) for _ in range(2))
                        for _ in range(25)]
                c.load(name, rows)
            c.prepare(memory=4 << 20)
            self.assertEqual(c.run(memory=4 << 20), c.expect())


if __name__ == "__main__":
    unittest.main()
