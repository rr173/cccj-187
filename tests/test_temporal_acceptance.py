"""Acceptance tests for the temporal (valid-time) product.

Every result is checked against the independent reference in
``temporal_oracle.py``, which reads only the raw interval JSONL and never
touches the temporal index, the event normalisation or the production
executor.  Hand-computed cases (including the statement's triangle and its
weighted duration) are additionally asserted literally.
"""
from __future__ import annotations

import json
import os
import random
import shutil
import sys
import tempfile
import unittest
from typing import Dict, List, Optional, Sequence, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import reljoin                                    # noqa: E402
from reljoin.temporal_build import build_temporal_index       # noqa: E402
from reljoin.temporal_open import open_temporal_index         # noqa: E402

import temporal_oracle                            # noqa: E402

INT64_MIN = -(1 << 63)
INT64_MAX = (1 << 63) - 1


def write_obj_jsonl(path: str, rows: Sequence[Dict]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


class TCase:
    """A self-contained temporal declaration + interval data directory."""

    def __init__(self, root: str, name: str) -> None:
        self.dir = os.path.join(root, name)
        os.makedirs(self.dir)
        self.data = os.path.join(self.dir, "data")
        os.makedirs(self.data)
        self.index = os.path.join(self.dir, "tindex")
        self.decl_path = os.path.join(self.dir, "query.json")
        self.declaration: Dict = {}

    def declare(self, order: Sequence[str],
                relations: Sequence[Tuple[str, Sequence[str]]]) -> None:
        self.declaration = {
            "version": 1,
            "order": list(order),
            "relations": [{"name": n, "vars": list(v)} for n, v in relations],
        }
        with open(self.decl_path, "w", encoding="utf-8") as fh:
            json.dump(self.declaration, fh)

    def load(self, relation: str, rows: Sequence[Dict]) -> None:
        write_obj_jsonl(os.path.join(self.data, relation + ".jsonl"), rows)

    def load_text(self, relation: str, text: str) -> None:
        name = relation if relation.endswith(".jsonl") else relation + ".jsonl"
        with open(os.path.join(self.data, name), "w") as fh:
            fh.write(text)

    def prepare(self, memory: int = 4 << 20, **kw) -> Dict:
        decl = reljoin.load_declaration(self.decl_path)
        kw.setdefault("force", True)
        return build_temporal_index(
            decl, data_dir=self.data, out_dir=self.index,
            memory_bytes=memory, **kw
        )

    def run(self, lo: int, hi: int, memory: int = 4 << 20,
            limit=None) -> List[Tuple[Tuple[int, ...], int, int, int]]:
        decl = reljoin.load_declaration(self.decl_path)
        idx = open_temporal_index(self.index, expected_declaration=decl)
        out: List[Tuple[Tuple[int, ...], int, int, int]] = []
        q = reljoin.temporal_query(
            idx, decl, window=(lo, hi), memory_bytes=memory, limit=limit
        )
        try:
            for values, (since, until, mult) in q:
                out.append((values, since, until, mult))
        finally:
            q.close()
        self.last_status = q.stats.status
        self.last_stats = q.report()
        return out

    def expect(self, lo: int, hi: int):
        decl = reljoin.load_declaration(self.decl_path)
        return temporal_oracle.temporal_brute_force(
            decl, self.data, lo, hi
        )


def triangle_case(c: TCase) -> None:
    c.declare(["x", "y", "z"],
              [("A", ["x", "y"]), ("B", ["y", "z"]), ("C", ["z", "x"])])


class HandSample(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="reljoin-thand-")

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_statement_triangle_window_0_10(self) -> None:
        c = TCase(self.tmp, "hand")
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
        c.prepare()
        got = c.run(0, 10)
        self.assertEqual(got, [
            ((1, 2, 3), 2, 3, 24),
            ((1, 2, 3), 3, 6, 36),
            ((1, 2, 3), 6, 7, 24),
        ])
        # No split at tick 5: the two C rows meet there with equal copies.
        self.assertEqual(
            [s for s in got if s[1] == 5], []
        )
        self.assertEqual(c.last_stats["weighted_duration"], 156)
        self.assertEqual(
            temporal_oracle.weighted_duration(c.expect(0, 10)), 156
        )
        self.assertEqual(c.last_status, "complete")

    def test_statement_triangle_window_4_8(self) -> None:
        c = TCase(self.tmp, "hand48")
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
        c.prepare()
        got = c.run(4, 8)
        self.assertEqual(got, [
            ((1, 2, 3), 4, 6, 36),
            ((1, 2, 3), 6, 7, 24),
        ])
        self.assertEqual(c.last_stats["weighted_duration"], 36 * 2 + 24)
        self.assertEqual(got, c.expect(4, 8))

    def test_segments_clipped_at_window_edges(self) -> None:
        c = TCase(self.tmp, "clip")
        triangle_case(c)
        c.load("A", [{"values": [1, 2], "since": -5, "until": 50}])
        c.load("B", [{"values": [2, 3], "since": -5, "until": 50}])
        c.load("C", [{"values": [3, 1], "since": -5, "until": 50}])
        c.prepare()
        self.assertEqual(c.run(3, 9), [((1, 2, 3), 3, 9, 1)])
        self.assertEqual(c.run(3, 9), c.expect(3, 9))

    def test_no_intersection_with_window_is_not_bad_data(self) -> None:
        c = TCase(self.tmp, "outside")
        triangle_case(c)
        c.load("A", [{"values": [1, 2], "since": 0, "until": 4}])
        c.load("B", [{"values": [2, 3], "since": 0, "until": 4}])
        c.load("C", [
            {"values": [3, 1], "since": 0, "until": 4},
            {"values": [9, 9], "since": 100, "until": 200},
        ])
        c.prepare()
        self.assertEqual(c.run(10, 20), [])
        self.assertEqual(c.last_status, "complete")
        self.assertEqual(c.last_stats["weighted_duration"], 0)


class Semantics(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="reljoin-tsem-")

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def check(self, c: TCase, lo: int, hi: int, memory: int = 1 << 20) -> None:
        c.prepare(memory=memory)
        self.assertEqual(c.run(lo, hi, memory=memory), c.expect(lo, hi),
                         msg=f"window [{lo},{hi})")
        self.assertEqual(c.last_status, "complete")

    def test_same_relation_aliases_count_independently(self) -> None:
        c = TCase(self.tmp, "alias")
        c.declare(["a", "b"],
                  [("E", ["a", "b"]), ("E", ["b", "a"])])
        c.load("E", [
            {"values": [1, 2], "since": 0, "until": 5, "copies": 2},
            {"values": [2, 1], "since": 1, "until": 6, "copies": 3},
            {"values": [2, 1], "since": 3, "until": 4, "copies": 1},
        ])
        for lo, hi in ((0, 10), (1, 5), (2, 4), (-3, 3)):
            self.check(c, lo, hi)

    def test_two_factors_swap_2_3_and_3_2_product_stays_6(self) -> None:
        # Alias 1 multiplicity goes 2 -> 3 at t=4; alias 2 goes 3 -> 2 at the
        # same tick.  The product stays 6, so exactly one merged segment.
        c = TCase(self.tmp, "swap")
        c.declare(["a", "b"],
                  [("E", ["a", "b"]), ("E", ["b", "a"])])
        c.load("E", [
            {"values": [1, 2], "since": 0, "until": 4, "copies": 2},
            {"values": [1, 2], "since": 4, "until": 9, "copies": 3},
            {"values": [2, 1], "since": 0, "until": 4, "copies": 3},
            {"values": [2, 1], "since": 4, "until": 9, "copies": 2},
        ])
        c.prepare()
        got = c.run(0, 10)
        # Two complete bindings exist ((a,b)=(1,2) and (2,1)); each keeps the
        # constant product 6 across the simultaneous swap, so one segment each.
        self.assertEqual(got, [
            ((1, 2), 0, 9, 6),
            ((2, 1), 0, 9, 6),
        ])
        stats = c.last_stats
        self.assertGreaterEqual(stats["segments_merged"], 1)
        self.assertEqual(got, c.expect(0, 10))

    def test_touching_endpoints_are_never_simultaneously_valid(self) -> None:
        # [0,5) and [5,10) touch at 5 but never overlap: no closed loop there.
        c = TCase(self.tmp, "touch")
        triangle_case(c)
        c.load("A", [{"values": [1, 2], "since": 0, "until": 5}])
        c.load("B", [{"values": [2, 3], "since": 5, "until": 10}])
        c.load("C", [
            {"values": [3, 1], "since": 0, "until": 10},
        ])
        c.prepare()
        self.assertEqual(c.run(0, 10), [])
        self.assertEqual(c.run(0, 10), c.expect(0, 10))

    def test_negative_time_and_int64_boundaries(self) -> None:
        c = TCase(self.tmp, "bounds")
        triangle_case(c)
        c.load("A", [
            {"values": [1, 2], "since": INT64_MIN, "until": 0, "copies": 2},
            {"values": [1, 2], "since": 0, "until": INT64_MAX, "copies": 5},
        ])
        c.load("B", [
            {"values": [2, 3], "since": INT64_MIN, "until": INT64_MAX,
             "copies": 7},
        ])
        c.load("C", [
            {"values": [3, 1], "since": INT64_MIN, "until": -3, "copies": 11},
            {"values": [3, 1], "since": -3, "until": INT64_MAX,
             "copies": 13},
        ])
        for lo, hi in (
            (INT64_MIN, INT64_MAX),
            (INT64_MIN, 0),
            (-100, 100),
            (INT64_MIN, -3),
            (-3, INT64_MAX),
        ):
            self.check(c, lo, hi)

    def test_overlapping_and_duplicate_rows_accumulate(self) -> None:
        c = TCase(self.tmp, "overlap")
        triangle_case(c)
        c.load("A", [
            {"values": [1, 2], "since": 0, "until": 8, "copies": 2},
            {"values": [1, 2], "since": 3, "until": 6},           # default 1
            {"values": [1, 2], "since": 0, "until": 8, "copies": 2},  # dup
        ])
        c.load("B", [{"values": [2, 3], "since": 1, "until": 7,
                      "copies": 3}])
        c.load("C", [
            {"values": [3, 1], "since": 2, "until": 9, "copies": 4},
        ])
        self.check(c, 0, 10)
        got = c.run(0, 10)
        # A multiplicity: 4 on [0,3), 5 on [3,6), 4 on [6,8); B*C = 12, but
        # B only covers [1,7), so the [7,8) tail of A never forms a loop.
        self.assertEqual(got, [
            ((1, 2, 3), 2, 3, 48),
            ((1, 2, 3), 3, 6, 60),
            ((1, 2, 3), 6, 7, 48),
        ])

    def test_multiplicity_beyond_64_bits(self) -> None:
        c = TCase(self.tmp, "bigmult")
        c.declare(["a", "b"],
                  [("E", ["a", "b"]), ("E", ["b", "a"])])
        huge = 10 ** 20
        c.load("E", [
            {"values": [1, 2], "since": 0, "until": 5, "copies": huge},
            {"values": [2, 1], "since": 0, "until": 5, "copies": huge},
        ])
        c.prepare()
        got = c.run(0, 5)
        self.assertEqual(got, [
            ((1, 2), 0, 5, huge * huge),
            ((2, 1), 0, 5, huge * huge),
        ])
        self.assertGreater(huge * huge, 1 << 64)
        self.assertEqual(c.last_stats["weighted_duration"],
                         2 * 5 * huge * huge)

    def test_hole_is_never_crossed_by_a_merge(self) -> None:
        c = TCase(self.tmp, "hole")
        triangle_case(c)
        c.load("A", [
            {"values": [1, 2], "since": 0, "until": 3},
            {"values": [1, 2], "since": 6, "until": 9},
        ])
        c.load("B", [{"values": [2, 3], "since": 0, "until": 9}])
        c.load("C", [{"values": [3, 1], "since": 0, "until": 9}])
        c.prepare()
        got = c.run(0, 10)
        self.assertEqual(got, [
            ((1, 2, 3), 0, 3, 1),
            ((1, 2, 3), 6, 9, 1),
        ])
        self.assertEqual(got, c.expect(0, 10))

    def test_distinct_bindings_are_never_merged(self) -> None:
        c = TCase(self.tmp, "bindings")
        triangle_case(c)
        c.load("A", [
            {"values": [1, 2], "since": 0, "until": 9},
            {"values": [4, 5], "since": 0, "until": 9},
        ])
        c.load("B", [
            {"values": [2, 3], "since": 0, "until": 9},
            {"values": [5, 6], "since": 0, "until": 9},
        ])
        c.load("C", [
            {"values": [3, 1], "since": 0, "until": 9},
            {"values": [6, 4], "since": 0, "until": 9},
        ])
        c.prepare()
        got = c.run(0, 10)
        self.assertEqual(got, [
            ((1, 2, 3), 0, 9, 1),
            ((4, 5, 6), 0, 9, 1),
        ])
        self.assertEqual(got, c.expect(0, 10))


class InputValidation(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="reljoin-tin-")
        self._seq = 0

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _case(self) -> TCase:
        c = TCase(self.tmp, f"bad{self._seq}")
        self._seq += 1
        triangle_case(c)
        c.load("A", [{"values": [1, 2], "since": 0, "until": 4}])
        c.load("B", [{"values": [2, 3], "since": 0, "until": 4}])
        c.load("C", [{"values": [3, 1], "since": 0, "until": 4}])
        return c

    def _expect_data_problem(self, c: TCase) -> None:
        decl = reljoin.load_declaration(c.decl_path)
        with self.assertRaises(reljoin.DataProblem) as ctx:
            build_temporal_index(
                decl, data_dir=c.data,
                out_dir=os.path.join(c.dir, "should-not-exist"),
                memory_bytes=1 << 20,
            )
        message = str(ctx.exception)
        self.assertIn("C", message)          # relation name present
        self.assertIn(".jsonl:", message)    # and a line number location
        self.assertFalse(os.path.exists(
            os.path.join(c.dir, "should-not-exist")))

    def test_bool_is_not_an_integer_anywhere(self) -> None:
        c = self._case()
        c.load_text("C.jsonl", '{"values":[3,1],"since":0,"until":4}\n'
                               '{"values":[true,1],"since":0,"until":4}\n')
        self._expect_data_problem(c)
        c = self._case()
        c.load_text("C.jsonl", '{"values":[3,1],"since":false,"until":4}\n')
        self._expect_data_problem(c)
        c = self._case()
        c.load_text("C.jsonl",
                    '{"values":[3,1],"since":0,"until":4,"copies":true}\n')
        self._expect_data_problem(c)

    def test_copies_must_be_positive(self) -> None:
        for bad in (0, -1):
            c = self._case()
            c.load_text(
                "C.jsonl",
                f'{{"values":[3,1],"since":0,"until":4,"copies":{bad}}}\n',
            )
            self._expect_data_problem(c)

    def test_since_must_be_below_until(self) -> None:
        c = self._case()
        c.load_text("C.jsonl",
                    '{"values":[3,1],"since":5,"until":5}\n')
        self._expect_data_problem(c)
        c = self._case()
        c.load_text("C.jsonl",
                    '{"values":[3,1],"since":9,"until":2}\n')
        self._expect_data_problem(c)

    def test_unknown_field_and_bad_shape_rejected(self) -> None:
        c = self._case()
        c.load_text("C.jsonl",
                    '{"values":[3,1],"since":0,"until":4,"extra":1}\n')
        self._expect_data_problem(c)
        c = self._case()
        c.load_text("C.jsonl", '[3, 1]\n')
        self._expect_data_problem(c)
        c = self._case()
        c.load_text("C.jsonl",
                    '{"values":[3],"since":0,"until":4}\n')
        self._expect_data_problem(c)

    def test_endpoint_outside_int64_rejected(self) -> None:
        c = self._case()
        c.load_text("C.jsonl",
                    '{"values":[3,1],"since":0,"until":9223372036854775808}\n')
        self._expect_data_problem(c)

    def test_error_message_names_relation_and_line(self) -> None:
        c = self._case()
        c.load_text("C.jsonl",
                    '{"values":[3,1],"since":0,"until":4}\n'
                    '\n'
                    'not json at all\n')
        decl = reljoin.load_declaration(c.decl_path)
        with self.assertRaises(reljoin.DataProblem) as ctx:
            build_temporal_index(
                decl, data_dir=c.data,
                out_dir=os.path.join(self.tmp, "nope2"),
                memory_bytes=1 << 20,
            )
        self.assertIn("C", str(ctx.exception))
        self.assertIn(":3:", str(ctx.exception))


class RandomOracleComparison(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="reljoin-trand-")

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_random_small_instances(self) -> None:
        rng = random.Random(20261117)
        shapes = [
            (["x", "y", "z"], [("A", ["x", "y"]), ("B", ["y", "z"]),
                                ("C", ["z", "x"])]),
            (["a", "b"], [("E", ["a", "b"]), ("E", ["b", "a"])]),
            (["a", "b", "c", "d"],
             [("AB", ["a", "b"]), ("BC", ["b", "c"]),
              ("CD", ["c", "d"]), ("DA", ["d", "a"])]),
            (["x", "y", "z"], [("E", ["x", "y"]), ("E", ["y", "z"]),
                               ("E", ["z", "x"])]),
        ]
        for trial in range(24):
            order, rels = shapes[trial % len(shapes)]
            c = TCase(self.tmp, f"rand{trial}")
            perm = list(order)
            rng.shuffle(perm)
            c.declare(perm, rels)
            arity_by_name = {n: len(v) for n, v in rels}
            for name in sorted(arity_by_name):
                arity = arity_by_name[name]
                rows = []
                for _ in range(rng.randint(0, 9)):
                    values = [rng.choice([-3, -1, 0, 1, 2, 5])
                              for _ in range(arity)]
                    s = rng.randint(-6, 6)
                    u = s + rng.randint(1, 6)
                    row = {"values": values, "since": s, "until": u}
                    if rng.random() < 0.4:
                        row["copies"] = rng.randint(1, 4)
                    rows.append(row)
                c.load(name, rows)
            lo = rng.randint(-8, 2)
            hi = rng.randint(lo + 1, 12)
            c.prepare(memory=1 << 20)
            got = c.run(lo, hi, memory=1 << 20)
            want = c.expect(lo, hi)
            self.assertEqual(got, want,
                             msg=f"trial {trial} window [{lo},{hi})")
            self.assertEqual(
                c.last_stats["weighted_duration"],
                temporal_oracle.weighted_duration(want),
            )


if __name__ == "__main__":
    unittest.main()
