"""Acceptance tests for the incremental maintenance product.

Every maintained view and every signed delta is checked against the
independent reference in ``maintain_oracle.py``, which rebuilds the relation
bags from raw interval + correction files and enumerates bindings from
scratch.  It never opens the maintenance database or calls a production
cursor, range finder or delta function.  The two hand examples from the brief
(triangle weighted change 48; self-alias +5) are asserted literally as well.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unittest
from typing import Any, Dict, List, Sequence, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import reljoin                                    # noqa: E402
from reljoin.errors import (                     # noqa: E402
    BatchConflict,
    CorrectionProblem,
    DataProblem,
)
from reljoin.temporal_build import build_temporal_index       # noqa: E402

import maintain_oracle                            # noqa: E402

INT64_MIN = -(1 << 63)
INT64_MAX = (1 << 63) - 1


def write_obj(path: str, rows: Sequence[Dict]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def write_corr(path: str, rows: Sequence[Dict]) -> None:
    write_obj(path, rows)


class MCase:
    def __init__(self, root: str, name: str, order, relations) -> None:
        self.dir = os.path.join(root, name)
        os.makedirs(self.dir)
        self.data = os.path.join(self.dir, "data")
        os.makedirs(self.data)
        self.tindex = os.path.join(self.dir, "tidx")
        self.maintain = os.path.join(self.dir, "mlib")
        self.decl_path = os.path.join(self.dir, "query.json")
        with open(self.decl_path, "w", encoding="utf-8") as fh:
            json.dump({
                "version": 1, "order": list(order),
                "relations": [
                    {"name": n, "vars": list(v)} for n, v in relations
                ],
            }, fh)

    def load(self, rel: str, rows: Sequence[Dict]) -> None:
        write_obj(os.path.join(self.data, rel + ".jsonl"), rows)

    def corr(self, name: str, rows: Sequence[Dict]) -> str:
        path = os.path.join(self.dir, name + ".jsonl")
        write_corr(path, rows)
        return path


def triangle_case(root: str, name: str = "tri") -> MCase:
    c = MCase(root, name, ("x", "y", "z"), [
        ("A", ("x", "y")), ("B", ("y", "z")), ("C", ("z", "x")),
    ])
    c.load("A", [{"values": [1, 2], "since": 0, "until": 10, "copies": 2}])
    c.load("B", [{"values": [2, 3], "since": 0, "until": 10, "copies": 3}])
    c.load("C", [{"values": [3, 1], "since": 0, "until": 10, "copies": 4}])
    return c


def build(c: MCase):
    decl = reljoin.load_declaration(c.decl_path)
    build_temporal_index(
        decl, data_dir=c.data, out_dir=c.tindex, memory_bytes=1 << 20
    )
    tidx = reljoin.open_temporal_index(c.tindex, expected_declaration=decl)
    try:
        reljoin.init_maintenance(
            decl, tidx, out_dir=c.maintain, memory_bytes=1 << 20
        )
    finally:
        tidx.close()
    return decl


def view_rows(store, decl, lo, hi):
    q = reljoin.MaintainQuery(store, decl, window=(lo, hi))
    out = [(tuple(v), s, u, m) for v, (s, u, m) in q]
    rev = q.revision
    q.close()
    return out, rev


def delta_rows(store, batch_id):
    return [
        (tuple(r["values"]), r["since"], r["until"],
         r["delta_multiplicity"])
        for r in reljoin.stream_batch_delta(store, batch_id)
    ]


class MaintenanceAcceptance(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="reljoin-maint-")

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    # -- hand examples ---------------------------------------------------
    def test_triangle_weighted_change_48(self) -> None:
        c = triangle_case(self.tmp)
        decl = build(c)
        store = reljoin.open_maintain_store(c.maintain, decl)
        self.assertEqual(store.revision(), 0)
        initial, rev = view_rows(store, decl, 0, 10)
        self.assertEqual(rev, 0)
        self.assertEqual(initial, [((1, 2, 3), 0, 10, 24)])

        corr = c.corr("b1", [
            {"relation": "A", "values": [1, 2], "since": 2, "until": 8,
             "delta": 1},
            {"relation": "B", "values": [2, 3], "since": 4, "until": 6,
             "delta": -1},
        ])
        result = reljoin.apply_corrections(
            store, batch_id="b1", expect_revision=0,
            correction_path=corr, memory_bytes=1 << 20,
        )
        self.assertEqual(result.receipt.revision, 1)
        self.assertEqual(result.receipt.weighted_change, 48)
        self.assertEqual(
            sorted(delta_rows(store, "b1")),
            sorted([
                ((1, 2, 3), 2, 4, 12),
                ((1, 2, 3), 6, 8, 12),
            ]),
        )
        view, rev = view_rows(store, decl, 0, 10)
        self.assertEqual(rev, 1)
        self.assertEqual(view, [
            ((1, 2, 3), 0, 2, 24),
            ((1, 2, 3), 2, 4, 36),
            ((1, 2, 3), 4, 6, 24),
            ((1, 2, 3), 6, 8, 36),
            ((1, 2, 3), 8, 10, 24),
        ])
        store.close()

    def test_self_alias_delta_five_not_four(self) -> None:
        c = MCase(self.tmp, "alias", ("x", "y"), [
            ("R", ("x", "y")), ("R", ("y", "x")),
        ])
        c.load("R", [{"values": [1, 1], "since": 0, "until": 4, "copies": 2}])
        decl = build(c)
        store = reljoin.open_maintain_store(c.maintain, decl)
        initial, _ = view_rows(store, decl, 0, 4)
        self.assertEqual(initial, [((1, 1), 0, 4, 4)])
        corr = c.corr("a1", [
            {"relation": "R", "values": [1, 1], "since": 1, "until": 3,
             "delta": 1},
        ])
        reljoin.apply_corrections(
            store, batch_id="a1", expect_revision=0,
            correction_path=corr, memory_bytes=1 << 20,
        )
        self.assertEqual(
            delta_rows(store, "a1"), [((1, 1), 1, 3, 5)]
        )
        view, _ = view_rows(store, decl, 0, 4)
        self.assertEqual(view, [
            ((1, 1), 0, 1, 4),
            ((1, 1), 1, 3, 9),
            ((1, 1), 3, 4, 4),
        ])
        store.close()

    # -- oracle-driven random/property cases -----------------------------
    def _oracle_drive(self, c: MCase, batches: List[List[Dict]],
                      lo=0, hi=20) -> None:
        decl = build(c)
        store = reljoin.open_maintain_store(c.maintain, decl)
        bags = maintain_oracle.load_initial_bags(decl, c.data)
        old_view = maintain_oracle.full_view(decl, bags, lo, hi)
        # Initial maintained view must match the reference.
        got_initial, _ = view_rows(store, decl, lo, hi)
        self.assertEqual(sorted(got_initial), sorted(old_view))
        expect = 0
        for bi, corrections in enumerate(batches):
            path = c.corr(f"batch{bi}", corrections)
            maintain_oracle.apply_corrections_to_bags(bags, corrections)
            violation = maintain_oracle.legality_violation(decl, bags)
            if violation is not None:
                with self.assertRaises(CorrectionProblem):
                    reljoin.apply_corrections(
                        store, batch_id=f"b{bi}", expect_revision=expect,
                        correction_path=path, memory_bytes=1 << 20,
                    )
                # Roll the oracle bags back to the pre-batch legal state by
                # negating this batch's corrections.
                rollback = [
                    {**x, "delta": -x["delta"]} for x in corrections
                ]
                maintain_oracle.apply_corrections_to_bags(bags, rollback)
                continue
            result = reljoin.apply_corrections(
                store, batch_id=f"b{bi}", expect_revision=expect,
                correction_path=path, memory_bytes=1 << 20,
            )
            expect += 1
            self.assertEqual(result.receipt.revision, expect)
            new_view = maintain_oracle.full_view(decl, bags, lo, hi)
            got_view, rev = view_rows(store, decl, lo, hi)
            self.assertEqual(rev, expect)
            self.assertEqual(sorted(got_view), sorted(new_view))
            want_delta = maintain_oracle.delta_between(old_view, new_view)
            got_delta = delta_rows(store, f"b{bi}")
            self.assertEqual(
                sorted(got_delta), sorted(want_delta),
                f"batch {bi} delta mismatch",
            )
            self.assertEqual(
                result.receipt.weighted_change,
                maintain_oracle.weighted_change(want_delta),
            )
            old_view = new_view
        store.close()

    def test_empty_library_completes_loop_in_one_batch(self) -> None:
        c = triangle_case(self.tmp, "emptyloop")
        # Remove all source rows, so the initial maintained view is empty.
        for rel in ("A", "B", "C"):
            with open(os.path.join(c.data, rel + ".jsonl"), "w"):
                pass
        self._oracle_drive(c, [[
            {"relation": "A", "values": [1, 2], "since": 0, "until": 10,
             "delta": 1},
            {"relation": "B", "values": [2, 3], "since": 0, "until": 10,
             "delta": 1},
            {"relation": "C", "values": [3, 1], "since": 0, "until": 10,
             "delta": 1},
        ]])

    def test_withdraw_to_zero(self) -> None:
        c = triangle_case(self.tmp, "zero")
        self._oracle_drive(c, [[
            {"relation": "A", "values": [1, 2], "since": 0, "until": 10,
             "delta": -2},
        ]])

    def test_mutual_cancellation_net_zero(self) -> None:
        c = triangle_case(self.tmp, "cancel")
        self._oracle_drive(c, [[
            {"relation": "A", "values": [1, 2], "since": 2, "until": 8,
             "delta": 1},
            {"relation": "A", "values": [1, 2], "since": 2, "until": 8,
             "delta": -1},
        ]])

    def test_illegal_negative_whole_batch_rollback(self) -> None:
        c = triangle_case(self.tmp, "neg")
        decl = build(c)
        store = reljoin.open_maintain_store(c.maintain, decl)
        # First line dips negative only temporarily inside the batch; the last
        # line leaves a genuinely negative standing bag.
        corr = c.corr("bad", [
            {"relation": "A", "values": [1, 2], "since": 1, "until": 2,
             "delta": -1},
            {"relation": "A", "values": [1, 2], "since": 1, "until": 2,
             "delta": 1},
            {"relation": "A", "values": [1, 2], "since": 3, "until": 9,
             "delta": -5},
        ])
        with self.assertRaises(CorrectionProblem) as ctx:
            reljoin.apply_corrections(
                store, batch_id="bad", expect_revision=0,
                correction_path=corr, memory_bytes=1 << 20,
            )
        self.assertIn("A", str(ctx.exception))
        self.assertIn("[3,", str(ctx.exception))
        self.assertEqual(store.revision(), 0)
        view, rev = view_rows(store, decl, 0, 10)
        self.assertEqual(rev, 0)
        self.assertEqual(view, [((1, 2, 3), 0, 10, 24)])
        store.close()

    def test_out_of_order_and_overlapping_corrections(self) -> None:
        import random
        rng = random.Random(2026)
        c = triangle_case(self.tmp, "ooo")
        batches = []
        for _ in range(4):
            rows = []
            for rel, tup in (("A", [1, 2]), ("B", [2, 3]), ("C", [3, 1])):
                for _ in range(3):
                    s = rng.randint(0, 9)
                    u = rng.randint(s + 1, 12)
                    d = rng.choice([-1, 1, 1, 2])
                    rows.append({
                        "relation": rel, "values": tup,
                        "since": s, "until": u, "delta": d,
                    })
            rng.shuffle(rows)
            batches.append(rows)
        self._oracle_drive(c, batches)

    def test_loop_introduced_by_multi_atom_batch_has_cross_terms(self) -> None:
        # The closed loop only exists after all three relations are introduced
        # in one batch; the product 2*3*4=24 must appear in a single apply.
        c = MCase(self.tmp, "cross", ("x", "y", "z"), [
            ("A", ("x", "y")), ("B", ("y", "z")), ("C", ("z", "x")),
        ])
        for rel in ("A", "B", "C"):
            with open(os.path.join(c.data, rel + ".jsonl"), "w"):
                pass
        self._oracle_drive(c, [[
            {"relation": "A", "values": [1, 2], "since": 0, "until": 10,
             "delta": 2},
            {"relation": "B", "values": [2, 3], "since": 0, "until": 10,
             "delta": 3},
            {"relation": "C", "values": [3, 1], "since": 0, "until": 10,
             "delta": 4},
        ]])

    def test_multi_column_and_disconnected_tuples(self) -> None:
        c = MCase(self.tmp, "multi", ("x", "y", "z"), [
            ("A", ("x", "y")), ("B", ("y", "z")), ("C", ("z", "x")),
        ])
        c.load("A", [
            {"values": [1, 2], "since": 0, "until": 10, "copies": 1},
            {"values": [5, 6], "since": 0, "until": 10, "copies": 1},
        ])
        c.load("B", [
            {"values": [2, 3], "since": 0, "until": 10, "copies": 1},
            {"values": [6, 7], "since": 0, "until": 10, "copies": 1},
        ])
        c.load("C", [
            {"values": [3, 1], "since": 0, "until": 10, "copies": 1},
            {"values": [7, 5], "since": 0, "until": 10, "copies": 1},
        ])
        self._oracle_drive(c, [
            [{"relation": "A", "values": [5, 6], "since": 2, "until": 8,
              "delta": 3}],
            [{"relation": "C", "values": [3, 1], "since": 1, "until": 9,
              "delta": -1},
             {"relation": "B", "values": [6, 7], "since": 4, "until": 6,
              "delta": 2}],
        ])

    # -- idempotency and conflicts ---------------------------------------
    def test_replay_and_conflicts(self) -> None:
        c = triangle_case(self.tmp, "replay")
        decl = build(c)
        store = reljoin.open_maintain_store(c.maintain, decl)
        corr = c.corr("r1", [
            {"relation": "A", "values": [1, 2], "since": 2, "until": 8,
             "delta": 1},
        ])
        first = reljoin.apply_corrections(
            store, batch_id="r1", expect_revision=0,
            correction_path=corr, memory_bytes=1 << 20,
        )
        # Identical re-send: same receipt, no revision bump.
        replay = reljoin.apply_corrections(
            store, batch_id="r1", expect_revision=0,
            correction_path=corr, memory_bytes=1 << 20,
        )
        self.assertTrue(replay.receipt.replayed)
        self.assertEqual(replay.receipt.revision, 1)
        self.assertEqual(
            replay.receipt.weighted_change, first.receipt.weighted_change
        )
        self.assertEqual(store.revision(), 1)
        # Same id, different content -> conflict.
        other = c.corr("r2", [
            {"relation": "A", "values": [1, 2], "since": 1, "until": 2,
             "delta": 1},
        ])
        with self.assertRaises(BatchConflict):
            reljoin.apply_corrections(
                store, batch_id="r1", expect_revision=1,
                correction_path=other, memory_bytes=1 << 20,
            )
        # New batch with a stale expect-revision -> conflict.
        with self.assertRaises(BatchConflict):
            reljoin.apply_corrections(
                store, batch_id="r3", expect_revision=0,
                correction_path=other, memory_bytes=1 << 20,
            )
        store.close()

    def test_net_zero_legal_batch_still_increments(self) -> None:
        c = triangle_case(self.tmp, "netzero")
        decl = build(c)
        store = reljoin.open_maintain_store(c.maintain, decl)
        corr = c.corr("z", [
            {"relation": "A", "values": [1, 2], "since": 2, "until": 8,
             "delta": 1},
            {"relation": "A", "values": [1, 2], "since": 2, "until": 8,
             "delta": -1},
        ])
        result = reljoin.apply_corrections(
            store, batch_id="z", expect_revision=0,
            correction_path=corr, memory_bytes=1 << 20,
        )
        self.assertEqual(result.receipt.revision, 1)
        self.assertEqual(result.receipt.delta_rows, 0)
        self.assertEqual(result.receipt.weighted_change, 0)
        store.close()

    def test_export_unknown_batch(self) -> None:
        c = triangle_case(self.tmp, "unk")
        decl = build(c)
        store = reljoin.open_maintain_store(c.maintain, decl)
        with self.assertRaises(DataProblem):
            list(reljoin.stream_batch_delta(store, "nope"))
        store.close()

    def test_correction_validation(self) -> None:
        c = triangle_case(self.tmp, "validation")
        decl = build(c)
        store = reljoin.open_maintain_store(c.maintain, decl)
        bad_cases = [
            [{"relation": "Z", "values": [1], "since": 0, "until": 1,
              "delta": 1}],
            [{"relation": "A", "values": [1], "since": 0, "until": 1,
              "delta": 1}],
            [{"relation": "A", "values": [1, 2], "since": 5, "until": 5,
              "delta": 1}],
            [{"relation": "A", "values": [1, 2], "since": 0, "until": 1,
              "delta": 0}],
            [{"relation": "A", "values": [1, True], "since": 0, "until": 1,
              "delta": 1}],
            [{"relation": "A", "values": [1, 2], "since": 0, "until": 1,
              "delta": True}],
        ]
        for i, rows in enumerate(bad_cases):
            path = c.corr(f"bad{i}", rows)
            with self.assertRaises(CorrectionProblem):
                reljoin.apply_corrections(
                    store, batch_id=f"bad{i}", expect_revision=0,
                    correction_path=path, memory_bytes=1 << 20,
                )
        self.assertEqual(store.revision(), 0)
        store.close()


if __name__ == "__main__":
    unittest.main()
