"""Verify the temporal engine only issues primary-key index work.

The candidate phase must use the same ordered ``LIMIT 1`` primary-key seeks as
the static engine (no table scans, no temp B-trees), and the time phase must
read each tuple's events through a bounded primary-key range — never a full
scan of the event table and never a SQL-side join.
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

import reljoin                                    # noqa: E402
from reljoin.temporal_build import build_temporal_index       # noqa: E402
from reljoin.temporal_engine import TemporalSolver            # noqa: E402
from reljoin.temporal_open import open_temporal_index         # noqa: E402

from test_temporal_acceptance import TCase, triangle_case     # noqa: E402


class TemporalPlans(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="reljoin-tplan-")
        c = TCase(self.tmp, "p")
        c.declare(["a", "b", "c", "d"], [
            ("R", ["d", "b"]), ("S", ["c", "b"]),
            ("T", ["a", "d"]), ("U", ["a", "c"]),
            ("R", ["a", "b"]),
        ])
        for name, arity in (("R", 2), ("S", 2), ("T", 2), ("U", 2)):
            rows = [
                {"values": [i, i + 1], "since": -2 * i, "until": 2 * i + 3,
                 "copies": 1 + (i % 2)}
                for i in range(40)
            ]
            c.load(name, rows)
        self.case = c
        decl = reljoin.load_declaration(c.decl_path)
        build_temporal_index(decl, data_dir=c.data, out_dir=c.index,
                             memory_bytes=1 << 20)
        self.decl = decl
        self.idx = open_temporal_index(c.index, expected_declaration=decl)

    def tearDown(self) -> None:
        self.idx.close()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_candidate_seeks_use_primary_key(self) -> None:
        solver = TemporalSolver(self.decl, self.idx, window=(-100, 100))
        checked = 0
        for ex in solver.atoms:
            for sql in ex.sql_first + ex.sql_ge + ex.sql_gt:
                if sql.count("?") == 0:
                    self.assertIn("LIMIT 1", sql)
                    plan = ex.conn.execute(
                        "EXPLAIN QUERY PLAN " + sql, ()
                    ).fetchall()
                    text = " ".join(str(r[-1]) for r in plan)
                    self.assertNotIn("TEMP B-TREE", text.upper())
                    continue
                for bound in (-(1 << 63), (1 << 63) - 1):
                    params = (bound,) * sql.count("?")
                    plan = ex.conn.execute(
                        "EXPLAIN QUERY PLAN " + sql, params
                    ).fetchall()
                    text = " ".join(str(r[-1]) for r in plan)
                    self.assertIn("SEARCH", text, msg=sql)
                    self.assertIn("PRIMARY KEY", text, msg=sql)
                    self.assertNotIn("TEMP B-TREE", text.upper(), msg=sql)
                    self.assertNotIn("SCAN t", text, msg=sql)
                    checked += 1
        self.assertGreater(checked, 10)

    def test_event_range_is_a_clustered_pk_range_not_a_scan(self) -> None:
        solver = TemporalSolver(self.decl, self.idx, window=(-100, 100))
        # Build the SQL the sweep would use for one atom and check its plan.
        ex = solver.atoms[0]
        sql = solver._event_sql_for(ex)
        params = tuple(0 for _ in ex.key_levels) + (100,)
        plan = ex.conn.execute(
            "EXPLAIN QUERY PLAN " + sql, params
        ).fetchall()
        text = " ".join(str(r[-1]) for r in plan)
        self.assertIn("SEARCH", text, msg=text)
        self.assertIn("PRIMARY KEY", text, msg=text)
        self.assertNotIn("SCAN t", text, msg=text)
        self.assertNotIn("TEMP B-TREE", text.upper(), msg=text)
        # The sweep must never hand SQL a join: exactly one table in the text.
        self.assertNotIn("JOIN", text.upper())

    def test_sweep_reads_only_events_below_the_upper_bound(self) -> None:
        # Use the hand triangle so at least one complete binding exists.
        hand = TCase(self.tmp, "hand")
        triangle_case(hand)
        hand.load("A", [
            {"values": [1, 2], "since": 0, "until": 8, "copies": 2},
            {"values": [1, 2], "since": 3, "until": 6, "copies": 1},
        ])
        hand.load("B", [{"values": [2, 3], "since": 1, "until": 7,
                         "copies": 3}])
        hand.load("C", [
            {"values": [3, 1], "since": 2, "until": 5, "copies": 4},
            {"values": [3, 1], "since": 5, "until": 9, "copies": 4},
        ])
        decl = reljoin.load_declaration(hand.decl_path)
        build_temporal_index(decl, data_dir=hand.data, out_dir=hand.index,
                             memory_bytes=1 << 20)
        idx = open_temporal_index(hand.index, expected_declaration=decl)
        try:
            solver = TemporalSolver(decl, idx, window=(0, 5))
            before = solver.stats.event_rows_read
            segments = list(solver.leaves_with_segments())
            self.assertGreater(solver.stats.event_rows_read, before)
            self.assertGreater(len(segments), 0)
            # Every segment emitted by the sweep lies inside the window.
            for _values, (a, b, _m) in segments:
                self.assertGreaterEqual(a, 0)
                self.assertLessEqual(b, 5)
                self.assertLess(a, b)
        finally:
            idx.close()


class TemporalAntiExplosion(unittest.TestCase):
    """Structural and scale observations on generated temporal data."""

    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="reljoin-tanti-")

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _run(self, case_dir: str, lo: int, hi: int, memory: int = 2 << 20):
        decl_path = os.path.join(case_dir, "query.json")
        index = os.path.join(case_dir, "idx")
        if not os.path.exists(index):
            decl = reljoin.load_declaration(decl_path)
            build_temporal_index(
                decl, data_dir=os.path.join(case_dir, "data"),
                out_dir=index, memory_bytes=memory,
            )
        decl = reljoin.load_declaration(decl_path)
        idx = open_temporal_index(index, expected_declaration=decl)
        q = reljoin.temporal_query(idx, decl, window=(lo, hi),
                                   memory_bytes=memory)
        out = list(q)
        q.close()
        return out, q

    def test_fragmented_trajectory_does_not_cache_everything(self) -> None:
        import gen_temporal_data
        d = gen_temporal_data.gen_frag(self.tmp, n_intervals=8000)
        # A deliberately tiny budget: the trajectory is far larger.
        out, q = self._run(d, 0, 2 * 8000, memory=1 << 20)
        stats = q.report()
        self.assertEqual(stats["status"], "complete")
        # Every interval contributes to the single binding; the output is a
        # sequence of short segments, never one giant buffered structure.
        self.assertGreater(len(out), 100)
        # Event rows are streamed in batches; the batch count is proportional
        # to the rows read while the fetch buffer stays small.
        self.assertGreater(stats["event_batches"], 1)
        self.assertLessEqual(stats["max_live_atoms"], 2)
        # Work is driven by actual events, not by the time span: ~48k events
        # for 24k interval rows (3 relations), orders of magnitude below the
        # 16k-wide integer timeline being scanned point by point.
        self.assertLess(stats["event_rows_read"], 6 * 8000)
        # Flat memory: no result accumulation before emission.
        self.assertLess(stats["peak_rss_kib"], 250 * 1024)

    def test_many_short_loops_order_merge_and_weighted_duration(self) -> None:
        import gen_temporal_data
        import temporal_oracle
        n = 1500
        d = gen_temporal_data.gen_many_loops(self.tmp, n=n)
        lo, hi = 0, n * 7 + 3
        out, q = self._run(d, lo, hi, memory=2 << 20)
        stats = q.report()
        self.assertEqual(stats["status"], "complete")
        # One segment per loop, ordered by binding then since.
        self.assertEqual(len(out), n)
        keys = [(seg[0], seg[1][0]) for seg in out]
        self.assertEqual(keys, sorted(keys))
        for values, (a, b, m) in out:
            self.assertEqual(b - a, 3)
            self.assertGreater(m, 0)
        decl = reljoin.load_declaration(os.path.join(d, "query.json"))
        want = temporal_oracle.temporal_brute_force(
            decl, os.path.join(d, "data"), lo, hi
        )
        self.assertEqual(
            [(v, a, b, m) for v, (a, b, m) in out], want
        )
        self.assertEqual(
            stats["weighted_duration"],
            temporal_oracle.weighted_duration(want),
        )

    def test_span_does_not_change_the_work(self) -> None:
        import gen_temporal_data
        d1 = gen_temporal_data.gen_span_pair(self.tmp, scale=1)
        d2 = gen_temporal_data.gen_span_pair(self.tmp, scale=10 ** 14)
        out1, q1 = self._run(d1, 0, 10)
        # The second window spans 10**15 ticks with the same endpoints.
        out2, q2 = self._run(d2, 0, 10 ** 15)
        s1, s2 = q1.report(), q2.report()
        # Identical event work even though the span grows by 10**14.
        self.assertEqual(s1["event_rows_read"], s2["event_rows_read"])
        self.assertEqual(s1["sweep_boundaries"], s2["sweep_boundaries"])
        self.assertEqual(s1["emitted_segments"], s2["emitted_segments"])
        # Segment shape identical after rescaling; weighted duration scales.
        self.assertEqual(len(out1), len(out2))
        for (v1, (a1, b1, m1)), (v2, (a2, b2, m2)) in zip(out1, out2):
            self.assertEqual(v1, v2)
            self.assertEqual(m1, m2)
            self.assertEqual(a2, a1 * 10 ** 14)
            self.assertEqual(b2, b1 * 10 ** 14)
        self.assertEqual(
            s2["weighted_duration"], s1["weighted_duration"] * 10 ** 14
        )

    def test_external_sort_spills_under_tiny_budget(self) -> None:
        import gen_temporal_data
        d = gen_temporal_data.gen_frag(self.tmp, n_intervals=12000)
        decl = reljoin.load_declaration(os.path.join(d, "query.json"))
        manifest = build_temporal_index(
            decl, data_dir=os.path.join(d, "data"),
            out_dir=os.path.join(d, "idx2"), memory_bytes=1 << 20,
        )
        self.assertGreater(manifest["build"]["sort_bytes_spilled"], 0)
        self.assertGreaterEqual(manifest["build"]["sort_runs_written"], 1)
        # The spilled build still answers exactly like the reference.
        import temporal_oracle
        idx = open_temporal_index(os.path.join(d, "idx2"),
                                  expected_declaration=decl)
        q = reljoin.temporal_query(idx, decl, window=(0, 6000),
                                   memory_bytes=1 << 20)
        got = [(v, a, b, m) for v, (a, b, m) in q]
        q.close()
        want = temporal_oracle.temporal_brute_force(
            decl, os.path.join(d, "data"), 0, 6000
        )
        self.assertEqual(got, want)


if __name__ == "__main__":
    unittest.main()
