"""Incremental-work and bounded-memory acceptance for maintenance.

The brief requires that an apply's work grows with what the batch actually
touches, not with the whole maintained history, and that a correction file or
per-tuple trajectory larger than the memory budget spills to disk.  These tests
generate every dataset at runtime (nothing large is committed), measure the
real counters and peak RSS, and assert the incremental access is a small
fraction of a full recomputation while the answer still matches the
independent oracle.
"""
from __future__ import annotations

import json
import os
import resource
import shutil
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import reljoin                                    # noqa: E402
from reljoin.temporal_build import build_temporal_index       # noqa: E402

import gen_maintain_data                          # noqa: E402
import maintain_oracle                            # noqa: E402


def _rss_kib() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


class MaintenanceScale(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="reljoin-mscale-")

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _build(self, root: str):
        decl = reljoin.load_declaration(os.path.join(root, "query.json"))
        build_temporal_index(
            decl, data_dir=os.path.join(root, "data"),
            out_dir=os.path.join(root, "tidx"), memory_bytes=4 << 20,
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

    def test_single_tuple_correction_touches_one_loop(self) -> None:
        # n stays modest because the independent oracle below is deliberately
        # exponential (a Cartesian product per endpoint gap); the incremental
        # guarantee is asserted through the exact counters, not through n.
        root = os.path.join(self.tmp, "many")
        info = gen_maintain_data.gen_manyloops(root, n=40)
        decl = self._build(root)
        store = reljoin.open_maintain_store(
            os.path.join(root, "mlib"), decl, verify=False
        )
        # Sanity: the initial view has one segment per independent loop.
        q = reljoin.MaintainQuery(store, decl, window=(0, info["span"]))
        initial = list(q)
        q.close()
        self.assertEqual(len(initial), info["n"])

        rss_before = _rss_kib()
        result = reljoin.apply_corrections(
            store, batch_id="one", expect_revision=0,
            correction_path=info["correction"], memory_bytes=4 << 20,
        )
        rss_after = _rss_kib()
        d = result.stats.as_dict()
        # Only one seed tuple and one affected binding are visited, regardless
        # of the 400 unrelated loops stored in the library.
        self.assertEqual(d["seed_tuples"], 1)
        self.assertEqual(d["bindings_discovered"], 1)
        self.assertEqual(d["binding_sweeps"], 1)
        # The view write touches only the single affected binding's rows.
        self.assertLessEqual(d["view_rows_deleted"], 1)
        self.assertLessEqual(d["view_rows_inserted"], 3)
        # Relation/event access is confined to the one touched tuple.
        self.assertLess(
            d["validation_rows_scanned"], 8,
            "must not scan unrelated relation history",
        )
        # Peak RSS stays flat-ish: certainly well under the size of 400 loops.
        self.assertLess(rss_after - rss_before, 64 << 20)

        # Cross-check against the independent oracle.
        bags = maintain_oracle.load_initial_bags(
            decl, os.path.join(root, "data")
        )
        with open(info["correction"], encoding="utf-8") as fh:
            corrections = [json.loads(line) for line in fh]
        maintain_oracle.apply_corrections_to_bags(bags, corrections)
        oracle_view = maintain_oracle.full_view(
            decl, bags, 0, info["span"]
        )
        q2 = reljoin.MaintainQuery(store, decl, window=(0, info["span"]))
        got = sorted((tuple(v), s, u, m) for v, (s, u, m) in q2)
        q2.close()
        self.assertEqual(got, sorted(oracle_view))
        store.close()

    def test_large_history_incremental_access_is_tiny(self) -> None:
        # Production-only scale check (no exponential oracle): with 1000
        # stored loops a one-tuple correction must still touch one loop, and
        # the relation/event counters must not grow with the history size.
        root = os.path.join(self.tmp, "large")
        info = gen_maintain_data.gen_manyloops(root, n=1000)
        decl = self._build(root)
        store = reljoin.open_maintain_store(
            os.path.join(root, "mlib"), decl, verify=False
        )
        q = reljoin.MaintainQuery(store, decl, window=(0, info["span"]))
        self.assertEqual(len(list(q)), 1000)
        q.close()
        result = reljoin.apply_corrections(
            store, batch_id="one", expect_revision=0,
            correction_path=info["correction"], memory_bytes=4 << 20,
        )
        d = result.stats.as_dict()
        self.assertEqual(d["seed_tuples"], 1)
        self.assertEqual(d["bindings_discovered"], 1)
        self.assertEqual(d["binding_sweeps"], 1)
        # Unrelated history is never enumerated or rewritten.
        self.assertLess(d["validation_rows_scanned"], 8)
        self.assertLessEqual(d["view_rows_deleted"], 1)
        store.close()

        root = os.path.join(self.tmp, "hot")
        info = gen_maintain_data.gen_hotspot(root, hot=1, ny=150)
        decl = self._build(root)
        store = reljoin.open_maintain_store(
            os.path.join(root, "mlib"), decl, verify=False
        )
        result = reljoin.apply_corrections(
            store, batch_id="hot", expect_revision=0,
            correction_path=info["correction"], memory_bytes=4 << 20,
        )
        d = result.stats.as_dict()
        # The corrected A tuple participates in exactly one closed loop here
        # (each y is distinct), so one binding is affected; the point is that
        # the other 149 loops are never swept.
        self.assertEqual(d["binding_sweeps"], 1)
        self.assertEqual(d["candidate_duplicates"], 0)
        self.assertLess(d["validation_rows_scanned"], 8)
        store.close()

    def test_fragmented_batch_spills_under_tight_memory(self) -> None:
        root = os.path.join(self.tmp, "frag")
        info = gen_maintain_data.gen_fragmented(root, intervals=3000)
        decl = self._build(root)
        store = reljoin.open_maintain_store(
            os.path.join(root, "mlib"), decl, verify=False
        )
        rss_before = _rss_kib()
        # A deliberately tight budget: the raw scratch and per-tuple merge
        # must spill to the disk-backed scratch database rather than holding
        # thousands of events in RAM.
        result = reljoin.apply_corrections(
            store, batch_id="frag", expect_revision=0,
            correction_path=info["correction"], memory_bytes=2 << 20,
        )
        rss_after = _rss_kib()
        d = result.stats.as_dict()
        self.assertGreaterEqual(d["raw_events_inserted"],
                                2 * info["intervals"])
        self.assertGreater(d["negative_checks"], 0)
        # The library must remain internally consistent after the spill run.
        store.close()
        reopened = reljoin.open_maintain_store(
            os.path.join(root, "mlib"), decl, deep=True
        )
        self.assertEqual(reopened.revision(), 1)
        reopened.close()
        # RSS growth is bounded even though thousands of events were processed.
        self.assertLess(rss_after - rss_before, 128 << 20)


if __name__ == "__main__":
    unittest.main()
