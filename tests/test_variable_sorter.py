"""Unit tests for the variable-width external merge sort.

The temporal builder relies on :class:`VariableExternalSorter` for bounded
memory; these tests force the multi-pass merge path with tiny synthetic
records instead of giant data, and verify framing, ordering and payload
integrity across passes.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from reljoin.sorter import VariableExternalSorter   # noqa: E402
from reljoin.temporal_format import decode_delta, encode_delta  # noqa: E402
from reljoin.jsonl import encode_int                 # noqa: E402


class VariableSorterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="reljoin-sort-")

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_multipass_merge_preserves_order_and_payloads(self) -> None:
        kw = 8
        # Smallest legal-ish buffer: 128 records per run, fan-in 2, so a few
        # hundred records force at least one merge pass.
        sorter = VariableExternalSorter(
            key_width=kw, cache_bytes=8 * kw + 4096, directory=self.tmp,
        )
        self.assertEqual(sorter.fan_in, 2)
        n = 1000
        for i in range(n):
            # Keys out of order; payloads are variable-width delta blobs,
            # including wide arbitrary-precision values.
            key = encode_int(((i * 37) % n) - 500)
            delta = (1 << (8 * (1 + (i % 20)))) * (1 if i % 2 else -1)
            sorter.add(key, encode_delta(delta))
        records = list(sorter.finish())
        self.assertEqual(len(records), n)
        self.assertGreaterEqual(sorter.merge_passes, 1)
        self.assertGreater(sorter.runs_written, sorter.fan_in)
        # Keys globally ordered, payloads intact.
        expected = {}
        for i in range(n):
            k = ((i * 37) % n) - 500
            delta = (1 << (8 * (1 + (i % 20)))) * (1 if i % 2 else -1)
            expected.setdefault(k, []).append(delta)
        from collections import defaultdict
        got = defaultdict(list)
        from reljoin.jsonl import decode_int
        prev = None
        for rec in records:
            key = rec[:kw]
            if prev is not None:
                self.assertLessEqual(prev, key)
            prev = key
            got[decode_int(key)].append(decode_delta(rec[kw:]))
        self.assertEqual({k: sorted(v) for k, v in got.items()},
                         {k: sorted(v) for k, v in expected.items()})

    def test_equal_keys_keep_all_records_and_round_trip_deltas(self) -> None:
        kw = 16
        sorter = VariableExternalSorter(
            key_width=kw, cache_bytes=8 * kw + 4096, directory=self.tmp,
        )
        key = encode_int(7) + encode_int(-9)
        values = [1, 5, -3, 10 ** 40, -(10 ** 50), 2, 2]
        for v in values:
            sorter.add(key, encode_delta(v))
        # Plus a different key to avoid a single-group stream.
        other = encode_int(0)  # second 8-byte column, smallest monotone value
        for j in range(300):
            sorter.add(encode_int(1000 + j) + other, encode_delta(j + 1))
        out = list(sorter.finish())
        first_group = [decode_delta(r[kw:]) for r in out
                       if r[:kw] == key]
        # Equal-key rows are never dropped or summed inside the sorter; the
        # builder owns coalescing.
        self.assertEqual(sorted(first_group), sorted(values))
        self.assertEqual(len(out), 300 + len(values))
        self.assertEqual(sum(first_group), sum(values))

    def test_close_removes_run_files(self) -> None:
        kw = 8
        sorter = VariableExternalSorter(
            key_width=kw, cache_bytes=8 * kw + 4096, directory=self.tmp,
        )
        for i in range(400):
            sorter.add(encode_int(i), encode_delta(i + 1))
        self.assertGreater(sorter.runs_written, 0)
        sorter.close()
        leftovers = [n for n in os.listdir(self.tmp)
                     if n.startswith("reljoin-vrun-")]
        self.assertEqual(leftovers, [])

    def test_delta_codec_round_trip(self) -> None:
        for v in (0, 1, -1, 2 ** 63, -(2 ** 63), 10 ** 100, -(10 ** 100)):
            self.assertEqual(decode_delta(encode_delta(v)), v)
        # Tag byte correctly encodes magnitude width up to the format limit.
        blob = encode_delta(10 ** 100)
        self.assertEqual(blob[0] & 0x7F, len(blob) - 1)


if __name__ == "__main__":
    unittest.main()
