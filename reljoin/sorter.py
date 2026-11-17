"""Bounded-memory external merge sort of fixed-width binary records.

Records are encoded so that plain byte-wise comparison of the key part is the
desired order; the payload (a multiplicity) is carried along untouched.  The
sorter buffers at most ``cache_bytes`` worth of records, sorts them, spills
them to a temporary run file, and finally merges the runs.

The merge is *multi-pass* with a fan-in derived from the memory budget, so the
number of simultaneously open runs and their read buffers also stay within the
budget even when the buffer can only hold a small fraction of the input.
"""
from __future__ import annotations

import heapq
import os
import tempfile
from typing import BinaryIO, Iterator, List, Optional, Tuple

from .errors import ResourceError

_READ_RECORDS = 256


class _Run:
    __slots__ = ("count", "fh", "name")

    def __init__(self, count: int, fh: BinaryIO, name: str) -> None:
        self.count = count
        self.fh = fh
        self.name = name


class ExternalSorter:
    """Sort ``bytes`` records lexicographically with a hard memory budget."""

    def __init__(self, key_width: int, payload_width: int, cache_bytes: int,
                 directory: Optional[str] = None) -> None:
        if key_width <= 0 or payload_width < 0:
            raise ValueError("invalid record widths")
        self.key_width = key_width
        self.payload_width = payload_width
        self.rec_width = key_width + payload_width
        if cache_bytes < 8 * self.rec_width + 4096:
            raise ResourceError(
                f"memory budget of {cache_bytes} bytes is too small to sort "
                f"{self.rec_width}-byte records"
            )
        self.cache_bytes = cache_bytes
        # A Python list of bytes costs 8 pointer bytes plus roughly 33 bytes
        # of bytes-object overhead on top of the payload.  Be conservative so
        # that a tight budget is actually honoured.
        per_item = self.rec_width + 48
        self.run_records = max(256, cache_bytes // per_item)
        per_reader = self.rec_width * _READ_RECORDS + 512
        self.fan_in = max(2, min(32, (cache_bytes // 4) // per_reader))
        self._buf: List[bytes] = []
        self._runs: List[_Run] = []
        self._dir = directory
        self._finished = False
        self.records_in = 0
        self.records_out = 0
        self.runs_written = 0
        self.merge_passes = 0

    # -- ingestion -------------------------------------------------------
    def add(self, key: bytes, payload: bytes = b"") -> None:
        if self._finished:
            raise RuntimeError("sorter already finished")
        if len(key) != self.key_width or len(payload) != self.payload_width:
            raise ValueError("record width mismatch")
        self._buf.append(key + payload)
        self.records_in += 1
        if len(self._buf) >= self.run_records:
            self._spill()

    def _new_run_file(self) -> Tuple[BinaryIO, str]:
        fd, name = tempfile.mkstemp(prefix="reljoin-run-", suffix=".bin",
                                    dir=self._dir)
        return os.fdopen(fd, "w+b", buffering=1 << 20), name

    def _spill(self) -> None:
        if not self._buf:
            return
        self._buf.sort()
        fh, name = self._new_run_file()
        run = _Run(len(self._buf), fh, name)
        self._runs.append(run)
        width = self.rec_width
        chunk: List[bytes] = []
        chunk_bytes = 0
        limit = max(64, min(self.cache_bytes // 2, 8 << 20))
        for rec in self._buf:
            chunk.append(rec)
            chunk_bytes += width
            if chunk_bytes >= limit:
                fh.write(b"".join(chunk))
                chunk.clear()
                chunk_bytes = 0
        if chunk:
            fh.write(b"".join(chunk))
        fh.flush()
        self.runs_written += 1
        self._buf = []

    # -- output ----------------------------------------------------------
    def finish(self) -> Iterator[bytes]:
        """Return an iterator over all records in sorted order."""
        if self._finished:
            raise RuntimeError("sorter already finished")
        self._finished = True
        if not self._runs:
            buf = self._buf
            self._buf = []
            buf.sort()
            try:
                for rec in buf:
                    self.records_out += 1
                    yield rec
            finally:
                buf.clear()
            return
        if self._buf:
            self._spill()
        try:
            self._reduce_runs()
            yield from self._merge_iter(
                [_reader(run.fh, self.rec_width) for run in self._runs]
            )
        finally:
            self._close_runs()
            self._unlink_runs()

    def _merge_iter(self, streams: List[Iterator[bytes]]) -> Iterator[bytes]:
        heap: List[Tuple[bytes, int]] = []
        for idx, stream in enumerate(streams):
            try:
                heap.append((next(stream), idx))
            except StopIteration:
                pass
        heapq.heapify(heap)
        while heap:
            rec, idx = heap[0]
            self.records_out += 1
            yield rec
            try:
                nxt = next(streams[idx])
            except StopIteration:
                heapq.heappop(heap)
            else:
                heapq.heapreplace(heap, (nxt, idx))

    def _reduce_runs(self) -> None:
        """Multi-pass merge until at most ``fan_in`` runs remain open."""
        while len(self._runs) > self.fan_in:
            self.merge_passes += 1
            group: List[_Run] = []
            new_runs: List[_Run] = []
            old = self._runs
            self._runs = []
            try:
                for run in old:
                    group.append(run)
                    if len(group) >= self.fan_in:
                        new_runs.append(self._merge_group(group))
                        group = []
                if group:
                    new_runs.append(self._merge_group(group))
            finally:
                for run in old:
                    self._close_run(run)
                self._runs = new_runs
                self._unlink_dead(old)

    def _merge_group(self, group: List[_Run]) -> _Run:
        out, name = self._new_run_file()
        count = sum(r.count for r in group)
        for rec in self._merge_iter([_reader(r.fh, self.rec_width)
                                    for r in group]):
            out.write(rec)
        out.flush()
        return _Run(count, out, name)

    def _close_run(self, run: _Run) -> None:
        try:
            run.fh.close()
        except OSError:
            pass

    def _close_runs(self) -> None:
        for run in self._runs:
            self._close_run(run)

    def _unlink_dead(self, dead: List[_Run]) -> None:
        live = {run.name for run in self._runs}
        for run in dead:
            if run.name in live:
                continue
            try:
                os.unlink(run.name)
            except OSError:
                pass

    def _unlink_runs(self) -> None:
        seen = set()
        for run in self._runs:
            if run.name in seen:
                continue
            seen.add(run.name)
            try:
                os.unlink(run.name)
            except OSError:
                pass
        self._runs = []

    def close(self) -> None:
        self._close_runs()
        self._buf = []
        for run in self._runs:
            try:
                os.unlink(run.name)
            except OSError:
                pass
        self._runs = []


def _reader(fh: BinaryIO, width: int) -> Iterator[bytes]:
    fh.seek(0)
    while True:
        data = fh.read(width * _READ_RECORDS)
        if not data:
            return
        if len(data) % width:
            raise ResourceError("temporary sort run is truncated")
        for i in range(0, len(data), width):
            yield data[i:i + width]


class VariableExternalSorter:
    """External merge sort for ``key`` + variable-width ``payload`` records.

    Identical memory contract to :class:`ExternalSorter` (bounded buffer,
    multi-pass merge, run files in ``directory``), but the payload may have any
    length (it is the temporal index's variable-length signed delta blob).
    Records are framed on disk with a 4-byte length prefix; ordering compares
    only the fixed-width key part, so equal keys keep a stable stream order.
    """

    _HDR = 4

    def __init__(self, key_width: int, cache_bytes: int,
                 directory: Optional[str] = None) -> None:
        if key_width <= 0:
            raise ValueError("invalid key width")
        self.key_width = key_width
        self.cache_bytes = cache_bytes
        # Python objects plus the framed bytes; stay deliberately conservative
        # so the memory budget is really honoured.
        per_item = key_width + 64
        self.run_records = max(128, cache_bytes // per_item)
        per_reader = (key_width + 64) * _READ_RECORDS
        self.fan_in = max(2, min(32, (cache_bytes // 4) // max(1, per_reader)))
        self._buf: List[bytes] = []
        self._runs: List[_Run] = []
        self._dir = directory
        self._finished = False
        self.records_in = 0
        self.records_out = 0
        self.runs_written = 0
        self.merge_passes = 0
        self.bytes_spilled = 0

    def add(self, key: bytes, payload: bytes = b"") -> None:
        if self._finished:
            raise RuntimeError("sorter already finished")
        if len(key) != self.key_width:
            raise ValueError("record key width mismatch")
        self._buf.append(key + payload)
        self.records_in += 1
        if len(self._buf) >= self.run_records:
            self._spill()

    @staticmethod
    def _frame(rec: bytes) -> bytes:
        if len(rec) > 0xFFFFFFFF:
            raise ResourceError("a sort record exceeds 4 GiB")
        return len(rec).to_bytes(4, "big") + rec

    def _new_run_file(self) -> Tuple[BinaryIO, str]:
        fd, name = tempfile.mkstemp(prefix="reljoin-vrun-", suffix=".bin",
                                    dir=self._dir)
        return os.fdopen(fd, "w+b", buffering=1 << 20), name

    def _spill(self) -> None:
        if not self._buf:
            return
        kw = self.key_width
        self._buf.sort(key=lambda r: r[:kw])
        fh, name = self._new_run_file()
        run = _Run(len(self._buf), fh, name)
        self._runs.append(run)
        chunk: List[bytes] = []
        chunk_bytes = 0
        limit = max(64, min(self.cache_bytes // 2, 8 << 20))
        for rec in self._buf:
            frame = self._frame(rec)
            chunk.append(frame)
            chunk_bytes += len(frame)
            if chunk_bytes >= limit:
                fh.write(b"".join(chunk))
                self.bytes_spilled += chunk_bytes
                chunk.clear()
                chunk_bytes = 0
        if chunk:
            fh.write(b"".join(chunk))
            self.bytes_spilled += chunk_bytes
        fh.flush()
        self.runs_written += 1
        self._buf = []

    def finish(self) -> Iterator[bytes]:
        """Yield full records (key + payload) in key order."""
        if self._finished:
            raise RuntimeError("sorter already finished")
        self._finished = True
        if not self._runs:
            buf = self._buf
            self._buf = []
            buf.sort(key=lambda r: r[:self.key_width])
            try:
                for rec in buf:
                    self.records_out += 1
                    yield rec
            finally:
                buf.clear()
            return
        if self._buf:
            self._spill()
        try:
            self._reduce_runs()
            yield from self._merge_iter(
                [_vreader(run.fh, self.key_width) for run in self._runs]
            )
        finally:
            self._close_runs()
            self._unlink_runs()

    def _merge_iter(self, streams: List[Iterator[bytes]]) -> Iterator[bytes]:
        kw = self.key_width
        heap: List[Tuple[bytes, int]] = []
        for idx, stream in enumerate(streams):
            try:
                heap.append((next(stream), idx))
            except StopIteration:
                pass
        heapq.heapify(heap)
        while heap:
            rec, idx = heap[0]
            self.records_out += 1
            yield rec
            try:
                nxt = next(streams[idx])
            except StopIteration:
                heapq.heappop(heap)
            else:
                # Order by key only; payloads of equal keys stay in run order.
                heapq.heapreplace(heap, (nxt, idx))

    def _reduce_runs(self) -> None:
        while len(self._runs) > self.fan_in:
            self.merge_passes += 1
            group: List[_Run] = []
            new_runs: List[_Run] = []
            old = self._runs
            self._runs = []
            try:
                for run in old:
                    group.append(run)
                    if len(group) >= self.fan_in:
                        new_runs.append(self._merge_group(group))
                        group = []
                if group:
                    new_runs.append(self._merge_group(group))
            finally:
                for run in old:
                    self._close_run(run)
                self._runs = new_runs
                self._unlink_dead(old)

    def _merge_group(self, group: List[_Run]) -> _Run:
        out, name = self._new_run_file()
        count = sum(r.count for r in group)
        written = 0
        for rec in self._merge_iter([_vreader(r.fh, self.key_width)
                                    for r in group]):
            frame = self._frame(rec)
            out.write(frame)
            written += len(frame)
        out.flush()
        self.bytes_spilled += written
        return _Run(count, out, name)

    def _close_run(self, run: _Run) -> None:
        try:
            run.fh.close()
        except OSError:
            pass

    def _close_runs(self) -> None:
        for run in self._runs:
            self._close_run(run)

    def _unlink_dead(self, dead: List[_Run]) -> None:
        live = {run.name for run in self._runs}
        for run in dead:
            if run.name in live:
                continue
            try:
                os.unlink(run.name)
            except OSError:
                pass

    def _unlink_runs(self) -> None:
        seen = set()
        for run in self._runs:
            if run.name in seen:
                continue
            seen.add(run.name)
            try:
                os.unlink(run.name)
            except OSError:
                pass
        self._runs = []

    def close(self) -> None:
        self._close_runs()
        self._buf = []
        for run in self._runs:
            try:
                os.unlink(run.name)
            except OSError:
                pass
        self._runs = []


def _vreader(fh: BinaryIO, key_width: int) -> Iterator[bytes]:
    fh.seek(0)
    while True:
        hdr = fh.read(VariableExternalSorter._HDR)
        if not hdr:
            return
        if len(hdr) != VariableExternalSorter._HDR:
            raise ResourceError("temporary sort run is truncated")
        length = int.from_bytes(hdr, "big")
        body = fh.read(length)
        if len(body) != length:
            raise ResourceError("temporary sort run is truncated")
        if len(body) < key_width:
            raise ResourceError("temporary sort record is too short")
        yield body
