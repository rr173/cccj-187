"""Read-side maintenance operations: view queries and batch export.

Both reads stream from already-committed tables only — they never recompute
the join from the relation state, so a ``maintain-run`` cannot masquerade as
maintenance by rebuilding the answer.  Each read pins one SQLite snapshot for
its whole lifetime: a revision read and the row cursor run inside the same
deferred transaction, hence a commit landing while an older reader is still
draining cannot mix two revisions into one stream.
"""
from __future__ import annotations

import sqlite3
import time
from typing import Any, Dict, Iterator, List, Optional, Tuple

from .declaration import Declaration
from .engine import _peak_rss_kib
from .errors import CorruptIndex, QueryAborted
from .maintain_format import decode_int_blob
from .maintain_solver import MaintStats
from .maintain_store import MaintainStore

_FETCH = 512


class MaintainQuery:
    """Streaming cursor over the committed view at one fixed revision."""

    def __init__(self, store: MaintainStore,
                 decl: Optional[Declaration] = None, *,
                 window: Tuple[int, int],
                 memory_bytes: Optional[int] = None,
                 limit: Optional[int] = None,
                 cancel=None) -> None:
        if limit is not None and limit < 0:
            raise ValueError("limit must be >= 0")
        lo, hi = window
        if lo >= hi:
            raise ValueError(
                f"window start ({lo}) must be strictly below end ({hi})"
            )
        self.store = store
        self.decl = decl or store.decl
        self.window = (lo, hi)
        self.limit = limit
        self.cancel = cancel
        self.depth = len(self.decl.order)
        self.stats = MaintStats()
        self.stats.window_from = lo
        self.stats.window_to = hi
        self._conn: Optional[sqlite3.Connection] = None
        self._cur: Optional[sqlite3.Cursor] = None
        self._gen: Optional[Iterator] = None
        self._closed = False
        self._exhausted = False
        self._finalized = False
        self.revision = -1
        if memory_bytes is not None:
            self._cache_kib = max(64, memory_bytes // 1024 // 4)
        else:
            self._cache_kib = 2048

    def _open(self) -> None:
        conn = self.store.connect(readonly=True)
        self._conn = conn
        conn.execute(f"PRAGMA cache_size=-{self._cache_kib}")
        # Pin one snapshot: the revision and every row come from the same
        # read transaction.
        conn.execute("BEGIN")
        row = conn.execute(
            "SELECT value FROM meta WHERE key='revision'"
        ).fetchone()
        if row is None:
            raise CorruptIndex("maintenance database has no revision row")
        self.revision = int(row[0])
        bcols = ", ".join(f"b{i}" for i in range(self.depth))
        lo, hi = self.window
        self._cur = conn.execute(
            f"SELECT {bcols}, since, until, multiplicity FROM view "
            f"WHERE until>? AND since<? ORDER BY {bcols}, since",
            (lo, hi),
        )

    def __iter__(self):
        if self._closed:
            raise QueryAborted("maintenance query is already closed")
        if self._gen is not None:
            raise QueryAborted("maintenance query was already consumed once")
        self._open()
        self._gen = self._iter()
        return self._gen

    def __next__(self):
        if self._gen is None:
            raise QueryAborted(
                "maintenance query has not been started; call iter(query)"
            )
        return next(self._gen)

    def _clip(self, since: int, until: int, mult: int,
              values: Tuple[int, ...]):
        lo, hi = self.window
        s, u = max(since, lo), min(until, hi)
        if s < u:
            return values, (s, u, mult)
        return None

    def _iter(self):
        lo, hi = self.window
        try:
            while True:
                if self.cancel is not None:
                    try:
                        stop = bool(self.cancel())
                    except Exception:
                        stop = True
                    if stop:
                        self.stats.status = "incomplete"
                        self.stats.status_reason = "cancelled by the caller"
                        self._finish()
                        raise QueryAborted("query cancelled by the caller")
                if self.limit is not None and self.stats.emitted >= self.limit:
                    self.stats.status = "limit-reached"
                    self.stats.status_reason = (
                        f"stopped after the limit of {self.limit} segments"
                    )
                    self._finish()
                    return
                rows = self._cur.fetchmany(_FETCH)
                if not rows:
                    break
                for row in rows:
                    values = tuple(row[:self.depth])
                    since, until = row[self.depth], row[self.depth + 1]
                    mult = decode_int_blob(row[self.depth + 2])
                    clipped = self._clip(since, until, mult, values)
                    if clipped is None:
                        continue
                    self.stats.emitted_segments += 1
                    self.stats.emitted += 1
                    self.stats.weighted_duration += (
                        clipped[1][1] - clipped[1][0]
                    ) * mult
                    yield clipped
            self._exhausted = True
            self.stats.status = "complete"
            self.stats.status_reason = (
                f"committed view at revision {self.revision} fully streamed"
            )
        except GeneratorExit:
            self.stats.status = "incomplete"
            self.stats.status_reason = "consumer closed the iterator early"
            raise
        finally:
            self._finish()

    def _finish(self) -> None:
        if self._finalized:
            return
        self._finalized = True
        self.stats.elapsed = time.time() - self.stats.started_at
        self.stats.peak_rss_kib = _peak_rss_kib()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        gen = self._gen
        if gen is not None and not self._exhausted:
            try:
                gen.close()
            except Exception:
                pass
        if self.stats.status == "running":
            self.stats.status = "incomplete"
            self.stats.status_reason = "query closed by caller"
        self._finish()
        if self._conn is not None:
            try:
                if self._exhausted:
                    self._conn.execute("COMMIT")
                else:
                    self._conn.execute("ROLLBACK")
            except sqlite3.DatabaseError:
                pass
            try:
                self._conn.close()
            except sqlite3.DatabaseError:
                pass

    def __enter__(self) -> "MaintainQuery":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def report(self, *, include_layers: bool = True) -> Dict[str, Any]:
        return self.stats.as_dict(
            self.decl if include_layers else None
        )


def stream_batch_delta(
    store: MaintainStore, batch_id: str
) -> Iterator[Dict[str, Any]]:
    """Yield the stored signed delta rows of one committed batch, in order.

    Raises :class:`BatchMissing` (a :class:`DataProblem`) if the batch id was
    never committed.  The receipt row is read first inside the same snapshot.
    """
    from .errors import DataProblem
    depth = len(store.decl.order)
    conn = store.connect(readonly=True)
    cur = None
    try:
        conn.execute("BEGIN")
        rec = conn.execute(
            "SELECT revision, delta_rows FROM batches WHERE batch_id=?",
            (batch_id,),
        ).fetchone()
        if rec is None:
            conn.execute("ROLLBACK")
            raise DataProblem(
                f"batch-id {batch_id!r} is not present in the maintenance "
                "library; nothing to export"
            )
        revision, expected_rows = rec
        bcols = ", ".join(f"b{i}" for i in range(depth))
        cur = conn.execute(
            f"SELECT {bcols}, since, until, delta_multiplicity, seq "
            f"FROM batch_delta WHERE batch_id=? ORDER BY seq",
            (batch_id,),
        )
        seen = 0
        while True:
            rows = cur.fetchmany(_FETCH)
            if not rows:
                break
            for row in rows:
                seen += 1
                yield {
                    "batch_id": batch_id,
                    "revision": revision,
                    "seq": row[depth + 3],
                    "values": list(row[:depth]),
                    "since": row[depth],
                    "until": row[depth + 1],
                    "delta_multiplicity": decode_int_blob(row[depth + 2]),
                }
        if seen != expected_rows:
            raise CorruptIndex(
                f"batch {batch_id!r} receipt claims {expected_rows} delta "
                f"rows but {seen} are stored"
            )
        conn.execute("COMMIT")
    except BaseException:
        try:
            conn.execute("ROLLBACK")
        except sqlite3.DatabaseError:
            pass
        raise
    finally:
        if cur is not None:
            try:
                cur.close()
            except sqlite3.DatabaseError:
                pass
        conn.close()


def fetch_receipt(store: MaintainStore, batch_id: str) -> Dict[str, Any]:
    """Return the stored receipt for a committed batch id."""
    from .errors import DataProblem
    conn = store.connect(readonly=True)
    try:
        row = conn.execute(
            "SELECT batch_id, revision, expect_revision, file_sha256, "
            "file_bytes, records, delta_rows, weighted_change, committed_at "
            "FROM batches WHERE batch_id=?",
            (batch_id,),
        ).fetchone()
        if row is None:
            raise DataProblem(
                f"batch-id {batch_id!r} is not present in the maintenance "
                "library"
            )
        return {
            "batch_id": row[0],
            "revision": row[1],
            "expect_revision": row[2],
            "file_sha256": row[3],
            "file_bytes": row[4],
            "records": row[5],
            "delta_rows": row[6],
            "weighted_change": int(row[7]),
            "committed_at": row[8],
        }
    finally:
        conn.close()


def list_batches(store: MaintainStore) -> List[Dict[str, Any]]:
    conn = store.connect(readonly=True)
    try:
        cur = conn.execute(
            "SELECT batch_id, revision, expect_revision, delta_rows, "
            "weighted_change FROM batches ORDER BY revision, batch_id"
        )
        return [
            {
                "batch_id": r[0],
                "revision": r[1],
                "expect_revision": r[2],
                "delta_rows": r[3],
                "weighted_change": int(r[4]),
            }
            for r in cur.fetchall()
        ]
    finally:
        conn.close()
