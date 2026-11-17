"""Valid-time closed-loop query engine.

The engine never re-runs the static query once per integer time point and
never builds the Cartesian product of the atoms' intervals.  It splits the
work into two parts:

1. **Tuple candidates** are found exactly as in the static engine: a layered
   leapfrog over the event tables' primary keys (tuple columns in global
   order), using ordered index seeks.  A tuple that appears in the table has at
   least one event, so its existence proves that the relation holds at some
   time; *when* it holds is decided afterwards.
2. **Time** is decided once per complete binding by a k-way merge of the
   atoms' event streams for the bound tuples.  Each stream is a contiguous
   primary-key range read with a small fetch buffer (the full valid-time
   trajectory may be larger than the memory budget).  The sweep keeps one
   running multiplicity per atom and the current product; it emits a segment
   on every boundary at which the product changes, clips segments to the query
   window, and merges adjacent segments whose multiplicity did not change —
   including boundaries where two factors changed but their product stayed the
   same and boundaries caused by an unrelated tuple's events (such boundaries
   simply never occur in this binding's streams).

All multiplicities and weighted durations are Python integers: no 64-bit
wrapping and no floating point anywhere.
"""
from __future__ import annotations

import heapq
import resource
import sqlite3
import time
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from .declaration import Declaration
from .engine import MAX_LAYERS, Stats, _AtomExec, _peak_rss_kib
from .errors import CorruptIndex, IndexProblem, QueryAborted, ResourceError
from .temporal_format import decode_delta
from .temporal_open import TemporalOpenIndex

INT64_MAX = (1 << 63) - 1
INT64_MIN = -(1 << 63)


class TemporalStats(Stats):
    """Exact counters for a temporal run; static counters are inherited."""

    __slots__ = (
        "window_from", "window_to", "event_rows_read", "event_batches",
        "sweep_boundaries", "raw_segments", "merged_segments",
        "emitted_segments", "weighted_duration", "bindings_with_segments",
        "fetch_buffer_rows",
    )

    def __init__(self) -> None:
        super().__init__()
        self.window_from: Optional[int] = None
        self.window_to: Optional[int] = None
        self.event_rows_read = 0
        self.event_batches = 0
        self.sweep_boundaries = 0
        self.raw_segments = 0
        self.merged_segments = 0
        self.emitted_segments = 0
        self.weighted_duration = 0
        self.bindings_with_segments = 0
        self.fetch_buffer_rows = 0

    def as_dict(self, decl: Optional[Declaration] = None) -> Dict[str, Any]:
        d = super().as_dict(decl)
        d.update({
            "window_from": self.window_from,
            "window_to": self.window_to,
            "event_rows_read": self.event_rows_read,
            "event_batches": self.event_batches,
            "sweep_boundaries": self.sweep_boundaries,
            "raw_segments_before_merge": self.raw_segments,
            "segments_merged": self.merged_segments,
            "emitted_segments": self.emitted_segments,
            "bindings_with_segments": self.bindings_with_segments,
            "weighted_duration": self.weighted_duration,
            "fetch_buffer_rows": self.fetch_buffer_rows,
        })
        return d


class _EventStream:
    """Bounded-buffer range cursor over one tuple's delta events.

    ``tick < upper`` is pushed into the SQL predicate so events at or beyond
    the window's right edge are never fetched; rows are pulled in fixed-size
    batches so a trajectory with more intervals than the memory budget is
    still read with a constant-size buffer.
    """

    __slots__ = ("conn", "sql", "params", "batch", "stats", "_cur", "_buf",
                 "_idx", "exhausted")

    def __init__(self, conn: sqlite3.Connection, sql: str,
                 params: Tuple[Any, ...], stats: TemporalStats,
                 batch_rows: int) -> None:
        self.conn = conn
        self.sql = sql
        self.params = params
        self.stats = stats
        self.batch = batch_rows
        self._cur: Optional[sqlite3.Cursor] = None
        self._buf: List[sqlite3.Row] = []
        self._idx = 0
        self.exhausted = False

    def open(self) -> None:
        try:
            self._cur = self.conn.execute(self.sql, self.params)
        except sqlite3.DatabaseError as exc:
            raise CorruptIndex(f"temporal index read failed: {exc}") from exc

    def _refill(self) -> bool:
        try:
            rows = self._cur.fetchmany(self.batch)
        except sqlite3.DatabaseError as exc:
            raise CorruptIndex(f"temporal index read failed: {exc}") from exc
        self.stats.event_batches += 1
        if not rows:
            self.exhausted = True
            self._buf = []
            self._idx = 0
            return False
        self.stats.event_rows_read += len(rows)
        self._buf = rows
        self._idx = 0
        return True

    def head(self) -> Optional[Tuple[int, int]]:
        """Next ``(tick, delta)`` or ``None`` at exhaustion."""
        if self._idx >= len(self._buf):
            if self.exhausted or not self._refill():
                return None
        row = self._buf[self._idx]
        return row[0], decode_delta(row[1])

    def advance(self) -> None:
        self._idx += 1

    def close(self) -> None:
        if self._cur is not None:
            try:
                self._cur.close()
            except sqlite3.DatabaseError:
                pass
            self._cur = None
        self._buf = []


class TemporalSolver:
    """Layered leapfrog for candidates + per-binding temporal sweep."""

    def __init__(self, decl: Declaration, idx: TemporalOpenIndex, *,
                 stats: Optional[TemporalStats] = None,
                 memory_bytes: Optional[int] = None,
                 window: Optional[Tuple[int, int]] = None,
                 fetch_buffer_rows: int = 512) -> None:
        self.decl = decl
        self.idx = idx
        self.stats = stats or TemporalStats()
        if window is not None:
            lo, hi = window
            self.stats.window_from = lo
            self.stats.window_to = hi
        self.window = window
        self.fetch_buffer_rows = fetch_buffer_rows
        depth = len(decl.order)
        if depth > MAX_LAYERS:
            raise ResourceError(
                f"declaration has {depth} variables; the solver supports at "
                f"most {MAX_LAYERS}"
            )
        self.stats.candidates = [0] * depth
        self.atoms: List[_AtomExec] = []
        self.active_at: List[List[_AtomExec]] = [[] for _ in range(depth)]
        self.completing_at: List[List[_AtomExec]] = [[] for _ in range(depth)]
        for atom in decl.atoms:
            table = idx.table_for_atom.get(atom.alias)
            if table is None:
                raise IndexProblem(
                    f"temporal index has no table for atom {atom.alias!r}"
                )
            ex = _AtomExec(idx, table, self.stats)
            self.atoms.append(ex)
            for lv in ex.key_levels:
                self.active_at[lv].append(ex)
            self.completing_at[ex.key_levels[-1]].append(ex)
        self._event_sql: Dict[int, str] = {}
        if memory_bytes is not None:
            n_conn = max(1, len(idx.table_for_atom))
            total_kib = max(256, memory_bytes // 1024 // 2)
            cache_kib = max(64, total_kib // n_conn)
            for conn in idx.connections():
                try:
                    conn.execute(f"PRAGMA cache_size=-{cache_kib}")
                except sqlite3.DatabaseError:
                    pass

    # -- candidate walk --------------------------------------------------
    def bindings(self) -> Iterator[Tuple[int, ...]]:
        bound: List[int] = []
        try:
            yield from self._walk(0, bound)
        finally:
            bound.clear()

    def _walk(self, level: int, bound: List[int]):
        if level == len(self.decl.order):
            yield tuple(bound)
            return
        stats = self.stats
        for value in self._candidates(level, bound):
            stats.candidates[level] += 1
            bound.append(value)
            try:
                if level + 1 > stats.max_prefix_len:
                    stats.max_prefix_len = level + 1
                yield from self._walk(level + 1, bound)
            finally:
                bound.pop()

    def _candidates(self, level: int, bound: Sequence[int]) -> Iterator[int]:
        live = self.active_at[level]
        stats = self.stats
        if len(live) > stats.max_live_atoms:
            stats.max_live_atoms = len(live)
        if len(live) == 1:
            ex = live[0]
            v = ex.first(level, bound)
            while v is not None:
                yield v
                v = ex.seek_gt(level, bound, v)
            return
        cands: List[Optional[int]] = [ex.first(level, bound) for ex in live]
        n = len(live)
        while True:
            top: Optional[int] = None
            for c in cands:
                if c is None:
                    return
                if top is None or c > top:
                    top = c
            assert top is not None
            lagging = [i for i in range(n) if cands[i] < top]
            if lagging:
                for i in lagging:
                    cands[i] = live[i].seek_ge(level, bound, top)
                continue
            yield top
            for i in range(n):
                cands[i] = live[i].seek_gt(level, bound, top)

    # -- temporal sweep --------------------------------------------------
    def _event_sql_for(self, ex: _AtomExec) -> str:
        sql = self._event_sql.get(id(ex))
        if sql is not None:
            return sql
        cols = [f"c{j}" for j in ex.key_columns]
        where = " AND ".join(f"{c}=?" for c in cols)
        # Events at tick == U only end intervals inside the window; they can
        # never start or change a segment inside [L, U), so exclude them.
        sql = (
            f"SELECT tick, d FROM t WHERE {where} AND tick<? "
            f"ORDER BY tick"
        )
        self._event_sql[id(ex)] = sql
        return sql

    def segments_for(
        self, values: Sequence[int]
    ) -> Iterator[Tuple[int, int, int]]:
        """Yield clipped, maximal ``(since, until, multiplicity)`` segments."""
        assert self.window is not None
        lo, hi = self.window
        stats = self.stats
        streams: List[_EventStream] = []
        try:
            for ex in self.atoms:
                params = tuple(values[lv] for lv in ex.key_levels) + (hi,)
                st = _EventStream(
                    ex.conn, self._event_sql_for(ex), params, stats,
                    self.fetch_buffer_rows,
                )
                st.open()
                streams.append(st)

            # Initial multiplicity of every atom on [lo, first tick >= lo).
            factors: List[int] = []
            heap: List[Tuple[int, int]] = []
            for si, st in enumerate(streams):
                total = 0
                head = st.head()
                while head is not None and head[0] < lo:
                    total += head[1]
                    st.advance()
                    head = st.head()
                if total < 0:
                    raise CorruptIndex(
                        "temporal index yields a negative multiplicity; the "
                        "index is damaged"
                    )
                factors.append(total)
                if head is not None:
                    heapq.heappush(heap, (head[0], si))
            product = 1
            for f in factors:
                product *= f

            cur_start = lo
            cur_mult = product
            produced_any = False
            while heap:
                tick = heap[0][0]
                stats.sweep_boundaries += 1
                # Apply every event at this tick (possibly from several atoms)
                # before judging the new product, so simultaneous changes make
                # one boundary, not two.
                while heap and heap[0][0] == tick:
                    _, si = heapq.heappop(heap)
                    st = streams[si]
                    head = st.head()
                    assert head is not None and head[0] == tick
                    factors[si] += head[1]
                    if factors[si] < 0:
                        raise CorruptIndex(
                            "temporal index yields a negative multiplicity; "
                            "the index is damaged"
                        )
                    st.advance()
                    nxt = st.head()
                    if nxt is not None:
                        heapq.heappush(heap, (nxt[0], si))
                new_product = 1
                for f in factors:
                    new_product *= f
                if new_product != cur_mult:
                    stats.raw_segments += 1
                    if cur_mult > 0 and tick > cur_start:
                        produced_any = True
                        stats.emitted_segments += 1
                        stats.weighted_duration += (tick - cur_start) * cur_mult
                        yield cur_start, tick, cur_mult
                    elif cur_mult > 0:
                        # Zero-length candidate segment; just drop it.
                        pass
                    cur_start = tick
                    cur_mult = new_product
                else:
                    stats.merged_segments += 1
            stats.raw_segments += 1
            if cur_mult > 0 and hi > cur_start:
                produced_any = True
                stats.emitted_segments += 1
                stats.weighted_duration += (hi - cur_start) * cur_mult
                yield cur_start, hi, cur_mult
            if produced_any:
                stats.bindings_with_segments += 1
        finally:
            for st in streams:
                st.close()

    def leaves_with_segments(
        self,
    ) -> Iterator[Tuple[Tuple[int, ...], Tuple[int, int, int]]]:
        """Stream every output segment in binding order, then ``a`` order."""
        for values in self.bindings():
            self.stats.leaves += 1
            for seg in self.segments_for(values):
                yield values, seg


def check_window(value: Any, name: str) -> int:
    """Validate a ``--from``/``--to`` endpoint as a signed 64-bit integer."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ResourceError(
            f"{name} must be a signed 64-bit integer, got "
            f"{type(value).__name__}: {value!r}"
        )
    if value < INT64_MIN or value > INT64_MAX:
        raise ResourceError(
            f"{name}={value} is outside the signed 64-bit range"
        )
    return value


class TemporalQuery:
    """Public iterator over valid-time segments.

    Iteration yields
    ``(values, (since, until, multiplicity))`` tuples, one JSON output object
    per segment, ordered by the global variable order and then by ``since``.
    Closing early, cancellation, a consumer exception or a write failure mark
    the run *incomplete*, close every cursor/connection and delete temporary
    files; only full exhaustion followed by a successful consumer finish marks
    it *complete*.
    """

    def __init__(self, decl: Declaration, idx: TemporalOpenIndex, *,
                 window: Tuple[int, int],
                 memory_bytes: Optional[int] = None,
                 limit: Optional[int] = None,
                 verify_mode: str = "deep",
                 cancel=None,
                 fetch_buffer_rows: int = 512) -> None:
        if limit is not None and limit < 0:
            raise ValueError("limit must be >= 0")
        lo = check_window(window[0], "window start (--from)")
        hi = check_window(window[1], "window end (--to)")
        if lo >= hi:
            raise ResourceError(
                f"--from ({lo}) must be strictly less than --to ({hi}); the "
                "query window is the half-open interval [from, to)"
            )
        self.decl = decl
        self.idx = idx
        self.window = (lo, hi)
        self.limit = limit
        self.cancel = cancel
        self.stats = TemporalStats()
        self.stats.limit = limit
        self.stats.verify_mode = verify_mode
        self.stats.window_from = lo
        self.stats.window_to = hi
        self._solver = TemporalSolver(
            decl, idx, stats=self.stats, memory_bytes=memory_bytes,
            window=(lo, hi), fetch_buffer_rows=fetch_buffer_rows,
        )
        self._gen: Optional[Iterator] = None
        self._closed = False
        self._exhausted = False
        self._finalized = False

    def __iter__(self):
        if self._closed:
            raise QueryAborted("temporal query is already closed")
        if self._gen is not None:
            raise QueryAborted(
                "temporal query results were already consumed once"
            )
        self._gen = self._iter_segments()
        return self

    def __next__(self):
        if self._gen is None:
            raise QueryAborted(
                "temporal query has not been started; call iter(query)"
            )
        if self.cancel is not None:
            try:
                stop = bool(self.cancel())
            except Exception:  # pragma: no cover - defensive
                stop = True
            if stop:
                self._mark_incomplete("query cancelled by the caller")
                self._finish_stats()
                raise QueryAborted("query cancelled by the caller")
        try:
            return next(self._gen)
        except StopIteration:
            raise
        except BaseException:
            self._mark_incomplete("query terminated by an exception")
            self._finish_stats()
            raise

    def _mark_incomplete(self, reason: str) -> None:
        if self.stats.status in ("running", "complete"):
            self.stats.status = "incomplete"
            self.stats.status_reason = reason

    def _finish_stats(self) -> None:
        if not self._finalized:
            self._finalized = True
            self.stats.elapsed = time.time() - self.stats.started_at
            self.stats.peak_rss_kib = _peak_rss_kib()

    def _iter_segments(self):
        try:
            for values, segment in self._solver.leaves_with_segments():
                if self.limit is not None and self.stats.emitted >= self.limit:
                    self.stats.status = "limit-reached"
                    self.stats.status_reason = (
                        f"stopped after the --limit {self.limit} segments; "
                        "the result set is not complete"
                    )
                    self._finish_stats()
                    return
                self.stats.emitted += 1
                yield values, segment
            self._exhausted = True
            self.stats.status = "complete"
            self.stats.status_reason = (
                "all valid-time segments in the window emitted"
            )
        except GeneratorExit:
            self._mark_incomplete("consumer closed the iterator early")
            raise
        except (BrokenPipeError, ConnectionResetError) as exc:
            self._mark_incomplete(f"output consumer went away ({exc})")
            raise
        except QueryAborted:
            self._mark_incomplete("query cancelled")
            raise
        except BaseException as exc:
            self._mark_incomplete(f"{type(exc).__name__}: {exc}")
            raise
        finally:
            self._finish_stats()
            if not self._exhausted and self.stats.status in (
                "running", "complete"
            ):
                self.stats.status = "incomplete"
                self.stats.status_reason = "query stopped before exhaustion"

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        gen = self._gen
        if gen is not None and not self._exhausted:
            try:
                gen.close()
            except (RuntimeError, QueryAborted):
                pass
            except Exception:  # pragma: no cover - defensive
                pass
        if self.stats.status == "running":
            self.stats.status = "incomplete"
            self.stats.status_reason = "query closed by caller"
        self._finish_stats()
        self.idx.close()

    def __enter__(self) -> "TemporalQuery":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    @property
    def finished(self) -> bool:
        return self.stats.status in ("complete", "limit-reached")

    def report(self, *, include_layers: bool = True) -> Dict[str, Any]:
        return self.stats.as_dict(self.decl if include_layers else None)
