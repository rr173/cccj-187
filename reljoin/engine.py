"""Layered closed-loop join engine.

The solver walks the *global variable order* one variable per layer.  It
never builds a two-relation intermediate relation and never hands the join to
SQL: at each layer it asks every atom that mentions the current variable for
its next distinct value given the prefix already bound, and advances only the
intersection of those ordered streams.  Values are found with ordered index
seeks, so an atom with millions of rows but only a few distinct values at a
layer costs a few seeks rather than one step per row.  Nothing but the
current prefix (fixed depth), one cursor per live atom and the current output
binding is held in memory.

Correctness argument
--------------------
Let the global levels be ``0 .. m-1``.

* An atom is *active* at level ``L`` exactly when the variable of level ``L``
  is one of its columns.  Its bound key prefix at ``L`` is the set of its key
  columns whose level is ``< L``; because the table's primary key is ordered
  by level (see ``index_build``), that is a contiguous index prefix and the
  seek ``prefix = … AND column >= value`` is an index range scan.
* The leapfrog intersection over the active atoms yields exactly the values
  ``v`` for which every atom mentioning level ``L`` has at least one row with
  the bound prefix and value ``v``.  So a value is never pushed down unless
  all *currently decidable* constraints accept it.
* An atom is *completing* at level ``L`` when ``L`` is its largest key level.
  At that point its entire key is bound; a point lookup proves the row exists
  and yields its bag multiplicity.  Atoms that completed at earlier levels
  already contributed their factor.
* Hence every leaf is a complete closed binding, every complete binding is
  reached exactly once, and ascending recursion emits leaves in lexicographic
  order of the global variable order.

Multiplicity is a Python integer product of the atom counts and is never
stored in a fixed-width field, so it cannot silently wrap.
"""
from __future__ import annotations

import resource
import sqlite3
import time
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from .declaration import Declaration
from .errors import CorruptIndex, IndexProblem, QueryAborted, ResourceError
from .index_open import OpenIndex

MAX_LAYERS = 256
INT64_MAX = (1 << 63) - 1
INT64_MIN = -(1 << 63)


class Stats:
    """Exact run counters.  Nothing here is an estimate."""

    __slots__ = (
        "started_at", "elapsed", "seeks", "point_probes", "probes",
        "candidates", "rejected", "accepted", "leaves", "emitted",
        "max_live_atoms", "max_prefix_len", "peak_rss_kib", "verify_mode",
        "status", "status_reason", "limit", "output_bytes", "tables_touched",
        "seek_ge", "seek_gt",
    )

    def __init__(self) -> None:
        self.started_at = time.time()
        self.elapsed = 0.0
        self.seeks = 0
        self.seek_ge = 0
        self.seek_gt = 0
        self.point_probes = 0
        self.probes = 0
        self.candidates: List[int] = []
        self.rejected = 0
        self.accepted = 0
        self.leaves = 0
        self.emitted = 0
        self.max_live_atoms = 0
        self.max_prefix_len = 0
        self.peak_rss_kib = 0
        self.verify_mode = "deep"
        self.status = "running"
        self.status_reason = ""
        self.limit: Optional[int] = None
        self.output_bytes = 0
        self.tables_touched: set = set()

    def as_dict(self, decl: Optional[Declaration] = None) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "status": self.status,
            "status_reason": self.status_reason,
            "elapsed_seconds": round(self.elapsed, 6),
            "index_lookups": self.probes,
            "index_seeks_first": self.seeks,
            "index_seeks_ge": self.seek_ge,
            "index_seeks_gt": self.seek_gt,
            "point_probes": self.point_probes,
            "candidate_values": list(self.candidates),
            "candidate_values_total": sum(self.candidates),
            "candidates_rejected": self.rejected,
            "candidates_accepted": self.accepted,
            "complete_bindings": self.leaves,
            "emitted": self.emitted,
            "limit": self.limit,
            "max_live_atoms": self.max_live_atoms,
            "max_prefix_len": self.max_prefix_len,
            "peak_rss_kib": self.peak_rss_kib,
            "tables_touched": len(self.tables_touched),
            "verify": self.verify_mode,
            "output_bytes": self.output_bytes,
        }
        if decl is not None:
            d["layers"] = [
                {
                    "level": i,
                    "var": v,
                    "candidate_values": (
                        self.candidates[i] if i < len(self.candidates) else 0
                    ),
                }
                for i, v in enumerate(decl.order)
            ]
        return d


def _blob_to_int(raw: Any) -> int:
    if not isinstance(raw, (bytes, bytearray, memoryview)):
        raise CorruptIndex(
            f"index multiplicity cell is {type(raw).__name__}, expected a BLOB"
        )
    raw = bytes(raw)
    if len(raw) != 16:
        raise CorruptIndex(
            f"index multiplicity cell has {len(raw)} bytes, expected 16"
        )
    return int.from_bytes(raw, "big")


class _AtomExec:
    """Ordered-seek access to one atom's index table."""

    __slots__ = ("key_columns", "key_levels", "pos", "conn", "sql_first",
                 "sql_ge", "sql_gt", "sql_exact", "stats")

    def __init__(self, idx: OpenIndex, table: Dict[str, Any],
                 stats: Stats) -> None:
        self.key_columns = tuple(table["key_columns"])
        self.key_levels = tuple(table["key_levels"])
        self.pos = {lv: i for i, lv in enumerate(self.key_levels)}
        self.conn = idx.connection(table["file"])
        self.stats = stats
        cols = [f"c{j}" for j in self.key_columns]
        order_by = ", ".join(cols)
        prefix_and = []
        self.sql_first: List[str] = []
        self.sql_ge: List[str] = []
        self.sql_gt: List[str] = []
        for p in range(len(cols)):
            cond = " AND ".join(prefix_and)
            where = f"WHERE {cond} AND " if cond else "WHERE "
            self.sql_first.append(
                f"SELECT {cols[p]} FROM t {where}1 ORDER BY {order_by} LIMIT 1"
            )
            self.sql_ge.append(
                f"SELECT {cols[p]} FROM t {where}{cols[p]}>=? "
                f"ORDER BY {order_by} LIMIT 1"
            )
            self.sql_gt.append(
                f"SELECT {cols[p]} FROM t {where}{cols[p]}>? "
                f"ORDER BY {order_by} LIMIT 1"
            )
            prefix_and.append(f"{cols[p]}=?")
        self.sql_exact = (
            "SELECT m FROM t WHERE "
            + " AND ".join(f"{c}=?" for c in cols)
            + " LIMIT 1"
        )

    def _prefix(self, level: int, bound: Sequence[int]) -> Tuple[int, ...]:
        p = self.pos[level]
        lv = self.key_levels
        return tuple(bound[lv[i]] for i in range(p))

    # -- ordered seeks ---------------------------------------------------
    def _one(self, sql: str, params: Tuple[Any, ...],
             counter: str = "seek_first") -> Optional[int]:
        self.stats.probes += 1
        if counter == "seek_first":
            self.stats.seeks += 1
        else:
            setattr(self.stats, counter, getattr(self.stats, counter) + 1)
        try:
            row = self.conn.execute(sql, params).fetchone()
        except sqlite3.DatabaseError as exc:
            raise CorruptIndex(f"index read failed: {exc}") from exc
        return None if row is None else row[0]

    def first(self, level: int, bound: Sequence[int]) -> Optional[int]:
        p = self.pos[level]
        return self._one(self.sql_first[p], self._prefix(level, bound))

    def seek_ge(self, level: int, bound: Sequence[int], value: int) -> Optional[int]:
        p = self.pos[level]
        params = self._prefix(level, bound) + (value,)
        return self._one(self.sql_ge[p], params, "seek_ge")

    def seek_gt(self, level: int, bound: Sequence[int], value: int) -> Optional[int]:
        if value >= INT64_MAX:
            return None
        p = self.pos[level]
        params = self._prefix(level, bound) + (value,)
        return self._one(self.sql_gt[p], params, "seek_gt")

    def count_exact(self, bound: Sequence[int]) -> Optional[int]:
        params = tuple(bound[lv] for lv in self.key_levels)
        self.stats.point_probes += 1
        self.stats.probes += 1
        try:
            row = self.conn.execute(self.sql_exact, params).fetchone()
        except sqlite3.DatabaseError as exc:
            raise CorruptIndex(f"index read failed: {exc}") from exc
        return None if row is None else _blob_to_int(row[0])


class Solver:
    """Generator walk over the global variable order."""

    def __init__(self, decl: Declaration, idx: OpenIndex, *,
                 stats: Optional[Stats] = None,
                 memory_bytes: Optional[int] = None) -> None:
        self.decl = decl
        self.idx = idx
        self.stats = stats or Stats()
        depth = len(decl.order)
        if depth > MAX_LAYERS:
            raise ResourceError(
                f"declaration has {depth} variables; the layered solver "
                f"supports at most {MAX_LAYERS}"
            )
        self.stats.candidates = [0] * depth
        self.atoms: List[_AtomExec] = []
        self.active_at: List[List[_AtomExec]] = [[] for _ in range(depth)]
        self.completing_at: List[List[_AtomExec]] = [[] for _ in range(depth)]
        for atom in decl.atoms:
            table = idx.table_for_atom.get(atom.alias)
            if table is None:
                raise IndexProblem(f"index has no table for atom {atom.alias!r}")
            ex = _AtomExec(idx, table, self.stats)
            self.atoms.append(ex)
            for lv in ex.key_levels:
                self.active_at[lv].append(ex)
            self.completing_at[ex.key_levels[-1]].append(ex)
        if memory_bytes is not None:
            # Split the page-cache allowance over the atom connections so the
            # total stays inside the budget.
            n_conn = max(1, len(idx.table_for_atom))
            total_kib = max(256, memory_bytes // 1024 // 2)
            cache_kib = max(64, total_kib // n_conn)
            for conn in idx.connections():
                try:
                    conn.execute(f"PRAGMA cache_size=-{cache_kib}")
                except sqlite3.DatabaseError:
                    pass

    # -- public ---------------------------------------------------------
    def bindings(self) -> Iterator[Tuple[Tuple[int, ...], int]]:
        bound: List[int] = []
        try:
            yield from self._walk(0, bound, 1)
        finally:
            bound.clear()

    def _walk(self, level: int, bound: List[int], product: int):
        if level == len(self.decl.order):
            yield tuple(bound), product
            return
        stats = self.stats
        completing = self.completing_at[level]
        for value in self._candidates(level, bound):
            stats.candidates[level] += 1
            bound.append(value)
            try:
                mult = product
                ok = True
                for ex in completing:
                    c = ex.count_exact(bound)
                    if c is None:      # defensive; the intersection proves it
                        ok = False
                        break
                    mult *= c
                if not ok:
                    stats.rejected += 1
                    continue
                stats.accepted += 1
                if level + 1 > stats.max_prefix_len:
                    stats.max_prefix_len = level + 1
                if level + 1 < len(self.decl.order):
                    yield from self._walk(level + 1, bound, mult)
                else:
                    stats.leaves += 1
                    yield tuple(bound), mult
            finally:
                bound.pop()

    def _candidates(self, level: int, bound: Sequence[int]) -> Iterator[int]:
        """Ascending intersection of the active atoms' value streams."""
        live = self.active_at[level]
        stats = self.stats
        if len(live) > stats.max_live_atoms:
            stats.max_live_atoms = len(live)
        if not live:  # pragma: no cover - every variable occurs in an atom
            raise IndexProblem(
                f"no atom references variable {self.decl.order[level]!r}"
            )
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


class Query:
    """Public iteration interface.

    ``iter(query)`` yields ``(values, multiplicity)`` where ``values`` is a
    tuple in the declaration's global variable order.  Closing the iterator
    early (consumer break/close, cancellation, write failure) always releases
    the index and marks the run *incomplete*; it never leaves a cursor or a
    connection behind and never pretends the result was finished.
    """

    def __init__(self, decl: Declaration, idx: OpenIndex, *,
                 memory_bytes: Optional[int] = None,
                 limit: Optional[int] = None,
                 verify_mode: str = "deep",
                 cancel=None) -> None:
        if limit is not None and limit < 0:
            raise ValueError("limit must be >= 0")
        self.decl = decl
        self.idx = idx
        self.limit = limit
        self.cancel = cancel
        self.stats = Stats()
        self.stats.limit = limit
        self.stats.verify_mode = verify_mode
        self._solver = Solver(decl, idx, stats=self.stats,
                              memory_bytes=memory_bytes)
        self._gen: Optional[Iterator[Tuple[Tuple[int, ...], int]]] = None
        self._closed = False
        self._exhausted = False
        self._finalized = False

    # -- iteration ------------------------------------------------------
    def __iter__(self) -> Iterator[Tuple[Tuple[int, ...], int]]:
        if self._closed:
            raise QueryAborted("query is already closed")
        if self._gen is not None:
            raise QueryAborted("query results were already consumed once")
        self._gen = self._iter_bindings()
        return self

    def __next__(self) -> Tuple[Tuple[int, ...], int]:
        if self._gen is None:
            raise QueryAborted("query has not been started; call iter(query)")
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

    def _iter_bindings(self) -> Iterator[Tuple[Tuple[int, ...], int]]:
        try:
            for values, mult in self._solver.bindings():
                if self.limit is not None and self.stats.emitted >= self.limit:
                    self.stats.status = "limit-reached"
                    self.stats.status_reason = (
                        f"stopped after the --limit {self.limit} bindings; "
                        "the result set is not complete"
                    )
                    self._finish_stats()
                    return
                self.stats.emitted += 1
                yield values, mult
            self._exhausted = True
            self.stats.status = "complete"
            self.stats.status_reason = "all bindings emitted"
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
        """Stop the run, release the index, record the outcome."""
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

    def __enter__(self) -> "Query":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    @property
    def finished(self) -> bool:
        return self.stats.status in ("complete", "limit-reached")

    def report(self, *, include_layers: bool = True) -> Dict[str, Any]:
        return self.stats.as_dict(self.decl if include_layers else None)


def _peak_rss_kib() -> int:
    try:
        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except (ValueError, OSError):  # pragma: no cover
        return 0
