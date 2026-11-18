"""Ordered-seek adapter and temporal sweep over a maintenance database.

The maintenance database keeps one physical event table per relation (not one
per atom signature), so the temporal engine's :class:`_AtomExec` — which
expects the table record to name the atom's access columns — needs a thin
adapter.  The seek machinery itself is identical: primary-key range scans on
the columns in global-variable order, leapfrog intersection, and per-binding
bounded event cursors.  No SQL join is ever issued.

The same :class:`MaintAtomExec` is used against the *current* relation state
(init, view reads) and, during ``maintain-apply``, against the in-transaction
state that already contains the batch's staged events.
"""
from __future__ import annotations

import heapq
import sqlite3
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .declaration import Declaration
from .engine import MAX_LAYERS
from .errors import CorruptIndex, IndexProblem, ResourceError
from .maintain_format import decode_int_blob, quote_table
from .temporal_engine import TemporalStats

INT64_MAX = (1 << 63) - 1


class MaintStats(TemporalStats):
    """Counters specific to maintenance work; temporal counters inherited."""

    __slots__ = (
        # init
        "relation_events_copied", "relation_rows_after",
        "initial_view_segments", "view_rows_written",
        # apply: startup / correction read
        "correction_lines", "correction_records", "raw_events_inserted",
        "stage_coalesced", "stage_tuples", "validation_rows_scanned",
        "negative_checks",
        # apply: candidate discovery
        "seed_tuples", "discovery_seeks", "discovery_probes",
        "bindings_discovered", "candidate_duplicates",
        # apply: delta synthesis
        "binding_sweeps", "delta_events_read", "delta_boundaries",
        "delta_segments", "delta_rows_written",
        # apply: view maintenance
        "view_rows_deleted", "view_rows_inserted", "relation_events_upserted",
        "relation_events_deleted",
        # spill / resources
        "scratch_bytes", "spill_files",
    )

    def __init__(self) -> None:
        super().__init__()
        self.relation_events_copied = 0
        self.relation_rows_after = 0
        self.initial_view_segments = 0
        self.view_rows_written = 0
        self.correction_lines = 0
        self.correction_records = 0
        self.raw_events_inserted = 0
        self.stage_coalesced = 0
        self.stage_tuples = 0
        self.validation_rows_scanned = 0
        self.negative_checks = 0
        self.seed_tuples = 0
        self.discovery_seeks = 0
        self.discovery_probes = 0
        self.bindings_discovered = 0
        self.candidate_duplicates = 0
        self.delta_events_read = 0
        self.delta_boundaries = 0
        self.binding_sweeps = 0
        self.delta_segments = 0
        self.delta_rows_written = 0
        self.view_rows_deleted = 0
        self.view_rows_inserted = 0
        self.relation_events_upserted = 0
        self.relation_events_deleted = 0
        self.scratch_bytes = 0
        self.spill_files = 0

    def as_dict(self, decl: Optional[Declaration] = None) -> Dict[str, Any]:
        d = super().as_dict(decl)
        d.update({
            "relation_events_copied": self.relation_events_copied,
            "relation_rows_after": self.relation_rows_after,
            "initial_view_segments": self.initial_view_segments,
            "view_rows_written": self.view_rows_written,
            "correction_lines": self.correction_lines,
            "correction_records": self.correction_records,
            "raw_events_inserted": self.raw_events_inserted,
            "stage_coalesced": self.stage_coalesced,
            "stage_tuples": self.stage_tuples,
            "validation_rows_scanned": self.validation_rows_scanned,
            "negative_checks": self.negative_checks,
            "seed_tuples": self.seed_tuples,
            "discovery_seeks": self.discovery_seeks,
            "discovery_probes": self.discovery_probes,
            "bindings_discovered": self.bindings_discovered,
            "candidate_duplicates": self.candidate_duplicates,
            "binding_sweeps": self.binding_sweeps,
            "delta_events_read": self.delta_events_read,
            "delta_boundaries": self.delta_boundaries,
            "delta_segments": self.delta_segments,
            "delta_rows_written": self.delta_rows_written,
            "view_rows_deleted": self.view_rows_deleted,
            "view_rows_inserted": self.view_rows_inserted,
            "relation_events_upserted": self.relation_events_upserted,
            "relation_events_deleted": self.relation_events_deleted,
            "scratch_bytes": self.scratch_bytes,
            "spill_files": self.spill_files,
        })
        return d


class MaintAtomExec:
    """Ordered-seek access to one alias's columns inside a maintenance table."""

    __slots__ = ("relation", "table", "key_columns", "key_levels", "pos",
                 "conn", "stats", "sql_first", "sql_ge", "sql_gt",
                 "sql_event", "sql_exact")

    def __init__(self, conn: sqlite3.Connection, atom_record: Dict[str, Any],
                 stats: MaintStats) -> None:
        self.relation = atom_record["relation"]
        self.table = atom_record["table"]
        self.key_columns = tuple(atom_record["key_columns"])
        self.key_levels = tuple(atom_record["key_levels"])
        self.pos = {lv: i for i, lv in enumerate(self.key_levels)}
        self.conn = conn
        self.stats = stats
        cols = [f"c{j}" for j in self.key_columns]
        order_by = ", ".join(cols)
        prefix_and: List[str] = []
        self.sql_first: List[str] = []
        self.sql_ge: List[str] = []
        self.sql_gt: List[str] = []
        t = quote_table(self.table)
        for p in range(len(cols)):
            cond = " AND ".join(prefix_and)
            where = f"WHERE {cond} AND " if cond else "WHERE "
            self.sql_first.append(
                f"SELECT {cols[p]} FROM {t} {where}1 "
                f"ORDER BY {order_by} LIMIT 1"
            )
            self.sql_ge.append(
                f"SELECT {cols[p]} FROM {t} {where}{cols[p]}>=? "
                f"ORDER BY {order_by} LIMIT 1"
            )
            self.sql_gt.append(
                f"SELECT {cols[p]} FROM {t} {where}{cols[p]}>? "
                f"ORDER BY {order_by} LIMIT 1"
            )
            prefix_and.append(f"{cols[p]}=?")
        self.sql_exact = (
            f"SELECT tick, d FROM {t} WHERE "
            + " AND ".join(f"{c}=?" for c in cols)
            + " ORDER BY tick"
        )
        self.sql_event = self.sql_exact

    def _prefix(self, level: int, bound: Sequence[int]) -> Tuple[int, ...]:
        p = self.pos[level]
        lv = self.key_levels
        return tuple(bound[lv[i]] for i in range(p))

    def _one(self, sql: str, params: Tuple[Any, ...]) -> Optional[int]:
        self.stats.discovery_seeks += 1
        try:
            row = self.conn.execute(sql, params).fetchone()
        except sqlite3.DatabaseError as exc:
            raise CorruptIndex(f"maintenance read failed: {exc}") from exc
        return None if row is None else row[0]

    def first(self, level: int, bound: Sequence[int]) -> Optional[int]:
        p = self.pos[level]
        return self._one(self.sql_first[p], self._prefix(level, bound))

    def seek_ge(self, level: int, bound: Sequence[int],
                value: int) -> Optional[int]:
        p = self.pos[level]
        params = self._prefix(level, bound) + (value,)
        return self._one(self.sql_ge[p], params)

    def seek_gt(self, level: int, bound: Sequence[int],
                value: int) -> Optional[int]:
        if value >= INT64_MAX:
            return None
        p = self.pos[level]
        params = self._prefix(level, bound) + (value,)
        return self._one(self.sql_gt[p], params)

    def open_events(self, bound: Sequence[int]
                    ) -> sqlite3.Cursor:
        """Cursor over ``(tick, delta_blob)`` for the fully bound tuple."""
        self.stats.discovery_probes += 1
        params = tuple(bound[lv] for lv in self.key_levels)
        try:
            return self.conn.execute(self.sql_event, params)
        except sqlite3.DatabaseError as exc:
            raise CorruptIndex(f"maintenance read failed: {exc}") from exc


class MaintSolver:
    """Layered leapfrog candidates + per-binding event sweep, maintenance DB."""

    def __init__(self, decl: Declaration, conn: sqlite3.Connection, *,
                 stats: Optional[MaintStats] = None,
                 window: Optional[Tuple[int, int]] = None,
                 memory_bytes: Optional[int] = None,
                 table_for_atom: Optional[Dict[str, Dict[str, Any]]] = None,
                 fetch_buffer_rows: int = 512) -> None:
        self.decl = decl
        self.conn = conn
        self.stats = stats or MaintStats()
        if window is not None:
            self.stats.window_from, self.stats.window_to = window
        self.window = window
        self.fetch_buffer_rows = fetch_buffer_rows
        depth = len(decl.order)
        if depth > MAX_LAYERS:
            raise ResourceError(
                f"declaration has {depth} variables; at most {MAX_LAYERS} are "
                "supported"
            )
        self.stats.candidates = [0] * depth
        if table_for_atom is None:
            table_for_atom = {
                atom.alias: {
                    "relation": atom.relation,
                    "table": "rel_" + atom.relation,
                    "key_columns": tuple(sorted(
                        range(atom.arity),
                        key=lambda i: decl.var_position[atom.vars[i]],
                    )),
                    "key_levels": tuple(
                        decl.var_position[atom.vars[i]]
                        for i in sorted(
                            range(atom.arity),
                            key=lambda i: decl.var_position[atom.vars[i]],
                        )
                    ),
                }
                for atom in decl.atoms
            }
        self.atoms: List[MaintAtomExec] = []
        self.active_at: List[List[MaintAtomExec]] = [[] for _ in range(depth)]
        for atom in decl.atoms:
            rec = table_for_atom.get(atom.alias)
            if rec is None:
                raise IndexProblem(
                    f"maintenance store has no table for atom {atom.alias!r}"
                )
            ex = MaintAtomExec(conn, rec, self.stats)
            self.atoms.append(ex)
            for lv in ex.key_levels:
                self.active_at[lv].append(ex)
        if memory_bytes is not None:
            total_kib = max(256, memory_bytes // 1024 // 2)
            cache_kib = max(64, total_kib // max(1, len(self.atoms)))
            try:
                conn.execute(f"PRAGMA cache_size=-{cache_kib}")
            except sqlite3.DatabaseError:
                pass

    # -- candidate walk --------------------------------------------------
    def bindings(self) -> List[Tuple[int, ...]]:
        out: List[Tuple[int, ...]] = []
        bound: List[int] = []
        gen = self._walk(0, bound)
        try:
            for b in gen:
                out.append(b)
        finally:
            bound.clear()
        return out

    def _walk(self, level: int, bound: List[int]):
        if level == len(self.decl.order):
            yield tuple(bound)
            return
        for value in self._candidates(level, bound):
            self.stats.candidates[level] += 1
            bound.append(value)
            try:
                yield from self._walk(level + 1, bound)
            finally:
                bound.pop()

    def _candidates(self, level: int, bound: Sequence[int]):
        live = self.active_at[level]
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

    # -- event sweep -----------------------------------------------------
    def segments_for(
        self, values: Sequence[int]
    ) -> List[Tuple[int, int, int]]:
        """All-time or window-clipped maximal segments for one binding."""
        assert self.window is not None
        lo, hi = self.window
        stats = self.stats
        cursors: List[sqlite3.Cursor] = []
        buffers: List[List[sqlite3.Row]] = []
        idxs: List[int] = []
        try:
            for ex in self.atoms:
                cur = ex.open_events(values)
                cursors.append(cur)
                buffers.append([])
                idxs.append(0)

            def head(si: int):
                if idxs[si] >= len(buffers[si]):
                    rows = cursors[si].fetchmany(self.fetch_buffer_rows)
                    buffers[si] = list(rows)
                    idxs[si] = 0
                    if not rows:
                        return None
                row = buffers[si][idxs[si]]
                return row[0], decode_int_blob(row[1])

            factors: List[int] = []
            heap: List[Tuple[int, int]] = []
            for si in range(len(cursors)):
                total = 0
                h = head(si)
                while h is not None and h[0] < lo:
                    total += h[1]
                    idxs[si] += 1
                    stats.delta_events_read += 1
                    h = head(si)
                if total < 0:
                    raise CorruptIndex(
                        "maintenance state yields a negative multiplicity; "
                        "the library is damaged"
                    )
                factors.append(total)
                if h is not None:
                    heapq.heappush(heap, (h[0], si))
            product = 1
            for f in factors:
                product *= f

            out: List[Tuple[int, int, int]] = []
            cur_start = lo
            cur_mult = product
            while heap:
                tick = heap[0][0]
                stats.delta_boundaries += 1
                while heap and heap[0][0] == tick:
                    _, si = heapq.heappop(heap)
                    h = head(si)
                    assert h is not None and h[0] == tick
                    factors[si] += h[1]
                    if factors[si] < 0:
                        raise CorruptIndex(
                            "maintenance state yields a negative multiplicity"
                        )
                    idxs[si] += 1
                    stats.delta_events_read += 1
                    nxt = head(si)
                    if nxt is not None:
                        heapq.heappush(heap, (nxt[0], si))
                new_product = 1
                for f in factors:
                    new_product *= f
                if new_product != cur_mult:
                    if cur_mult > 0 and tick > cur_start and tick <= hi:
                        out.append((cur_start, min(tick, hi), cur_mult))
                    cur_start = max(tick, lo)
                    cur_mult = new_product
            if cur_mult > 0 and hi > cur_start:
                out.append((cur_start, hi, cur_mult))
            return out
        finally:
            for cur in cursors:
                try:
                    cur.close()
                except sqlite3.DatabaseError:
                    pass

    def leaves_with_segments(
        self,
    ) -> List[Tuple[Tuple[int, ...], Tuple[int, int, int]]]:
        out = []
        for values in self.bindings():
            self.stats.leaves += 1
            for seg in self.segments_for(values):
                out.append((values, seg))
        return out
