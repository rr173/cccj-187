"""Apply one correction batch to a maintained library, incrementally.

The applier never rebuilds the index and never runs the full query twice to
diff the answers.  Work is proportional to what the batch actually touches:

1. **Correction read** — the JSONL batch is streamed, validated and written as
   raw ``+delta``/``-delta`` events into a private, disk-backed scratch
   database, so a batch larger than the memory budget never accumulates in the
   process.
2. **Coalescing + legality** — per touched tuple the batch's raw events and
   the tuple's current relation events are merge-joined in tick order (both
   sides are ordered primary-key ranges, read in fetch buffers).  The running
   final multiplicity is checked tick by tick: the whole batch is rejected if
   any period would be negative, even though intermediate raw lines may dip.
   The coalesced, non-zero final events become the tuple's staged trajectory.
3. **Affected binding discovery** — each staged tuple seeds the layered
   leapfrog walk with its values clamped into every alias of its relation.
   Only the first alias of a relation seeds (aliases share the physical table
   and its key permutation), so a binding reached through several atoms is
   produced once; untouched relations and bindings are never scanned.
4. **Delta synthesis** — for each affected binding the old and staged event
   streams are heap-merged across all atoms.  Same-tick events from several
   atoms make one boundary; the running product difference produces the
   signed, maximal delta segments directly — no mechanical old/new pairing
   and the multi-atom cross terms fall out of the product rule.
5. **Atomic commit** — relation state, the view, the signed delta and the
   batch receipt change inside one SQLite transaction.  A kill or rollback
   leaves the library exactly at the previous revision.

Idempotency is decided before the revision conflict: an identical re-send
(same batch id, identical file bytes, identical original expect-revision)
returns the stored receipt and delta without bumping the revision.
"""
from __future__ import annotations

import hashlib
import heapq
import os
import sqlite3
import tempfile
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .declaration import Declaration
from .errors import BatchConflict, CorrectionProblem
from .maintain_corrections import scan_corrections
from .maintain_format import (
    MAINTAIN_TMP_PREFIX,
    decode_int_blob,
    encode_int_blob,
    physical_column_spec,
    quote_ident,
    relation_table_name,
)
from .maintain_solver import MaintAtomExec, MaintStats
from .maintain_store import MaintainStore

_INSERT_CHUNK = 512
_FETCH = 256


@dataclass
class MaintainReceipt:
    batch_id: str
    revision: int
    expect_revision: int
    file_sha256: str
    file_bytes: int
    records: int
    delta_rows: int
    weighted_change: int
    committed_at: float
    replayed: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "revision": self.revision,
            "expect_revision": self.expect_revision,
            "file_sha256": self.file_sha256,
            "file_bytes": self.file_bytes,
            "records": self.records,
            "delta_rows": self.delta_rows,
            "weighted_change": self.weighted_change,
            "committed_at": self.committed_at,
            "replayed": self.replayed,
        }


@dataclass
class ApplyResult:
    receipt: MaintainReceipt
    stats: MaintStats


class _Scratch:
    """A disk-backed ATTACH database; never held in the main db file.

    The scratch database lives in the process's temporary directory (or the
    caller's ``sort_dir``), never inside the committed library directory, so a
    SIGKILL cannot leave foreign files next to ``state.db`` and reopening the
    library does not depend on the scratch surviving.  Normal completion
    deletes the file; an OS crash leaves it in the temp area for the OS to
    reclaim.
    """

    def __init__(self, sort_dir: Optional[str] = None) -> None:
        fd, self.path = tempfile.mkstemp(
            prefix=MAINTAIN_TMP_PREFIX, suffix="-scratch.db",
            dir=sort_dir,
        )
        os.close(fd)
        os.unlink(self.path)  # ATTACH creates it fresh.

    def attach(self, main: sqlite3.Connection, cache_kib: int) -> None:
        # ATTACH is allowed inside a transaction; only cache sizing is set on
        # the scratch schema (journal/synchronous changes are forbidden once
        # a transaction is open, and MEMORY journal would also risk the
        # rollback durability we rely on).
        main.execute(
            "ATTACH DATABASE " + _sql_str(self.path) + " AS scratch"
        )
        try:
            main.execute(f"PRAGMA scratch.cache_size=-{int(cache_kib)}")
        except sqlite3.DatabaseError:
            pass

    def detach(self, main: sqlite3.Connection) -> None:
        try:
            main.execute("DETACH DATABASE scratch")
        except sqlite3.DatabaseError:
            pass

    def cleanup(self) -> None:
        for suffix in ("", "-wal", "-shm", "-journal"):
            try:
                os.unlink(self.path + suffix)
            except OSError:
                pass


def _sql_str(text: str) -> str:
    return "'" + text.replace("'", "''") + "'"


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------
def apply_corrections(
    store: MaintainStore,
    *,
    batch_id: str,
    expect_revision: int,
    correction_path: str,
    memory_bytes: int,
    sqlite_cache_kib: Optional[int] = None,
    sort_dir: Optional[str] = None,
    kill_hook: Optional[str] = None,
    progress=None,
) -> ApplyResult:
    if not isinstance(batch_id, str) or not batch_id:
        raise CorrectionProblem("batch-id must be a non-empty string")
    if "\x00" in batch_id:
        raise CorrectionProblem("batch-id contains a NUL character")
    if isinstance(expect_revision, bool) or not isinstance(expect_revision, int):
        raise CorrectionProblem("expect-revision must be an integer")
    if expect_revision < 0:
        raise CorrectionProblem("expect-revision must be non-negative")
    if sqlite_cache_kib is None:
        sqlite_cache_kib = max(256, int(memory_bytes * 0.4) // 1024)

    conn = store.connect(readonly=False, busy_timeout_ms=10000)
    scratch = _Scratch(sort_dir)
    stats = MaintStats()
    # PRAGMAs that touch journal/safety settings must run before BEGIN.
    conn.execute("PRAGMA temp_store=FILE")
    try:
        # Single-writer lock; replay beats the revision-conflict check.
        conn.execute("BEGIN IMMEDIATE")
        existing = conn.execute(
            "SELECT revision, expect_revision, file_sha256, file_bytes, "
            "records, delta_rows, weighted_change, committed_at "
            "FROM batches WHERE batch_id=?",
            (batch_id,),
        ).fetchone()
        if existing is not None:
            (rev, exp_rev, sha, fbytes, records, delta_rows,
             weighted, committed_at) = existing
            file_sha, size = _hash_file(correction_path)
            if (file_sha == sha and size == fbytes
                    and exp_rev == expect_revision):
                conn.execute("COMMIT")
                receipt = MaintainReceipt(
                    batch_id=batch_id, revision=rev,
                    expect_revision=exp_rev, file_sha256=sha,
                    file_bytes=fbytes, records=records,
                    delta_rows=delta_rows, weighted_change=int(weighted),
                    committed_at=committed_at, replayed=True,
                )
                return ApplyResult(receipt=receipt, stats=stats)
            raise BatchConflict(
                f"batch-id {batch_id!r} was already used with different "
                "correction bytes or a different original expect-revision"
            )
        current_rev = store.revision(conn)
        if expect_revision != current_rev:
            raise BatchConflict(
                f"expect-revision {expect_revision} does not match the "
                f"current maintenance revision {current_rev}"
            )

        scratch.attach(conn, sqlite_cache_kib)
        try:
            result = _apply_locked(
                store, conn, scratch, stats, batch_id=batch_id,
                expect_revision=expect_revision,
                correction_path=correction_path,
                kill_hook=kill_hook, progress=progress,
            )
        except BaseException:
            try:
                scratch.detach(conn)
            except sqlite3.DatabaseError:
                pass
            raise
        return result
    except BaseException:
        try:
            conn.execute("ROLLBACK")
        except sqlite3.DatabaseError:
            pass
        raise
    finally:
        try:
            conn.close()
        except sqlite3.Error:
            pass
        scratch.cleanup()


def _hash_file(path: str) -> Tuple[str, int]:
    h = hashlib.sha256()
    size = 0
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
            size += len(chunk)
    return h.hexdigest(), size


# --------------------------------------------------------------------------
# Transactional body
# --------------------------------------------------------------------------
def _apply_locked(store, conn, scratch, stats, *, batch_id, expect_revision,
                  correction_path, kill_hook, progress) -> ApplyResult:
    decl: Declaration = store.decl
    depth = len(decl.order)
    arity_of = decl.relation_arity
    table_for_atom = store.table_for_atom
    raw_names: Dict[str, str] = {}
    stage_names: Dict[str, str] = {}

    # -- 1. scratch tables ------------------------------------------------
    for rel, arity in arity_of.items():
        raw = f"raw_{rel}"
        stg = f"stg_{rel}"
        raw_names[rel] = raw
        stage_names[rel] = stg
        cols = ", ".join(f"c{i} INTEGER NOT NULL" for i in range(arity))
        ordered = ", ".join(f"c{i}" for i in range(arity))
        # Raw rows may share a (tuple, tick) when several correction lines
        # overlap, so the raw table is a plain heap with an ordering index;
        # the coalescer sums such collisions before they reach the stage.
        conn.execute(
            f"CREATE TABLE scratch.{quote_ident(raw)} ({cols}, "
            f"tick INTEGER NOT NULL, d BLOB NOT NULL)"
        )
        conn.execute(
            f"CREATE INDEX scratch.{quote_ident(raw + '_idx')} ON "
            f"{quote_ident(raw)} ({ordered}, tick)"
        )
        pk = ", ".join([*(f"c{i}" for i in range(arity)), "tick"])
        conn.execute(
            f"CREATE TABLE scratch.{quote_ident(stg)} ({cols}, "
            f"tick INTEGER NOT NULL, d BLOB NOT NULL, "
            f"PRIMARY KEY ({pk})) WITHOUT ROWID"
        )

    # -- 2. stream corrections into scratch raw tables --------------------
    corrections, digest = scan_corrections(correction_path, decl)
    pending: Dict[str, List[tuple]] = {rel: [] for rel in arity_of}
    insert_sql: Dict[str, str] = {}
    for rel, arity in arity_of.items():
        cols = ", ".join(physical_column_spec(arity))
        ph = ",".join("?" * (arity + 2))
        insert_sql[rel] = (
            f"INSERT INTO scratch.{quote_ident(raw_names[rel])} "
            f"({cols}, tick, d) VALUES ({ph})"
        )

    def flush(rel: str) -> None:
        rows = pending[rel]
        if rows:
            conn.executemany(insert_sql[rel], rows)
            stats.raw_events_inserted += len(rows)
            rows.clear()

    saw_any = False
    for corr in corrections:
        saw_any = True
        stats.correction_records += 1
        base = tuple(corr.values)
        pending[corr.relation].append(
            base + (corr.since, encode_int_blob(corr.delta))
        )
        pending[corr.relation].append(
            base + (corr.until, encode_int_blob(-corr.delta))
        )
        if len(pending[corr.relation]) >= _INSERT_CHUNK:
            flush(corr.relation)
    for rel in arity_of:
        flush(rel)
    stats.correction_lines = digest.lines
    if not saw_any:
        raise CorrectionProblem("correction batch contains no records")
    file_sha = digest.sha256.hexdigest()
    file_bytes = digest.bytes
    if progress:
        progress(
            f"read {digest.records} correction(s), "
            f"{stats.raw_events_inserted} raw events"
        )

    # -- 3. per-relation coalescing + legality ----------------------------
    staged_keys: Dict[str, List[Tuple[int, ...]]] = {}
    for rel in sorted(arity_of):
        arity = arity_of[rel]
        keys = _coalesce_relation(
            conn, rel, arity, raw_names[rel], stage_names[rel], stats
        )
        staged_keys[rel] = keys
        stats.stage_tuples += len(keys)
    if progress:
        progress(
            f"coalesced {stats.stage_tuples} touched tuple(s); "
            f"{stats.negative_checks} legality checks"
        )

    # Seed views expose the absolute post-batch trajectory (current relation
    # events unioned with the balanced batch deltas), so discovery can find a
    # binding whose tuple is brand new in this batch as well as one that merely
    # changed.  Both sides are balanced event streams, so their union is too.
    seed_view_of_rel: Dict[str, str] = {}
    for rel in sorted(arity_of):
        arity = arity_of[rel]
        view_name = f"seed_{rel}"
        cols = ", ".join(physical_column_spec(arity))
        main_tbl = relation_table_name(rel)
        # A TEMP view may reference both the main database and the attached
        # scratch database; an ordinary scratch view is forbidden from reading
        # main.  The view name itself is unqualified so it resolves in temp.
        conn.execute(
            f"CREATE TEMP VIEW {quote_ident(view_name)} AS "
            f"SELECT {cols}, tick, d FROM main.{quote_ident(main_tbl)} "
            f"UNION ALL "
            f"SELECT {cols}, tick, d FROM scratch.{quote_ident(stage_names[rel])}"
        )
        seed_view_of_rel[rel] = view_name

    # -- 4. affected binding discovery ------------------------------------
    # Dedup lives in the disk-backed scratch database (a unique-key table), so
    # an affected set larger than the memory budget never accumulates in RAM.
    bcols_def = ", ".join(f"b{i} INTEGER NOT NULL" for i in range(depth))
    bcols = ", ".join(f"b{i}" for i in range(depth))
    conn.execute(
        f"CREATE TABLE scratch.affected ({bcols_def}, "
        f"PRIMARY KEY ({bcols})) WITHOUT ROWID"
    )
    affected_insert = (
        f"INSERT OR IGNORE INTO scratch.affected ({bcols}) VALUES ("
        + ",".join("?" * depth) + ")"
    )
    atom_records = [table_for_atom[a.alias] for a in decl.atoms]
    # Discovery reads the post-batch seed views rather than the pre-batch main
    # tables, so newly introduced tuples are discoverable in the same batch.
    discovery_records = [
        {
            **rec,
            "table": seed_view_of_rel[rec["relation"]],
        }
        for rec in atom_records
    ]
    executors = [MaintAtomExec(conn, rec, stats) for rec in discovery_records]
    active_at: List[List[int]] = [[] for _ in range(depth)]
    for ei, ex in enumerate(executors):
        for lv in ex.key_levels:
            active_at[lv].append(ei)
    first_alias_of_rel: Dict[str, int] = {}
    for ai, atom in enumerate(decl.atoms):
        first_alias_of_rel.setdefault(atom.relation, ai)
    seed_atom_of_rel = {
        rel: first_alias_of_rel[rel] for rel in arity_of
    }

    for rel in sorted(arity_of):
        ex = executors[seed_atom_of_rel[rel]]
        for key in staged_keys[rel]:
            stats.seed_tuples += 1
            # key is in physical file-column order; the executor reads its
            # columns in key_columns order, so the seed value for key
            # position p is key[key_columns[p]].  Building it by level would
            # scramble atoms whose file and global orders differ (e.g. C(z,x)).
            seed_key = tuple(key[fc] for fc in ex.key_columns)
            for binding in _seed_walk(
                conn, decl, executors, active_at, ex, seed_key, stats
            ):
                before = conn.total_changes
                conn.execute(affected_insert, binding)
                if conn.total_changes == before:
                    # The unique key rejected it: this binding was already
                    # discovered through another changed atom.
                    stats.candidate_duplicates += 1
                else:
                    stats.bindings_discovered += 1
    if progress:
        progress(
            f"discovery: {stats.seed_tuples} seed tuple(s) -> "
            f"{stats.bindings_discovered} affected binding(s)"
        )
    affected_cur = conn.execute(
        f"SELECT {bcols} FROM scratch.affected ORDER BY {bcols}"
    )
    affected: List[Tuple[int, ...]] = []

    def _iter_affected():
        while True:
            rows = affected_cur.fetchmany(128)
            if not rows:
                return
            for row in rows:
                yield tuple(row)

    # -- 5. delta synthesis + view replacement ----------------------------
    # Deltas are streamed straight into the committed batch_delta table in
    # bounded chunks, so a result larger than the memory budget is never held
    # in RAM; the CLI re-reads it via maintain-export.
    weighted_change = 0
    delta_insert = (
        "INSERT INTO batch_delta (batch_id, seq, "
        + ", ".join(f"b{i}" for i in range(depth))
        + ", since, until, delta_multiplicity) VALUES ("
        + ",".join("?" * (depth + 5)) + ")"
    )
    seq = 0
    delta_count = 0
    delta_pending: List[tuple] = []
    for binding in _iter_affected():
        stats.binding_sweeps += 1
        first_tick, last_tick, segments = _sweep_binding(
            conn, atom_records, stage_names, binding, stats
        )
        if first_tick is None:
            continue
        _replace_view(
            conn, depth, binding, segments, stats,
            covered=(first_tick, last_tick),
        )
        for since, until, delta, _new_mult in segments:
            if delta == 0:
                continue
            stats.delta_segments += 1
            weighted_change += (until - since) * delta
            delta_count += 1
            delta_pending.append(
                (batch_id, seq, *binding, since, until,
                 encode_int_blob(delta))
            )
            seq += 1
            if len(delta_pending) >= _INSERT_CHUNK:
                conn.executemany(delta_insert, delta_pending)
                stats.delta_rows_written += len(delta_pending)
                delta_pending.clear()
    if delta_pending:
        conn.executemany(delta_insert, delta_pending)
        stats.delta_rows_written += len(delta_pending)
    affected_cur.close()

    # -- 6. promote staged trajectories into the relation tables ----------
    for rel in sorted(arity_of):
        _promote_relation(
            conn, rel, arity_of[rel], stage_names[rel],
            staged_keys[rel], stats,
        )

    # -- 7. revision + receipt, atomic commit -----------------------------
    if kill_hook == "before-commit":
        # Test-only: an uncatchable SIGKILL abandons the transaction exactly
        # like an OS crash; reopening must show the previous revision.
        os.kill(os.getpid(), 9)
    new_rev = expect_revision + 1
    conn.execute(
        "UPDATE meta SET value=? WHERE key='revision'", (str(new_rev),)
    )
    conn.execute(
        "INSERT INTO batches (batch_id, revision, expect_revision, "
        "file_sha256, file_bytes, records, delta_rows, weighted_change, "
        "committed_at) VALUES (?,?,?,?,?,?,?,?,?)",
        (batch_id, new_rev, expect_revision, file_sha, file_bytes,
         digest.records, delta_count, str(weighted_change),
         time.time()),
    )
    scratch.detach(conn)
    conn.execute("COMMIT")
    try:
        conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
    except sqlite3.DatabaseError:
        pass

    receipt = MaintainReceipt(
        batch_id=batch_id, revision=new_rev,
        expect_revision=expect_revision, file_sha256=file_sha,
        file_bytes=file_bytes, records=digest.records,
        delta_rows=delta_count, weighted_change=weighted_change,
        committed_at=time.time(), replayed=False,
    )
    stats.elapsed = time.time() - stats.started_at
    from .engine import _peak_rss_kib
    stats.peak_rss_kib = _peak_rss_kib()
    return ApplyResult(receipt=receipt, stats=stats)


# --------------------------------------------------------------------------
# Coalesce one relation
# --------------------------------------------------------------------------
class _BufCursor:
    """Fetch-buffered cursor over (tick, delta-int) rows."""

    __slots__ = ("cur", "buf", "idx", "fetch")

    def __init__(self, cur: sqlite3.Cursor, fetch: int = _FETCH) -> None:
        self.cur = cur
        self.buf: List[sqlite3.Row] = []
        self.idx = 0
        self.fetch = fetch

    def head(self) -> Optional[Tuple[int, int]]:
        if self.idx >= len(self.buf):
            self.buf = list(self.cur.fetchmany(self.fetch))
            self.idx = 0
            if not self.buf:
                return None
        row = self.buf[self.idx]
        return row[0], decode_int_blob(row[1])

    def advance(self) -> None:
        self.idx += 1

    def close(self) -> None:
        try:
            self.cur.close()
        except sqlite3.DatabaseError:
            pass


def _coalesce_relation(conn, rel, arity, raw_name, stage_name,
                       stats: MaintStats) -> List[Tuple[int, ...]]:
    """Coalesce raw batch events and prove the batch keeps every bag legal.

    The stage stores the tuple's *balanced batch deltas*: the raw correction
    events coalesced per tick (a zero net change at a tick is dropped).  A
    correction interval contributes +delta at its start and -delta at its end,
    so the per-tuple stage stream always sums back to zero.  Legality is
    checked against the current relation state by merge-walking both ordered
    streams and confirming ``old_running + batch_running`` never goes negative,
    even when an early correction line temporarily dips.
    """
    main_tbl = relation_table_name(rel)
    cols = physical_column_spec(arity)
    col_list = ", ".join(cols)
    key_match = " AND ".join(f"{c}=?" for c in cols)
    stg_qualified = f"scratch.{quote_ident(stage_name)}"
    insert_sql = (
        f"INSERT INTO {stg_qualified} ({col_list}, tick, d) VALUES "
        + f"({','.join('?' * (arity + 2))})"
    )
    distinct = conn.execute(
        f"SELECT DISTINCT {col_list} FROM scratch.{quote_ident(raw_name)} "
        f"ORDER BY {col_list}"
    )
    staged: List[Tuple[int, ...]] = []
    pending: List[tuple] = []
    while True:
        key_rows = distinct.fetchmany(128)
        if not key_rows:
            break
        for krow in key_rows:
            key = tuple(krow)
            main_cur = conn.execute(
                f"SELECT tick, d FROM {quote_ident(main_tbl)} "
                f"WHERE {key_match} ORDER BY tick",
                key,
            )
            raw_cur = conn.execute(
                f"SELECT tick, d FROM scratch.{quote_ident(raw_name)} "
                f"WHERE {key_match} ORDER BY tick",
                key,
            )
            old = _BufCursor(main_cur)
            new = _BufCursor(raw_cur)
            old_running = 0
            batch_running = 0
            violation: Optional[Tuple[int, Optional[int]]] = None
            produced = 0
            # Coalesce raw events that share a tick before they reach the
            # stage, so a zero net batch delta at a tick is simply absent.
            tick_buffer: List[Tuple[int, int]] = []

            def flush_tick(tick: int, batch_delta: int) -> None:
                nonlocal produced
                if batch_delta == 0:
                    return
                pending.append(
                    key + (tick, encode_int_blob(batch_delta))
                )
                produced += 1
                if len(pending) >= _INSERT_CHUNK:
                    conn.executemany(insert_sql, pending)
                    pending.clear()

            try:
                ho, hn = old.head(), new.head()
                while ho is not None or hn is not None:
                    if hn is None or (ho is not None and ho[0] < hn[0]):
                        tick, delta = ho
                        old_running += delta
                        stats.validation_rows_scanned += 1
                        old.advance()
                    else:
                        # All raw events at this tick share one batch delta.
                        tick = hn[0]
                        merged = 0
                        while hn is not None and hn[0] == tick:
                            merged += hn[1]
                            stats.stage_coalesced += 1
                            new.advance()
                            hn = new.head()
                        if ho is not None and ho[0] == tick:
                            old_running += ho[1]
                            stats.validation_rows_scanned += 1
                            old.advance()
                        batch_running += merged
                        final_running = old_running + batch_running
                        stats.negative_checks += 1
                        if final_running < 0 and violation is None:
                            violation = (tick, None)
                        flush_tick(tick, merged)
                    ho, hn = old.head(), new.head()
            finally:
                old.close()
                new.close()
            if violation is not None:
                bad_tick = violation[0]
                nxt = conn.execute(
                    f"SELECT MIN(tick) FROM {stg_qualified} WHERE "
                    f"{key_match} AND tick>?",
                    key + (bad_tick,),
                ).fetchone()
                end_tick = nxt[0] if nxt and nxt[0] is not None \
                    else bad_tick + 1
                raise CorrectionProblem(
                    f"correction would make relation {rel!r} tuple "
                    f"{list(key)} negative on [{bad_tick}, {end_tick}); "
                    "the whole batch is rejected"
                )
            if produced:
                staged.append(key)
    if pending:
        conn.executemany(insert_sql, pending)
    return staged


# --------------------------------------------------------------------------
# Seeded leapfrog discovery
# --------------------------------------------------------------------------
def _seed_walk(conn, decl: Declaration,
               executors: List[MaintAtomExec],
               active_at: List[List[int]],
               seed_ex: MaintAtomExec,
               seed_key: Tuple[int, ...],
               stats: MaintStats) -> List[Tuple[int, ...]]:
    """Complete bindings compatible with one seeded relation tuple.

    Every alias of the seeded relation shares its table and key permutation,
    so the seed clamps all of them to the seed tuple's values; the caller only
    seeds from the first alias, which makes each binding appear once without a
    materialised candidate set.  Seeded atoms expose exactly one value (the
    seed column value) via a >= seek, while the other atoms use their ordinary
    ordered first/ge/gt streams, so the leapfrog intersection still drives the
    whole walk and no full relation is ever scanned.
    """
    depth = len(decl.order)
    seed_relation = seed_ex.relation
    bound: List[int] = []
    out: List[Tuple[int, ...]] = []

    def make_stream(ei: int, level: int):
        ex = executors[ei]
        if ex.relation != seed_relation:
            return _FreeSeedStream(ex, level, bound)
        wanted = seed_key[ex.pos[level]]
        return _ClampedSeedStream(ex, level, bound, wanted)

    def walk(level: int) -> None:
        if level == depth:
            out.append(tuple(bound))
            return
        live = active_at[level]
        streams = [make_stream(ei, level) for ei in live]
        cands = [s.first() for s in streams]
        n = len(streams)
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
                    cands[i] = streams[i].seek_ge(top)
                continue
            bound.append(top)
            try:
                walk(level + 1)
            finally:
                bound.pop()
            for i in range(n):
                cands[i] = streams[i].seek_gt(top)

    walk(0)
    return out


class _FreeSeedStream:
    __slots__ = ("ex", "level", "bound")

    def __init__(self, ex: MaintAtomExec, level: int,
                 bound: List[int]) -> None:
        self.ex = ex
        self.level = level
        self.bound = bound

    def first(self):
        return self.ex.first(self.level, self.bound)

    def seek_ge(self, value):
        return self.ex.seek_ge(self.level, self.bound, value)

    def seek_gt(self, value):
        return self.ex.seek_gt(self.level, self.bound, value)


class _ClampedSeedStream:
    """An atom stream pinned to exactly one seed column value."""

    __slots__ = ("ex", "level", "bound", "wanted", "_emitted")

    def __init__(self, ex: MaintAtomExec, level: int,
                 bound: List[int], wanted: int) -> None:
        self.ex = ex
        self.level = level
        self.bound = bound
        self.wanted = wanted
        self._emitted = False

    def first(self):
        # Seek directly to the seed value; never take the table minimum.
        v = self.ex.seek_ge(self.level, self.bound, self.wanted)
        if v is None or v != self.wanted:
            return None
        return v

    def seek_ge(self, value):
        if value > self.wanted:
            return None
        v = self.ex.seek_ge(self.level, self.bound, self.wanted)
        if v is None or v != self.wanted:
            return None
        return v

    def seek_gt(self, value):
        # A clamped column contributes exactly one distinct value.
        return None


# --------------------------------------------------------------------------
# Per-binding delta + new-view sweep
# --------------------------------------------------------------------------
class _Evt:
    __slots__ = ("cur", "buf", "idx", "stats")

    def __init__(self, cur: sqlite3.Cursor, stats: MaintStats) -> None:
        self.cur = cur
        self.buf: List[sqlite3.Row] = []
        self.idx = 0
        self.stats = stats

    def head(self) -> Optional[Tuple[int, int]]:
        if self.idx >= len(self.buf):
            self.buf = list(self.cur.fetchmany(_FETCH))
            self.idx = 0
            if not self.buf:
                return None
        row = self.buf[self.idx]
        return row[0], decode_int_blob(row[1])

    def advance(self) -> None:
        self.idx += 1
        self.stats.delta_events_read += 1

    def close(self) -> None:
        try:
            self.cur.close()
        except sqlite3.DatabaseError:
            pass


def _open_events(conn, qualified: str, ex: MaintAtomExec,
                 binding: Tuple[int, ...]) -> sqlite3.Cursor:
    cols = [f"c{j}" for j in ex.key_columns]
    sql = (
        f"SELECT tick, d FROM {qualified} WHERE "
        + " AND ".join(f"{c}=?" for c in cols)
        + " ORDER BY tick"
    )
    params = tuple(binding[lv] for lv in ex.key_levels)
    return conn.execute(sql, params)


def _sweep_binding(conn, atom_records, stage_names,
                   binding: Tuple[int, ...], stats: MaintStats
                   ) -> Tuple[int, int, List[Tuple[int, int, int, int]]]:
    """Sweep one affected binding over its full boundary partition.

    Returns ``(first_tick, last_tick, segments)`` where the segments cover the
    whole interval ``[first_tick, last_tick)`` partitioned at every event tick
    of either the old relation streams or the balanced batch streams.  Each
    segment is ``(since, until, delta_multiplicity, new_multiplicity)``; a zero
    new multiplicity is a normal segment the view merge drops, and a zero delta
    segment is filtered out of the signed output but still returned so the
    view replacement knows the binding's multiplicity over the whole range.
    Same-tick events from several atoms form one boundary, so cross terms fall
    out of the product rule without enumeration.
    """
    streams: List[Tuple[_Evt, _Evt]] = []
    try:
        old_factors: List[int] = []
        batch_factors: List[int] = []
        heap: List[Tuple[int, int, int]] = []
        for ai, rec in enumerate(atom_records):
            ex = MaintAtomExec(conn, rec, stats)
            old_s = _Evt(
                _open_events(conn, quote_ident(rec["table"]), ex, binding),
                stats,
            )
            bat_s = _Evt(
                _open_events(
                    conn, "scratch." + quote_ident(stage_names[rec["relation"]]),
                    ex, binding,
                ),
                stats,
            )
            streams.append((old_s, bat_s))
            old_factors.append(0)
            batch_factors.append(0)
            ho, hb = old_s.head(), bat_s.head()
            if ho is not None:
                heapq.heappush(heap, (ho[0], ai, 0))
            if hb is not None:
                heapq.heappush(heap, (hb[0], ai, 1))

        def product(factors: List[int]) -> int:
            p = 1
            for f in factors:
                p *= f
            return p

        first_tick: Optional[int] = None
        last_tick: Optional[int] = None
        # (start, delta, new_multiplicity) per constant run; the until is the
        # next run's start, or last_tick for the final run.
        runs: List[Tuple[int, int, int]] = []
        while heap:
            tick = heap[0][0]
            stats.delta_boundaries += 1
            if first_tick is None:
                first_tick = tick
            last_tick = tick
            while heap and heap[0][0] == tick:
                _, ai, side = heapq.heappop(heap)
                old_s, bat_s = streams[ai]
                if side == 0:
                    h = old_s.head()
                    assert h is not None and h[0] == tick
                    old_factors[ai] += h[1]
                    old_s.advance()
                    nxt = old_s.head()
                    if nxt is not None:
                        heapq.heappush(heap, (nxt[0], ai, 0))
                else:
                    h = bat_s.head()
                    assert h is not None and h[0] == tick
                    batch_factors[ai] += h[1]
                    bat_s.advance()
                    nxt = bat_s.head()
                    if nxt is not None:
                        heapq.heappush(heap, (nxt[0], ai, 1))
            new_factors = [
                old_factors[i] + batch_factors[i]
                for i in range(len(atom_records))
            ]
            old_p = product(old_factors)
            new_p = product(new_factors)
            delta = new_p - old_p
            if runs and runs[-1][1] == delta and runs[-1][2] == new_p:
                continue
            runs.append((tick, delta, new_p))

        segments: List[Tuple[int, int, int, int]] = []
        for i, (start, delta, new_p) in enumerate(runs):
            until = runs[i + 1][0] if i + 1 < len(runs) else last_tick
            segments.append((start, until, delta, new_p))
        return first_tick, last_tick, segments
    finally:
        for old_s, bat_s in streams:
            old_s.close()
            bat_s.close()


def _replace_view(conn, depth: int, binding: Tuple[int, ...],
                  segments: List[Tuple[int, int, int, int]],
                  stats: MaintStats, *,
                  covered: Tuple[int, int]) -> None:
    """Replace one binding's view rows over the swept range.

    ``covered`` is the exact ``[first_tick, last_tick)`` range the sweep
    partitioned, so the new multiplicity (including an explicit zero) is
    authoritative everywhere inside it; outside the range the binding keeps
    its previously stored segments.  Old and new segment endpoints are
    unioned, adjacent equal positive runs are re-merged, and zero/empty runs
    are dropped.  A binding that withdraws to zero over the whole covered
    range simply loses those view rows.  Only this binding's rows are touched.
    """
    covered_start, covered_end = covered
    where = " AND ".join(f"b{i}=?" for i in range(depth))
    old_rows = conn.execute(
        f"SELECT since, until, multiplicity FROM view WHERE {where} "
        f"ORDER BY since",
        binding,
    ).fetchall()
    old_segments = [(s, u, decode_int_blob(b)) for s, u, b in old_rows]
    new_segments = [(s, u, m) for s, u, _d, m in segments]

    # Index the new (sorted, disjoint) segments by start tick so a gap's
    # multiplicity is a bisect lookup instead of an O(n) scan per gap.
    new_starts = [s for s, _u, _m in new_segments]

    def new_at(t: int) -> int:
        import bisect
        i = bisect.bisect_right(new_starts, t) - 1
        if i < 0:
            return 0
        s, u, m = new_segments[i]
        return m if s <= t < u else 0

    endpoints = {covered_start, covered_end}
    for s, u, _m in old_segments + new_segments:
        endpoints.add(s)
        endpoints.add(u)
    cuts = sorted(e for e in endpoints if covered_start <= e <= covered_end)
    merged: List[Tuple[int, int, int]] = []
    for a, b in zip(cuts, cuts[1:]):
        if a >= b:
            continue
        m = new_at(a)
        if m <= 0:
            continue
        if merged and merged[-1][2] == m and merged[-1][1] == a:
            ps, _pu, pm = merged[-1]
            merged[-1] = (ps, b, pm)
        else:
            merged.append((a, b, m))

    # Keep old segments that lie entirely outside the covered range; clip any
    # that straddle a boundary so the replacement stays interval-consistent.
    kept: List[Tuple[int, int, int]] = []
    for s, u, m in old_segments:
        if u <= covered_start or s >= covered_end:
            kept.append((s, u, m))
            continue
        if s < covered_start:
            kept.append((s, covered_start, m))
        if u > covered_end:
            kept.append((covered_end, u, m))
    final_segments = sorted(kept + merged)
    # Re-merge across the kept/replaced seam.
    re_merged: List[Tuple[int, int, int]] = []
    for s, u, m in final_segments:
        if re_merged and re_merged[-1][2] == m and re_merged[-1][1] == s:
            ps, _pu, pm = re_merged[-1]
            re_merged[-1] = (ps, u, pm)
        else:
            re_merged.append((s, u, m))

    deleted = conn.execute(
        f"DELETE FROM view WHERE {where}", binding
    ).rowcount
    stats.view_rows_deleted += int(deleted)
    insert_sql = (
        "INSERT INTO view ("
        + ", ".join(f"b{i}" for i in range(depth))
        + ", since, until, multiplicity) VALUES ("
        + ",".join("?" * (depth + 3)) + ")"
    )
    pending = [
        tuple(binding) + (s, u, encode_int_blob(m))
        for s, u, m in re_merged
    ]
    if pending:
        conn.executemany(insert_sql, pending)
        stats.view_rows_inserted += len(pending)


def _promote_relation(conn, rel, arity, stage_name,
                      keys: List[Tuple[int, ...]],
                      stats: MaintStats) -> None:
    """Rebuild each touched tuple's absolute event trajectory.

    Both streams are balanced deltas summed from zero, so the tuple's new
    multiplicity at every tick is ``old_running + batch_running``.  Promotion
    merge-walks them in tick order and writes the change in that running
    multiplicity as the new absolute events, dropping zero changes.  Untouched
    tuples are never read or rewritten.
    """
    tbl = relation_table_name(rel)
    cols = physical_column_spec(arity)
    col_list = ", ".join(cols)
    key_match = " AND ".join(f"{c}=?" for c in cols)
    insert_sql = (
        f"INSERT INTO {quote_ident(tbl)} ({col_list}, tick, d) VALUES "
        f"({','.join('?' * (arity + 2))})"
    )
    for key in keys:
        old_cur = conn.execute(
            f"SELECT tick, d FROM {quote_ident(tbl)} WHERE {key_match} "
            f"ORDER BY tick",
            key,
        )
        stg_cur = conn.execute(
            f"SELECT tick, d FROM scratch.{quote_ident(stage_name)} "
            f"WHERE {key_match} ORDER BY tick",
            key,
        )
        old = _BufCursor(old_cur)
        stg = _BufCursor(stg_cur)
        replacements: List[tuple] = []
        prev_running = 0
        old_running = 0
        batch_running = 0
        try:
            ho, hs = old.head(), stg.head()
            while ho is not None or hs is not None:
                if hs is None or (ho is not None and ho[0] < hs[0]):
                    tick, delta = ho
                    old_running += delta
                    old.advance()
                elif ho is None or hs[0] < ho[0]:
                    tick, delta = hs
                    batch_running += delta
                    stg.advance()
                else:
                    tick = ho[0]
                    old_running += ho[1]
                    batch_running += hs[1]
                    old.advance()
                    stg.advance()
                running = old_running + batch_running
                abs_delta = running - prev_running
                if abs_delta != 0:
                    replacements.append(
                        key + (tick, encode_int_blob(abs_delta))
                    )
                prev_running = running
                ho, hs = old.head(), stg.head()
        finally:
            old.close()
            stg.close()
        deleted = conn.execute(
            f"DELETE FROM {quote_ident(tbl)} WHERE {key_match}", key
        ).rowcount
        stats.relation_events_deleted += int(deleted)
        if replacements:
            conn.executemany(insert_sql, replacements)
            stats.relation_events_upserted += len(replacements)

