"""Create an independent maintenance library from a temporal index.

``maintain-init`` is the *only* step allowed full computation.  It:

1. opens the existing verified temporal index;
2. copies every relation's current event state into the maintenance database
   (one table per physical relation, transcoded into the unbounded maintenance
   integer codec) — aliases with different variable assignments share that one
   table;
3. builds the initial closed-loop view once, with the same layered
   leapfrog + per-binding event sweep as ``temporal-run`` over the all-time
   window, and stores it;
4. records ``revision = 0`` and publishes the whole directory atomically.

After a successful initialisation the temporal index and every original
interval file can be deleted: the maintenance library is self-contained and
reopening it never touches the source again.
"""
from __future__ import annotations

import json
import os
import shutil
import sqlite3
import tempfile
import time
from typing import Any, Dict, List, Optional

from .declaration import Atom, Declaration, fingerprint
from .errors import IndexProblem, ResourceError
from .index_build import _fsync_dir
from .maintain_format import (
    MAINTAIN_DB,
    MAINTAIN_FORMAT_VERSION,
    MAINTAIN_INDEX_FORMAT,
    MAINTAIN_MANIFEST,
    MAINTAIN_TMP_PREFIX,
    encode_int_blob,
    physical_column_spec,
    quote_ident,
    relation_table_name,
)
from .maintain_solver import MaintSolver, MaintStats
from .temporal_format import decode_delta
from .temporal_open import TemporalOpenIndex

MIN_INIT_MEMORY = 1 << 20
INT64_MIN = -(1 << 63)
INT64_MAX = (1 << 63) - 1


def _key_columns_for(decl: Declaration, atom: Atom) -> List[int]:
    return list(sorted(
        range(atom.arity),
        key=lambda i: decl.var_position[atom.vars[i]],
    ))


def init_maintenance(
    decl: Declaration,
    temporal_index: TemporalOpenIndex,
    *,
    out_dir: str,
    memory_bytes: int,
    sort_dir: Optional[str] = None,
    sqlite_cache_kib: Optional[int] = None,
    force: bool = False,
    progress=None,
) -> Dict[str, Any]:
    """Build and atomically publish a maintenance library at ``out_dir``."""
    if memory_bytes < MIN_INIT_MEMORY:
        raise ResourceError(
            f"--memory must be at least {MIN_INIT_MEMORY >> 20} MiB"
        )
    out_dir = os.path.abspath(out_dir)
    parent = os.path.dirname(out_dir) or "."
    os.makedirs(parent, exist_ok=True)
    if os.path.exists(out_dir):
        if not os.path.isdir(out_dir):
            raise IndexProblem(
                f"--maintain {out_dir!r} exists and is not a directory"
            )
        if os.listdir(out_dir) and not force:
            raise IndexProblem(
                f"--maintain {out_dir!r} already exists and is not empty; "
                "remove it or pass --force to replace it atomically"
            )
    staging = tempfile.mkdtemp(prefix=MAINTAIN_TMP_PREFIX, dir=parent)
    started = time.time()
    stats = MaintStats()
    try:
        manifest = _build_into(
            decl, temporal_index, staging, memory_bytes, sqlite_cache_kib,
            progress, stats,
        )
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    replaced: Optional[str] = None
    try:
        if os.path.isdir(out_dir):
            replaced = os.path.join(
                parent, ".reljoin-old-" + os.path.basename(staging)
            )
            os.rename(out_dir, replaced)
        os.rename(staging, out_dir)
        _fsync_dir(parent)
    except BaseException:
        if replaced is not None and not os.path.exists(out_dir):
            try:
                os.rename(replaced, out_dir)
                replaced = None
            except OSError:
                pass
        shutil.rmtree(staging, ignore_errors=True)
        raise
    finally:
        if replaced is not None:
            shutil.rmtree(replaced, ignore_errors=True)
    manifest["init"] = {
        "seconds": round(time.time() - started, 3),
        "memory_bytes": memory_bytes,
        "report": stats.as_dict(decl),
    }
    # Re-write the manifest inside the published directory with the report.
    mpath = os.path.join(out_dir, MAINTAIN_MANIFEST)
    with open(mpath, "wb") as fh:
        fh.write(json.dumps(manifest, sort_keys=True, indent=2,
                            ensure_ascii=True).encode("utf-8"))
        fh.flush()
        os.fsync(fh.fileno())
    _fsync_dir(out_dir)
    return manifest


def _connect_staging(db_path: str, cache_kib: int) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, isolation_level=None)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=FULL")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute(f"PRAGMA cache_size=-{int(cache_kib)}")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _build_into(decl, temporal_index, staging, memory_bytes, sqlite_cache_kib,
                progress, stats: MaintStats) -> Dict[str, Any]:
    db_path = os.path.join(staging, MAINTAIN_DB)
    if sqlite_cache_kib is None:
        sqlite_cache_kib = max(256, int(memory_bytes * 0.4) // 1024)
    conn = _connect_staging(db_path, sqlite_cache_kib)
    depth = len(decl.order)
    try:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            "CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        conn.execute("INSERT INTO meta VALUES ('revision', '0')")

        relation_records: List[Dict[str, Any]] = []
        copied_relations = set()
        # One physical table per relation name, keyed in *file-column* order
        # (c0, c1, ...).  Every alias carries its own key_columns permutation
        # mapping the global variable order onto these physical columns, so
        # aliases with different variable assignments (e.g. R(x,y) and R(y,x))
        # share one table while each seeking it in its own access order.
        for atom in decl.atoms:
            if atom.relation in copied_relations:
                continue
            copied_relations.add(atom.relation)
            arity = decl.relation_arity[atom.relation]
            tname = relation_table_name(atom.relation)
            coldefs = ", ".join(
                f"c{i} INTEGER NOT NULL" for i in range(arity)
            )
            key_cols = list(range(arity))
            pk = ", ".join([*(f"c{i}" for i in key_cols), "tick"])
            conn.execute(
                f"CREATE TABLE {quote_ident(tname)} ({coldefs}, "
                f"tick INTEGER NOT NULL, d BLOB NOT NULL, "
                f"PRIMARY KEY ({pk})) WITHOUT ROWID"
            )
            relation_records.append({
                "relation": atom.relation,
                "table": tname,
                "arity": arity,
                "key_columns": key_cols,
                "key_vars": [atom.vars[i] for i in key_cols],
                "key_levels": [
                    decl.var_position[atom.vars[i]] for i in key_cols
                ],
            })

        view_cols = ", ".join(f"b{i} INTEGER NOT NULL" for i in range(depth))
        view_pk = ", ".join([*(f"b{i}" for i in range(depth)), "since"])
        conn.execute(
            f"CREATE TABLE view ({view_cols}, since INTEGER NOT NULL, "
            f"until INTEGER NOT NULL, multiplicity BLOB NOT NULL, "
            f"PRIMARY KEY ({view_pk})) WITHOUT ROWID"
        )
        conn.execute(
            "CREATE TABLE batches (batch_id TEXT PRIMARY KEY, "
            "revision INTEGER NOT NULL, expect_revision INTEGER NOT NULL, "
            "file_sha256 TEXT NOT NULL, file_bytes INTEGER NOT NULL, "
            "records INTEGER NOT NULL, delta_rows INTEGER NOT NULL, "
            "weighted_change TEXT NOT NULL, committed_at REAL NOT NULL)"
        )
        conn.execute(
            "CREATE TABLE batch_delta (batch_id TEXT NOT NULL, "
            "seq INTEGER NOT NULL, "
            + ", ".join(f"b{i} INTEGER NOT NULL" for i in range(depth))
            + ", since INTEGER NOT NULL, until INTEGER NOT NULL, "
            "delta_multiplicity BLOB NOT NULL, "
            "PRIMARY KEY (batch_id, seq)) WITHOUT ROWID"
        )

        # -- phase 1: copy relation state from the temporal index --------
        _copy_relation_state(
            decl, temporal_index, conn, stats, progress
        )

        # -- phase 2: full initial closed-loop view ----------------------
        _build_initial_view(
            decl, conn, depth, memory_bytes, stats, progress
        )
        conn.execute("COMMIT")
        # Fold the WAL into the main db file before publishing.
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()

    atom_records = []
    for atom in decl.atoms:
        key_columns = _key_columns_for(decl, atom)
        key_vars = [atom.vars[i] for i in key_columns]
        key_levels = [decl.var_position[v] for v in key_vars]
        atom_records.append({
            "alias": atom.alias,
            "relation": atom.relation,
            "vars": list(atom.vars),
            "table": relation_table_name(atom.relation),
            "key_columns": key_columns,
            "key_vars": key_vars,
            "key_levels": key_levels,
        })

    manifest: Dict[str, Any] = {
        "format": MAINTAIN_INDEX_FORMAT,
        "format_version": MAINTAIN_FORMAT_VERSION,
        "kind": "maintain",
        "declaration": {
            "version": 1,
            "order": list(decl.order),
            "relations": [
                {"name": a.relation, "vars": list(a.vars)}
                for a in decl.atoms
            ],
        },
        "declaration_fingerprint": fingerprint(decl),
        "order": list(decl.order),
        "relations": relation_records,
        "atoms": atom_records,
        "created": time.time(),
    }
    mpath = os.path.join(staging, MAINTAIN_MANIFEST)
    with open(mpath, "wb") as fh:
        fh.write(json.dumps(manifest, sort_keys=True, indent=2,
                            ensure_ascii=True).encode("utf-8"))
        fh.flush()
        os.fsync(fh.fileno())
    _fsync_dir(staging)
    return manifest


def _copy_relation_state(decl, temporal_index: TemporalOpenIndex,
                         conn: sqlite3.Connection, stats: MaintStats,
                         progress) -> None:
    """Stream each relation's events out of the source and into the store.

    The source table's columns are ordered by that atom signature's global
    variable order, while the maintenance table stores physical file columns
    c0..c{n-1}.  The source's ``key_columns`` permutation maps each selected
    column back to its file-column index, so rows are reordered on copy.
    """
    copied = set()
    for atom in decl.atoms:
        rel = atom.relation
        if rel in copied:
            continue
        copied.add(rel)
        arity = decl.relation_arity[rel]
        src_rec = temporal_index.tables[(rel, atom.vars)]
        src_conn = temporal_index.connection(src_rec["file"])
        tname = relation_table_name(rel)
        cols = ", ".join(physical_column_spec(arity))
        placeholders = ",".join("?" * (arity + 2))
        insert_sql = (
            f"INSERT INTO {quote_ident(tname)} ({cols}, tick, d) "
            f"VALUES ({placeholders})"
        )
        src_cols = ", ".join(f"c{j}" for j in src_rec["key_columns"])
        select_sql = (
            f"SELECT {src_cols}, tick, d FROM t "
            f"ORDER BY {src_cols}, tick"
        )
        # selected position -> physical file column
        inv = [0] * arity
        for selected_pos, file_col in enumerate(src_rec["key_columns"]):
            inv[selected_pos] = file_col
        cur = src_conn.execute(select_sql)
        pending: List[Any] = []
        rows = 0
        while True:
            batch = cur.fetchmany(1024)
            if not batch:
                break
            for row in batch:
                selected = list(row[:arity])
                values = [0] * arity
                for selected_pos, file_col in enumerate(inv):
                    values[file_col] = selected[selected_pos]
                tick = row[arity]
                delta = decode_delta(row[arity + 1])
                pending.append(
                    tuple(values) + (tick, encode_int_blob(delta))
                )
                rows += 1
            conn.executemany(insert_sql, pending)
            stats.relation_events_copied += len(pending)
            pending.clear()
        stats.relation_rows_after = stats.relation_events_copied
        if progress:
            progress(f"copied relation {rel!r}: {rows} events")


def _build_initial_view(decl, conn, depth, memory_bytes,
                        stats: MaintStats, progress) -> None:
    solver = MaintSolver(
        decl, conn, stats=stats,
        window=(INT64_MIN, INT64_MAX),
        memory_bytes=memory_bytes,
    )
    placeholders = ",".join("?" * (depth + 3))
    bcols = ", ".join([*(f"b{i}" for i in range(depth)),
                       "since", "until", "multiplicity"])
    insert_sql = f"INSERT INTO view ({bcols}) VALUES ({placeholders})"
    pending: List[Any] = []
    segments = 0
    for values, (since, until, mult) in solver.leaves_with_segments():
        pending.append(
            tuple(values) + (since, until, encode_int_blob(mult))
        )
        segments += 1
        if len(pending) >= 512:
            conn.executemany(insert_sql, pending)
            stats.view_rows_written += len(pending)
            pending.clear()
    if pending:
        conn.executemany(insert_sql, pending)
        stats.view_rows_written += len(pending)
    stats.initial_view_segments = segments
    if progress:
        progress(f"initial view: {segments} segment(s) at revision 0")
