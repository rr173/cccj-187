"""Build a read-only *temporal* (valid-time) index from interval JSONL files.

The on-disk layout mirrors the static index but lives under ``tables/`` and
uses a different manifest format, so the two index kinds can never be opened
for the wrong command::

    final/
      manifest.json                   # reljoin-temporal-index-v1; written last
      tables/<relation>.<digest>.db   # event table per distinct atom signature

Every input interval ``[since, until)`` with ``copies`` copies becomes two
events, ``+copies`` at ``since`` and ``-copies`` at ``until``.  Events are run
through the same bounded external merge sort machinery as the static builder;
the merge step sums deltas that share a tuple and a tick and drops groups whose
sum is zero.  The resulting table is keyed by the tuple columns in global
variable order plus the tick, so the whole valid-time trajectory of one tuple
is one contiguous primary-key range.

Builds happen in a hidden staging directory and are published by rename: a
failure (bad line, out-of-range value, cancellation) removes the staging tree,
and a forced rebuild keeps the previous index until the new one is complete.
"""
from __future__ import annotations

import json
import os
import shutil
import sqlite3
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple

from .declaration import Atom, Declaration, fingerprint
from .errors import DataProblem, IndexProblem, ResourceError
from .index_build import MIN_BUILD_MEMORY, _fsync_dir, _hash_file, table_filename
from .jsonl import decode_int, encode_int
from .sorter import VariableExternalSorter
from .temporal_format import (
    TEMPORAL_FORMAT_VERSION,
    TEMPORAL_INDEX_FORMAT,
    TEMPORAL_MANIFEST,
    TEMPORAL_TABLES_DIR,
    encode_delta,
)
from .temporal_jsonl import scan_temporal_source


class TemporalTableSpec:
    """One physical temporal table: a distinct (relation, vars) signature."""

    __slots__ = ("relation", "vars", "arity", "file", "key_columns",
                 "key_vars", "key_levels")

    def __init__(self, decl: Declaration, atom: Atom) -> None:
        self.relation = atom.relation
        self.vars = atom.vars
        self.arity = atom.arity
        self.file = table_filename(atom.relation, atom.vars)
        self.key_columns = tuple(
            sorted(range(self.arity),
                   key=lambda i: decl.var_position[atom.vars[i]])
        )
        self.key_vars = tuple(atom.vars[i] for i in self.key_columns)
        self.key_levels = tuple(decl.var_position[v] for v in self.key_vars)


def build_temporal_index(
    decl: Declaration,
    *,
    data_dir: str,
    out_dir: str,
    memory_bytes: int,
    sort_dir: Optional[str] = None,
    sqlite_cache_kib: Optional[int] = None,
    force: bool = False,
    progress=None,
) -> Dict[str, Any]:
    """Build a temporal index and atomically publish it at ``out_dir``."""
    if memory_bytes < MIN_BUILD_MEMORY:
        raise ResourceError(
            f"--memory must be at least {MIN_BUILD_MEMORY >> 20} MiB for a build"
        )
    if not os.path.isdir(data_dir):
        raise DataProblem(f"data directory {data_dir!r} does not exist")
    out_dir = os.path.abspath(out_dir)
    parent = os.path.dirname(out_dir) or "."
    os.makedirs(parent, exist_ok=True)
    if os.path.exists(out_dir):
        if not os.path.isdir(out_dir):
            raise IndexProblem(f"--index {out_dir!r} exists and is not a directory")
        if os.listdir(out_dir) and not force:
            raise IndexProblem(
                f"--index {out_dir!r} already exists and is not empty; "
                "remove it or pass --force to replace it atomically"
            )
    staging = tempfile.mkdtemp(prefix=".reljoin-tbuild-", dir=parent)
    try:
        manifest = _build_into(
            decl, data_dir, staging, memory_bytes, sort_dir, sqlite_cache_kib,
            progress,
        )
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    replaced: Optional[str] = None
    try:
        if os.path.isdir(out_dir):
            replaced = parent + "/.reljoin-old-" + os.path.basename(staging)
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
    return manifest


def _build_into(decl, data_dir, staging, memory_bytes, sort_dir,
                sqlite_cache_kib, progress) -> Dict[str, Any]:
    tables_dir = os.path.join(staging, TEMPORAL_TABLES_DIR)
    os.makedirs(tables_dir)

    seen: Dict[Tuple[str, Tuple[str, ...]], TemporalTableSpec] = {}
    tables: List[TemporalTableSpec] = []
    for atom in decl.atoms:
        sig = (atom.relation, atom.vars)
        if sig in seen:
            continue
        spec = TemporalTableSpec(decl, atom)
        seen[sig] = spec
        tables.append(spec)

    sort_budget = max(1 << 20, int(memory_bytes * 0.5))
    if sqlite_cache_kib is None:
        sqlite_cache_kib = max(256, int(memory_bytes * 0.4) // 1024)

    t_start = time.time()
    table_records: List[Dict[str, Any]] = []
    relation_stats: Dict[str, Dict[str, Any]] = {}
    total_spill_bytes = 0
    total_runs = 0
    total_passes = 0

    for spec in tables:
        src = os.path.join(data_dir, spec.relation + ".jsonl")
        # Sort key: tuple columns in global order + tick.  Payload: delta blob.
        key_width = 8 * (spec.arity + 1)
        sorter = VariableExternalSorter(
            key_width=key_width, cache_bytes=sort_budget, directory=sort_dir,
        )
        it, digest = scan_temporal_source(spec.relation, src, spec.arity)
        w = 8
        cols = spec.key_columns
        try:
            for file_key, tick, delta in it:
                ordered = b"".join(
                    file_key[i * w:(i + 1) * w] for i in cols
                )
                sorter.add(ordered + encode_int(tick), encode_delta(delta))
        except BaseException:
            sorter.close()
            raise
        db_path = os.path.join(tables_dir, spec.file)
        stats = {"coalesced": 0, "dropped_zero": 0, "min_tick": None,
                 "max_tick": None}
        try:
            rows = _merge_into_event_table(
                sorter, spec, db_path, sqlite_cache_kib, stats
            )
        finally:
            sorter.close()
        total_spill_bytes += sorter.bytes_spilled
        total_runs += sorter.runs_written
        total_passes += sorter.merge_passes
        sha, size = _hash_file(db_path)
        table_records.append({
            "relation": spec.relation,
            "vars": list(spec.vars),
            "file": spec.file,
            "arity": spec.arity,
            "key_columns": list(spec.key_columns),
            "key_vars": list(spec.key_vars),
            "key_levels": list(spec.key_levels),
            "rows": rows,
            "source_records": digest.records,
            "interval_events": digest.interval_events,
            "coalesced_events": stats["coalesced"],
            "dropped_zero_events": stats["dropped_zero"],
            "min_tick": stats["min_tick"],
            "max_tick": stats["max_tick"],
            "sha256": sha,
            "bytes": size,
        })
        cur = {
            "file": spec.relation + ".jsonl",
            "sha256": digest.sha256.hexdigest(),
            "bytes": digest.bytes,
            "records": digest.records,
            "blank_lines": digest.blank_lines,
            "arity": spec.arity,
        }
        prev = relation_stats.get(spec.relation)
        if prev is not None and (
            prev["sha256"] != cur["sha256"] or prev["records"] != cur["records"]
        ):
            raise DataProblem(
                f"relation {spec.relation!r} changed while it was being indexed"
            )
        relation_stats[spec.relation] = cur
        if progress:
            progress(
                f"{spec.relation} (key {' '.join(spec.key_vars)}, tick): "
                f"{digest.records} intervals / {digest.interval_events} events "
                f"-> {rows} stored events"
            )

    atom_records = []
    for atom in decl.atoms:
        spec = seen[(atom.relation, atom.vars)]
        atom_records.append({
            "alias": atom.alias,
            "relation": atom.relation,
            "vars": list(atom.vars),
            "table": spec.file,
            "key_vars": list(spec.key_vars),
            "key_levels": list(spec.key_levels),
        })

    manifest: Dict[str, Any] = {
        "format": TEMPORAL_INDEX_FORMAT,
        "format_version": TEMPORAL_FORMAT_VERSION,
        "kind": "temporal",
        "declaration": {
            "version": 1,
            "order": list(decl.order),
            "relations": [
                {"name": a.relation, "vars": list(a.vars)} for a in decl.atoms
            ],
        },
        "declaration_fingerprint": fingerprint(decl),
        "declaration_source": {
            "path": decl.path,
            "sha256": _hash_file(decl.path)[0],
        },
        "order": list(decl.order),
        "tables": table_records,
        "atoms": atom_records,
        "relations": relation_stats,
        "build": {
            "memory_bytes": memory_bytes,
            "sqlite_cache_kib": sqlite_cache_kib,
            "sort_runs_written": total_runs,
            "sort_merge_passes": total_passes,
            "sort_bytes_spilled": total_spill_bytes,
            "started_at": t_start,
            "finished_at": time.time(),
            "seconds": round(time.time() - t_start, 3),
        },
    }
    mpath = os.path.join(staging, TEMPORAL_MANIFEST)
    with open(mpath, "wb") as fh:
        fh.write(json.dumps(manifest, sort_keys=True, indent=2,
                            ensure_ascii=True).encode("utf-8"))
        fh.flush()
        os.fsync(fh.fileno())
    _fsync_dir(tables_dir)
    _fsync_dir(staging)
    return manifest


def _merge_into_event_table(sorter: VariableExternalSorter,
                            spec: TemporalTableSpec, db_path: str,
                            cache_kib: int, stats: Dict[str, Any]) -> int:
    """Drain sorted events, coalesce same (tuple, tick), insert survivors."""
    arity = spec.arity
    tuple_w = 8 * arity
    cols = ", ".join(f"c{i}" for i in range(arity))
    coldefs = ", ".join(f"c{i} INTEGER NOT NULL" for i in range(arity))
    pk = ", ".join([*(f"c{i}" for i in spec.key_columns), "tick"])
    placeholders = ",".join("?" * (arity + 2))
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA journal_mode=OFF")
        conn.execute("PRAGMA synchronous=OFF")
        conn.execute("PRAGMA temp_store=MEMORY")
        conn.execute(f"PRAGMA cache_size=-{int(cache_kib)}")
        conn.execute(
            f"CREATE TABLE t ({coldefs}, tick INTEGER NOT NULL, "
            f"d BLOB NOT NULL, PRIMARY KEY ({pk})) WITHOUT ROWID"
        )
        insert_sql = f"INSERT INTO t ({cols}, tick, d) VALUES ({placeholders})"
        pending: List[Tuple[Any, ...]] = []
        pending_bytes = 0
        limit = max(2048, (cache_kib * 1024) // 8)
        rows = 0

        def flush() -> None:
            nonlocal pending, pending_bytes
            if pending:
                conn.executemany(insert_sql, pending)
                pending = []
                pending_bytes = 0

        def emit(tuple_key: bytes, tick: int, total: int) -> None:
            nonlocal pending_bytes, rows
            physical = [0] * arity
            for i, col in enumerate(spec.key_columns):
                physical[col] = decode_int(tuple_key[i * 8:(i + 1) * 8])
            pending.append(
                tuple(physical) + (tick, encode_delta(total))
            )
            pending_bytes += 8 * (arity + 1) + 32
            rows += 1
            if stats["min_tick"] is None or tick < stats["min_tick"]:
                stats["min_tick"] = tick
            if stats["max_tick"] is None or tick > stats["max_tick"]:
                stats["max_tick"] = tick
            if pending_bytes >= limit:
                flush()

        from .temporal_format import decode_delta
        prev_kt: Optional[bytes] = None
        prev_total = 0
        prev_group = 0
        for rec in sorter.finish():
            tuple_key = rec[:tuple_w]
            tick = decode_int(rec[tuple_w:tuple_w + 8])
            delta = decode_delta(rec[tuple_w + 8:])
            kt = rec[:tuple_w + 8]
            if prev_kt is not None and kt == prev_kt:
                prev_total += delta
                prev_group += 1
                continue
            if prev_kt is not None:
                if prev_group > 1:
                    stats["coalesced"] += 1
                if prev_total != 0:
                    emit(prev_tuple, prev_tick, prev_total)
                else:
                    stats["dropped_zero"] += 1
            prev_kt = kt
            prev_tuple = tuple_key
            prev_tick = tick
            prev_total = delta
            prev_group = 1
        if prev_kt is not None:
            if prev_group > 1:
                stats["coalesced"] += 1
            if prev_total != 0:
                emit(prev_tuple, prev_tick, prev_total)
            else:
                stats["dropped_zero"] += 1
        flush()
        conn.commit()
        return rows
    finally:
        conn.close()
