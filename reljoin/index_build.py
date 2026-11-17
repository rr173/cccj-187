"""Build a read-only on-disk index from JSONL relations.

Layout produced by :func:`build_index`::

    final/
      manifest.json                  # written last; presence means "complete"
      atoms/<relation>.<digest>.db   # one WITHOUT ROWID table per distinct
                                     # (relation, variable assignment) atom

Each table's primary key is the atom's variables **projected onto the global
variable order**, so every prefix of the global order is an index prefix and
the atom's access order (its "layer order") never depends on how the JSONL
file happened to order its columns.  The payload ``m`` is a 16-byte
big-endian multiplicity; it never enters SQLite integer arithmetic, so the
bag count cannot silently overflow 64 bits.

Nothing is written in place.  Relations are streamed through a bounded
external merge sort and assembled in a private temporary directory that is
renamed into place only after every file is flushed and the manifest (with
content hashes) is written.  A failed or cancelled build removes the staging
directory, so a half-built index can never be mistaken for a complete one.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple

from .declaration import Atom, Declaration, fingerprint
from .errors import DataProblem, IndexProblem, ResourceError
from .jsonl import decode_int, scan_source
from .sorter import ExternalSorter

INDEX_FORMAT = "reljoin-index-v1"
INDEX_FORMAT_VERSION = 1
MANIFEST = "manifest.json"
ATOMS_DIR = "atoms"
COUNT_BYTES = 16
MAX_COUNT = 1 << (8 * COUNT_BYTES)
MIN_BUILD_MEMORY = 1 << 20


def quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def table_filename(relation: str, vars_: Tuple[str, ...]) -> str:
    digest = hashlib.sha256("\x1f".join(vars_).encode("utf-8")).hexdigest()[:16]
    safe = "".join(ch if (ch.isalnum() or ch in "-_.") else "_" for ch in relation)
    return f"{safe}.{digest}.db"


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


def _fsync_dir(path: str) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


class TableSpec:
    """One physical table: a distinct (relation, vars) atom signature."""

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


def build_index(
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
    """Build a read-only index and atomically publish it at ``out_dir``."""
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
    staging = tempfile.mkdtemp(prefix=".reljoin-build-", dir=parent)
    try:
        manifest = _build_into(
            decl, data_dir, staging, memory_bytes, sort_dir, sqlite_cache_kib,
            progress,
        )
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    # Publish.  With --force the old index is moved aside first, so at any
    # instant out_dir is either the old complete index or the new complete
    # index, never a partial one.
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
    atoms_dir = os.path.join(staging, ATOMS_DIR)
    os.makedirs(atoms_dir)

    seen: Dict[Tuple[str, Tuple[str, ...]], TableSpec] = {}
    tables: List[TableSpec] = []
    for atom in decl.atoms:
        sig = (atom.relation, atom.vars)
        if sig in seen:
            continue
        spec = TableSpec(decl, atom)
        seen[sig] = spec
        tables.append(spec)

    sort_budget = max(1 << 20, int(memory_bytes * 0.5))
    if sqlite_cache_kib is None:
        sqlite_cache_kib = max(256, int(memory_bytes * 0.4) // 1024)

    t_start = time.time()
    table_records: List[Dict[str, Any]] = []
    relation_stats: Dict[str, Dict[str, Any]] = {}

    for spec in tables:
        src = os.path.join(data_dir, spec.relation + ".jsonl")
        sorter = ExternalSorter(
            key_width=8 * spec.arity,
            payload_width=COUNT_BYTES,
            cache_bytes=sort_budget,
            directory=sort_dir,
        )
        it, digest = scan_source(src, spec.arity)
        payload = (1).to_bytes(COUNT_BYTES, "big")
        try:
            if spec.key_columns == tuple(range(spec.arity)):
                for key in it:
                    sorter.add(key, payload)
            else:
                w = 8
                cols = spec.key_columns
                for key in it:
                    sorter.add(b"".join(key[i * w:(i + 1) * w] for i in cols),
                               payload)
        except BaseException:
            sorter.close()
            raise
        db_path = os.path.join(atoms_dir, spec.file)
        stats = {"duplicate_groups": 0, "duplicate_rows": 0}
        try:
            rows = _merge_into_table(sorter, spec, db_path, sqlite_cache_kib,
                                     stats)
        finally:
            sorter.close()
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
            "duplicate_rows": stats["duplicate_rows"],
            "duplicate_groups": stats["duplicate_groups"],
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
                f"{spec.relation} (key {' '.join(spec.key_vars)}): "
                f"{digest.records} source rows -> {rows} index rows"
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
        "format": INDEX_FORMAT,
        "format_version": INDEX_FORMAT_VERSION,
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
            "started_at": t_start,
            "finished_at": time.time(),
            "seconds": round(time.time() - t_start, 3),
        },
    }
    mpath = os.path.join(staging, MANIFEST)
    with open(mpath, "wb") as fh:
        fh.write(json.dumps(manifest, sort_keys=True, indent=2,
                            ensure_ascii=True).encode("utf-8"))
        fh.flush()
        os.fsync(fh.fileno())
    _fsync_dir(atoms_dir)
    _fsync_dir(staging)
    return manifest


def _merge_into_table(sorter: ExternalSorter, spec: TableSpec, db_path: str,
                      cache_kib: int, stats: Dict[str, int]) -> int:
    """Drain a sorted record stream into a SQLite table, summing duplicates."""
    arity = spec.arity
    key_w = 8 * arity
    cols = ", ".join(f"c{i}" for i in range(arity))
    coldefs = ", ".join(f"c{i} INTEGER NOT NULL" for i in range(arity))
    pk = ", ".join(f"c{i}" for i in spec.key_columns)
    placeholders = ",".join("?" * (arity + 1))
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA journal_mode=OFF")
        conn.execute("PRAGMA synchronous=OFF")
        conn.execute("PRAGMA temp_store=MEMORY")
        conn.execute(f"PRAGMA cache_size=-{int(cache_kib)}")
        conn.execute(
            f"CREATE TABLE t ({coldefs}, m BLOB NOT NULL, "
            f"PRIMARY KEY ({pk})) WITHOUT ROWID"
        )
        insert_sql = f"INSERT INTO t ({cols}, m) VALUES ({placeholders})"
        pending: List[Tuple[Any, ...]] = []
        pending_bytes = 0
        limit = max(4096, (cache_kib * 1024) // 8)
        rows = 0

        def flush() -> None:
            nonlocal pending, pending_bytes
            if pending:
                conn.executemany(insert_sql, pending)
                pending = []
                pending_bytes = 0

        def emit(key: bytes, count: int) -> None:
            nonlocal pending_bytes, rows
            # ``key`` is in primary-key (level) order; scatter it back into
            # physical file-column order for the INSERT.
            physical = [0] * arity
            for i, col in enumerate(spec.key_columns):
                physical[col] = decode_int(key[i * 8:(i + 1) * 8])
            pending.append(tuple(physical) + (count.to_bytes(COUNT_BYTES, "big"),))
            pending_bytes += COUNT_BYTES + 8 * arity + 8
            rows += 1
            if pending_bytes >= limit:
                flush()

        prev_key: Optional[bytes] = None
        prev_count = 0
        for rec in sorter.finish():
            key = rec[:key_w]
            count = int.from_bytes(rec[key_w:], "big")
            if prev_key is not None and key == prev_key:
                prev_count += count
                if prev_count >= MAX_COUNT:
                    raise DataProblem(
                        f"multiplicity overflow while indexing "
                        f"{spec.relation!r}: a tuple occurs at least "
                        f"{MAX_COUNT} times"
                    )
                continue
            if prev_key is not None:
                if prev_count > 1:
                    stats["duplicate_groups"] += 1
                    stats["duplicate_rows"] += prev_count - 1
                emit(prev_key, prev_count)
            prev_key = key
            prev_count = count
        if prev_key is not None:
            if prev_count > 1:
                stats["duplicate_groups"] += 1
                stats["duplicate_rows"] += prev_count - 1
            emit(prev_key, prev_count)
        flush()
        conn.commit()
        return rows
    finally:
        conn.close()
