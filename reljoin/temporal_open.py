"""Open and verify read-only temporal indexes.

The temporal manifest format is distinct from the static one
(``reljoin-temporal-index-v1`` and a ``tables/`` directory).  Opening a static
index through this module, or a temporal one through the static opener, raises
:class:`WrongIndexKind`, so a mistaken command reports a clear error instead of
silently returning an empty result.

Verification covers the manifest shape, file sizes and SHA-256 hashes, a
read-only SQLite integrity check, and — in deep mode — a streamed consistency
proof over every event table: delta cells must decode, ticks must be ordered
per tuple, the running multiplicity must never be negative and must return to
zero at the end of every tuple.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from typing import Any, Dict, List

from .declaration import declaration_from_payload, fingerprint
from .errors import (
    CorruptIndex,
    IndexProblem,
    MissingIndex,
    WrongIndexKind,
)
from .index_open import _check_filename, _uri_escape
from .temporal_format import (
    TEMPORAL_FORMAT_VERSION,
    TEMPORAL_INDEX_FORMAT,
    TEMPORAL_MANIFEST,
    TEMPORAL_TABLES_DIR,
    decode_delta,
)

_MANIFEST_KEYS = {
    "format", "format_version", "kind", "declaration",
    "declaration_fingerprint", "order", "tables", "atoms", "relations",
    "build",
}


class TemporalOpenIndex:
    """An opened, verified temporal index with its embedded declaration."""

    def __init__(self, path: str, manifest: Dict[str, Any], decl) -> None:
        self.path = path
        self.manifest = manifest
        self.decl = decl
        self.tables: Dict = {}
        for rec in manifest["tables"]:
            self.tables[(rec["relation"], tuple(rec["vars"]))] = rec
        self.table_for_atom: Dict[str, Dict[str, Any]] = {}
        for atom in decl.atoms:
            self.table_for_atom[atom.alias] = self.tables[
                (atom.relation, atom.vars)
            ]
        self._conns: Dict[str, sqlite3.Connection] = {}
        self._closed = False

    # -- verification ----------------------------------------------------
    def verify(self, *, deep: bool = True, progress=None) -> Dict[str, Any]:
        if not os.path.isdir(self.path):
            raise MissingIndex(f"temporal index {self.path!r} disappeared")
        report = {"tables": 0, "bytes": 0, "deep": deep}
        for rec in self.manifest["tables"]:
            fn = rec["file"]
            _check_filename(fn)
            fp = os.path.join(self.path, TEMPORAL_TABLES_DIR, fn)
            _check_file(fp, rec)
            report["tables"] += 1
            report["bytes"] += rec["bytes"]
            if deep:
                self._verify_table(fp, rec)
            if progress:
                progress(f"verified {fn}")
        self._verify_no_extras()
        return report

    def _verify_table(self, fp: str, rec: Dict[str, Any]) -> None:
        uri = "file:" + _uri_escape(fp) + "?mode=ro&immutable=1"
        try:
            conn = sqlite3.connect(uri, uri=True)
        except sqlite3.Error as exc:
            raise CorruptIndex(
                f"cannot open temporal index file {fp!r}: {exc}"
            ) from exc
        try:
            try:
                conn.execute("PRAGMA query_only=ON")
                row = conn.execute("PRAGMA quick_check(1)").fetchone()
            except sqlite3.DatabaseError as exc:
                raise CorruptIndex(
                    f"temporal index file {fp!r} is not a readable SQLite "
                    f"database: {exc}"
                ) from exc
            if row is None or row[0] != "ok":
                raise CorruptIndex(
                    f"temporal index file {fp!r} fails SQLite's integrity "
                    f"check: {row!r}"
                )
            arity = rec["arity"]
            key_cols = rec["key_columns"]
            tcols = ", ".join(f"c{i}" for i in key_cols)
            try:
                n = conn.execute("SELECT COUNT(*) FROM t").fetchone()[0]
            except sqlite3.DatabaseError as exc:
                raise CorruptIndex(
                    f"temporal index file {fp!r} has no usable main table: {exc}"
                ) from exc
            if n != rec["rows"]:
                raise CorruptIndex(
                    f"temporal index file {fp!r} holds {n} events, manifest "
                    f"says {rec['rows']}; the index is damaged"
                )
            # Streamed prefix-sum proof: per tuple, deltas must never make the
            # running multiplicity negative and must return exactly to zero.
            sql = f"SELECT {tcols}, tick, d FROM t ORDER BY {tcols}, tick"
            try:
                cur = conn.execute(sql)
            except sqlite3.DatabaseError as exc:
                raise CorruptIndex(
                    f"temporal index file {fp!r} cannot stream its events: {exc}"
                ) from exc
            prev_key = None
            running = 0
            seen = 0
            while True:
                batch = cur.fetchmany(2048)
                if not batch:
                    break
                for row in batch:
                    key = tuple(row[:arity])
                    if key != prev_key:
                        if prev_key is not None and running != 0:
                            raise CorruptIndex(
                                f"temporal index file {fp!r}: tuple {prev_key} "
                                f"ends with multiplicity {running}, not zero"
                            )
                        prev_key = key
                        running = 0
                    delta = decode_delta(row[arity + 1])
                    running += delta
                    seen += 1
                    if running < 0:
                        raise CorruptIndex(
                            f"temporal index file {fp!r}: tuple {key} has a "
                            f"negative multiplicity at tick {row[arity]} "
                            f"(running sum {running})"
                        )
            if seen != n:
                raise CorruptIndex(
                    f"temporal index file {fp!r}: streamed {seen} events but "
                    f"the table holds {n}"
                )
            if prev_key is not None and running != 0:
                raise CorruptIndex(
                    f"temporal index file {fp!r}: tuple {prev_key} ends with "
                    f"multiplicity {running}, not zero"
                )
        finally:
            conn.close()

    def _verify_no_extras(self) -> None:
        tables_dir = os.path.join(self.path, TEMPORAL_TABLES_DIR)
        expected = {rec["file"] for rec in self.manifest["tables"]}
        try:
            names = set(os.listdir(tables_dir))
        except FileNotFoundError as exc:
            raise CorruptIndex(
                f"temporal index {self.path!r} has no {TEMPORAL_TABLES_DIR}/ "
                "directory"
            ) from exc
        extras = sorted(names - expected)
        if extras:
            raise CorruptIndex(
                "temporal index contains unexpected file(s): "
                + ", ".join(extras[:5])
            )
        missing = sorted(expected - names)
        if missing:
            raise CorruptIndex(
                "temporal index is missing table file(s): "
                + ", ".join(missing[:5])
            )

    # -- query support ---------------------------------------------------
    def connection(self, file: str) -> sqlite3.Connection:
        if self._closed:
            raise IndexProblem("temporal index is closed")
        conn = self._conns.get(file)
        if conn is not None:
            return conn
        fp = os.path.join(self.path, TEMPORAL_TABLES_DIR, file)
        if not os.path.isfile(fp):
            raise CorruptIndex(
                f"temporal index file {fp!r} disappeared during the query"
            )
        uri = "file:" + _uri_escape(fp) + "?mode=ro&immutable=1"
        try:
            conn = sqlite3.connect(uri, uri=True, isolation_level=None)
            conn.execute("PRAGMA query_only=ON")
            conn.execute("PRAGMA cache_size=-8192")
        except sqlite3.Error as exc:
            raise CorruptIndex(
                f"cannot open temporal index file {fp!r}: {exc}"
            ) from exc
        self._conns[file] = conn
        return conn

    def connections(self) -> List[sqlite3.Connection]:
        return list(self._conns.values())

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for conn in self._conns.values():
            try:
                conn.close()
            except sqlite3.Error:
                pass
        self._conns.clear()

    def __enter__(self) -> "TemporalOpenIndex":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def describe(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "kind": "temporal",
            "declaration": self.decl.describe(),
            "fingerprint": self.manifest["declaration_fingerprint"],
            "build": self.manifest["build"],
        }


def _check_file(fp: str, rec: Dict[str, Any]) -> None:
    try:
        st = os.stat(fp)
    except FileNotFoundError as exc:
        raise CorruptIndex(
            f"temporal index file {fp!r} named by the manifest is missing; the "
            "index is incomplete"
        ) from exc
    except OSError as exc:
        raise CorruptIndex(f"cannot stat temporal index file {fp!r}: {exc}") from exc
    if not os.path.isfile(fp):
        raise CorruptIndex(f"temporal index file {fp!r} is not a regular file")
    if st.st_size != rec["bytes"]:
        raise CorruptIndex(
            f"temporal index file {fp!r} has {st.st_size} bytes, manifest says "
            f"{rec['bytes']}; the index is damaged or truncated"
        )
    h = hashlib.sha256()
    with open(fp, "rb") as fh:
        while True:
            chunk = fh.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    if h.hexdigest() != rec["sha256"]:
        raise CorruptIndex(
            f"temporal index file {fp!r} fails its SHA-256 check; the index "
            "is damaged"
        )


def open_temporal_index(path: str, *, verify: bool = True, deep: bool = True,
                        expected_declaration=None, progress=None
                        ) -> TemporalOpenIndex:
    """Open and verify a temporal index; raise on missing/damaged/wrong kind."""
    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise MissingIndex(
            f"temporal index {path!r} does not exist; run 'temporal-prepare' "
            "first"
        )
    if not os.path.isdir(path):
        raise IndexProblem(f"index path {path!r} is not a directory")
    mpath = os.path.join(path, TEMPORAL_MANIFEST)
    if not os.path.exists(mpath):
        # The static manifest would sit at the same name; distinguish the two.
        static_manifest = os.path.join(path, "manifest.json")
        if os.path.exists(static_manifest):
            try:
                with open(static_manifest, "rb") as fh:
                    other = json.loads(fh.read().decode("utf-8"))
                if other.get("format") != TEMPORAL_INDEX_FORMAT:
                    raise WrongIndexKind(
                        f"{path!r} is a static (validity-less) index; use "
                        "'prepare'/'run' for it, or build a temporal index "
                        "with 'temporal-prepare'"
                    )
            except (UnicodeDecodeError, json.JSONDecodeError):
                pass
        raise MissingIndex(
            f"temporal index {path!r} has no {TEMPORAL_MANIFEST}; it was "
            "never completed"
        )
    try:
        with open(mpath, "rb") as fh:
            raw = fh.read()
    except OSError as exc:
        raise IndexProblem(
            f"cannot read temporal manifest {mpath!r}: {exc}"
        ) from exc
    try:
        manifest = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CorruptIndex(
            f"temporal manifest {mpath!r} is not valid JSON: {exc}"
        ) from exc
    _check_manifest(manifest, mpath)

    decl = declaration_from_payload(manifest["declaration"], path=mpath,
                                    embedded=True)
    if fingerprint(decl) != manifest["declaration_fingerprint"]:
        raise CorruptIndex(
            "temporal manifest declaration does not match its fingerprint; "
            "the index is damaged"
        )
    if expected_declaration is not None:
        if fingerprint(expected_declaration) != manifest["declaration_fingerprint"]:
            raise IndexProblem(
                "temporal index was built for a different declaration "
                f"(index {manifest['declaration_fingerprint'][:16]}…, query "
                f"{fingerprint(expected_declaration)[:16]}…); run "
                "'temporal-prepare' for the current declaration"
            )
    idx = TemporalOpenIndex(path, manifest, decl)
    if verify:
        try:
            idx.verify(deep=deep, progress=progress)
        except BaseException:
            idx.close()
            raise
    return idx


def _check_manifest(obj: Any, where: str) -> None:
    if not isinstance(obj, dict):
        raise CorruptIndex(f"{where}: temporal manifest is not a JSON object")
    if obj.get("format") == "reljoin-index-v1":
        raise WrongIndexKind(
            f"{where}: this is a static index (format {obj['format']!r}); "
            "temporal commands need a temporal index built with "
            "'temporal-prepare'"
        )
    missing = _MANIFEST_KEYS - set(obj)
    if missing:
        raise CorruptIndex(
            f"{where}: temporal manifest is missing key(s): "
            + ", ".join(sorted(missing))
        )
    if obj["format"] != TEMPORAL_INDEX_FORMAT:
        if obj["format"] == "reljoin-index-v1":
            raise WrongIndexKind(
                f"{where}: this is a static index (format {obj['format']!r}); "
                "temporal commands need a temporal index built with "
                "'temporal-prepare'"
            )
        raise CorruptIndex(
            f"{where}: unsupported index format {obj['format']!r}"
        )
    if obj["format_version"] != TEMPORAL_FORMAT_VERSION:
        raise CorruptIndex(
            f"{where}: unsupported temporal format version "
            f"{obj['format_version']!r}"
        )
    if obj.get("kind") != "temporal":
        raise CorruptIndex(f"{where}: manifest kind is not 'temporal'")
    tables = obj["tables"]
    if not isinstance(tables, list) or not tables:
        raise CorruptIndex(f"{where}: temporal manifest has no table entries")
    seen_files = set()
    for i, rec in enumerate(tables):
        if not isinstance(rec, dict):
            raise CorruptIndex(f"{where}: tables[{i}] is not an object")
        for key in ("relation", "vars", "file", "arity", "key_columns",
                    "key_vars", "key_levels", "rows", "sha256", "bytes",
                    "source_records", "interval_events"):
            if key not in rec:
                raise CorruptIndex(f"{where}: tables[{i}] lacks {key!r}")
        if not isinstance(rec["vars"], list) or not rec["vars"]:
            raise CorruptIndex(f"{where}: tables[{i}].vars is empty")
        if len(rec["vars"]) != rec["arity"]:
            raise CorruptIndex(f"{where}: tables[{i}] arity mismatch")
        if len(rec["key_columns"]) != rec["arity"]:
            raise CorruptIndex(
                f"{where}: tables[{i}] key_columns length mismatch"
            )
        if sorted(rec["key_columns"]) != list(range(rec["arity"])):
            raise CorruptIndex(
                f"{where}: tables[{i}].key_columns is not a permutation"
            )
        if rec["key_levels"] != sorted(rec["key_levels"]):
            raise CorruptIndex(
                f"{where}: tables[{i}] key_levels is not sorted"
            )
        if rec["key_vars"] != [rec["vars"][j] for j in rec["key_columns"]]:
            raise CorruptIndex(
                f"{where}: tables[{i}] key_vars disagree with key_columns"
            )
        if not isinstance(rec["sha256"], str) or len(rec["sha256"]) != 64:
            raise CorruptIndex(
                f"{where}: tables[{i}].sha256 is not a SHA-256 digest"
            )
        if not isinstance(rec["bytes"], int) or rec["bytes"] <= 0:
            raise CorruptIndex(f"{where}: tables[{i}].bytes is invalid")
        if not isinstance(rec["rows"], int) or rec["rows"] < 0:
            raise CorruptIndex(f"{where}: tables[{i}].rows is invalid")
        if rec["file"] in seen_files:
            raise CorruptIndex(
                f"{where}: duplicate table file {rec['file']!r}"
            )
        seen_files.add(rec["file"])
    order = obj["order"]
    if not isinstance(order, list) or not order:
        raise CorruptIndex(f"{where}: manifest has no variable order")
    for i, atom in enumerate(obj["atoms"]):
        if not isinstance(atom, dict):
            raise CorruptIndex(f"{where}: atoms[{i}] is not an object")
        for key in ("alias", "relation", "vars", "table"):
            if key not in atom:
                raise CorruptIndex(f"{where}: atoms[{i}] lacks {key!r}")
        if atom["table"] not in seen_files:
            raise CorruptIndex(
                f"{where}: atoms[{i}] points at unknown table "
                f"{atom['table']!r}"
            )
