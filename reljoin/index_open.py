"""Read-only index opening and integrity verification.

A *missing* index and a *damaged* index are deliberately different errors.
Opening never creates a database ("queryable but empty") as a side effect:
every atom table is verified against the manifest (size, SHA-256, and a
read-only ``PRAGMA quick_check``) before the first binding can be produced.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from typing import Any, Dict, List, Tuple

from .declaration import (
    Declaration,
    declaration_from_payload,
    fingerprint,
)
from .errors import CorruptIndex, IndexProblem, MissingIndex, ReljoinError, WrongIndexKind
from .index_build import (
    ATOMS_DIR,
    INDEX_FORMAT,
    INDEX_FORMAT_VERSION,
    MANIFEST,
)
from .temporal_format import TEMPORAL_INDEX_FORMAT

MANIFEST_KEYS = {
    "format", "format_version", "declaration", "declaration_fingerprint",
    "order", "tables", "atoms", "relations", "build",
}


class OpenIndex:
    """An opened, verified index plus the declaration it was built for."""

    def __init__(self, path: str, manifest: Dict[str, Any],
                 decl: Declaration) -> None:
        self.path = path
        self.manifest = manifest
        self.decl = decl
        self.tables: Dict[Tuple[str, Tuple[str, ...]], Dict[str, Any]] = {}
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
        """Check every table file against the manifest; raise on mismatch."""
        if not os.path.isdir(self.path):
            raise MissingIndex(f"index {self.path!r} disappeared")
        report = {"tables": 0, "bytes": 0, "deep": deep}
        for rec in self.manifest["tables"]:
            fn = rec["file"]
            _check_filename(fn)
            fp = os.path.join(self.path, ATOMS_DIR, fn)
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
            raise CorruptIndex(f"cannot open index file {fp!r}: {exc}") from exc
        try:
            try:
                conn.execute("PRAGMA query_only=ON")
                row = conn.execute("PRAGMA quick_check(1)").fetchone()
            except sqlite3.DatabaseError as exc:
                raise CorruptIndex(
                    f"index file {fp!r} is not a readable SQLite database: {exc}"
                ) from exc
            if row is None or row[0] != "ok":
                raise CorruptIndex(
                    f"index file {fp!r} fails SQLite's integrity check: {row!r}"
                )
            try:
                n = conn.execute("SELECT COUNT(*) FROM t").fetchone()[0]
            except sqlite3.DatabaseError as exc:
                raise CorruptIndex(
                    f"index file {fp!r} has no usable main table: {exc}"
                ) from exc
            if n != rec["rows"]:
                raise CorruptIndex(
                    f"index file {fp!r} holds {n} rows, manifest says "
                    f"{rec['rows']}; the index is damaged"
                )
            try:
                bad = conn.execute(
                    "SELECT COUNT(*) FROM t WHERE typeof(m) != 'blob' "
                    "OR length(m) != 16"
                ).fetchone()[0]
            except sqlite3.DatabaseError as exc:
                raise CorruptIndex(
                    f"index file {fp!r} has an unusable multiplicity column: {exc}"
                ) from exc
            if bad:
                raise CorruptIndex(
                    f"index file {fp!r} has {bad} malformed multiplicity cells"
                )
        finally:
            conn.close()

    def _verify_no_extras(self) -> None:
        """Reject unexpected files: a stray journal/WAL means a half-written index."""
        atoms_dir = os.path.join(self.path, ATOMS_DIR)
        expected = {rec["file"] for rec in self.manifest["tables"]}
        try:
            names = set(os.listdir(atoms_dir))
        except FileNotFoundError as exc:
            raise CorruptIndex(
                f"index {self.path!r} has no {ATOMS_DIR}/ directory"
            ) from exc
        extras = sorted(names - expected)
        if extras:
            raise CorruptIndex(
                f"index {self.path!r} contains unexpected file(s): "
                + ", ".join(extras[:5])
                + ("; a half-written or concurrently modified index was detected"
                   if any(x.endswith(("-journal", "-wal", "-shm", ".tmp",
                                       ".partial")) for x in extras) else "")
            )
        missing = sorted(expected - names)
        if missing:
            raise CorruptIndex(
                f"index {self.path!r} is missing table file(s): "
                + ", ".join(missing[:5])
            )

    # -- query support ---------------------------------------------------
    def connection(self, file: str) -> sqlite3.Connection:
        if self._closed:
            raise IndexProblem("index is closed")
        conn = self._conns.get(file)
        if conn is not None:
            return conn
        fp = os.path.join(self.path, ATOMS_DIR, file)
        if not os.path.isfile(fp):
            raise CorruptIndex(f"index file {fp!r} disappeared during the query")
        uri = "file:" + _uri_escape(fp) + "?mode=ro&immutable=1"
        try:
            conn = sqlite3.connect(uri, uri=True, isolation_level=None)
            conn.execute("PRAGMA query_only=ON")
            conn.execute("PRAGMA cache_size=-8192")
        except sqlite3.Error as exc:
            raise CorruptIndex(f"cannot open index file {fp!r}: {exc}") from exc
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

    def __enter__(self) -> "OpenIndex":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def describe(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "declaration": self.decl.describe(),
            "fingerprint": self.manifest["declaration_fingerprint"],
            "build": self.manifest["build"],
        }


def _check_filename(fn: Any) -> None:
    if (not isinstance(fn, str) or not fn or "/" in fn or "\\" in fn
            or fn in (".", "..") or "\x00" in fn):
        raise CorruptIndex(f"manifest names an unsafe index file: {fn!r}")


def _check_file(fp: str, rec: Dict[str, Any]) -> None:
    try:
        st = os.stat(fp)
    except FileNotFoundError as exc:
        raise CorruptIndex(
            f"index file {fp!r} named by the manifest is missing; the index "
            "is incomplete"
        ) from exc
    except OSError as exc:
        raise CorruptIndex(f"cannot stat index file {fp!r}: {exc}") from exc
    if not os.path.isfile(fp):
        raise CorruptIndex(f"index file {fp!r} is not a regular file")
    if st.st_size != rec["bytes"]:
        raise CorruptIndex(
            f"index file {fp!r} has {st.st_size} bytes, manifest says "
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
            f"index file {fp!r} fails its SHA-256 check; the index is damaged"
        )


def _uri_escape(path: str) -> str:
    out = []
    for ch in os.path.abspath(path):
        if ch in "?#% ":
            out.append("%%%02X" % ord(ch))
        else:
            out.append(ch)
    return "".join(out)


def open_index(path: str, *, verify: bool = True, deep: bool = True,
               expected_declaration: Declaration | None = None,
               progress=None) -> OpenIndex:
    """Open and verify an index.  Raises MissingIndex/CorruptIndex/IndexProblem."""
    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise MissingIndex(f"index {path!r} does not exist; run 'prepare' first")
    if not os.path.isdir(path):
        raise IndexProblem(f"index path {path!r} is not a directory")
    mpath = os.path.join(path, MANIFEST)
    try:
        with open(mpath, "rb") as fh:
            raw = fh.read()
    except FileNotFoundError as exc:
        raise MissingIndex(
            f"index {path!r} has no {MANIFEST}; it was never completed"
        ) from exc
    except OSError as exc:
        raise IndexProblem(f"cannot read index manifest {mpath!r}: {exc}") from exc
    try:
        manifest = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CorruptIndex(
            f"index manifest {mpath!r} is not valid JSON: {exc}"
        ) from exc
    if isinstance(manifest, dict) and manifest.get("format") == TEMPORAL_INDEX_FORMAT:
        raise WrongIndexKind(
            f"{path!r} is a temporal (valid-time) index; use "
            "'temporal-run'/'temporal-explain' for it, or build a static "
            "index with 'prepare'"
        )
    _check_manifest(manifest, mpath)

    decl = declaration_from_payload(manifest["declaration"], path=mpath,
                                    embedded=True)
    if fingerprint(decl) != manifest["declaration_fingerprint"]:
        raise CorruptIndex(
            "index manifest declaration does not match its fingerprint; the "
            "index is damaged"
        )
    if expected_declaration is not None:
        if fingerprint(expected_declaration) != manifest["declaration_fingerprint"]:
            raise IndexProblem(
                "index was built for a different declaration "
                f"(index {manifest['declaration_fingerprint'][:16]}…, query "
                f"{fingerprint(expected_declaration)[:16]}…); run 'prepare' "
                "for the current declaration"
            )
    idx = OpenIndex(path, manifest, decl)
    if verify:
        try:
            idx.verify(deep=deep, progress=progress)
        except BaseException:
            idx.close()
            raise
    return idx


def _check_manifest(obj: Any, where: str) -> None:
    if not isinstance(obj, dict):
        raise CorruptIndex(f"{where}: manifest is not a JSON object")
    missing = MANIFEST_KEYS - set(obj)
    if missing:
        raise CorruptIndex(
            f"{where}: manifest is missing key(s): " + ", ".join(sorted(missing))
        )
    if obj["format"] != INDEX_FORMAT:
        raise CorruptIndex(f"{where}: unsupported index format {obj['format']!r}")
    if obj["format_version"] != INDEX_FORMAT_VERSION:
        raise CorruptIndex(
            f"{where}: unsupported index format version "
            f"{obj['format_version']!r}"
        )
    tables = obj["tables"]
    if not isinstance(tables, list) or not tables:
        raise CorruptIndex(f"{where}: manifest has no table entries")
    seen_files = set()
    for i, rec in enumerate(tables):
        if not isinstance(rec, dict):
            raise CorruptIndex(f"{where}: tables[{i}] is not an object")
        for key in ("relation", "vars", "file", "arity", "key_columns",
                    "key_vars", "key_levels", "rows", "sha256", "bytes",
                    "source_records"):
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
        if (len(rec["key_vars"]) != rec["arity"]
                or len(rec["key_levels"]) != rec["arity"]):
            raise CorruptIndex(
                f"{where}: tables[{i}] key metadata length mismatch"
            )
        if rec["key_levels"] != sorted(rec["key_levels"]):
            raise CorruptIndex(
                f"{where}: tables[{i}] key_levels is not sorted; prefix "
                "access would be unsound"
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
            raise CorruptIndex(f"{where}: duplicate table file {rec['file']!r}")
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
                f"{where}: atoms[{i}] points at unknown table {atom['table']!r}"
            )
