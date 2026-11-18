"""Open, verify and share a maintenance library.

A maintenance library is self-describing: its manifest carries the query
declaration and the physical table layout.  Reads (``maintain-run``, batch
export) open the database read-only in WAL mode and pin one snapshot for the
whole stream, so a commit arriving while a reader is still draining cannot
mix revisions inside one result.  Writers (``maintain-apply``) open a normal
read-write connection; SQLite's single-writer lock plus an in-transaction
revision check make two same-expect-revision writers conflict rather than both
succeed.
"""
from __future__ import annotations

import json
import os
import sqlite3
from typing import Any, Dict, List, Optional

from .declaration import (
    Declaration,
    declaration_from_payload,
    fingerprint,
)
from .errors import (
    CorruptIndex,
    IndexProblem,
    MissingIndex,
    WrongIndexKind,
)
from .index_open import _uri_escape
from .maintain_format import (
    MAINTAIN_DB,
    MAINTAIN_FORMAT_VERSION,
    MAINTAIN_INDEX_FORMAT,
    MAINTAIN_MANIFEST,
    physical_column_spec,
    quote_ident,
)

_MANIFEST_KEYS = {
    "format", "format_version", "kind", "declaration",
    "declaration_fingerprint", "order", "relations", "atoms", "created",
}


class MaintainStore:
    """An opened maintenance library with its embedded declaration."""

    def __init__(self, path: str, manifest: Dict[str, Any],
                 decl: Declaration) -> None:
        self.path = os.path.abspath(path)
        self.manifest = manifest
        self.decl = decl
        # One physical table per declared relation name.
        self.relation_tables: Dict[str, Dict[str, Any]] = {}
        for rec in manifest["relations"]:
            self.relation_tables[rec["relation"]] = rec
        self.table_for_atom: Dict[str, Dict[str, Any]] = {}
        for atom_rec in manifest["atoms"]:
            self.table_for_atom[atom_rec["alias"]] = dict(atom_rec)
        self._conns: List[sqlite3.Connection] = []
        self._closed = False

    # -- verification ----------------------------------------------------
    def verify(self, *, deep: bool = True, progress=None) -> Dict[str, Any]:
        if not os.path.isdir(self.path):
            raise MissingIndex(
                f"maintenance library {self.path!r} disappeared"
            )
        db_path = os.path.join(self.path, MAINTAIN_DB)
        if not os.path.isfile(db_path):
            raise CorruptIndex(
                f"maintenance library {self.path!r} has no {MAINTAIN_DB}; it "
                "was never fully initialised"
            )
        report: Dict[str, Any] = {"tables": 0, "deep": deep,
                                  "integrity_rows": 0}
        conn = self.connect()
        try:
            try:
                row = conn.execute("PRAGMA quick_check(1)").fetchone()
            except sqlite3.DatabaseError as exc:
                raise CorruptIndex(
                    f"maintenance database is not readable: {exc}"
                ) from exc
            if row is None or row[0] != "ok":
                raise CorruptIndex(
                    f"maintenance database fails SQLite's integrity check: "
                    f"{row!r}"
                )
            # Shape checks: every declared table exists with the right columns.
            for rel, rec in self.relation_tables.items():
                tname = rec["table"]
                cols = {r[1] for r in conn.execute(
                    f"PRAGMA table_info({quote_ident(tname)})"
                )}
                expected = set(physical_column_spec(rec["arity"])) | {
                    "tick", "d"
                }
                if cols != expected:
                    raise CorruptIndex(
                        f"maintenance table {tname!r} has columns "
                        f"{sorted(cols)}, expected {sorted(expected)}"
                    )
            for required in ("view", "batches", "batch_delta", "meta"):
                names = {r[0] for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )}
                if required not in names:
                    raise CorruptIndex(
                        f"maintenance database is missing table {required!r}"
                    )
            rev = self.revision(conn)
            if rev < 0:
                raise CorruptIndex("maintenance revision is negative")
            if deep:
                report["integrity_rows"] = self._deep_check(conn)
            report["tables"] = len(self.relation_tables)
            if progress:
                progress(f"verified {MAINTAIN_DB} at revision {rev}")
        finally:
            pass
        return report

    def _deep_check(self, conn: sqlite3.Connection) -> int:
        """Bounded prefix-sum proof over every maintained relation.

        Streams each relation in primary-key order and proves the running
        multiplicity of every tuple never goes negative.  Unlike the read-only
        temporal index, a maintained tuple's event stream is not required to
        return to zero at its last event: corrections persist a standing bag,
        so a tuple may end while still positive (it simply has no later
        events).  Then view and delta multiplicity blobs must decode and be
        non-zero.  Everything is streamed in batches so the check itself never
        loads a table.
        """
        from .maintain_format import decode_int_blob
        seen = 0
        for rel, rec in self.relation_tables.items():
            tname = rec["table"]
            arity = rec["arity"]
            cols = ", ".join(
                quote_ident(c) for c in physical_column_spec(arity)
            )
            cur = conn.execute(
                f"SELECT {cols}, tick, d FROM {quote_ident(tname)} "
                f"ORDER BY {cols}, tick"
            )
            prev_key = None
            running = 0
            while True:
                batch = cur.fetchmany(2048)
                if not batch:
                    break
                for row in batch:
                    key = tuple(row[:arity])
                    if key != prev_key:
                        prev_key = key
                        running = 0
                    running += decode_int_blob(row[arity + 1])
                    seen += 1
                    if running < 0:
                        raise CorruptIndex(
                            f"maintenance relation {rel!r}: tuple {key} has a "
                            f"negative multiplicity at tick {row[arity]} "
                            f"(running sum {running})"
                        )
            cur.close()
        # View and delta blobs must decode and be positive/non-zero.
        for table, column in (("view", "multiplicity"),
                              ("batch_delta", "delta_multiplicity")):
            cur = conn.execute(
                f"SELECT {quote_ident(column)} FROM {quote_ident(table)}"
            )
            while True:
                batch = cur.fetchmany(2048)
                if not batch:
                    break
                for (blob,) in batch:
                    value = decode_int_blob(blob)
                    if value == 0:
                        raise CorruptIndex(
                            f"maintenance table {table!r} holds a zero "
                            "multiplicity segment"
                        )
                    seen += 1
        return seen

    # -- connections -----------------------------------------------------
    def connect(self, *, readonly: bool = False,
                busy_timeout_ms: int = 30000) -> sqlite3.Connection:
        """Open an independent connection.

        Every call returns a fresh connection; callers own its lifecycle and
        close it themselves.  Connections are therefore never shared between a
        writer and a long-lived reader, and closing one query cannot break a
        concurrent one.  The store's own bookkeeping list is kept only for a
        best-effort shutdown and does not pool these connections.
        """
        if self._closed:
            raise IndexProblem("maintenance library is closed")
        db_path = os.path.join(self.path, MAINTAIN_DB)
        if readonly:
            uri = "file:" + _uri_escape(db_path) + "?mode=ro"
            conn = sqlite3.connect(uri, uri=True, isolation_level=None)
            conn.execute("PRAGMA query_only=ON")
        else:
            conn = sqlite3.connect(db_path, isolation_level=None)
        conn.execute(f"PRAGMA busy_timeout={int(busy_timeout_ms)}")
        return conn

    def revision(self, conn: Optional[sqlite3.Connection] = None) -> int:
        own = conn is None
        if own:
            conn = self.connect(readonly=True)
        try:
            row = conn.execute(
                "SELECT value FROM meta WHERE key='revision'"
            ).fetchone()
            if row is None:
                raise CorruptIndex("maintenance database has no revision row")
            return int(row[0])
        finally:
            if own:
                conn.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for conn in self._conns:
            try:
                conn.close()
            except sqlite3.Error:
                pass
        self._conns.clear()

    def __enter__(self) -> "MaintainStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def open_maintain_store(path: str,
                        expected_declaration: Optional[Declaration] = None,
                        *, verify: bool = True,
                        deep: bool = True,
                        progress=None) -> MaintainStore:
    """Open and verify a maintenance library."""
    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise MissingIndex(
            f"maintenance library {path!r} does not exist; run "
            "'maintain-init' first"
        )
    if not os.path.isdir(path):
        raise IndexProblem(f"maintenance path {path!r} is not a directory")
    mpath = os.path.join(path, MAINTAIN_MANIFEST)
    if not os.path.exists(mpath):
        # Distinguish the other two index kinds for a clear diagnostic.
        other = os.path.join(path, "manifest.json")
        if os.path.exists(other):
            try:
                with open(other, "rb") as fh:
                    payload = json.loads(fh.read().decode("utf-8"))
                kind = payload.get("format", "")
                if kind in ("reljoin-index-v1", "reljoin-temporal-index-v1"):
                    raise WrongIndexKind(
                        f"{path!r} is a {kind} index, not a maintenance "
                        "library; run 'maintain-init' on a temporal index"
                    )
            except (UnicodeDecodeError, json.JSONDecodeError):
                pass
        raise MissingIndex(
            f"maintenance library {path!r} has no {MAINTAIN_MANIFEST}; it was "
            "never fully initialised"
        )
    try:
        with open(mpath, "rb") as fh:
            raw = fh.read()
    except OSError as exc:
        raise IndexProblem(
            f"cannot read maintenance manifest {mpath!r}: {exc}"
        ) from exc
    try:
        manifest = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CorruptIndex(
            f"maintenance manifest {mpath!r} is not valid JSON: {exc}"
        ) from exc
    _check_manifest(manifest, mpath)
    decl = declaration_from_payload(manifest["declaration"], path=mpath,
                                    embedded=True)
    if fingerprint(decl) != manifest["declaration_fingerprint"]:
        raise CorruptIndex(
            "maintenance manifest declaration does not match its fingerprint; "
            "the library is damaged"
        )
    if expected_declaration is not None and fingerprint(
        expected_declaration
    ) != manifest["declaration_fingerprint"]:
        raise IndexProblem(
            "maintenance library was built for a different declaration"
        )
    store = MaintainStore(path, manifest, decl)
    if verify:
        try:
            store.verify(deep=deep, progress=progress)
        except BaseException:
            store.close()
            raise
    return store


def _check_manifest(obj: Any, where: str) -> None:
    if not isinstance(obj, dict):
        raise CorruptIndex(f"{where}: maintenance manifest is not a JSON object")
    if obj.get("format") in ("reljoin-index-v1",
                             "reljoin-temporal-index-v1"):
        raise WrongIndexKind(
            f"{where}: this is a {obj['format']} index; maintenance commands "
            "need a library created with 'maintain-init'"
        )
    missing = _MANIFEST_KEYS - set(obj)
    if missing:
        raise CorruptIndex(
            f"{where}: maintenance manifest is missing key(s): "
            + ", ".join(sorted(missing))
        )
    if obj["format"] != MAINTAIN_INDEX_FORMAT:
        raise CorruptIndex(
            f"{where}: unsupported maintenance format {obj['format']!r}"
        )
    if obj["format_version"] != MAINTAIN_FORMAT_VERSION:
        raise CorruptIndex(
            f"{where}: unsupported maintenance format version "
            f"{obj['format_version']!r}"
        )
    if obj.get("kind") != "maintain":
        raise CorruptIndex(f"{where}: manifest kind is not 'maintain'")
    rels = obj["relations"]
    if not isinstance(rels, list) or not rels:
        raise CorruptIndex(f"{where}: maintenance manifest has no relations")
    for i, rec in enumerate(rels):
        for key in ("relation", "table", "arity", "key_columns",
                    "key_vars", "key_levels"):
            if key not in rec:
                raise CorruptIndex(
                    f"{where}: relations[{i}] lacks {key!r}"
                )
        if len(rec["key_columns"]) != rec["arity"]:
            raise CorruptIndex(
                f"{where}: relations[{i}] key column count disagrees with arity"
            )
    if not isinstance(obj["atoms"], list) or not obj["atoms"]:
        raise CorruptIndex(f"{where}: maintenance manifest has no atoms")
