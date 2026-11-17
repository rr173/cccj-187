"""Streaming ingestion of temporal (valid-time) relation files.

Every non-blank line of ``<relation>.jsonl`` is a JSON object::

    {"values": [col0, col1, ...], "since": s, "until": u, "copies": c}

where ``values`` are the relation columns (in declaration order), ``[s, u)``
is a half-open integer-time interval and ``copies`` is an optional positive
arbitrary-precision bag multiplicity (default 1).  Booleans are never accepted
as integers, ``since`` must be strictly below ``until``, and both endpoints
are signed 64-bit values; there are no infinite endpoints and no time zones.

Each valid row is yielded as two delta events ``(key, tick, delta)``: a
``+copies`` event at ``since`` and a ``-copies`` event at ``until``.  Nothing
here is indexed or normalised beyond validation and encoding, so the reference
oracle can re-implement the same rules straight from the raw lines.
"""
from __future__ import annotations

import hashlib
import json
from typing import Iterator, Tuple

from .declaration import INT64_MAX, INT64_MIN
from .errors import DataProblem
from .jsonl import encode_int

_ROW_KEYS = {"values", "since", "until", "copies"}


class TemporalSourceDigest:
    __slots__ = ("sha256", "bytes", "lines", "blank_lines", "records",
                 "interval_events")

    def __init__(self) -> None:
        self.sha256 = hashlib.sha256()
        self.bytes = 0
        self.lines = 0
        self.blank_lines = 0
        self.records = 0
        self.interval_events = 0


def _fail(relation: str, path: str, lineno: int, message: str) -> None:
    raise DataProblem(
        f"relation {relation!r} {path}:{lineno}: {message}"
    )


def _strict_int(value, relation: str, path: str, lineno: int, field: str,
                *, allow_big: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(relation, path, lineno,
              f"{field} must be an integer, got {type(value).__name__}: "
              f"{value!r}")
    if not allow_big and (value < INT64_MIN or value > INT64_MAX):
        _fail(relation, path, lineno,
              f"{field}={value} is outside the signed 64-bit range")
    return value


def scan_temporal_source(
    relation: str, path: str, arity: int
) -> Tuple[Iterator[Tuple[bytes, int, int]], TemporalSourceDigest]:
    """Return ``(iterator of (tuple-key, tick, delta) events, digest)``.

    The key is the monotone-encoded tuple in *file-column* order; the builder
    reorders it onto the global variable order before sorting.
    """
    digest = TemporalSourceDigest()
    return _iter_temporal(relation, path, arity, digest), digest


def _iter_temporal(relation: str, path: str, arity: int,
                   digest: TemporalSourceDigest
                   ) -> Iterator[Tuple[bytes, int, int]]:
    try:
        fh = open(path, "rb")
    except FileNotFoundError as exc:
        raise DataProblem(
            f"temporal source relation file is missing: {path!r}"
        ) from exc
    except OSError as exc:
        raise DataProblem(
            f"cannot open temporal source file {path!r}: {exc}"
        ) from exc
    with fh:
        for lineno, raw in enumerate(fh, 1):
            digest.lines += 1
            digest.bytes += len(raw)
            digest.sha256.update(raw)
            stripped = raw.strip()
            if not stripped:
                digest.blank_lines += 1
                continue
            try:
                row = json.loads(stripped)
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                _fail(relation, path, lineno, f"invalid JSON ({exc})")
            if not isinstance(row, dict):
                _fail(relation, path, lineno,
                      f"expected a JSON object, got {type(row).__name__}")
            extra = set(row) - _ROW_KEYS
            if extra:
                _fail(relation, path, lineno,
                      "unknown field(s): " + ", ".join(sorted(extra)))
            if "values" not in row:
                _fail(relation, path, lineno, "missing 'values'")
            values = row["values"]
            if not isinstance(values, list):
                _fail(relation, path, lineno,
                      f"'values' must be an array, got {type(values).__name__}")
            if len(values) != arity:
                _fail(relation, path, lineno,
                      f"'values' has {len(values)} columns, expected {arity}")
            if "since" not in row:
                _fail(relation, path, lineno, "missing 'since'")
            if "until" not in row:
                _fail(relation, path, lineno, "missing 'until'")
            since = _strict_int(row["since"], relation, path, lineno, "since")
            until = _strict_int(row["until"], relation, path, lineno, "until")
            if since >= until:
                _fail(relation, path, lineno,
                      f"since ({since}) must be strictly less than until "
                      f"({until}); intervals are half-open [since, until)")
            copies = 1
            if "copies" in row:
                copies = _strict_int(row["copies"], relation, path, lineno,
                                     "copies", allow_big=True)
                if copies <= 0:
                    _fail(relation, path, lineno,
                          f"copies must be a positive integer, got {copies}")
            key = bytearray(8 * arity)
            for i, value in enumerate(values):
                # bool rejection and the int64 check happen in _strict_int.
                _strict_int(value, relation, path, lineno,
                            f"values[{i}]")
                key[i * 8:(i + 1) * 8] = encode_int(value)
            digest.records += 1
            digest.interval_events += 2
            yield bytes(key), since, copies
            yield bytes(key), until, -copies
