"""Streaming ingestion and validation of correction batches.

A correction file is UTF-8 JSONL; every non-blank line is exactly::

    {"relation": "A", "values": [1, 2], "since": 2, "until": 8, "delta": 1}

Rules enforced while streaming (the file is never read whole):

* ``relation`` must be declared and the number of ``values`` must match the
  relation's arity; values and endpoints are signed ``int64`` and booleans are
  not integers;
* ``since < until`` — half-open intervals only;
* ``delta`` is a *non-zero* arbitrary-precision signed integer (it is the only
  integer allowed to exceed 64 bits);
* no extra fields.

The same tuple may be corrected by several lines, possibly on overlapping
intervals; nothing is summed here.  The applier sums same-tuple corrections
before composing them with the pre-commit bag, so a legal batch is never
rejected just because an early line temporarily drives a running total
negative.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Iterator, Tuple

from .declaration import INT64_MAX, INT64_MIN, Declaration
from .errors import CorrectionProblem

ROW_KEYS = {"relation", "values", "since", "until", "delta"}


@dataclass(frozen=True)
class Correction:
    relation: str
    values: Tuple[int, ...]
    since: int
    until: int
    delta: int


class CorrectionDigest:
    __slots__ = ("sha256", "bytes", "lines", "blank_lines", "records")

    def __init__(self) -> None:
        self.sha256 = hashlib.sha256()
        self.bytes = 0
        self.lines = 0
        self.blank_lines = 0
        self.records = 0


def _is_int64(value) -> bool:
    return (isinstance(value, int) and not isinstance(value, bool)
            and INT64_MIN <= value <= INT64_MAX)


def scan_corrections(path: str, decl: Declaration
                     ) -> Tuple[Iterator[Correction], CorrectionDigest]:
    """Return ``(iterator of validated corrections, running digest)``.

    The digest accumulates while the iterator is consumed, so it is only
    complete after the iterator has been exhausted.
    """
    digest = CorrectionDigest()
    return _iter_corrections(path, decl, digest), digest


def _fail(path: str, lineno: int, message: str) -> None:
    raise CorrectionProblem(f"correction file {path!r} line {lineno}: {message}")


def _iter_corrections(path: str, decl: Declaration,
                      digest: CorrectionDigest) -> Iterator[Correction]:
    try:
        fh = open(path, "rb")
    except OSError as exc:
        raise CorrectionProblem(
            f"cannot open correction file {path!r}: {exc}"
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
                text = stripped.decode("utf-8")
            except UnicodeDecodeError as exc:
                _fail(path, lineno, f"not valid UTF-8 ({exc})")
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                _fail(path, lineno, f"invalid JSON ({exc})")
            if not isinstance(row, dict):
                _fail(path, lineno,
                      f"expected a JSON object, got {type(row).__name__}")
            extra = set(row) - ROW_KEYS
            if extra:
                _fail(path, lineno,
                      "unknown field(s): " + ", ".join(sorted(extra)))
            missing = ROW_KEYS - set(row)
            if missing:
                _fail(path, lineno,
                      "missing field(s): " + ", ".join(sorted(missing)))
            relation = row["relation"]
            if not isinstance(relation, str) or not relation:
                _fail(path, lineno, "'relation' must be a non-empty string")
            arity = decl.relation_arity.get(relation)
            if arity is None:
                _fail(path, lineno,
                      f"relation {relation!r} is not in the declaration")
            values = row["values"]
            if not isinstance(values, list) or len(values) != arity:
                _fail(path, lineno,
                      f"'values' must be an array of {arity} integers, got "
                      f"{values!r}")
            if not all(_is_int64(v) for v in values):
                _fail(path, lineno,
                      "'values' must be signed 64-bit integers "
                      "(booleans are not integers)")
            since, until = row["since"], row["until"]
            if not _is_int64(since) or not _is_int64(until):
                _fail(path, lineno,
                      "'since'/'until' must be signed 64-bit integers "
                      "(booleans are not integers)")
            if since >= until:
                _fail(path, lineno,
                      f"since ({since}) must be strictly less than until "
                      f"({until}); intervals are half-open [since, until)")
            delta = row["delta"]
            if isinstance(delta, bool) or not isinstance(delta, int):
                _fail(path, lineno,
                      f"'delta' must be a non-zero integer, got "
                      f"{type(delta).__name__}: {delta!r}")
            if delta == 0:
                _fail(path, lineno, "'delta' must be non-zero")
            digest.records += 1
            yield Correction(
                relation=relation,
                values=tuple(values),
                since=since,
                until=until,
                delta=delta,
            )
