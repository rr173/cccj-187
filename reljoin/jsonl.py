"""Streaming JSONL source ingestion.

Each line of a relation file is a JSON array of signed 64-bit integers, one
entry per declared column, in the declaration's column order.  Blank lines
are ignored; any other malformed line is a hard error (a damaged or
mismatched source must never silently look like an empty relation).
"""
from __future__ import annotations

import hashlib
import json
import struct
from typing import Iterator, Tuple

from .declaration import INT64_MAX, INT64_MIN
from .errors import DataProblem

_SIGN = 1 << 63


def encode_int(value: int) -> bytes:
    """Monotone encoding: signed 64-bit int -> 8 big-endian bytes.

    Adding 2**63 maps [-2**63, 2**63-1] onto [0, 2**64-1], so byte-wise
    lexicographic order equals numeric order, including the extremes.
    """
    return struct.pack(">Q", value + _SIGN)


def decode_int(raw: bytes) -> int:
    return struct.unpack(">Q", raw)[0] - _SIGN


class SourceDigest:
    __slots__ = ("sha256", "bytes", "lines", "blank_lines", "records")

    def __init__(self) -> None:
        self.sha256 = hashlib.sha256()
        self.bytes = 0
        self.lines = 0
        self.blank_lines = 0
        self.records = 0


def scan_source(path: str, arity: int) -> Tuple[Iterator[bytes], SourceDigest]:
    """Return ``(iterator of fixed-width encoded keys, digest)``.

    The iterator validates every value eagerly as it is consumed; the digest
    accumulates while iterating, so consume the iterator fully before reading
    the counters.
    """
    digest = SourceDigest()
    return _iter_source(path, arity, digest), digest


def _iter_source(path: str, arity: int, digest: SourceDigest) -> Iterator[bytes]:
    rec = 8 * arity
    try:
        fh = open(path, "rb")
    except FileNotFoundError as exc:
        raise DataProblem(f"source relation file is missing: {path!r}") from exc
    except OSError as exc:
        raise DataProblem(f"cannot open source file {path!r}: {exc}") from exc
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
                raise DataProblem(f"{path}:{lineno}: invalid JSON ({exc})") from exc
            if not isinstance(row, list):
                raise DataProblem(
                    f"{path}:{lineno}: expected a JSON array, got "
                    f"{type(row).__name__}"
                )
            if len(row) != arity:
                raise DataProblem(
                    f"{path}:{lineno}: expected {arity} columns, got {len(row)}"
                )
            key = bytearray(rec)
            for i, value in enumerate(row):
                if isinstance(value, bool) or not isinstance(value, int):
                    raise DataProblem(
                        f"{path}:{lineno}: column {i} is not an integer "
                        f"({type(value).__name__}: {value!r})"
                    )
                if value < INT64_MIN or value > INT64_MAX:
                    raise DataProblem(
                        f"{path}:{lineno}: column {i} value {value} is outside "
                        "the signed 64-bit range"
                    )
                key[i * 8:(i + 1) * 8] = encode_int(value)
            digest.records += 1
            yield bytes(key)
