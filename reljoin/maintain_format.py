"""On-disk format details for an incremental maintenance library.

A maintenance library is an *independent* directory created by
``maintain-init`` from an existing temporal index.  After initialisation the
source index and the original interval files are no longer needed: everything
required to answer ``maintain-run`` queries and to apply further correction
batches lives inside this directory::

    <store>/
      manifest.json     # reljoin-maintain-v1; presence means "published"
      state.db          # one SQLite database (WAL mode) holding
                        #   - the current per-relation event state,
                        #   - the current closed-loop view,
                        #   - batch receipts and per-batch signed deltas.

Multiplicity arithmetic is exact Python-integer arithmetic everywhere.  Deltas
and multiplicities are stored as self-describing signed big-endian BLOBs with
no fixed width (a single maintained tuple may carry more copies than the 16
bytes the static builder allows), so a dedicated codec lives here rather than
reusing the temporal index's 255-magnitude-byte codec.
"""
from __future__ import annotations

from typing import Tuple

from .errors import CorruptIndex

MAINTAIN_INDEX_FORMAT = "reljoin-maintain-v1"
MAINTAIN_FORMAT_VERSION = 1
MAINTAIN_MANIFEST = "manifest.json"
MAINTAIN_DB = "state.db"
MAINTAIN_WAL = "state.db-wal"
MAINTAIN_SHM = "state.db-shm"
MAINTAIN_TMP_PREFIX = ".reljoin-tmp-"


def encode_int_blob(value: int) -> bytes:
    """Signed arbitrary-precision integer -> self-describing BLOB.

    Layout: a one-byte header per group whose low six bits give the number of
    magnitude bytes in the group; bit 7 means "another group follows" and the
    terminal group's bit 6 carries the sign.  Zero is the single byte
    ``0x00``.  There is no fixed width, so deltas and multiplicities may be
    arbitrarily large.
    """
    if value == 0:
        return b"\x00"
    negative = value < 0
    magnitude = -value if negative else value
    body = magnitude.to_bytes(
        max(1, (magnitude.bit_length() + 7) // 8), "big"
    )
    groups = [body[i:i + 63] for i in range(0, len(body), 63)]
    out = bytearray()
    for gi, group in enumerate(groups):
        last = gi == len(groups) - 1
        header = len(group)            # 1..63: only the low six bits
        if not last:
            header |= 0x80  # continuation
        elif negative:
            header |= 0x40  # terminal + sign
        out.append(header)
        out.extend(group)
    return bytes(out)


def decode_int_blob(raw: bytes) -> int:
    if not isinstance(raw, (bytes, bytearray, memoryview)):
        raise CorruptIndex(
            f"maintenance integer cell is {type(raw).__name__}, expected a BLOB"
        )
    raw = bytes(raw)
    if not raw:
        raise CorruptIndex("maintenance integer cell is empty")
    if raw == b"\x00":
        return 0
    pos = 0
    negative = False
    mag = bytearray()
    while True:
        if pos >= len(raw):
            raise CorruptIndex("maintenance integer cell is truncated")
        header = raw[pos]
        pos += 1
        cont = bool(header & 0x80)
        sign = bool(header & 0x40)
        length = header & 0x3F  # group length lives in the low six bits
        if pos + length > len(raw):
            raise CorruptIndex("maintenance integer cell is truncated")
        if length == 0:
            raise CorruptIndex("maintenance integer cell has an empty group")
        mag.extend(raw[pos:pos + length])
        pos += length
        if not cont:
            negative = sign
            break
    if pos != len(raw):
        raise CorruptIndex("maintenance integer cell has trailing bytes")
    if not mag:
        if negative:
            raise CorruptIndex("maintenance integer cell marks zero negative")
        return 0
    value = int.from_bytes(bytes(mag), "big")
    return -value if negative else value


def relation_table_name(relation: str) -> str:
    """SQLite table name for one physical relation's current event state."""
    return "rel_" + relation


def stage_table_name(relation: str) -> str:
    """Scratch table (in the attached scratch DB) for one batch's events."""
    return "stage_" + relation


def quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def quote_table(name: str) -> str:
    """Quote a possibly schema-qualified table name (``schema.table``)."""
    return ".".join(quote_ident(part) for part in name.split("."))


def q(table: str, column: str) -> str:
    return f"{quote_ident(table)}.{quote_ident(column)}"


def physical_column_spec(arity: int) -> Tuple[str, ...]:
    """Physical table columns ``c0..c{n-1}`` in file-column order."""
    return tuple(f"c{i}" for i in range(arity))
