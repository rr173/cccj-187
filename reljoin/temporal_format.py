"""Shared on-disk format details for the temporal (valid-time) index.

Each input row ``{"values": [...], "since": s, "until": u, "copies": c}`` is a
half-open interval ``[s, u)`` carrying an arbitrary-precision positive bag
multiplicity ``c``.  The builder turns every interval into two *events*:

* at tick ``s`` the multiplicity delta is ``+c`` (the interval starts);
* at tick ``u`` the multiplicity delta is ``-c`` (the interval ends).

At query time the per-tuple multiplicity at time ``t`` is the running sum of
all deltas whose tick is ``<= t``; deltas with the same tuple and the same
tick are already coalesced at build time, so zero-sum events never reach the
index.  Deltas are stored as variable-length signed big-endian BLOBs because a
single tuple may carry more copies than fit in 16 bytes, and the multiplicity
of a complete binding is a Python integer product that may be wider still.
"""
from __future__ import annotations

from typing import Tuple

TEMPORAL_INDEX_FORMAT = "reljoin-temporal-index-v1"
TEMPORAL_FORMAT_VERSION = 1
TEMPORAL_TABLES_DIR = "tables"
TEMPORAL_MANIFEST = "manifest.json"

# A delta blob is [tag byte][big-endian magnitude bytes].  The tag packs the
# sign and the number of magnitude bytes that follow, so decode is unambiguous
# without a separate length field (SQLite keeps the BLOB length anyway).
_MAX_MAG_LEN = 255


def encode_delta(value: int) -> bytes:
    """Signed arbitrary-precision integer -> self-describing BLOB."""
    if value == 0:
        return b"\x00"
    negative = value < 0
    magnitude = -value if negative else value
    mag = magnitude.to_bytes(max(1, (magnitude.bit_length() + 7) // 8), "big")
    if len(mag) > _MAX_MAG_LEN:
        from .errors import ResourceError
        raise ResourceError(
            f"a coalesced multiplicity delta needs {len(mag)} bytes; the "
            f"temporal format supports at most {_MAX_MAG_LEN}"
        )
    tag = len(mag) | (0x80 if negative else 0)
    return bytes((tag,)) + mag


def decode_delta(raw: bytes) -> int:
    if not isinstance(raw, (bytes, bytearray, memoryview)):
        from .errors import CorruptIndex
        raise CorruptIndex(
            f"temporal delta cell is {type(raw).__name__}, expected a BLOB"
        )
    raw = bytes(raw)
    if not raw:
        from .errors import CorruptIndex
        raise CorruptIndex("temporal delta cell is empty")
    tag = raw[0]
    mag_len = tag & 0x7F
    if len(raw) != 1 + mag_len:
        from .errors import CorruptIndex
        raise CorruptIndex(
            f"temporal delta cell has {len(raw) - 1} magnitude bytes but its "
            f"tag declares {mag_len}"
        )
    if mag_len == 0:
        return 0
    value = int.from_bytes(raw[1:], "big")
    return -value if tag & 0x80 else value


def split_event(record: bytes, key_width: int) -> Tuple[bytes, int, int]:
    """Split a framed sorter record into ``(key, tick, delta)``."""
    from .jsonl import decode_int
    key = record[:key_width]
    tick = decode_int(record[key_width:key_width + 8])
    delta = decode_delta(record[key_width + 8:])
    return key, tick, delta
