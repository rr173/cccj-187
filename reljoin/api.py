"""Python iteration interface (the stable, documented entry point).

Typical use::

    import reljoin

    decl = reljoin.load_declaration("query.json")
    idx = reljoin.open_index("idx", expected_declaration=decl)
    with reljoin.query(idx, decl) as q:
        for values, multiplicity in q:
            ...
    print(q.report())

``query()`` accepts the declaration *embedded in the index* if you omit
``decl``, so a prepared index is self-describing.  The iterator holds only the
current prefix and one cursor per atom; closing it early is safe and marks
the run incomplete.
"""
from __future__ import annotations

from typing import Any, Dict, Iterator, Optional, Tuple

from .declaration import Declaration, declaration_from_payload, parse_declaration
from .engine import Query, Stats, Solver
from .errors import (
    BatchConflict,
    CorruptIndex,
    CorrectionProblem,
    DataProblem,
    DeclarationProblem,
    IndexProblem,
    MissingIndex,
    QueryAborted,
    ReljoinError,
    ResourceError,
    WrongIndexKind,
)
from .explain import explain
from .explain_text import render_explain
from .index_build import build_index
from .index_open import OpenIndex, open_index
from .temporal_build import build_temporal_index
from .temporal_engine import TemporalQuery, TemporalStats, check_window
from .temporal_explain import render_temporal_explain, temporal_explain
from .temporal_open import TemporalOpenIndex, open_temporal_index
from .maintain_apply import (
    ApplyResult,
    MaintainReceipt,
    apply_corrections,
)
from .maintain_explain import maintain_explain, render_maintain_explain
from .maintain_init import init_maintenance
from .maintain_query import (
    MaintainQuery,
    fetch_receipt,
    list_batches,
    stream_batch_delta,
)
from .maintain_store import MaintainStore, open_maintain_store

__all__ = [
    "load_declaration",
    "load_index_declaration",
    "open_index",
    "build_index",
    "query",
    "explain",
    "render_explain",
    "open_temporal_index",
    "build_temporal_index",
    "temporal_query",
    "temporal_query_segments",
    "temporal_explain",
    "render_temporal_explain",
    "check_window",
    "Binding",
    "TemporalSegment",
    "Query",
    "TemporalQuery",
    "TemporalStats",
    "OpenIndex",
    "TemporalOpenIndex",
    "Declaration",
    "ReljoinError",
    "DeclarationProblem",
    "IndexProblem",
    "MissingIndex",
    "CorruptIndex",
    "WrongIndexKind",
    "DataProblem",
    "ResourceError",
    "QueryAborted",
    "CorrectionProblem",
    "BatchConflict",
    "init_maintenance",
    "open_maintain_store",
    "MaintainStore",
    "MaintainQuery",
    "MaintainReceipt",
    "ApplyResult",
    "apply_corrections",
    "maintain_query_segments",
    "stream_batch_delta",
    "fetch_receipt",
    "list_batches",
    "maintain_explain",
    "render_maintain_explain",
]


def load_declaration(path: str) -> Declaration:
    return parse_declaration(path)


def load_index_declaration(index_dir: str) -> Declaration:
    """Read the declaration embedded in an index without verifying tables."""
    return open_index(index_dir, verify=False).decl


class Binding(Tuple[Tuple[int, ...], int]):
    """``(values, multiplicity)`` with named accessors."""

    __slots__ = ()

    def __new__(cls, values: Tuple[int, ...], multiplicity: int) -> "Binding":
        return super().__new__(cls, (values, multiplicity))

    @property
    def values(self) -> Tuple[int, ...]:
        return self[0]

    @property
    def multiplicity(self) -> int:
        return self[1]

    def __repr__(self) -> str:
        return f"Binding(values={self[0]!r}, multiplicity={self[1]!r})"


def query(
    idx: OpenIndex,
    decl: Optional[Declaration] = None,
    *,
    memory_bytes: Optional[int] = None,
    limit: Optional[int] = None,
    verify_mode: str = "deep",
    cancel=None,
) -> Query:
    """Create an iterator over closed-loop bindings of ``idx``.

    ``cancel`` is an optional zero-argument callable polled before each
    binding; returning true stops the run and reports it incomplete.
    """
    if decl is None:
        decl = idx.decl
    return Query(decl, idx, memory_bytes=memory_bytes, limit=limit,
                 verify_mode=verify_mode, cancel=cancel)


def query_bindings(
    idx: OpenIndex,
    decl: Optional[Declaration] = None,
    **kwargs: Any,
) -> Iterator[Binding]:
    q = query(idx, decl, **kwargs)
    try:
        for values, mult in q:
            yield Binding(values, mult)
    finally:
        q.close()


class TemporalSegment(Tuple[Tuple[int, ...], int, int, int]):
    """``(values, since, until, multiplicity)`` with named accessors."""

    __slots__ = ()

    def __new__(cls, values: Tuple[int, ...], since: int, until: int,
                multiplicity: int) -> "TemporalSegment":
        return super().__new__(cls, (values, since, until, multiplicity))

    @property
    def values(self) -> Tuple[int, ...]:
        return self[0]

    @property
    def since(self) -> int:
        return self[1]

    @property
    def until(self) -> int:
        return self[2]

    @property
    def multiplicity(self) -> int:
        return self[3]

    def __repr__(self) -> str:
        return (
            f"TemporalSegment(values={self[0]!r}, since={self[1]}, "
            f"until={self[2]}, multiplicity={self[3]})"
        )


def temporal_query(
    idx: TemporalOpenIndex,
    decl: Optional[Declaration] = None,
    *,
    window: Tuple[int, int],
    memory_bytes: Optional[int] = None,
    limit: Optional[int] = None,
    verify_mode: str = "deep",
    cancel=None,
) -> TemporalQuery:
    """Create an iterator over valid-time segments of ``idx``.

    The query covers the half-open window ``[window[0], window[1])``.  Each
    iteration yields ``(values, (since, until, multiplicity))``; segments
    arrive in global variable order and then ``since`` order.  The returned
    object's :meth:`TemporalQuery.report` gives exact counters including the
    arbitrary-precision ``weighted_duration``.  Closing early is safe and
    truthful: the run is reported as ``incomplete``.
    """
    if decl is None:
        decl = idx.decl
    return TemporalQuery(
        decl, idx, window=window, memory_bytes=memory_bytes, limit=limit,
        verify_mode=verify_mode, cancel=cancel,
    )


def temporal_query_segments(
    idx: TemporalOpenIndex,
    decl: Optional[Declaration] = None,
    **kwargs: Any,
) -> Iterator[TemporalSegment]:
    """Convenience generator yielding :class:`TemporalSegment` objects."""
    q = temporal_query(idx, decl, **kwargs)
    try:
        for values, (since, until, mult) in q:
            yield TemporalSegment(values, since, until, mult)
    finally:
        q.close()


class MaintainSegment(Tuple[Tuple[int, ...], int, int, int]):
    """``(values, since, until, multiplicity)`` from a maintained view."""

    __slots__ = ()

    def __new__(cls, values: Tuple[int, ...], since: int, until: int,
                multiplicity: int) -> "MaintainSegment":
        return super().__new__(cls, (values, since, until, multiplicity))

    @property
    def values(self) -> Tuple[int, ...]:
        return self[0]

    @property
    def since(self) -> int:
        return self[1]

    @property
    def until(self) -> int:
        return self[2]

    @property
    def multiplicity(self) -> int:
        return self[3]


def maintain_query_segments(
    store: MaintainStore,
    decl: Optional[Declaration] = None,
    **kwargs: Any,
) -> Iterator[MaintainSegment]:
    """Convenience generator yielding maintained view segments."""
    q = MaintainQuery(store, decl, **kwargs)
    try:
        for values, (since, until, mult) in q:
            yield MaintainSegment(values, since, until, mult)
    finally:
        q.close()
