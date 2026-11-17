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
    CorruptIndex,
    DataProblem,
    DeclarationProblem,
    IndexProblem,
    MissingIndex,
    QueryAborted,
    ReljoinError,
    ResourceError,
)
from .explain import explain
from .explain_text import render_explain
from .index_build import build_index
from .index_open import OpenIndex, open_index

__all__ = [
    "load_declaration",
    "load_index_declaration",
    "open_index",
    "build_index",
    "query",
    "explain",
    "render_explain",
    "Binding",
    "Query",
    "OpenIndex",
    "Declaration",
    "ReljoinError",
    "DeclarationProblem",
    "IndexProblem",
    "MissingIndex",
    "CorruptIndex",
    "DataProblem",
    "ResourceError",
    "QueryAborted",
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
