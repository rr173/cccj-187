"""Independent brute-force oracle: Cartesian enumeration over the raw JSONL.

This module is intentionally dumb.  It reads the original ``<relation>.jsonl``
files, enumerates the Cartesian product of the atoms' rows, keeps the
combinations that agree on every shared variable, and adds their bag counts.
It shares only the declaration parser with the production code; it never
opens the index, never uses SQLite, and never touches the production solver
or its cursors.

That makes it a usable acceptance test for cyclic queries where a two-way
join would explode but the closed loop is tiny.
"""
from __future__ import annotations

import itertools
import json
from collections import defaultdict
from typing import Dict, Iterator, List, Sequence, Tuple

from reljoin.declaration import Declaration


def load_relations(decl: Declaration, data_dir: str) -> Dict[str, List[Tuple[int, ...]]]:
    """Read every relation file referenced by ``decl`` (raw JSON, no index)."""
    out: Dict[str, List[Tuple[int, ...]]] = {}
    for name, arity in decl.relation_arity.items():
        rows: List[Tuple[int, ...]] = []
        path = f"{data_dir}/{name}.jsonl"
        with open(path, "r", encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if not isinstance(row, list) or len(row) != arity:
                    raise ValueError(
                        f"{path}:{lineno}: expected {arity} columns, got {row!r}"
                    )
                for v in row:
                    if not isinstance(v, int) or isinstance(v, bool):
                        raise ValueError(f"{path}:{lineno}: non-integer value")
                rows.append(tuple(row))
        out[name] = rows
    return out


def brute_force(decl: Declaration, data_dir: str) -> List[Tuple[Tuple[int, ...], int]]:
    """Return ``[(values_in_global_order, multiplicity), ...]`` sorted.

    Pure Cartesian enumeration of atom rows with equality checks on shared
    variables and product-bag counting; exponential by design and meant only
    for small acceptance cases.
    """
    rels = load_relations(decl, data_dir)
    per_atom_rows = [rels[a.relation] for a in decl.atoms]
    counts: Dict[Tuple[int, ...], int] = defaultdict(int)
    order = decl.order
    pos = decl.var_position
    for combo in itertools.product(*per_atom_rows):
        assignment: Dict[str, int] = {}
        ok = True
        for atom, row in zip(decl.atoms, combo):
            for var, val in zip(atom.vars, row):
                prev = assignment.get(var)
                if prev is None:
                    assignment[var] = val
                elif prev != val:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            continue
        if len(assignment) != len(order):
            raise AssertionError("declaration has an unconstrained variable")
        values = tuple(assignment[v] for v in order)
        counts[values] += 1
    return sorted(counts.items())


def brute_force_counts(decl: Declaration, data_dir: str) -> Dict[Tuple[int, ...], int]:
    return dict(brute_force(decl, data_dir))


def iter_brute_force(decl: Declaration, data_dir: str) -> Iterator[Tuple[Tuple[int, ...], int]]:
    """Same result as :func:`brute_force` but streamed in sorted order."""
    rels = load_relations(decl, data_dir)
    per_atom_rows = [rels[a.relation] for a in decl.atoms]
    order = decl.order
    product_size = 1
    for rows in per_atom_rows:
        product_size *= len(rows)
    counts: Dict[Tuple[int, ...], int] = defaultdict(int)
    for combo in itertools.product(*per_atom_rows):
        assignment: Dict[str, int] = {}
        ok = True
        for atom, row in zip(decl.atoms, combo):
            for var, val in zip(atom.vars, row):
                prev = assignment.get(var)
                if prev is None:
                    assignment[var] = val
                elif prev != val:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            counts[tuple(assignment[v] for v in order)] += 1
    for values in sorted(counts):
        yield values, counts[values]
