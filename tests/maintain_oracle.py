"""Independent brute-force oracle for incremental maintenance.

The reference shares nothing with the maintenance production code: it never
opens the maintenance database, never uses the applier, the seed walk, the
delta sweep or any SQLite cursor.  Given only

* the query declaration,
* the initial temporal interval files (``{relation}.jsonl``), and
* an ordered list of correction batches (raw JSONL correction lines),

it rebuilds every relation as a bag of ``(tuple, since, until, copies)``
records, applies the corrections as additional signed interval records, then
for each batch enumerates the complete bindings and their per-gap multiplicity
between every endpoint, exactly like ``temporal_oracle``.  It returns both the
full post-batch view and the signed new-minus-old delta, so a test can compare
the maintained library's committed view and stored delta against a result
computed from scratch — without calling any production cursor, range finder or
delta function.
"""
from __future__ import annotations

import itertools
import json
from collections import defaultdict
from typing import Any, Dict, List, Tuple

from reljoin.declaration import INT64_MAX, INT64_MIN, Declaration


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def load_initial_bags(
    decl: Declaration, data_dir: str
) -> Dict[str, List[Tuple[Tuple[int, ...], int, int, int]]]:
    """Read the initial interval files as bag records."""
    bags: Dict[str, List[Tuple[Tuple[int, ...], int, int, int]]] = {}
    for name, arity in decl.relation_arity.items():
        path = f"{data_dir}/{name}.jsonl"
        rows: List[Tuple[Tuple[int, ...], int, int, int]] = []
        try:
            fh = open(path, "r", encoding="utf-8")
        except FileNotFoundError:
            bags[name] = rows
            continue
        with fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                rows.append((
                    tuple(row["values"]),
                    row["since"], row["until"], int(row.get("copies", 1)),
                ))
        bags[name] = rows
    return bags


def apply_corrections_to_bags(
    bags: Dict[str, List[Tuple[Tuple[int, ...], int, int, int]]],
    corrections: List[Dict[str, Any]],
) -> None:
    """Fold correction lines into the in-memory bags as signed records."""
    for c in corrections:
        bags[c["relation"]].append((
            tuple(c["values"]), c["since"], c["until"], int(c["delta"]),
        ))


def tuple_multiplicity(records, target, t):
    return sum(
        copies for values, since, until, copies in records
        if values == target and since <= t < until
    )


def full_view(
    decl: Declaration,
    bags: Dict[str, List[Tuple[Tuple[int, ...], int, int, int]]],
    lo: int,
    hi: int,
) -> List[Tuple[Tuple[int, ...], int, int, int]]:
    """Reference closed-loop view as maximal positive-multiplicity segments."""
    endpoints = {lo, hi}
    for records in bags.values():
        for _v, since, until, _c in records:
            if lo < since < hi:
                endpoints.add(since)
            if lo < until < hi:
                endpoints.add(until)
    cuts = sorted(endpoints)
    per_atom = [bags[a.relation] for a in decl.atoms]
    raw: List[Tuple[Tuple[int, ...], int, int, int]] = []
    for a, b in zip(cuts, cuts[1:]):
        if a >= b:
            continue
        t = a
        options = []
        for atom, records in zip(decl.atoms, per_atom):
            bag = defaultdict(int)
            for values, since, until, copies in records:
                if since <= t < until:
                    bag[values] += copies
            options.append(list(bag.items()))
        counts: Dict[Tuple[int, ...], int] = defaultdict(int)
        for combo in itertools.product(*options):
            assignment: Dict[str, int] = {}
            product = 1
            ok = True
            for atom, (values, mult) in zip(decl.atoms, combo):
                product *= mult
                for var, val in zip(atom.vars, values):
                    if var in assignment and assignment[var] != val:
                        ok = False
                        break
                    assignment[var] = val
                if not ok:
                    break
            if ok:
                binding = tuple(assignment[v] for v in decl.order)
                counts[binding] += product
        for binding, mult in counts.items():
            if mult > 0:
                raw.append((binding, a, b, mult))
    raw.sort(key=lambda r: (r[0], r[1]))
    merged = []
    for binding, a, b, mult in raw:
        if (merged and merged[-1][0] == binding and merged[-1][3] == mult
                and merged[-1][2] == a):
            ps, _pu, pm = merged[-1][1], merged[-1][2], merged[-1][3]
            merged[-1] = (binding, ps, b, pm)
        else:
            merged.append((binding, a, b, mult))
    return merged


def view_as_map(
    segments: List[Tuple[Tuple[int, ...], int, int, int]]
) -> Dict[Tuple[Tuple[int, ...], int, int], int]:
    return {(binding, s, u): m for binding, s, u, m in segments}


def delta_between(
    old_segments: List[Tuple[Tuple[int, ...], int, int, int]],
    new_segments: List[Tuple[Tuple[int, ...], int, int, int]],
) -> List[Tuple[Tuple[int, ...], int, int, int]]:
    """Signed new-minus-old delta over the elementary endpoint partition.

    Both views are projected onto the union of their endpoints so the result
    is independent of how either side merged adjacent equal runs.  Zero-delta
    pieces are dropped and adjacent equal-delta pieces of the same binding are
    re-merged into maximal segments.
    """
    endpoints = set()
    for _b, s, u, _m in old_segments + new_segments:
        endpoints.add(s)
        endpoints.add(u)
    cuts = sorted(endpoints)
    old_map = view_as_map(old_segments)
    new_map = view_as_map(new_segments)

    def lookup(seg_map, binding, t):
        total = 0
        for (b, s, u), m in seg_map.items():
            if b == binding and s <= t < u:
                total += m
        return total

    bindings = {b for b, _s, _u, _m in old_segments + new_segments}
    pieces = []
    for binding in bindings:
        for a, b in zip(cuts, cuts[1:]):
            if a >= b:
                continue
            old_m = lookup(old_map, binding, a)
            new_m = lookup(new_map, binding, a)
            d = new_m - old_m
            if d != 0:
                pieces.append((binding, a, b, d))
    pieces.sort(key=lambda r: (r[0], r[1]))
    merged = []
    for binding, a, b, d in pieces:
        if (merged and merged[-1][0] == binding and merged[-1][3] == d
                and merged[-1][2] == a):
            ps, _pu, pd = merged[-1][1], merged[-1][2], merged[-1][3]
            merged[-1] = (binding, ps, b, pd)
        else:
            merged.append((binding, a, b, d))
    return merged


def weighted_change(
    delta: List[Tuple[Tuple[int, ...], int, int, int]]
) -> int:
    return sum((u - s) * d for _b, s, u, d in delta)


def legality_violation(
    decl: Declaration,
    bags: Dict[str, List[Tuple[Tuple[int, ...], int, int, int]]],
) -> Tuple[str, Tuple[int, ...], int, int] | None:
    """Return one (relation, tuple, since, until) with a negative bag, or None.

    The check is done over every endpoint gap of the relation's records, so a
    temporary dip caused by an early correction line is observed just as the
    production legality check must observe it.
    """
    for rel, records in bags.items():
        endpoints = set()
        for _v, s, u, _c in records:
            endpoints.add(s)
            endpoints.add(u)
        cuts = sorted(endpoints)
        tuples = {v for v, _s, _u, _c in records}
        for tup in tuples:
            for a, b in zip(cuts, cuts[1:]):
                if tuple_multiplicity(records, tup, a) < 0:
                    return rel, tup, a, b
    return None
