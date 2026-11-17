"""Independent brute-force oracle for valid-time (temporal) queries.

The reference deliberately shares nothing with the temporal production code:
it does not open the index, does not use the event normalisation, the external
sorter, SQLite, or the production executor.  It only reuses the declaration
*parser* (the query text itself) and then re-implements the bag semantics from
the raw JSONL lines:

1. every interval row is validated independently and kept as a bag record
   (tuples are repeated conceptually — copies are multiplied as integers, the
   rows themselves are never physically expanded);
2. all interval endpoints are collected; between two consecutive endpoints the
   membership of every bag record is constant, so one probe point decides the
   multiplicity of the whole gap;
3. at each probe point the atoms' bag records are enumerated as a Cartesian
   product, keeping the combinations whose shared variables agree, and the
   complete binding's multiplicity is the product of per-tuple bag sums;
4. adjacent gaps with the same binding and the same multiplicity are merged
   into maximal half-open segments clipped to the query window.

It is exponential on purpose and used only on small random acceptance cases.
"""
from __future__ import annotations

import itertools
import json
from collections import defaultdict
from typing import Any, Dict, List, Tuple

from reljoin.declaration import INT64_MAX, INT64_MIN, Declaration


class OracleInputError(ValueError):
    pass


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def load_temporal_relations(
    decl: Declaration, data_dir: str
) -> Dict[str, List[Tuple[Tuple[int, ...], int, int, int]]]:
    """Read temporal rows as ``(values, since, until, copies)`` records."""
    out: Dict[str, List[Tuple[Tuple[int, ...], int, int, int]]] = {}
    for name, arity in decl.relation_arity.items():
        path = f"{data_dir}/{name}.jsonl"
        rows: List[Tuple[Tuple[int, ...], int, int, int]] = []
        with open(path, "r", encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise OracleInputError(
                        f"relation {name!r} {path}:{lineno}: invalid JSON ({exc})"
                    )
                if not isinstance(row, dict):
                    raise OracleInputError(
                        f"relation {name!r} {path}:{lineno}: expected an object"
                    )
                if set(row) - {"values", "since", "until", "copies"}:
                    raise OracleInputError(
                        f"relation {name!r} {path}:{lineno}: unknown field(s)"
                    )
                values = row.get("values")
                if not isinstance(values, list) or len(values) != arity:
                    raise OracleInputError(
                        f"relation {name!r} {path}:{lineno}: values must be an "
                        f"array of {arity} integers"
                    )
                if not all(_is_int(v) and INT64_MIN <= v <= INT64_MAX
                           for v in values):
                    raise OracleInputError(
                        f"relation {name!r} {path}:{lineno}: values must be "
                        "signed 64-bit integers (booleans rejected)"
                    )
                since, until = row.get("since"), row.get("until")
                if not _is_int(since) or not _is_int(until):
                    raise OracleInputError(
                        f"relation {name!r} {path}:{lineno}: since/until must "
                        "be signed 64-bit integers"
                    )
                if not (INT64_MIN <= since <= INT64_MAX
                        and INT64_MIN <= until <= INT64_MAX):
                    raise OracleInputError(
                        f"relation {name!r} {path}:{lineno}: endpoint outside "
                        "the signed 64-bit range"
                    )
                if since >= until:
                    raise OracleInputError(
                        f"relation {name!r} {path}:{lineno}: since must be < "
                        "until (half-open intervals)"
                    )
                copies = row.get("copies", 1)
                if not _is_int(copies) or copies <= 0:
                    raise OracleInputError(
                        f"relation {name!r} {path}:{lineno}: copies must be a "
                        "positive integer (booleans rejected)"
                    )
                rows.append((tuple(values), since, until, copies))
        out[name] = rows
    return out


def _tuple_multiplicity_at(
    records: List[Tuple[Tuple[int, ...], int, int, int]],
    target: Tuple[int, ...], t: int
) -> int:
    """Bag multiplicity of one tuple at time ``t``: sum of covering copies."""
    total = 0
    for values, since, until, copies in records:
        if values == target and since <= t < until:
            total += copies
    return total


def temporal_brute_force(
    decl: Declaration, data_dir: str, lo: int, hi: int
) -> List[Tuple[Tuple[int, ...], int, int, int]]:
    """Reference result as ``[(values, since, until, multiplicity), ...]``.

    Segments are clipped to the half-open window ``[lo, hi)``, carry only
    positive multiplicities and are maximal (adjacent equal runs merged).
    """
    if not (_is_int(lo) and _is_int(hi) and lo < hi):
        raise OracleInputError("window must satisfy int64 and lo < hi")
    rels = load_temporal_relations(decl, data_dir)
    per_atom_records = [rels[a.relation] for a in decl.atoms]

    # Every membership change happens at an interval endpoint; restrict the
    # sweep to endpoints that can change something inside the window.
    endpoints = {lo, hi}
    for records in per_atom_records:
        for _v, since, until, _c in records:
            if lo < since < hi:
                endpoints.add(since)
            if lo < until < hi:
                endpoints.add(until)
    cuts = sorted(endpoints)

    order = decl.order
    raw: List[Tuple[Tuple[int, ...], int, int, int]] = []
    for a, b in zip(cuts, cuts[1:]):
        if a >= b:
            continue
        t = a  # left-closed: the probe belongs to [a, b)
        # Distinct tuple options per atom at time t, with their bag sums.
        per_atom_options: List[List[Tuple[Tuple[int, ...], int]]] = []
        for atom, records in zip(decl.atoms, per_atom_records):
            bag: Dict[Tuple[int, ...], int] = defaultdict(int)
            for values, since, until, copies in records:
                if since <= t < until:
                    bag[values] += copies
            per_atom_options.append(list(bag.items()))
        counts: Dict[Tuple[int, ...], int] = defaultdict(int)
        for combo in itertools.product(*per_atom_options):
            assignment: Dict[str, int] = {}
            product = 1
            ok = True
            for atom, (values, mult) in zip(decl.atoms, combo):
                product *= mult
                for var, val in zip(atom.vars, values):
                    prev = assignment.get(var)
                    if prev is None:
                        assignment[var] = val
                    elif prev != val:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                if len(assignment) != len(order):
                    raise AssertionError(
                        "declaration has an unconstrained variable"
                    )
                binding = tuple(assignment[v] for v in order)
                counts[binding] += product
        for binding, mult in counts.items():
            if mult > 0:
                raw.append((binding, a, b, mult))

    # Merge adjacent, same-binding, same-multiplicity runs into maximal ones.
    raw.sort(key=lambda r: (r[0], r[1]))
    merged: List[Tuple[Tuple[int, ...], int, int, int]] = []
    for binding, a, b, mult in raw:
        if (merged and merged[-1][0] == binding and merged[-1][3] == mult
                and merged[-1][2] == a):
            prev = merged[-1]
            merged[-1] = (binding, prev[1], b, mult)
        else:
            merged.append((binding, a, b, mult))
    return merged


def weighted_duration(
    segments: List[Tuple[Tuple[int, ...], int, int, int]]
) -> int:
    """Exact integer sum of ``(until - since) * multiplicity``."""
    return sum((b - a) * m for _v, a, b, m in segments)
