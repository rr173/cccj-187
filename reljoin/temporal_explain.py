"""Explanation of how a valid-time query is solved and what was pre-built.

The document is derived only from the declaration and the verified temporal
manifest (no query is run): how tuple candidates are sought, how time validity
is recovered from the event tables, how aliases multiply independently, and
which work the ``temporal-prepare`` step already completed (external sort,
same-tick coalescing, zero-event elimination, prefix-sum validation).
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

from .declaration import Declaration
from .temporal_open import TemporalOpenIndex


def temporal_explain(decl: Declaration, idx: TemporalOpenIndex,
                     *, window: Tuple[int, int] | None = None) -> Dict[str, Any]:
    layers: List[Dict[str, Any]] = []
    atom_rows = []
    for atom in decl.atoms:
        table = idx.table_for_atom[atom.alias]
        key_vars = tuple(table["key_vars"])
        key_levels = tuple(table["key_levels"])
        atom_rows.append({
            "alias": atom.alias,
            "relation": atom.relation,
            "file": atom.relation + ".jsonl",
            "vars_in_file_order": list(atom.vars),
            "index_table": table["file"],
            "index_columns_in_file_order": list(table["key_columns"]),
            "access_order": list(key_vars),
            "access_levels": list(key_levels),
            "source_intervals": table["source_records"],
            "interval_events": table["interval_events"],
            "stored_events": table["rows"],
            "same_tick_events_coalesced": table["coalesced_events"],
            "zero_sum_events_dropped": table["dropped_zero_events"],
            "tick_span": [table["min_tick"], table["max_tick"]],
            "validity_source": (
                "each input [since, until) interval with c copies is stored as "
                "two events (+c at since, -c at until); the multiplicity at "
                "time t is the running sum of the bound tuple's events with "
                "tick <= t; every alias contributes its own factor"
            ),
            "first_layer": key_levels[0],
            "last_layer": key_levels[-1],
        })
    for lv, var in enumerate(decl.order):
        active = [
            a.alias for a in decl.atoms
            if lv in idx.table_for_atom[a.alias]["key_levels"]
        ]
        prefix: List[Dict[str, Any]] = []
        for a in decl.atoms:
            levels = idx.table_for_atom[a.alias]["key_levels"]
            if lv not in levels:
                continue
            p = levels.index(lv)
            prefix.append({
                "alias": a.alias,
                "bound_key_prefix": [decl.order[x] for x in levels[:p]],
                "seek_column_level": lv,
                "access": (
                    "primary-key range seek on the bound prefix; distinct "
                    "tuple values are intersected by leapfrog, time is not a "
                    "join variable"
                ),
            })
        layers.append({
            "level": lv,
            "variable": var,
            "active_atoms": active,
            "seeks": prefix,
            "note": (
                "tuple candidates only: the event columns (tick, delta) are "
                "not part of the candidate intersection"
            ),
        })
    build = idx.manifest["build"]
    return {
        "mode": "temporal",
        "declaration": decl.describe(),
        "window": (
            None if window is None else {"from": window[0], "to": window[1]}
        ),
        "index": {
            "path": idx.path,
            "kind": idx.manifest["kind"],
            "fingerprint": idx.manifest["declaration_fingerprint"],
            "build": build,
        },
        "layers": layers,
        "atoms": atom_rows,
        "preparation": {
            "external_sort": (
                "interval events were externally merge-sorted on the tuple "
                "key in global order plus the tick, bounded by --memory"
            ),
            "sort_runs_written": build.get("sort_runs_written"),
            "sort_merge_passes": build.get("sort_merge_passes"),
            "sort_bytes_spilled": build.get("sort_bytes_spilled"),
            "coalescing": (
                "events with the same tuple and tick were summed at build "
                "time; groups summing to zero were removed"
            ),
            "validation": (
                "on open, a streamed prefix-sum check proves every tuple's "
                "multiplicity never goes negative and returns to zero"
            ),
        },
        "algorithm": {
            "name": "layered leapfrog candidates + k-way event sweep per binding",
            "global_order_is_emission_order": True,
            "sql_join_used": False,
            "two_relation_intermediate": False,
            "interval_cartesian_product": False,
            "per_integer_time_point_rerun": False,
            "candidate_phase": (
                "the same ordered primary-key seeks as the static engine find "
                "each complete tuple binding exactly once; time never appears "
                "as an extra column"
            ),
            "time_phase": (
                "for one complete binding, each alias opens a bounded range "
                "cursor over its bound tuple's events with tick < to; the "
                "cursors are heap-merged in tick order, maintaining one "
                "running multiplicity per alias and their product"
            ),
            "segment_rules": (
                "segments are clipped to [from, to); only positive "
                "multiplicity is emitted; adjacent segments with the same "
                "multiplicity are merged into one maximal interval, even when "
                "several factors change on the same tick; holes and different "
                "bindings are never merged"
            ),
            "multiplicity": (
                "exact Python integer sums and products of arbitrary-precision "
                "deltas; weighted duration sum((until-since)*multiplicity) is "
                "an exact integer, never floating point"
            ),
            "memory": (
                "fixed-depth prefix, one candidate cursor per atom, and per "
                "binding one small fetch buffer per alias; trajectories larger "
                "than the budget are streamed, never cached whole"
            ),
        },
    }


def render_temporal_explain(doc: Dict[str, Any]) -> str:
    out = []
    d = doc["declaration"]
    out.append("TEMPORAL (VALID-TIME) DECLARATION")
    out.append(f"  file:  {d['path']}{' (embedded in index)' if d.get('embedded') else ''}")
    out.append("  order: " + " -> ".join(d["order"]))
    if doc["window"]:
        out.append(
            f"  window: [{doc['window']['from']}, {doc['window']['to']})"
        )
    out.append("")
    out.append("ATOMS (relation occurrences; aliases are independent factors)")
    for a in doc["atoms"]:
        out.append(
            f"  {a['alias']:<12} {a['relation']:<16} "
            f"cols={a['vars_in_file_order']} -> {a['file']}"
        )
        out.append(
            f"    index table {a['index_table']}  access order "
            f"{a['access_order']} (levels {a['access_levels']})"
        )
        out.append(
            f"    intervals: {a['source_intervals']} source -> "
            f"{a['interval_events']} events -> {a['stored_events']} stored "
            f"({a['same_tick_events_coalesced']} coalesced, "
            f"{a['zero_sum_events_dropped']} zero-sum dropped)"
        )
        out.append(f"    tick span: {a['tick_span'][0]} .. {a['tick_span'][1]}")
        out.append(f"    validity: {a['validity_source']}")
    out.append("")
    out.append("LAYERS (tuple candidates; emission is lexicographic)")
    for layer in doc["layers"]:
        out.append(
            f"  L{layer['level']:<3} {layer['variable']:<12} "
            f"active={layer['active_atoms']}"
        )
        for s in layer["seeks"]:
            bound = " ".join(s["bound_key_prefix"]) or "-"
            out.append(
                f"       {s['alias']}: bound prefix [{bound}], seek on level "
                f"{s['seek_column_level']}"
            )
    out.append("")
    out.append("PREPARATION (already completed by temporal-prepare)")
    prep = doc["preparation"]
    out.append(f"  {prep['external_sort']}")
    out.append(
        f"  sort runs written: {prep['sort_runs_written']}, merge passes: "
        f"{prep['sort_merge_passes']}, bytes spilled: "
        f"{prep['sort_bytes_spilled']}"
    )
    out.append(f"  coalescing: {prep['coalescing']}")
    out.append(f"  validation: {prep['validation']}")
    out.append("")
    algo = doc["algorithm"]
    out.append("ALGORITHM")
    out.append(f"  {algo['name']}")
    out.append(f"  SQL join used:              {algo['sql_join_used']}")
    out.append(f"  two-relation intermediate:  {algo['two_relation_intermediate']}")
    out.append(f"  interval Cartesian product: {algo['interval_cartesian_product']}")
    out.append(f"  per-time-point re-run:      {algo['per_integer_time_point_rerun']}")
    out.append(f"  candidate phase:            {algo['candidate_phase']}")
    out.append(f"  time phase:                 {algo['time_phase']}")
    out.append(f"  segments:                   {algo['segment_rules']}")
    out.append(f"  multiplicity:               {algo['multiplicity']}")
    out.append(f"  memory:                     {algo['memory']}")
    out.append(f"  index fingerprint:          {doc['index']['fingerprint']}")
    b = doc["index"]["build"]
    out.append(
        f"  built with memory_bytes={b['memory_bytes']} in {b['seconds']}s"
    )
    return "\n".join(out) + "\n"
