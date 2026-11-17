"""Explanation of how a query will be solved and where counts come from.

Everything printed here is derived from the declaration and the verified
manifest: layer order, which atoms are active/completing at each layer, which
index file serves each atom, the index prefix used at each layer, and the
provenance of each multiplicity factor.  No cost estimate is fabricated.
"""
from __future__ import annotations

from typing import Any, Dict, List

from .declaration import Declaration
from .index_open import OpenIndex


def explain(decl: Declaration, idx: OpenIndex) -> Dict[str, Any]:
    depth = len(decl.order)
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
            "index_rows": table["rows"],
            "source_rows": table["source_records"],
            "duplicate_rows_collapsed": table["duplicate_rows"],
            "multiplicity_source": (
                f"COUNT of equal tuples in {atom.relation}.jsonl, stored as a "
                f"16-byte big-endian BLOB in column m of {table['file']}; "
                "every alias contributes its own factor"
            ),
            "first_layer": key_levels[0],
            "last_layer": key_levels[-1],
        })
    for lv, var in enumerate(decl.order):
        active = [
            a.alias for a in decl.atoms
            if lv in idx.table_for_atom[a.alias]["key_levels"]
        ]
        completing = [
            a.alias for a in decl.atoms
            if idx.table_for_atom[a.alias]["key_levels"][-1] == lv
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
                    "primary-key range seek on the bound prefix; the value "
                    "stream is read ascending without scanning rows"
                ),
            })
        layers.append({
            "level": lv,
            "variable": var,
            "active_atoms": active,
            "completing_atoms": completing,
            "seeks": prefix,
            "note": (
                "intersection of the active atoms' ascending distinct values; "
                "completing atoms are point-probed for their bag count"
            ),
        })
    return {
        "declaration": decl.describe(),
        "index": {
            "path": idx.path,
            "fingerprint": idx.manifest["declaration_fingerprint"],
            "build": idx.manifest["build"],
        },
        "layers": layers,
        "atoms": atom_rows,
        "algorithm": {
            "name": "layered leapfrog over ordered index seeks",
            "global_order_is_emission_order": True,
            "sql_join_used": False,
            "two_relation_intermediate": False,
            "per_layer_state": (
                "current prefix (fixed depth) + one ascending value stream per "
                "active atom; no per-layer value set is materialised"
            ),
            "multiplicity": (
                "exact Python integer product of per-atom bag counts; no "
                "64-bit wrapping"
            ),
        },
    }
