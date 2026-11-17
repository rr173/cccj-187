"""Text rendering of an explain document for humans."""
from __future__ import annotations

from typing import Any, Dict


def render_explain(doc: Dict[str, Any]) -> str:
    out = []
    d = doc["declaration"]
    out.append("DECLARATION")
    out.append(f"  file:  {d['path']}{' (embedded in index)' if d.get('embedded') else ''}")
    out.append("  order: " + " -> ".join(d["order"]))
    out.append("")
    out.append("ATOMS (relation occurrences; aliases are independent)")
    for a in doc["atoms"]:
        out.append(
            f"  {a['alias']:<12} {a['relation']:<16} "
            f"cols={a['vars_in_file_order']} -> {a['file']}"
        )
        out.append(
            f"    index table {a['index_table']}  "
            f"access order {a['access_order']} (levels {a['access_levels']})"
        )
        out.append(
            f"    rows: {a['source_rows']} source, {a['index_rows']} distinct; "
            f"{a['duplicate_rows_collapsed']} duplicate rows collapsed into m"
        )
        out.append(f"    multiplicity: {a['multiplicity_source']}")
    out.append("")
    out.append("LAYERS (one variable per level; emission is lexicographic)")
    for layer in doc["layers"]:
        out.append(
            f"  L{layer['level']:<3} {layer['variable']:<12} "
            f"active={layer['active_atoms']} completing={layer['completing_atoms']}"
        )
        for s in layer["seeks"]:
            bound = " ".join(s["bound_key_prefix"]) or "-"
            out.append(
                f"       {s['alias']}: bound prefix [{bound}], "
                f"seek on level {s['seek_column_level']}"
            )
    out.append("")
    algo = doc["algorithm"]
    out.append("ALGORITHM")
    out.append(f"  {algo['name']}")
    out.append(f"  SQL join used:            {algo['sql_join_used']}")
    out.append(f"  two-relation intermediate: {algo['two_relation_intermediate']}")
    out.append(f"  per-layer state:          {algo['per_layer_state']}")
    out.append(f"  multiplicity:             {algo['multiplicity']}")
    out.append(f"  index fingerprint:        {doc['index']['fingerprint']}")
    b = doc["index"]["build"]
    out.append(
        f"  built with memory_bytes={b['memory_bytes']} in {b['seconds']}s"
    )
    return "\n".join(out) + "\n"
