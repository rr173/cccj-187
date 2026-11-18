"""Structured and text explanations of the maintenance machinery.

``maintain-explain`` derives everything from the maintenance manifest and the
live library state (no batch is applied): how affected work is found, how the
signed multiplicity delta is synthesised, how the view stays in step with the
relation state inside one transaction, and which counters the run report
splits out.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from .declaration import Declaration
from .maintain_store import MaintainStore


def maintain_explain(store: MaintainStore,
                     decl: Optional[Declaration] = None) -> Dict[str, Any]:
    decl = decl or store.decl
    conn = store.connect(readonly=True)
    try:
        revision = store.revision(conn)
        n_view = conn.execute("SELECT COUNT(*) FROM view").fetchone()[0]
        n_batches = conn.execute("SELECT COUNT(*) FROM batches").fetchone()[0]
        n_delta = conn.execute(
            "SELECT COUNT(*) FROM batch_delta"
        ).fetchone()[0]
        rel_sizes = {}
        for rec in store.manifest["relations"]:
            tname = rec["table"]
            rel_sizes[rec["relation"]] = conn.execute(
                f"SELECT COUNT(*) FROM {_q(tname)}"
            ).fetchone()[0]
    finally:
        conn.close()

    atoms = []
    for atom_rec in store.manifest["atoms"]:
        atoms.append({
            "alias": atom_rec["alias"],
            "relation": atom_rec["relation"],
            "vars_in_file_order": atom_rec["vars"],
            "state_table": atom_rec["table"],
            "access_order": atom_rec["key_vars"],
            "access_levels": atom_rec["key_levels"],
            "note": (
                "aliases of one relation share one physical event table; each "
                "alias still contributes its own multiplicity factor"
            ),
        })

    return {
        "mode": "maintain",
        "declaration": decl.describe(),
        "library": {
            "path": store.path,
            "revision": revision,
            "committed_batches": n_batches,
            "view_segments": n_view,
            "stored_delta_rows": n_delta,
            "relation_event_rows": rel_sizes,
            "snapshot_isolation": (
                "WAL mode; every reader pins one snapshot for its whole "
                "stream, so an old reader never sees a revision boundary in "
                "the middle of its result"
            ),
        },
        "atoms": atoms,
        "affected_scope": {
            "principle": (
                "only tuples named by the correction batch are touched; "
                "unrelated history is never scanned, copied or refilled"
            ),
            "steps": [
                "correction lines are streamed into a disk-backed scratch "
                "database as balanced +delta/-delta events",
                "for each touched tuple, batch events are merge-joined with "
                "that tuple's current events in tick order, proving the final "
                "per-period multiplicity is nowhere negative",
                "each surviving tuple seeds the layered leapfrog walk with "
                "its values clamped into its relation's aliases; only the "
                "first alias seeds, so one binding is discovered at most once "
                "even when several of its atoms change in the same batch",
                "bindings that no changed tuple can reach are never visited",
            ],
        },
        "delta_synthesis": {
            "rule": (
                "for every affected binding, old and staged event streams of "
                "all atoms are heap-merged; factors are summed at ticks and "
                "the complete binding multiplicity is the product of all "
                "atom factors (each alias contributes one factor)"
            ),
            "cross_terms": (
                "same-tick events from several atoms form one boundary, so "
                "the product difference includes every cross term without "
                "enumerating combinations; old/new segments are never paired "
                "mechanically"
            ),
            "output": (
                "emit new-minus-old multiplicity only where it is non-zero; "
                "maximal adjacent segments with the same delta are merged; "
                "ordering is binding lexicographic then segment start"
            ),
            "weighted_change": (
                "exact signed integer sum of (until-since)*delta_multiplicity"
            ),
        },
        "view_sync": {
            "atomicity": (
                "relation state, view segments, batch delta and the receipt "
                "change in one SQLite transaction; a kill or rollback leaves "
                "all four at the previous revision together"
            ),
            "view_updates": (
                "only affected bindings have their view rows deleted and "
                "replaced; every other stored segment stays byte-for-byte "
                "untouched"
            ),
            "revision": (
                "a successful commit (including a net-zero legal batch) "
                "increments revision by one; an identical re-send returns the "
                "original receipt without incrementing"
            ),
        },
        "bounded_memory": {
            "correction_file": "streamed line by line; never read whole",
            "tuple_trajectories": "merge-joined through fetch buffers",
            "affected_bindings": (
                "seed walk returns bindings per seed; no global candidate "
                "list is materialised"
            ),
            "delta_and_view": (
                "rows are inserted in bounded chunks; spill lives in the "
                "scratch database under the library directory, not in RAM"
            ),
            "streaming_reads": (
                "maintain-run and batch export fetch fixed-size row batches"
            ),
        },
        "report_sections": [
            "init", "startup_verification", "correction_read",
            "relation_and_event_access", "candidate_dedup",
            "delta_output", "view_write",
        ],
    }


def _q(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def render_maintain_explain(doc: Dict[str, Any]) -> str:
    out = []
    d = doc["declaration"]
    lib = doc["library"]
    out.append("MAINTENANCE LIBRARY (incremental valid-time upkeep)")
    out.append(f"  path:     {lib['path']}")
    out.append(f"  revision: {lib['revision']}")
    out.append(
        f"  batches:  {lib['committed_batches']} committed, "
        f"{lib['stored_delta_rows']} stored delta rows, "
        f"{lib['view_segments']} view segments"
    )
    out.append("  order:    " + " -> ".join(d["order"]))
    out.append("")
    out.append("ATOMS (one shared event table per relation; aliases are "
               "independent factors)")
    for a in doc["atoms"]:
        out.append(
            f"  {a['alias']:<12} {a['relation']:<12} cols={a['vars_in_file_order']} "
            f"-> {a['state_table']} access={a['access_order']} "
            f"levels={a['access_levels']}"
        )
    out.append("")
    out.append("HOW THE AFFECTED SCOPE IS FOUND")
    for i, step in enumerate(doc["affected_scope"]["steps"], 1):
        out.append(f"  {i}. {step}")
    out.append("")
    out.append("HOW THE MULTIPLICITY DELTA IS SYNTHESISED")
    out.append(f"  {doc['delta_synthesis']['rule']}")
    out.append(f"  cross terms: {doc['delta_synthesis']['cross_terms']}")
    out.append(f"  output:      {doc['delta_synthesis']['output']}")
    out.append(f"  weighted:    {doc['delta_synthesis']['weighted_change']}")
    out.append("")
    out.append("HOW THE VIEW STAYS IN STEP WITH THE RELATIONS")
    out.append(f"  atomicity:  {doc['view_sync']['atomicity']}")
    out.append(f"  view writes: {doc['view_sync']['view_updates']}")
    out.append(f"  revision:   {doc['view_sync']['revision']}")
    out.append("")
    out.append("BOUNDED MEMORY")
    for k, v in doc["bounded_memory"].items():
        out.append(f"  {k}: {v}")
    out.append("")
    out.append("REPORT SECTIONS (exact counters, no hidden full scans)")
    for section in doc["report_sections"]:
        out.append(f"  - {section}")
    return "\n".join(out) + "\n"
