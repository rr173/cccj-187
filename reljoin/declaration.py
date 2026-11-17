"""Query declaration parsing and validation.

A declaration is a small JSON object::

    {
      "version": 1,
      "order": ["a", "b", "c"],
      "relations": [
        {"name": "A", "vars": ["a", "b"]},
        {"name": "B", "vars": ["b", "c"]},
        {"name": "C", "vars": ["c", "a"]}
      ]
    }

Terms
-----
* ``order`` is the *global variable order*.  It is both the order in which
  the join is solved (one variable per layer) and the lexicographic order in
  which complete bindings are emitted.
* Each relation entry is an *atom*: the relation name plus the variable name
  of every file column, in file-column order.  The same relation may occur
  more than once (aliases, e.g. ``A(x,y)`` and ``A(y,z)`` for a self-join).
  Each alias may name its columns independently; every alias of one relation
  must read the same file, so all aliases of a relation must have the same
  arity.
* Variables inside one atom must be distinct.  Every variable named in
  ``order`` must occur in at least one atom and vice versa; otherwise the
  declaration is rejected rather than silently ignored.

Nothing in this module depends on SQLite or on the index, so it can be used
by an independent oracle.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from .errors import DeclarationProblem

DECL_KEYS = {"version", "order", "relations"}
ATOM_KEYS = {"name", "vars"}
SUPPORTED_VERSION = 1

INT64_MIN = -(1 << 63)
INT64_MAX = (1 << 63) - 1

# Identifiers are used verbatim as SQLite column names after quoting; reject
# anything that would make diagnostics ambiguous or that is not representable.
# The empty string is rejected because SQLite would treat it as "rowid".
_MAX_IDENT = 512


class DeclarationError(DeclarationProblem):
    code = "declaration"


def _ident(value: Any, what: str) -> str:
    if not isinstance(value, str):
        raise DeclarationError(f"{what} must be a string, got {type(value).__name__}")
    if not value:
        raise DeclarationError(f"{what} must not be empty")
    if len(value) > _MAX_IDENT:
        raise DeclarationError(f"{what} is longer than {_MAX_IDENT} characters")
    if "\x00" in value:
        raise DeclarationError(f"{what} contains a NUL character")
    return value


@dataclass(frozen=True)
class Atom:
    """One relation occurrence (alias) in the query."""

    alias: str               # unique occurrence id, e.g. "A#0"
    relation: str            # relation/file name shared by aliases
    vars: Tuple[str, ...]    # variable per file column, in file-column order

    @property
    def arity(self) -> int:
        return len(self.vars)


@dataclass
class Declaration:
    path: str
    order: Tuple[str, ...]
    atoms: Tuple[Atom, ...]
    relation_arity: Dict[str, int]
    var_position: Dict[str, int] = field(default_factory=dict)
    canonical: bytes = b""                  # canonical payload for fingerprint
    embedded: bool = False                  # loaded from an index manifest

    # -- access order ----------------------------------------------------
    def key_vars(self, atom: Atom) -> Tuple[str, ...]:
        """Atom columns in *global-order* order: the index access order."""
        return tuple(sorted(atom.vars, key=self.var_position.__getitem__))

    def key_levels(self, atom: Atom) -> Tuple[int, ...]:
        return tuple(self.var_position[v] for v in self.key_vars(atom))

    def describe(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "embedded": self.embedded,
            "order": list(self.order),
            "atoms": [
                {"alias": a.alias, "relation": a.relation, "vars": list(a.vars)}
                for a in self.atoms
            ],
            "relation_arity": dict(self.relation_arity),
        }


def _load_json_object(path: str) -> Dict[str, Any]:
    try:
        with open(path, "rb") as fh:
            raw = fh.read()
    except OSError as exc:
        raise DeclarationError(f"cannot read declaration {path!r}: {exc}") from exc
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise DeclarationError(f"declaration {path!r} is not valid UTF-8: {exc}") from exc
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        raise DeclarationError(f"declaration {path!r} is not valid JSON: {exc}") from exc
    if not isinstance(obj, dict):
        raise DeclarationError(f"declaration {path!r} must be a JSON object")
    return obj


def parse_declaration(path: str) -> Declaration:
    """Load and validate a declaration file.  Raises DeclarationError."""
    obj = _load_json_object(path)
    decl = declaration_from_payload(obj, path=os.path.abspath(path))
    decl.embedded = False
    return decl


def declaration_from_payload(obj: Dict[str, Any], *, path: str,
                             embedded: bool = False) -> Declaration:
    """Validate a declaration already decoded from JSON."""
    extra = set(obj) - DECL_KEYS
    if extra:
        raise DeclarationError(
            "unknown declaration key(s): " + ", ".join(sorted(map(str, extra)))
        )
    for req in ("order", "relations"):
        if req not in obj:
            raise DeclarationError(f"declaration is missing {req!r}")

    version = obj.get("version", SUPPORTED_VERSION)
    if version != SUPPORTED_VERSION:
        raise DeclarationError(
            f"unsupported declaration version {version!r} "
            f"(expected {SUPPORTED_VERSION})"
        )

    order_raw = obj["order"]
    if not isinstance(order_raw, list):
        raise DeclarationError("'order' must be a JSON array of variable names")
    order: List[str] = []
    seen_order = set()
    for item in order_raw:
        v = _ident(item, "variable in 'order'")
        if v in seen_order:
            raise DeclarationError(f"variable {v!r} appears twice in 'order'")
        seen_order.add(v)
        order.append(v)
    if not order:
        raise DeclarationError("'order' must contain at least one variable")

    rel_raw = obj["relations"]
    if not isinstance(rel_raw, list):
        raise DeclarationError("'relations' must be a JSON array")
    if not rel_raw:
        raise DeclarationError("declaration has no relations")
    relation_arity: Dict[str, int] = {}
    atoms: List[Atom] = []
    counts: Dict[str, int] = {}
    used_vars = set()
    for pos, rel in enumerate(rel_raw):
        if not isinstance(rel, dict):
            raise DeclarationError(f"relations[{pos}] must be a JSON object")
        extra = set(rel) - ATOM_KEYS
        if extra:
            raise DeclarationError(
                f"relations[{pos}] has unknown key(s): "
                + ", ".join(sorted(map(str, extra)))
            )
        name = _ident(rel.get("name"), f"relations[{pos}].name")
        cols = rel.get("vars")
        if not isinstance(cols, list):
            raise DeclarationError(f"relations[{pos}].vars must be an array")
        if not cols:
            raise DeclarationError(f"relation {name!r} must have at least one column")
        cols_t = tuple(_ident(c, f"column of relation {name!r}") for c in cols)
        if len(set(cols_t)) != len(cols_t):
            raise DeclarationError(
                f"relation {name!r} lists a variable more than once: {list(cols_t)}"
            )
        if name in relation_arity:
            if relation_arity[name] != len(cols_t):
                raise DeclarationError(
                    f"relation {name!r} is used with {len(cols_t)} columns here "
                    f"but with {relation_arity[name]} elsewhere; every alias of "
                    "one relation must read the same file layout"
                )
        else:
            relation_arity[name] = len(cols_t)
        for v in cols_t:
            if v not in seen_order:
                raise DeclarationError(
                    f"variable {v!r} used by relation {name!r} "
                    "is missing from 'order'"
                )
        used_vars.update(cols_t)
        n = counts.get(name, 0)
        counts[name] = n + 1
        atoms.append(Atom(alias=f"{name}#{n}", relation=name, vars=cols_t))

    for v in order:
        if v not in used_vars:
            raise DeclarationError(
                f"variable {v!r} from 'order' is not used by any relation"
            )

    decl = Declaration(
        path=path,
        order=tuple(order),
        atoms=tuple(atoms),
        relation_arity=relation_arity,
        var_position={v: i for i, v in enumerate(order)},
        embedded=embedded,
    )
    decl.canonical = _canonical_bytes(decl)
    return decl


def _canonical_payload(decl: Declaration) -> Dict[str, Any]:
    return {
        "version": SUPPORTED_VERSION,
        "order": list(decl.order),
        "atoms": [
            {"alias": a.alias, "relation": a.relation, "vars": list(a.vars)}
            for a in decl.atoms
        ],
    }


def _canonical_bytes(decl: Declaration) -> bytes:
    text = json.dumps(_canonical_payload(decl), sort_keys=True,
                      separators=(",", ":"), ensure_ascii=True)
    return text.encode("utf-8")


def fingerprint(decl: Declaration) -> str:
    return hashlib.sha256(decl.canonical).hexdigest()


def check_value(value: Any, *, where: str) -> int:
    """Validate that a JSON scalar is a signed 64-bit integer."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise DeclarationError(f"{where}: expected a signed 64-bit integer, "
                               f"got {type(value).__name__}: {value!r}")
    if value < INT64_MIN or value > INT64_MAX:
        raise DeclarationError(f"{where}: {value} is outside the signed 64-bit range")
    return value
