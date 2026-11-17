"""reljoin: closed-loop relational queries over bag relations.

See README.md for the command-line interface and the design notes.
"""
from .api import (  # noqa: F401
    Binding,
    CorruptIndex,
    DataProblem,
    Declaration,
    DeclarationProblem,
    IndexProblem,
    MissingIndex,
    OpenIndex,
    Query,
    QueryAborted,
    ReljoinError,
    ResourceError,
    build_index,
    explain,
    load_declaration,
    load_index_declaration,
    open_index,
    query,
    query_bindings,
    render_explain,
)

__version__ = "1.0.0"
