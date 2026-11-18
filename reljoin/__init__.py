"""reljoin: closed-loop relational queries over bag relations.

The package supports two index kinds:

* static indexes (``prepare`` / ``explain`` / ``run``) answer whether sources
  form a closed loop and with what bag multiplicity;
* temporal indexes (``temporal-prepare`` / ``temporal-explain`` /
  ``temporal-run``) additionally answer *when* the loop holds, over
  half-open integer-time intervals.

See README.md for the command-line interface and the design notes.
"""
from .api import (  # noqa: F401
    ApplyResult,
    BatchConflict,
    Binding,
    CorruptIndex,
    CorrectionProblem,
    DataProblem,
    Declaration,
    DeclarationProblem,
    IndexProblem,
    MaintainQuery,
    MaintainReceipt,
    MaintainSegment,
    MaintainStore,
    MissingIndex,
    OpenIndex,
    Query,
    QueryAborted,
    ReljoinError,
    ResourceError,
    TemporalOpenIndex,
    TemporalQuery,
    TemporalSegment,
    TemporalStats,
    WrongIndexKind,
    apply_corrections,
    build_index,
    build_temporal_index,
    check_window,
    explain,
    fetch_receipt,
    init_maintenance,
    list_batches,
    load_declaration,
    load_index_declaration,
    maintain_explain,
    maintain_query_segments,
    open_index,
    open_maintain_store,
    open_temporal_index,
    query,
    query_bindings,
    render_explain,
    render_maintain_explain,
    render_temporal_explain,
    stream_batch_delta,
    temporal_explain,
    temporal_query,
    temporal_query_segments,
)

__version__ = "1.2.0"
