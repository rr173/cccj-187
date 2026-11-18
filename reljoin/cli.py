"""Command line interface: prepare, explain, run.

Exit codes (stable):

  0  success (including an empty result set)
  2  usage error (bad arguments)
  3  declaration problem
  4  data problem (source files)
  5  missing index
  6  corrupt / damaged index
  7  incomplete run (consumer went away, cancelled, --limit or --max-output
     hit before exhaustion, or a write failed)
  8  resource error
  9  internal error
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
from typing import Any, Dict, List, Optional

from . import api
from .errors import (
    BatchConflict,
    CorrectionProblem,
    CorruptIndex,
    DataProblem,
    DeclarationProblem,
    IndexProblem,
    MissingIndex,
    QueryAborted,
    ReljoinError,
    ResourceError,
)
from .explain import explain as explain_doc
from .explain_text import render_explain
from .index_build import build_index
from .index_open import open_index
from .maintain_apply import apply_corrections
from .maintain_explain import maintain_explain, render_maintain_explain
from .maintain_init import init_maintenance
from .maintain_query import (
    fetch_receipt,
    stream_batch_delta,
)
from .maintain_store import open_maintain_store
from .temporal_build import build_temporal_index
from .temporal_engine import check_window
from .temporal_explain import render_temporal_explain, temporal_explain
from .temporal_open import open_temporal_index

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_DECLARATION = 3
EXIT_DATA = 4
EXIT_MISSING_INDEX = 5
EXIT_CORRUPT_INDEX = 6
EXIT_INCOMPLETE = 7
EXIT_RESOURCE = 8
EXIT_INTERNAL = 9
EXIT_BATCH_CONFLICT = 10


class _Cancelled(Exception):
    """Raised from the SIGINT/SIGTERM handler inside the run loop."""


def _exit_code_for(exc: BaseException) -> int:
    if isinstance(exc, QueryAborted):
        return EXIT_INCOMPLETE
    if isinstance(exc, BatchConflict):
        return EXIT_BATCH_CONFLICT
    if isinstance(exc, MissingIndex):
        return EXIT_MISSING_INDEX
    if isinstance(exc, CorruptIndex):
        return EXIT_CORRUPT_INDEX
    if isinstance(exc, (DataProblem, CorrectionProblem)):
        return EXIT_DATA
    if isinstance(exc, DeclarationProblem):
        return EXIT_DECLARATION
    if isinstance(exc, IndexProblem):
        return EXIT_CORRUPT_INDEX
    if isinstance(exc, ResourceError):
        return EXIT_RESOURCE
    if isinstance(exc, ReljoinError):
        return EXIT_INTERNAL
    return EXIT_INTERNAL


def _err(msg: str, code: str) -> None:
    print(f"reljoin: {code}: {msg}", file=sys.stderr)


def _parse_memory(text: str) -> int:
    text = text.strip()
    mult = 1
    if text and text[-1] in "kKmMgGtT":
        mult = {"k": 1 << 10, "m": 1 << 20, "g": 1 << 30, "t": 1 << 40}[
            text[-1].lower()
        ]
        text = text[:-1]
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid memory size {text!r}")
    if value <= 0:
        raise argparse.ArgumentTypeError("memory size must be positive")
    return value * mult


def _memory_str(n: int) -> str:
    if n % (1 << 30) == 0:
        return f"{n // (1 << 30)}GiB"
    if n % (1 << 20) == 0:
        return f"{n // (1 << 20)}MiB"
    return f"{n}B"


# --------------------------------------------------------------------------
def cmd_prepare(args: argparse.Namespace) -> int:
    decl = api.load_declaration(args.decl)
    manifest = build_index(
        decl,
        data_dir=args.data,
        out_dir=args.index,
        memory_bytes=args.memory,
        sort_dir=args.sort_dir,
        sqlite_cache_kib=args.sqlite_cache_kib,
        force=args.force,
        progress=_progress_printer(args.verbose),
    )
    if not args.quiet:
        total_src = sum(r["records"] for r in manifest["relations"].values())
        total_idx = sum(t["rows"] for t in manifest["tables"])
        print(
            f"prepared {args.index}: {len(manifest['tables'])} index table(s), "
            f"{total_src} source rows -> {total_idx} distinct rows, "
            f"{manifest['build']['seconds']}s, "
            f"memory budget {_memory_str(args.memory)}",
            file=sys.stderr,
        )
    if args.json:
        print(json.dumps(manifest, sort_keys=True, indent=2))
    return EXIT_OK


def _progress_printer(verbose: bool):
    if not verbose:
        return None

    def show(msg: str) -> None:
        print(f"  {msg}", file=sys.stderr)

    return show


# --------------------------------------------------------------------------
def cmd_explain(args: argparse.Namespace) -> int:
    decl = api.load_declaration(args.decl)
    idx = open_index(args.index, expected_declaration=decl,
                     deep=not args.no_deep_verify)
    try:
        doc = explain_doc(decl, idx)
        doc["run_defaults"] = {
            "memory_bytes": args.memory,
            "limit": args.limit,
        }
        if args.json:
            print(json.dumps(doc, sort_keys=True, indent=2))
        else:
            sys.stdout.write(render_explain(doc))
            run = doc["run_defaults"]
            print("")
            print("RUN")
            print(f"  order:      " + " -> ".join(decl.order))
            print(f"  memory:     {_memory_str(run['memory_bytes'])}")
            print(
                "  output:     one JSON array [values..., multiplicity] per "
                "line, ascending in the order above"
            )
        return EXIT_OK
    finally:
        idx.close()


# --------------------------------------------------------------------------
def cmd_run(args: argparse.Namespace) -> int:
    decl = api.load_declaration(args.decl)
    # Install handlers before index verification so a signal during startup is
    # still reported as an incomplete run rather than terminating the process.
    previous = _install_signal_handlers()
    idx = None
    q = None
    out = None
    close_out = False
    try:
        idx = open_index(
            args.index,
            expected_declaration=decl,
            deep=not args.no_deep_verify,
            progress=_progress_printer(args.verbose),
        )
        q = api.query(idx, decl, memory_bytes=args.memory, limit=args.limit,
                      verify_mode="deep" if not args.no_deep_verify else "shallow")

        if args.out == "-":
            out = sys.stdout.buffer
        else:
            out = open(args.out, "wb")
            close_out = True
        try:
            for values, mult in q:
                line = _format_binding(values, mult)
                out.write(line)
                q.stats.output_bytes += len(line)
                if (args.max_output_bytes is not None
                        and q.stats.output_bytes > args.max_output_bytes):
                    q.close()
                    q.stats.status = "incomplete"
                    q.stats.status_reason = (
                        f"output budget of {args.max_output_bytes} bytes "
                        "exhausted before the result set was complete"
                    )
                    _emit_stats(args, q)
                    _err("output budget exhausted before the result set was "
                         "complete", "incomplete")
                    return EXIT_INCOMPLETE
        finally:
            if out is sys.stdout.buffer:
                try:
                    out.flush()
                except (BrokenPipeError, OSError):
                    # Suppress the interpreter's own "BrokenPipeError ignored"
                    # warning: the outcome is already reported below.
                    try:
                        devnull = os.open(os.devnull, os.O_WRONLY)
                        os.dup2(devnull, sys.stdout.fileno())
                    except OSError:
                        pass
            elif close_out:
                try:
                    out.close()
                except OSError as exc:
                    q.close()
                    raise ReljoinError(
                        f"cannot finish writing {args.out!r}: {exc}",
                        code="output",
                    ) from exc
        q.close()
        if q.stats.status == "complete":
            _emit_stats(args, q)
            return EXIT_OK
        if q.stats.status == "limit-reached":
            _emit_stats(args, q)
            _err(
                f"stopped after --limit {args.limit} bindings; the result set "
                "is incomplete", "incomplete"
            )
            return EXIT_INCOMPLETE
        _emit_stats(args, q)
        _err(f"run did not complete: {q.stats.status_reason}", "incomplete")
        return EXIT_INCOMPLETE
    except _Cancelled:
        if q is not None:
            q.close()
            q.stats.status = "incomplete"
            q.stats.status_reason = "cancelled by signal"
            _emit_stats(args, q)
            _err("cancelled by signal; the result set is incomplete",
                 "incomplete")
        else:
            if idx is not None:
                idx.close()
            _err("cancelled by signal during startup; the result set is "
                 "incomplete", "incomplete")
        return EXIT_INCOMPLETE
    except BrokenPipeError:
        q.close()
        q.stats.status = "incomplete"
        q.stats.status_reason = "output consumer closed the pipe"
        _emit_stats(args, q)
        _err("output consumer went away; the result set is incomplete",
             "incomplete")
        return EXIT_INCOMPLETE
    except QueryAborted as exc:
        _emit_stats(args, q)
        _err(str(exc), "incomplete")
        return EXIT_INCOMPLETE
    except OSError as exc:
        if q is not None:
            q.close()
            q.stats.status = "incomplete"
            q.stats.status_reason = f"write failed: {exc}"
            _emit_stats(args, q)
        elif idx is not None:
            idx.close()
        _err(f"write failed: {exc}; the result set is incomplete", "incomplete")
        return EXIT_INCOMPLETE
    finally:
        _restore_signal_handlers(previous)
        if q is not None:
            q.close()
        elif idx is not None:
            idx.close()


def _format_binding(values, mult: int) -> bytes:
    # Compact, unambiguous JSON array: values then the exact multiplicity.
    return ("[" + ",".join(str(v) for v in values) + "," + str(mult) + "]\n"
            ).encode("ascii")


def _emit_stats(args: argparse.Namespace, q) -> None:
    report = q.report(include_layers=not args.no_layer_stats)
    report["output_lines"] = q.stats.emitted
    if args.stats:
        try:
            with open(args.stats, "w", encoding="utf-8") as fh:
                json.dump(report, fh, sort_keys=True, indent=2)
                fh.write("\n")
        except OSError as exc:
            print(f"reljoin: cannot write --stats file: {exc}", file=sys.stderr)
    if args.stats_stdout:
        print(json.dumps(report, sort_keys=True))
    if args.verbose:
        print(json.dumps(report, sort_keys=True, indent=2), file=sys.stderr)


def _install_signal_handlers():
    previous = {}

    def handler(signum, frame):
        raise _Cancelled()

    for name in ("SIGINT", "SIGTERM", "SIGHUP"):
        sig = getattr(signal, name, None)
        if sig is None:
            continue
        try:
            previous[sig] = signal.signal(sig, handler)
        except (ValueError, OSError):
            pass
    return previous


def _restore_signal_handlers(previous) -> None:
    for sig, old in previous.items():
        try:
            signal.signal(sig, old)
        except (ValueError, OSError):
            pass


# --------------------------------------------------------------------------
def cmd_temporal_prepare(args: argparse.Namespace) -> int:
    decl = api.load_declaration(args.decl)
    manifest = build_temporal_index(
        decl,
        data_dir=args.data,
        out_dir=args.index,
        memory_bytes=args.memory,
        sort_dir=args.sort_dir,
        sqlite_cache_kib=args.sqlite_cache_kib,
        force=args.force,
        progress=_progress_printer(args.verbose),
    )
    if not args.quiet:
        total_src = sum(r["records"] for r in manifest["relations"].values())
        total_events = sum(t["rows"] for t in manifest["tables"])
        print(
            f"prepared temporal {args.index}: {len(manifest['tables'])} event "
            f"table(s), {total_src} interval rows -> {total_events} stored "
            f"events, {manifest['build']['seconds']}s, memory budget "
            f"{_memory_str(args.memory)}",
            file=sys.stderr,
        )
    if args.json:
        print(json.dumps(manifest, sort_keys=True, indent=2))
    return EXIT_OK


def cmd_temporal_explain(args: argparse.Namespace) -> int:
    decl = api.load_declaration(args.decl)
    idx = open_temporal_index(args.index, expected_declaration=decl,
                              deep=not args.no_deep_verify)
    try:
        window = None
        if args.from_ is not None or args.to is not None:
            if args.from_ is None or args.to is None:
                raise ResourceError(
                    "temporal-explain needs both --from and --to when either "
                    "is given"
                )
            window = (check_window(args.from_, "--from"),
                      check_window(args.to, "--to"))
        doc = temporal_explain(decl, idx, window=window)
        doc["run_defaults"] = {
            "memory_bytes": args.memory,
            "limit": args.limit,
            "window": window,
        }
        if args.json:
            print(json.dumps(doc, sort_keys=True, indent=2))
        else:
            sys.stdout.write(render_temporal_explain(doc))
            run = doc["run_defaults"]
            print("")
            print("TEMPORAL RUN")
            print("  order:      " + " -> ".join(decl.order))
            print(f"  memory:     {_memory_str(run['memory_bytes'])}")
            print(
                "  output:     one JSON object per positive-multiplicity "
                'segment: {"values":[...], "since":a, "until":b, '
                '"multiplicity":m}, ascending by values then since, clipped '
                "to [from, to)"
            )
        return EXIT_OK
    finally:
        idx.close()


def _format_segment(values, since, until, mult) -> bytes:
    vals = ",".join(str(v) for v in values)
    return (
        f'{{"values":[{vals}],"since":{since},"until":{until},'
        f'"multiplicity":{mult}}}\n'
    ).encode("ascii")


def cmd_temporal_run(args: argparse.Namespace) -> int:
    decl = api.load_declaration(args.decl)
    lo = check_window(args.from_, "--from")
    hi = check_window(args.to, "--to")
    if lo >= hi:
        raise ResourceError(
            f"--from ({lo}) must be strictly less than --to ({hi}); the "
            "query window is the half-open interval [from, to)"
        )
    # Install the signal handlers before any potentially long work (index
    # verification included), so a cancellation during startup is still
    # reported as an incomplete run instead of killing the process outright.
    previous = _install_signal_handlers()
    idx = None
    q = None
    out = None
    close_out = False
    try:
        idx = open_temporal_index(
            args.index,
            expected_declaration=decl,
            deep=not args.no_deep_verify,
            progress=_progress_printer(args.verbose),
        )
        q = api.temporal_query(
            idx, decl, window=(lo, hi), memory_bytes=args.memory,
            limit=args.limit,
            verify_mode="deep" if not args.no_deep_verify else "shallow",
        )

        if args.out == "-":
            out = sys.stdout.buffer
        else:
            out = open(args.out, "wb")
            close_out = True
        try:
            for values, (since, until, mult) in q:
                line = _format_segment(values, since, until, mult)
                out.write(line)
                q.stats.output_bytes += len(line)
                if (args.max_output_bytes is not None
                        and q.stats.output_bytes > args.max_output_bytes):
                    q.close()
                    q.stats.status = "incomplete"
                    q.stats.status_reason = (
                        f"output budget of {args.max_output_bytes} bytes "
                        "exhausted before the result set was complete"
                    )
                    _emit_temporal_stats(args, q)
                    _err("output budget exhausted before the result set was "
                         "complete", "incomplete")
                    return EXIT_INCOMPLETE
        finally:
            if out is sys.stdout.buffer:
                try:
                    out.flush()
                except (BrokenPipeError, OSError):
                    try:
                        devnull = os.open(os.devnull, os.O_WRONLY)
                        os.dup2(devnull, sys.stdout.fileno())
                    except OSError:
                        pass
            elif close_out:
                try:
                    out.close()
                except OSError as exc:
                    q.close()
                    raise ReljoinError(
                        f"cannot finish writing {args.out!r}: {exc}",
                        code="output",
                    ) from exc
        q.close()
        if q.stats.status == "complete":
            _emit_temporal_stats(args, q)
            return EXIT_OK
        if q.stats.status == "limit-reached":
            _emit_temporal_stats(args, q)
            _err(
                f"stopped after --limit {args.limit} segments; the result set "
                "is incomplete", "incomplete"
            )
            return EXIT_INCOMPLETE
        _emit_temporal_stats(args, q)
        _err(f"run did not complete: {q.stats.status_reason}", "incomplete")
        return EXIT_INCOMPLETE
    except _Cancelled:
        if q is not None:
            q.close()
            q.stats.status = "incomplete"
            q.stats.status_reason = "cancelled by signal"
            _emit_temporal_stats(args, q)
            _err("cancelled by signal; the result set is incomplete",
                 "incomplete")
        else:
            if idx is not None:
                idx.close()
            _err("cancelled by signal during startup; the result set is "
                 "incomplete", "incomplete")
        return EXIT_INCOMPLETE
    except BrokenPipeError:
        q.close()
        q.stats.status = "incomplete"
        q.stats.status_reason = "output consumer closed the pipe"
        _emit_temporal_stats(args, q)
        _err("output consumer went away; the result set is incomplete",
             "incomplete")
        return EXIT_INCOMPLETE
    except QueryAborted as exc:
        _emit_temporal_stats(args, q)
        _err(str(exc), "incomplete")
        return EXIT_INCOMPLETE
    except OSError as exc:
        if q is not None:
            q.close()
            q.stats.status = "incomplete"
            q.stats.status_reason = f"write failed: {exc}"
            _emit_temporal_stats(args, q)
        elif idx is not None:
            idx.close()
        _err(f"write failed: {exc}; the result set is incomplete", "incomplete")
        return EXIT_INCOMPLETE
    finally:
        _restore_signal_handlers(previous)
        if q is not None:
            q.close()
        elif idx is not None:
            idx.close()


def _emit_temporal_stats(args: argparse.Namespace, q) -> None:
    report = q.report(include_layers=not args.no_layer_stats)
    report["output_lines"] = q.stats.emitted
    if args.stats:
        try:
            with open(args.stats, "w", encoding="utf-8") as fh:
                json.dump(report, fh, sort_keys=True, indent=2)
                fh.write("\n")
        except OSError as exc:
            print(f"reljoin: cannot write --stats file: {exc}", file=sys.stderr)
    if args.stats_stdout:
        print(json.dumps(report, sort_keys=True))
    if args.verbose:
        print(json.dumps(report, sort_keys=True, indent=2), file=sys.stderr)


# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# Maintenance (incremental correction) commands
# --------------------------------------------------------------------------
def cmd_maintain_init(args: argparse.Namespace) -> int:
    decl = api.load_declaration(args.decl)
    tidx = open_temporal_index(
        args.index, expected_declaration=decl,
        deep=not args.no_deep_verify,
        progress=_progress_printer(args.verbose),
    )
    try:
        manifest = init_maintenance(
            decl, tidx, out_dir=args.maintain, memory_bytes=args.memory,
            sort_dir=args.sort_dir, sqlite_cache_kib=args.sqlite_cache_kib,
            force=args.force, progress=_progress_printer(args.verbose),
        )
    finally:
        tidx.close()
    if not args.quiet:
        print(
            f"initialised maintenance library {args.maintain} at revision 0 "
            f"with {manifest['init']['report']['initial_view_segments']} "
            "view segment(s); the temporal index and originals can now be "
            "removed",
            file=sys.stderr,
        )
    if args.json:
        print(json.dumps(manifest, sort_keys=True, indent=2))
    return EXIT_OK


def _format_delta_row(row: Dict[str, Any]) -> bytes:
    vals = ",".join(str(v) for v in row["values"])
    return (
        f'{{"values":[{vals}],"since":{row["since"]},"until":{row["until"]},'
        f'"delta_multiplicity":{row["delta_multiplicity"]}}}\n'
    ).encode("ascii")


def _open_output(path: str):
    if path == "-":
        return sys.stdout.buffer, False
    return open(path, "wb"), True


def cmd_maintain_apply(args: argparse.Namespace) -> int:
    decl = api.load_declaration(args.decl)
    store = open_maintain_store(
        args.maintain, expected_declaration=decl,
        deep=not args.no_deep_verify,
        progress=_progress_printer(args.verbose),
    )
    out = None
    close_out = False
    try:
        result = apply_corrections(
            store, batch_id=args.batch_id,
            expect_revision=args.expect_revision,
            correction_path=args.corrections, memory_bytes=args.memory,
            sqlite_cache_kib=args.sqlite_cache_kib,
            sort_dir=args.sort_dir,
            kill_hook=args.kill_hook,
            progress=_progress_printer(args.verbose),
        )
        receipt = result.receipt
        # The commit is durable regardless of whether the export below
        # succeeds.  The delta is re-read from the committed batch table (never
        # re-applied), so a write failure cannot masquerade as a failed apply.
        committed_note = None
        try:
            out, close_out = _open_output(args.out)
            for row in stream_batch_delta(store, receipt.batch_id):
                out.write(_format_delta_row(row))
            if out is sys.stdout.buffer:
                out.flush()
            elif close_out:
                out.close()
        except OSError as exc:
            committed_note = exc
        if committed_note is not None:
            _err(
                f"batch {receipt.batch_id!r} is committed at revision "
                f"{receipt.revision}, but exporting the delta to "
                f"{args.out!r} failed: {committed_note}; the batch was NOT "
                "re-applied — re-fetch it with 'maintain-export --batch-id "
                f"{receipt.batch_id}'",
                "incomplete",
            )
            if args.stats:
                _write_stats_file(args, result, receipt, exported=False)
            return EXIT_INCOMPLETE
        if not args.quiet:
            verb = "replayed" if receipt.replayed else "committed"
            print(
                f"{verb} batch {receipt.batch_id!r} at revision "
                f"{receipt.revision}: {receipt.delta_rows} delta row(s), "
                f"weighted change {receipt.weighted_change}",
                file=sys.stderr,
            )
        if args.stats:
            _write_stats_file(args, result, receipt, exported=True)
        if args.json:
            print(json.dumps(receipt.as_dict(), sort_keys=True))
        return EXIT_OK
    finally:
        store.close()


def _write_stats_file(args, result, receipt, *, exported: bool) -> None:
    d = result.stats.as_dict()
    # Group the exact counters into the audited run phases, so no full scan is
    # hidden inside an opaque total.
    report = {
        "receipt": receipt.as_dict(),
        "delta_exported": exported,
        "run_report": {
            "startup_verification": {
                "status": d["status"],
                "verify": d["verify"],
            },
            "correction_read": {
                "correction_lines": d["correction_lines"],
                "correction_records": d["correction_records"],
                "raw_events_inserted": d["raw_events_inserted"],
            },
            "relation_and_event_access": {
                "stage_tuples": d["stage_tuples"],
                "stage_coalesced": d["stage_coalesced"],
                "validation_rows_scanned": d["validation_rows_scanned"],
                "negative_checks": d["negative_checks"],
                "delta_events_read": d["delta_events_read"],
                "relation_events_upserted": d["relation_events_upserted"],
                "relation_events_deleted": d["relation_events_deleted"],
            },
            "candidate_dedup": {
                "seed_tuples": d["seed_tuples"],
                "discovery_seeks": d["discovery_seeks"],
                "discovery_probes": d["discovery_probes"],
                "bindings_discovered": d["bindings_discovered"],
                "candidate_duplicates": d["candidate_duplicates"],
                "binding_sweeps": d["binding_sweeps"],
            },
            "delta_output": {
                "delta_boundaries": d["delta_boundaries"],
                "delta_segments": d["delta_segments"],
                "delta_rows_written": d["delta_rows_written"],
            },
            "view_write": {
                "view_rows_deleted": d["view_rows_deleted"],
                "view_rows_inserted": d["view_rows_inserted"],
            },
            "resources": {
                "peak_rss_kib": d["peak_rss_kib"],
                "elapsed_seconds": d["elapsed_seconds"],
                "scratch_bytes": d["scratch_bytes"],
                "spill_files": d["spill_files"],
            },
        },
        "counters": d,
    }
    try:
        with open(args.stats, "w", encoding="utf-8") as fh:
            json.dump(report, fh, sort_keys=True, indent=2)
            fh.write("\n")
    except OSError as exc:
        print(f"reljoin: cannot write --stats file: {exc}", file=sys.stderr)


def cmd_maintain_run(args: argparse.Namespace) -> int:
    decl = api.load_declaration(args.decl)
    lo = check_window(args.from_, "--from")
    hi = check_window(args.to, "--to")
    if lo >= hi:
        raise ResourceError(
            f"--from ({lo}) must be strictly less than --to ({hi})"
        )
    store = open_maintain_store(
        args.maintain, expected_declaration=decl,
        deep=not args.no_deep_verify,
    )
    q = None
    out = None
    close_out = False
    try:
        q = api.MaintainQuery(
            store, decl, window=(lo, hi), memory_bytes=args.memory,
            limit=args.limit,
        )
        out, close_out = _open_output(args.out)
        try:
            for values, (since, until, mult) in q:
                line = _format_segment(values, since, until, mult)
                out.write(line)
                q.stats.output_bytes += len(line)
                if (args.max_output_bytes is not None
                        and q.stats.output_bytes > args.max_output_bytes):
                    q.close()
                    q.stats.status = "incomplete"
                    q.stats.status_reason = "output budget exhausted"
                    _emit_maintain_stats(args, q, revision=q.revision)
                    return EXIT_INCOMPLETE
        finally:
            if out is sys.stdout.buffer:
                try:
                    out.flush()
                except (BrokenPipeError, OSError):
                    try:
                        devnull = os.open(os.devnull, os.O_WRONLY)
                        os.dup2(devnull, sys.stdout.fileno())
                    except OSError:
                        pass
            elif close_out:
                out.close()
        revision = q.revision
        q.close()
        if not args.quiet:
            print(
                f"maintain-run read revision {revision} "
                f"({q.stats.emitted_segments} segment(s)) from the committed "
                "view; no relation recomputation was performed",
                file=sys.stderr,
            )
        _emit_maintain_stats(args, q, revision=revision)
        if q.stats.status == "complete":
            return EXIT_OK
        return EXIT_INCOMPLETE
    except BrokenPipeError:
        if q is not None:
            q.close()
        return EXIT_INCOMPLETE
    finally:
        # The query owns its own connection; closing it is idempotent.  The
        # store connection list is no longer pooled, so closing the store only
        # flips its own state and never breaks another reader.
        if q is not None:
            q.close()
        store.close()


def _emit_maintain_stats(args, q, *, revision: int) -> None:
    if not args.stats and not args.stats_stdout:
        return
    report = q.report(include_layers=not args.no_layer_stats)
    report["revision_read"] = revision
    report["source"] = "committed view table (no join recomputation)"
    if args.stats:
        try:
            with open(args.stats, "w", encoding="utf-8") as fh:
                json.dump(report, fh, sort_keys=True, indent=2)
                fh.write("\n")
        except OSError as exc:
            print(f"reljoin: cannot write --stats file: {exc}", file=sys.stderr)
    if args.stats_stdout:
        print(json.dumps(report, sort_keys=True))


def cmd_maintain_explain(args: argparse.Namespace) -> int:
    decl = api.load_declaration(args.decl)
    store = open_maintain_store(
        args.maintain, expected_declaration=decl,
        deep=not args.no_deep_verify,
    )
    try:
        doc = maintain_explain(store, decl)
        if args.json:
            print(json.dumps(doc, sort_keys=True, indent=2))
        else:
            sys.stdout.write(render_maintain_explain(doc))
        return EXIT_OK
    finally:
        store.close()


def cmd_maintain_export(args: argparse.Namespace) -> int:
    decl = api.load_declaration(args.decl)
    store = open_maintain_store(
        args.maintain, expected_declaration=decl,
        deep=not args.no_deep_verify,
    )
    out = None
    close_out = False
    try:
        receipt = fetch_receipt(store, args.batch_id)
        out, close_out = _open_output(args.out)
        try:
            for row in stream_batch_delta(store, args.batch_id):
                out.write(_format_delta_row(row))
            if out is sys.stdout.buffer:
                out.flush()
            elif close_out:
                out.close()
        except OSError as exc:
            _err(
                f"export write failed for committed batch "
                f"{args.batch_id!r}: {exc}; the stored delta is untouched and "
                "can be re-fetched",
                "incomplete",
            )
            return EXIT_INCOMPLETE
        if not args.quiet:
            print(
                f"exported {receipt['delta_rows']} delta row(s) of batch "
                f"{args.batch_id!r} (revision {receipt['revision']}, "
                f"weighted change {receipt['weighted_change']})",
                file=sys.stderr,
            )
        if args.json:
            print(json.dumps(receipt, sort_keys=True))
        return EXIT_OK
    finally:
        store.close()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="reljoin",
        description=(
            "Closed-loop (cyclic) relational queries over JSONL bag relations "
            "using layered ordered index seeks."
        ),
    )
    sub = p.add_subparsers(dest="command", required=True)

    common_load = argparse.ArgumentParser(add_help=False)
    common_load.add_argument("--index", required=True, metavar="DIR",
                             help="index directory")
    common_load.add_argument("--no-deep-verify", action="store_true",
                             help="skip SQLite-level verification of table files"
                                  " (size+SHA-256 are always checked)")

    # -- prepare ---------------------------------------------------------
    pp = sub.add_parser(
        "prepare", help="turn JSONL relations into a read-only disk index",
        description="Build a read-only index. The directory is published "
                    "atomically; on failure nothing is left behind.",
    )
    pp.add_argument("--decl", required=True, metavar="FILE",
                    help="query declaration JSON")
    pp.add_argument("--data", required=True, metavar="DIR",
                    help="directory containing <relation>.jsonl files")
    pp.add_argument("--index", required=True, metavar="DIR",
                    help="index directory to create")
    pp.add_argument("--memory", required=True, type=_parse_memory,
                    metavar="SIZE",
                    help="memory budget, e.g. 32M or 1G (external sort spills)")
    pp.add_argument("--sort-dir", default=None, metavar="DIR",
                    help="directory for external-sort run files (default: temp)")
    pp.add_argument("--sqlite-cache-kib", type=int, default=None,
                    help="SQLite page cache per table, in KiB "
                         "(default: 40%% of --memory)")
    pp.add_argument("--force", action="store_true",
                    help="atomically replace an existing non-empty index")
    pp.add_argument("--json", action="store_true",
                    help="print the manifest as JSON to stdout")
    pp.add_argument("--verbose", action="store_true")
    pp.add_argument("--quiet", action="store_true")
    pp.set_defaults(func=cmd_prepare, limit=None, memory=None)

    # -- explain ---------------------------------------------------------
    ep = sub.add_parser(
        "explain", help="show layers, index access and multiplicity sources",
    )
    ep.add_argument("--decl", required=True, metavar="FILE")
    ep.add_argument("--index", required=True, metavar="DIR")
    ep.add_argument("--memory", type=_parse_memory, default=64 << 20,
                    metavar="SIZE", help="memory budget shown for `run` "
                                         "(default 64M)")
    ep.add_argument("--limit", type=int, default=None,
                    help="limit shown for `run`")
    ep.add_argument("--json", action="store_true")
    ep.add_argument("--no-deep-verify", action="store_true")
    ep.set_defaults(func=cmd_explain)

    # -- run -------------------------------------------------------------
    rp = sub.add_parser(
        "run", help="emit closed-loop bindings in global variable order",
        description=(
            "Emit one line per complete binding: a JSON array of the variable "
            "values in declaration order followed by the exact bag "
            "multiplicity. Exit code 0 means the enumeration completed (which "
            "includes an empty result); 7 means it did not."
        ),
    )
    rp.add_argument("--decl", required=True, metavar="FILE")
    rp.add_argument("--index", required=True, metavar="DIR")
    rp.add_argument("--memory", type=_parse_memory, default=64 << 20,
                    metavar="SIZE")
    rp.add_argument("--out", default="-", metavar="FILE",
                    help="output file (default: stdout)")
    rp.add_argument("--limit", type=int, default=None,
                    help="stop after N bindings and report the run incomplete")
    rp.add_argument("--max-output-bytes", type=int, default=None,
                    help="stop after this many output bytes and report the run "
                         "incomplete")
    rp.add_argument("--stats", default=None, metavar="FILE",
                    help="write exact run statistics as JSON")
    rp.add_argument("--stats-stdout", action="store_true",
                    help="print run statistics as one JSON line on stdout")
    rp.add_argument("--no-layer-stats", action="store_true",
                    help="omit the per-layer candidate table from --stats")
    rp.add_argument("--no-deep-verify", action="store_true")
    rp.add_argument("--verbose", action="store_true")
    rp.set_defaults(func=cmd_run)

    # -- temporal-prepare ------------------------------------------------
    tp = sub.add_parser(
        "temporal-prepare",
        help="turn interval JSONL relations into a valid-time disk index",
        description=(
            "Build a read-only temporal index from objects "
            '{"values":[...], "since":s, "until":u, "copies":c} (copies is '
            "optional and defaults to 1). Intervals are half-open [s, u); "
            "the directory is published atomically and a failure leaves no "
            "half-built index."
        ),
    )
    tp.add_argument("--decl", required=True, metavar="FILE",
                    help="query declaration JSON (unchanged static format)")
    tp.add_argument("--data", required=True, metavar="DIR",
                    help="directory containing temporal <relation>.jsonl files")
    tp.add_argument("--index", required=True, metavar="DIR",
                    help="temporal index directory to create")
    tp.add_argument("--memory", required=True, type=_parse_memory,
                    metavar="SIZE",
                    help="memory budget, e.g. 32M or 1G (event sort spills)")
    tp.add_argument("--sort-dir", default=None, metavar="DIR",
                    help="directory for external-sort run files (default: temp)")
    tp.add_argument("--sqlite-cache-kib", type=int, default=None)
    tp.add_argument("--force", action="store_true",
                    help="atomically replace an existing non-empty index")
    tp.add_argument("--json", action="store_true",
                    help="print the manifest as JSON to stdout")
    tp.add_argument("--verbose", action="store_true")
    tp.add_argument("--quiet", action="store_true")
    tp.set_defaults(func=cmd_temporal_prepare)

    # -- temporal-explain ------------------------------------------------
    te = sub.add_parser(
        "temporal-explain",
        help="show candidate seeks, validity handling, alias factors and "
             "preparation work",
    )
    te.add_argument("--decl", required=True, metavar="FILE")
    te.add_argument("--index", required=True, metavar="DIR")
    te.add_argument("--from", dest="from_", type=int, default=None,
                    metavar="L",
                    help="optional window start shown in the plan (with --to)")
    te.add_argument("--to", dest="to", type=int, default=None, metavar="U")
    te.add_argument("--memory", type=_parse_memory, default=64 << 20,
                    metavar="SIZE")
    te.add_argument("--limit", type=int, default=None)
    te.add_argument("--json", action="store_true")
    te.add_argument("--no-deep-verify", action="store_true")
    te.set_defaults(func=cmd_temporal_explain)

    # -- temporal-run ----------------------------------------------------
    tr = sub.add_parser(
        "temporal-run",
        help="emit valid-time closed-loop segments inside [--from, --to)",
        description=(
            "Emit one JSON object per positive-multiplicity, non-empty "
            'segment: {"values":[...], "since":a, "until":b, '
            '"multiplicity":m}. Bindings are ordered by the global variable '
            "order and segments by since; adjacent segments with the same "
            "multiplicity are merged into one maximal interval. Exit 0 means "
            "the enumeration completed; 7 means it did not."
        ),
    )
    tr.add_argument("--decl", required=True, metavar="FILE")
    tr.add_argument("--index", required=True, metavar="DIR")
    tr.add_argument("--from", dest="from_", required=True, type=int,
                    metavar="L", help="window start (inclusive)")
    tr.add_argument("--to", dest="to", required=True, type=int, metavar="U",
                    help="window end (exclusive); must exceed --from")
    tr.add_argument("--memory", type=_parse_memory, default=64 << 20,
                    metavar="SIZE")
    tr.add_argument("--out", default="-", metavar="FILE",
                    help="output file (default: stdout)")
    tr.add_argument("--limit", type=int, default=None,
                    help="stop after N segments and report the run incomplete")
    tr.add_argument("--max-output-bytes", type=int, default=None,
                    help="stop after this many output bytes and report the run "
                         "incomplete")
    tr.add_argument("--stats", default=None, metavar="FILE",
                    help="write exact run statistics as JSON, including the "
                         "integer weighted_duration")
    tr.add_argument("--stats-stdout", action="store_true",
                    help="print run statistics as one JSON line on stdout")
    tr.add_argument("--no-layer-stats", action="store_true")
    tr.add_argument("--no-deep-verify", action="store_true")
    tr.add_argument("--verbose", action="store_true")
    tr.set_defaults(func=cmd_temporal_run)

    # -- maintain-init ---------------------------------------------------
    mi = sub.add_parser(
        "maintain-init",
        help="create an independent maintenance library from a temporal index",
        description=(
            "Copy the temporal index's relation state into a new self-"
            "contained maintenance library and build the initial closed-loop "
            "view (revision 0). Afterwards the temporal index and the original "
            "files can be deleted."
        ),
    )
    mi.add_argument("--decl", required=True, metavar="FILE")
    mi.add_argument("--index", required=True, metavar="DIR",
                    help="existing temporal index directory")
    mi.add_argument("--maintain", required=True, metavar="DIR",
                    help="maintenance library directory to create")
    mi.add_argument("--memory", required=True, type=_parse_memory,
                    metavar="SIZE")
    mi.add_argument("--sort-dir", default=None, metavar="DIR")
    mi.add_argument("--sqlite-cache-kib", type=int, default=None)
    mi.add_argument("--force", action="store_true")
    mi.add_argument("--json", action="store_true")
    mi.add_argument("--verbose", action="store_true")
    mi.add_argument("--quiet", action="store_true")
    mi.add_argument("--no-deep-verify", action="store_true")
    mi.set_defaults(func=cmd_maintain_init)

    # -- maintain-apply --------------------------------------------------
    ma = sub.add_parser(
        "maintain-apply",
        help="apply one correction batch and emit the signed result delta",
        description=(
            "Incrementally apply a UTF-8 JSONL correction batch. The relation "
            "state, the closed-loop view, the signed delta and the batch "
            "receipt commit atomically and revision increments by one. An "
            "identical re-send (same batch id, bytes and expect-revision) "
            "returns the original receipt without another revision."
        ),
    )
    ma.add_argument("--decl", required=True, metavar="FILE")
    ma.add_argument("--maintain", required=True, metavar="DIR")
    ma.add_argument("--batch-id", required=True, metavar="ID")
    ma.add_argument("--expect-revision", required=True, type=int, metavar="N")
    ma.add_argument("--corrections", required=True, metavar="FILE",
                    help="UTF-8 JSONL correction batch")
    ma.add_argument("--memory", required=True, type=_parse_memory,
                    metavar="SIZE")
    ma.add_argument("--sqlite-cache-kib", type=int, default=None)
    ma.add_argument("--sort-dir", default=None, metavar="DIR",
                    help="directory for the disk-backed scratch database "
                         "(default: system temp; never inside the library)")
    ma.add_argument("--out", default="-", metavar="FILE",
                    help="delta JSONL output (default: stdout)")
    ma.add_argument("--stats", default=None, metavar="FILE")
    ma.add_argument("--stats-stdout", action="store_true")
    ma.add_argument("--no-layer-stats", action="store_true")
    ma.add_argument("--no-deep-verify", action="store_true")
    ma.add_argument("--kill-hook", default=None, choices=["before-commit"],
                    help=argparse.SUPPRESS)
    ma.add_argument("--json", action="store_true")
    ma.add_argument("--verbose", action="store_true")
    ma.add_argument("--quiet", action="store_true")
    ma.set_defaults(func=cmd_maintain_apply)

    # -- maintain-run ----------------------------------------------------
    mr = sub.add_parser(
        "maintain-run",
        help="stream the committed closed-loop view at its revision",
        description=(
            "Read the currently committed view inside [--from, --to) and "
            "report the revision it was read at. Only stored view rows are "
            "read; the join is never recomputed from the relation state."
        ),
    )
    mr.add_argument("--decl", required=True, metavar="FILE")
    mr.add_argument("--maintain", required=True, metavar="DIR")
    mr.add_argument("--from", dest="from_", required=True, type=int)
    mr.add_argument("--to", dest="to", required=True, type=int)
    mr.add_argument("--memory", type=_parse_memory, default=64 << 20,
                    metavar="SIZE")
    mr.add_argument("--out", default="-", metavar="FILE")
    mr.add_argument("--limit", type=int, default=None)
    mr.add_argument("--max-output-bytes", type=int, default=None)
    mr.add_argument("--stats", default=None, metavar="FILE")
    mr.add_argument("--stats-stdout", action="store_true")
    mr.add_argument("--no-layer-stats", action="store_true")
    mr.add_argument("--no-deep-verify", action="store_true")
    mr.add_argument("--quiet", action="store_true")
    mr.set_defaults(func=cmd_maintain_run)

    # -- maintain-explain ------------------------------------------------
    me = sub.add_parser(
        "maintain-explain",
        help="explain affected-scope discovery, delta synthesis and sync",
    )
    me.add_argument("--decl", required=True, metavar="FILE")
    me.add_argument("--maintain", required=True, metavar="DIR")
    me.add_argument("--json", action="store_true")
    me.add_argument("--no-deep-verify", action="store_true")
    me.set_defaults(func=cmd_maintain_explain)

    # -- maintain-export -------------------------------------------------
    mx = sub.add_parser(
        "maintain-export",
        help="stream a committed batch's signed delta by batch-id",
        description=(
            "Re-fetch the stored delta of an already committed batch without "
            "re-applying anything. Used to recover a delivery whose export "
            "failed after the commit."
        ),
    )
    mx.add_argument("--decl", required=True, metavar="FILE")
    mx.add_argument("--maintain", required=True, metavar="DIR")
    mx.add_argument("--batch-id", required=True, metavar="ID")
    mx.add_argument("--out", default="-", metavar="FILE")
    mx.add_argument("--json", action="store_true")
    mx.add_argument("--quiet", action="store_true")
    mx.add_argument("--no-deep-verify", action="store_true")
    mx.set_defaults(func=cmd_maintain_export)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except BrokenPipeError:
        try:
            sys.stderr.write("reljoin: incomplete: output consumer went away\n")
        except OSError:
            pass
        return EXIT_INCOMPLETE
    except ReljoinError as exc:
        _err(str(exc), exc.code)
        return _exit_code_for(exc)
    except KeyboardInterrupt:
        _err("interrupted; nothing was assumed complete", "incomplete")
        return EXIT_INCOMPLETE
    except Exception as exc:  # pragma: no cover - last resort
        _err(f"internal error: {type(exc).__name__}: {exc}", "internal")
        return EXIT_INTERNAL


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
