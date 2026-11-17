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

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_DECLARATION = 3
EXIT_DATA = 4
EXIT_MISSING_INDEX = 5
EXIT_CORRUPT_INDEX = 6
EXIT_INCOMPLETE = 7
EXIT_RESOURCE = 8
EXIT_INTERNAL = 9


class _Cancelled(Exception):
    """Raised from the SIGINT/SIGTERM handler inside the run loop."""


def _exit_code_for(exc: BaseException) -> int:
    if isinstance(exc, QueryAborted):
        return EXIT_INCOMPLETE
    if isinstance(exc, MissingIndex):
        return EXIT_MISSING_INDEX
    if isinstance(exc, CorruptIndex):
        return EXIT_CORRUPT_INDEX
    if isinstance(exc, DataProblem):
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
    idx = open_index(
        args.index,
        expected_declaration=decl,
        deep=not args.no_deep_verify,
        progress=_progress_printer(args.verbose),
    )
    q = api.query(idx, decl, memory_bytes=args.memory, limit=args.limit,
                  verify_mode="deep" if not args.no_deep_verify else "shallow")

    previous = _install_signal_handlers()
    out = None
    close_out = False
    try:
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
        q.close()
        q.stats.status = "incomplete"
        q.stats.status_reason = "cancelled by signal"
        _emit_stats(args, q)
        _err("cancelled by signal; the result set is incomplete", "incomplete")
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
        q.close()
        q.stats.status = "incomplete"
        q.stats.status_reason = f"write failed: {exc}"
        _emit_stats(args, q)
        _err(f"write failed: {exc}; the result set is incomplete", "incomplete")
        return EXIT_INCOMPLETE
    finally:
        _restore_signal_handlers(previous)
        q.close()


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
