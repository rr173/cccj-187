"""Ergonomic error types shared by the library and the CLI.

Every expected failure carries a stable machine-readable ``code`` so the CLI
can classify it and tests can assert on it without matching prose.
"""
from __future__ import annotations


class ReljoinError(Exception):
    """Base class for all expected (diagnosable) failures."""

    code = "error"

    def __init__(self, message: str, *, code: str | None = None):
        super().__init__(message)
        if code is not None:
            self.code = code


class DeclarationProblem(ReljoinError):
    code = "declaration"


class IndexProblem(ReljoinError):
    code = "index"


class MissingIndex(IndexProblem):
    code = "index-missing"


class CorruptIndex(IndexProblem):
    code = "index-corrupt"


class WrongIndexKind(IndexProblem):
    """A static index was opened as temporal (or vice versa)."""

    code = "index-wrong-kind"


class BuildIncomplete(ReljoinError):
    code = "build-incomplete"


class DataProblem(ReljoinError):
    code = "data"


class ResourceError(ReljoinError):
    code = "resource"


class QueryAborted(ReljoinError):
    code = "query-aborted"
