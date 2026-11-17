"""Read/write helpers for JSONL test data (oracle-safe, no production code)."""
import json
from typing import Iterable, Sequence


def write_jsonl(path: str, rows: Iterable[Sequence[int]]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(list(row)))
            fh.write("\n")
