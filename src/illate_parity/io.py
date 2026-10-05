"""JSONL helpers. Every tool in this repo reads and writes the same two record shapes.

Gold record:        {"id": str, "text": str, "label": str}
Prediction record:  {"id": str, "label": str, "latency_ms"?: float,
                     "input_tokens"?: int, "output_tokens"?: int}
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path


def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_no}: invalid JSON ({e.msg})") from e
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def index_by_id(rows: list[dict], what: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for row in rows:
        if "id" not in row:
            raise ValueError(f"{what}: a record has no 'id' field: {row!r}")
        key = str(row["id"])
        if key in out:
            raise ValueError(f"{what}: duplicate id {key!r}")
        out[key] = row
    return out


def align(gold: list[dict], preds: list[dict], what: str = "predictions") -> list[dict]:
    """Return predictions in gold order. Missing ids are an error, not silently dropped:
    a model that skipped the hard items would otherwise look better than it is."""
    by_id = index_by_id(preds, what)
    missing = [str(g["id"]) for g in gold if str(g["id"]) not in by_id]
    if missing:
        head = ", ".join(missing[:5])
        raise ValueError(f"{what}: {len(missing)} gold ids have no prediction (e.g. {head})")
    return [by_id[str(g["id"])] for g in gold]
