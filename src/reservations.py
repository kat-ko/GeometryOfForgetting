"""The two arrangement populations in `results/reserved_arrangements.json`.

The reserved set never informs `lr0`, the rank grid, or the stopping criterion.
`tests/test_reserved_arrangements.py` asserts that provenance directly.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
RESERVATION_PATH = ROOT / "results" / "reserved_arrangements.json"

_STREAM_ID_KEYS = frozenset({"stream_id", "stream_ids"})


def load() -> dict:
    return json.loads(RESERVATION_PATH.read_text())


def reserved_stream_ids() -> frozenset[int]:
    return frozenset(int(i) for i in load()["populations"]["reserved"]["stream_ids"])


def old_init_seeds() -> tuple[int, ...]:
    return tuple(int(a["init_seed"]) for a in load()["populations"]["old"]["arrangements"])


def _ints_from(value: Any) -> list[int]:
    if isinstance(value, bool) or value is None:
        return []
    if isinstance(value, int):
        return [value]
    if isinstance(value, list):
        out: list[int] = []
        for v in value:
            out.extend(_ints_from(v))
        return out
    return []


def stream_ids_in_json(obj: Any) -> set[int]:
    found: set[int] = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in _STREAM_ID_KEYS:
                found.update(_ints_from(v))
            else:
                found.update(stream_ids_in_json(v))
    elif isinstance(obj, list):
        for v in obj:
            found.update(stream_ids_in_json(v))
    return found


def stream_ids_in_python(source: str) -> set[int]:
    """Assignments and keywords named stream_id / stream_ids / STREAM_IDS."""
    found: set[int] = set()
    tree = ast.parse(source)

    def take(name: str | None, value: ast.AST) -> None:
        if name not in {"stream_id", "stream_ids", "STREAM_IDS"}:
            return
        if isinstance(value, ast.Constant) and isinstance(value.value, int):
            found.add(value.value)
        elif isinstance(value, (ast.List, ast.Tuple, ast.Set)):
            for elt in value.elts:
                if isinstance(elt, ast.Constant) and isinstance(elt.value, int):
                    found.add(elt.value)
        elif (isinstance(value, ast.Call) and isinstance(value.func, ast.Name)
              and value.func.id == "range"):
            args = [a.value for a in value.args
                    if isinstance(a, ast.Constant) and isinstance(a.value, int)]
            if len(args) >= 2:
                found.update(range(args[0], args[1]))

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    take(t.id, node.value)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value:
            take(node.target.id, node.value)
        elif isinstance(node, ast.keyword):
            take(node.arg, node.value)
    return found


def design_decision_paths(record: dict | None = None) -> list[Path]:
    rec = record if record is not None else load()
    paths: list[Path] = []
    seen: set[Path] = set()
    for rel in rec["design_decision_artifacts"]:
        p = ROOT / rel
        if p.exists() and p.resolve() not in seen:
            seen.add(p.resolve())
            paths.append(p)
    for pattern in rec["design_decision_globs"]:
        for p in ROOT.glob(pattern):
            if p.resolve() not in seen and p.resolve() != RESERVATION_PATH.resolve():
                seen.add(p.resolve())
                paths.append(p)
    return paths


def reserved_ids_in_path(path: Path, reserved: Iterable[int]) -> set[int]:
    reserved_set = set(reserved)
    text = path.read_text()
    if path.suffix == ".json":
        found = stream_ids_in_json(json.loads(text))
    elif path.suffix == ".py":
        found = stream_ids_in_python(text)
    else:
        found = set()
    return found & reserved_set
