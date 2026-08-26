"""Which code produced this number? — version stamping and stale-fork detection.

Three silent failures in this project have shared one shape: clean, plausible output
that measured the wrong thing. Accuracy-stopping measured a saturated metric, the DTW
coverage condition measured a degenerate alignment, and a forked worker pool measured
a **superseded solver** — the parent had imported the module before the edit, and
`fork` copies the parent's already-imported modules, so the workers ran old code while
the repository held new code.

The fix has to be structural. Two mechanisms:

**Import-time hashing, which is the part that is easy to get wrong.** Each estimation
module records a hash of its own source *at import*. Checking staleness then means
comparing that inherited value against the file as it stands on disk. Hashing the file
at worker start would not work — a forked child re-reading the file sees the *new*
bytes and reports agreement while executing the old bytes in memory. Only a value
captured at import time travels with the fork.

**A stamp on every record**, carrying the git SHA, whether the tree was dirty, and the
per-module hashes, so "which code produced this" is answerable from the result file
alone rather than from memory of what was running.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# The modules whose version a measurement depends on. Declared rather than inferred,
# because `_AT_IMPORT` only contains what a process happened to import — so an empty
# `stale` would otherwise mean either "nothing is stale" or "nothing was checked", and
# those must not look alike. `code_stamp` reports which of these are missing.
EXPECTED = ("core", "pipeline", "attribution")

# module name -> (source path, hash captured when that module was first imported)
_AT_IMPORT: dict[str, tuple[Path, str]] = {}


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def register(module_file: str) -> str:
    """Call at module scope: `_SOURCE = provenance.register(__file__)`.

    Evaluated once, at import. A forked child inherits the parent's value, which is
    what makes `assert_current` able to see that the child is running stale code.
    """
    path = Path(module_file).resolve()
    digest = _hash_file(path)
    _AT_IMPORT[path.stem] = (path, digest)
    return digest


def git_sha() -> str:
    try:
        out = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def git_dirty() -> bool:
    try:
        out = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--", "."],
                             capture_output=True, text=True, timeout=10)
        return bool(out.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        return False


def stale_modules() -> dict[str, dict[str, str]]:
    """Modules whose in-memory source differs from the file on disk."""
    out = {}
    for name, (path, at_import) in sorted(_AT_IMPORT.items()):
        try:
            now = _hash_file(path)
        except OSError:
            continue
        if now != at_import:
            out[name] = {"imported": at_import, "on_disk": now}
    return out


def code_stamp() -> dict:
    """Attach to every result record.

    `unregistered` lists any `EXPECTED` module absent from the registry, so a stamp
    cannot claim currency for code it never looked at.
    """
    return {
        "git_sha": git_sha(),
        "git_dirty": git_dirty(),
        "modules": {n: d for n, (_, d) in sorted(_AT_IMPORT.items())},
        "stale": stale_modules(),
        "unregistered": [n for n in EXPECTED if n not in _AT_IMPORT],
    }


def assert_current(*, strict: bool = True) -> dict:
    """Fail fast if the running process holds code older than the working tree.

    Called at worker start. `strict=False` warns instead, for interactive use where
    an edit mid-session is expected and harmless.
    """
    missing = [n for n in EXPECTED if n not in _AT_IMPORT]
    if missing and strict:
        raise RuntimeError(
            f"cannot certify this process: {missing} were never registered, so their "
            f"version is unknown. An empty staleness report here would mean 'not "
            f"checked', not 'not stale'. Import them before measuring, or amend "
            f"`provenance.EXPECTED` if they are genuinely not in the measurement path."
        )
    stale = stale_modules()
    if stale and strict:
        detail = ", ".join(
            f"{n} (imported {d['imported']}, on disk {d['on_disk']})"
            for n, d in stale.items())
        raise RuntimeError(
            "this process is running stale code — the source changed after it was "
            f"imported: {detail}. A forked worker pool inherits the parent's modules, "
            "so restart the pool (or use the 'spawn' start method) before measuring. "
            "Any timing or geometry produced here would describe superseded code."
        )
    return code_stamp()
