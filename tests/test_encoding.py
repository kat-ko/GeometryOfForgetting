"""Error registry I14: repo text files are UTF-8, and Windows defaults to cp1252.

Every text-mode `open`, `read_text` and `write_text` must name its encoding, or the
same code reads differently on the laptop and on the cluster. This scans the code
statically, so a new call site without `encoding=` fails here and not later at a
UnicodeDecodeError.
"""

from __future__ import annotations

import ast
import warnings
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ROOTS = ("src", "scripts", "tests", "notebooks/core")


def _mode(call: ast.Call) -> str | None:
    """The mode string of an `open` call, 'r' when absent, None when not a literal."""
    f = call.func
    pos = call.args[0] if isinstance(f, ast.Attribute) and call.args else (
        call.args[1] if len(call.args) >= 2 else None)
    m = next((k.value for k in call.keywords if k.arg == "mode"), pos)
    if m is None:
        return "r"
    return m.value if isinstance(m, ast.Constant) and isinstance(m.value, str) else None


def unencoded_text_io(path: Path) -> list[str]:
    with warnings.catch_warnings():               # escape-sequence warnings in scanned files
        warnings.simplefilter("ignore")
        tree = ast.parse(path.read_text(encoding="utf-8"))
    out = []
    for n in ast.walk(tree):
        if not isinstance(n, ast.Call):
            continue
        kw = {k.arg for k in n.keywords}
        if "encoding" in kw or None in kw:          # None: **kwargs may carry it
            continue
        f = n.func
        name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None)
        if name in ("read_text", "write_text"):
            out.append(f"{path.relative_to(REPO)}:{n.lineno} {name}")
        elif name == "open":
            mode = _mode(n)
            if mode is not None and "b" not in mode:
                out.append(f"{path.relative_to(REPO)}:{n.lineno} open({mode!r})")
    return out


def test_every_text_io_call_names_its_encoding():
    bad = [hit for root in ROOTS for p in sorted((REPO / root).rglob("*.py"))
           if "__pycache__" not in p.parts for hit in unencoded_text_io(p)]
    assert not bad, "text I/O without encoding= (Windows reads cp1252):\n" + "\n".join(bad)


def test_the_scan_catches_a_bare_read_text():
    p = REPO / "tests" / "_encoding_probe_tmp.py"
    try:
        p.write_text("from pathlib import Path\nPath('x').read_text()\nopen('y', 'w')\n"
                     "open('z', 'rb')\nPath('x').read_text(encoding='utf-8')\n", encoding="utf-8")
        hits = unencoded_text_io(p)
    finally:
        p.unlink(missing_ok=True)
    assert [h.split(" ", 1)[1] for h in hits] == ["read_text", "open('w')"]
