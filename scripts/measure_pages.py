"""How many pages, and where do they go?

The length decisions all turn on this number and it was UNKNOWN until the paper was first compiled.
It will be asked repeatedly while cutting, so it is a script rather than a sequence of shell
invocations, and it reports the *decomposition* rather than one total: the useful question is not
"how long is it" but "what would deleting this buy".

Variants are built from `paper/harness/main.tex` by substitution, compiled into a scratch directory,
and counted with `check_submission.count_pages`, which inflates PDF object streams -- a plain regex
over the bytes returns 0 for this engine's output.

**Float placement is chaotic near the boundary.** Whether a float shares a page with text depends on
where the surrounding text happens to break, so a single count can move by a page for reasons that
have nothing to do with length. Counts are therefore reported together, and a difference of one page
between two variants should not be read as a real saving without checking the float warnings too.

    python scripts/measure_pages.py
"""

from __future__ import annotations

import importlib.util
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HARNESS = ROOT / "paper" / "harness" / "main.tex"
TECTONIC = shutil.which("tectonic") or "/tmp/tectonic"

LIMIT_PAGES = 4


def _check_submission():
    spec = importlib.util.spec_from_file_location("cs", ROOT / "scripts" / "check_submission.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def stub_captions(text: str) -> str:
    """Replace every caption body with a stub, keeping the float and its graphic."""
    return re.sub(r"\{\\caption\{.*?\}\}\n(\s*\{\\includegraphics)", r"{\\caption{Stub.}}\n\1",
                  text, flags=re.S)


def caption_words() -> dict[str, int]:
    """Words per body caption. The label precedes the caption in `\\floatconts`, so the label is
    captured first -- reversing the two silently attributed one figure's count to the other."""
    text = (ROOT / "paper" / "figures.tex").read_text()
    out = {}
    for m in re.finditer(r"\{fig:(\w+)\}\s*\n\s*\{\\caption\{(.*?)\}\}\s*\n\s*\{\\includegraphics",
                         text, re.S):
        out[m.group(1)] = len(m.group(2).split())
    return out


def main() -> None:
    cs = _check_submission()
    harness = HARNESS.read_text()
    figures = (ROOT / "paper" / "figures.tex").read_text()

    # (label, keep figures, keep references, stub the captions)
    variants = [
        ("body text alone", False, False, False),
        ("body + references", False, True, False),
        ("body + figures, captions stubbed", True, False, True),
        ("body + figures", True, False, False),
        ("body + figures + references  (the submission)", True, True, False),
    ]

    # The variants have to be compiled *in* paper/harness/, because the harness reaches out with
    # `\input{../body}`, `\bibliography{../references}` and `\graphicspath{{../../}}`. Compiling a
    # copy in a temp directory silently produces a 0-page PDF instead of failing.
    scratch = HARNESS.parent
    written: list[Path] = []
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        stubfile = scratch / "_measure_figstub.tex"
        stubfile.write_text(stub_captions(figures))
        written.append(stubfile)
        rows = []
        try:
            for label, figs, refs, stub in variants:
                src = harness
                if stub:
                    src = src.replace("\\input{../figures}", "\\input{_measure_figstub}")
                elif not figs:
                    src = src.replace("\\input{../figures}", "%")
                if not refs:
                    src = src.replace("\\bibliography{../references}", "%")
                name = "_measure_" + re.sub(r"\W+", "_", label).strip("_")
                tex = scratch / f"{name}.tex"
                tex.write_text(src)
                written.append(tex)
                subprocess.run([TECTONIC, "-X", "compile", str(tex),
                                "--outdir", str(td), "--keep-logs"],
                               capture_output=True, check=False)
                pdf, log = td / f"{name}.pdf", td / f"{name}.log"
                pages = cs.count_pages(pdf) if pdf.exists() else 0
                over = len(re.findall(r"Float too large", log.read_text(errors="replace"))) \
                    if log.exists() else 0
                rows.append((label, pages, over))
        finally:
            for p in written:
                p.unlink(missing_ok=True)

        width = max(len(r[0]) for r in rows)
        print(f"{'variant':<{width}}  pages  floats over")
        for label, pages, over in rows:
            flag = "" if not over else f"  <- {over} float(s) exceed the text block"
            print(f"{label:<{width}}  {pages:>5}  {over:>11}{flag}")

        print(f"\nlimit is {LIMIT_PAGES} pages excluding references and appendices.")
        cw = caption_words()
        total = sum(cw.values())
        print(f"body captions: {', '.join(f'{k} {v} words' for k, v in sorted(cw.items()))} "
              f"-- {total} words total")
        stubbed = next(p for lbl, p, _ in rows if "stubbed" in lbl)
        full = next(p for lbl, p, _ in rows if lbl == "body + figures")
        print(f"captions cost {full - stubbed} page(s) of the {full} in `body + figures`.")


if __name__ == "__main__":
    main()
