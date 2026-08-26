"""Check the submission the way a reviewer's first ten seconds would, and refuse to pass quietly.

A submission checklist written as prose in a log gets read once, on the day it is written, and then
ticked from memory at 3am on the deadline. This is the same checklist as a script, so the ticking is
done by something that actually looks.

Each check reports one of three states, and the distinction is the point:

    PASS     the check ran and the condition holds
    FAIL     the check ran and the condition does not hold
    UNKNOWN  the check could not run --- no LaTeX toolchain, no built PDF, no main.tex yet

`UNKNOWN` is not a pass. Several items on the venue checklist (page count, bibliography compiles)
cannot be verified without a TeX installation, and this machine has none. Printing them as `PASS`
because nothing objected would be the same fault the appendix documents four times: a guard that
fails open. They are printed as `UNKNOWN` with the reason, they are counted separately, and the
exit code distinguishes them --- 0 clean, 1 something failed, 2 nothing failed but something could
not be checked and a human still has to.

    python scripts/check_submission.py [--paper-dir paper]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import zlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Strings that would identify the authors, their machine, or their institution if they survived
# into a submitted file. The self-citation is the subtle one: an anonymised paper that cites
# "our prior work" by arXiv id has deanonymised itself as thoroughly as a name on the title page.
IDENTIFYING = {
    "/home/": "an absolute path from the author's machine",
    "kati": "an author's name",
    "Structure-Function-Analysis": "the repository name",
    "2604.27656": "the authors' own prior arXiv submission, cited by id",
    "github.com": "a link to the authors' code",
    "Overleaf": "an editing-tool artifact",
}

PLACEHOLDERS = (r"\\todo\{", r"\\tc\{", r"\\cn\{", r"\\TODO\{", r"\\fixme\{", r"\\citeneeded\b")
WATERMARKS = (r"\\watermark", r"draftonly", r"\bDRAFT\b", r"\\usepackage\{draftwatermark\}")

PAGE_LIMIT = 4  # NeurReps Extended Abstract, excluding references and appendices


class Report:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str]] = []

    def add(self, state: str, name: str, detail: str = "") -> None:
        self.rows.append((state, name, detail))

    def ok(self, name: str, detail: str = "") -> None:
        self.add("PASS", name, detail)

    def bad(self, name: str, detail: str) -> None:
        self.add("FAIL", name, detail)

    def unknown(self, name: str, detail: str) -> None:
        self.add("UNKNOWN", name, detail)

    def render(self) -> int:
        width = max(len(n) for _s, n, _d in self.rows)
        for state, name, detail in self.rows:
            print(f"  {state:<8} {name:<{width}}  {detail}")
        fails = sum(s == "FAIL" for s, _n, _d in self.rows)
        unknowns = sum(s == "UNKNOWN" for s, _n, _d in self.rows)
        print(f"\n{len(self.rows)} checks: {len(self.rows) - fails - unknowns} pass, "
              f"{fails} fail, {unknowns} could not be checked")
        if fails:
            return 1
        return 2 if unknowns else 0


def tex_sources(paper: Path) -> list[Path]:
    return sorted(p for p in paper.glob("*.tex"))


def strip_comments(text: str) -> str:
    r"""Drop TeX comments, keeping escaped \% .

    Structural checks must read what LaTeX reads, or a file explaining that it contains no
    `\begin{document}` counts as containing one -- which is how this function came to exist. The
    anonymisation check deliberately does NOT use it: a comment naming the author's machine ships
    with the source even though it never reaches the PDF.
    """
    out = []
    for line in text.splitlines():
        cut = None
        for i, ch in enumerate(line):
            if ch == "%" and (i == 0 or line[i - 1] != "\\"):
                cut = i
                break
        out.append(line if cut is None else line[:cut])
    return "\n".join(out)


def read_all(paths: list[Path]) -> str:
    """Concatenated source with comments removed -- for structural checks only."""
    return "\n".join(strip_comments(p.read_text(errors="replace")) for p in paths)


def check_placeholders(rep: Report, tex: list[Path]) -> None:
    hits = []
    for p in tex:
        for i, line in enumerate(strip_comments(p.read_text(errors="replace")).splitlines(), 1):
            for pat in PLACEHOLDERS:
                if re.search(pat, line):
                    hits.append(f"{p.name}:{i}")
    if hits:
        rep.bad("placeholder macros", f"{len(hits)} remaining: {', '.join(hits[:6])}")
    else:
        rep.ok("placeholder macros", "no \\todo, \\tc, \\cn or \\fixme")


def check_watermark(rep: Report, tex: list[Path]) -> None:
    hits = [f"{p.name}" for p in tex
            if any(re.search(w, strip_comments(p.read_text(errors="replace")))
                   for w in WATERMARKS)]
    if hits:
        rep.bad("watermark removed", f"draft watermark still set in {', '.join(hits)}")
    else:
        rep.ok("watermark removed")


def check_anonymity(rep: Report, paper: Path, tex: list[Path]) -> None:
    """Scan submitted text and every figure the paper embeds, including binary metadata."""
    targets: list[tuple[str, bytes]] = [(p.name, p.read_bytes()) for p in tex]
    targets += [(p.name, p.read_bytes()) for p in paper.glob("*.bib")]
    figdir = ROOT / "figures"
    targets += [(p.name, p.read_bytes()) for p in sorted(figdir.glob("*.pdf"))]

    hits = []
    for name, blob in targets:
        low = blob.lower()
        for needle, why in IDENTIFYING.items():
            if needle.lower().encode() in low:
                hits.append(f"{name} contains {needle!r} ({why})")
    if hits:
        rep.bad("anonymisation", "; ".join(hits[:5]))
    else:
        rep.ok("anonymisation",
               f"{len(targets)} files scanned incl. figure PDF metadata; no identifying strings")


def check_figures(rep: Report, tex: list[Path]) -> None:
    """Every embedded graphic must be a file the manifest currently claims."""
    man_path = ROOT / "figures" / "MANIFEST.json"
    if not man_path.exists():
        rep.unknown("figures traceable", "no figures/MANIFEST.json")
        return
    man = json.loads(man_path.read_text())
    claimed = {n: d for e in man["figures"].values() for n, d in e["outputs"].items()}

    used = set()
    for m in re.finditer(r"\\includegraphics(?:\[[^\]]*\])?\s*\{\s*([^}]+?)\s*\}",
                         read_all(tex), re.S):
        used.add(re.sub(r"\s+", "", m.group(1)))
    if not used:
        rep.unknown("figures traceable", "no \\includegraphics found in the paper sources")
        return

    problems = []
    for ref in sorted(used):
        name = Path(ref).name
        if "superseded" in ref:
            problems.append(f"{name} is a superseded variant")
        elif name not in claimed:
            problems.append(f"{name} is not in the manifest")
        else:
            on_disk = ROOT / "figures" / name
            digest = hashlib.sha256(on_disk.read_bytes()).hexdigest()[:16]
            if digest != claimed[name]:
                problems.append(f"{name} differs from the manifest")
    if problems:
        rep.bad("figures traceable", "; ".join(problems))
    else:
        rep.ok("figures traceable",
               f"{len(used)} graphics, all in the manifest at the current digest")

    if man.get("git_dirty"):
        rep.bad("figures from a committed tree",
                f"manifest records git_dirty at {man.get('git_sha')} — the scripts that drew "
                f"these figures are not the scripts at that commit")
    else:
        rep.ok("figures from a committed tree", f"git {man.get('git_sha')}")


def check_bibliography(rep: Report, paper: Path, tex: list[Path]) -> None:
    bibs = sorted(paper.glob("*.bib"))
    if not bibs:
        rep.unknown("bibliography", "no .bib file in the paper directory")
        return
    # Raw, not comment-stripped. BibTeX does not treat `%` as a comment anyway, and reading this
    # through strip_comments once made the check pass by not looking, which is the failure the whole
    # appendix is about.
    text = "\n".join(p.read_text(errors="replace") for p in bibs)

    # Anywhere inside an entry, in any syntax. This used to require the marker to be an indented
    # `%` comment. Then the markers were moved into `note` fields -- because `%` does not comment
    # inside a BibTeX entry and was silently deleting three references -- and this check went green
    # while four unverified fields sat in the file. Same fault as the strip_comments one and the
    # PDF page counter: a check that stops seeing rather than starts failing. So: match the token
    # itself, not the syntax that happens to carry it, and exclude only the file header.
    body = text.split("@", 1)[1] if "@" in text else text
    unverified = re.findall(r"^.*TODO-VERIFY.*$", body, re.M)
    if unverified:
        rep.bad("bibliography fields verified",
                f"{len(unverified)} entries have a field nobody has looked up")
    else:
        rep.ok("bibliography fields verified")

    keys = set(re.findall(r"@\w+\{\s*([^,\s]+)\s*,", text))
    body = read_all(tex)
    cited = set()
    for m in re.finditer(r"\\cite[a-zA-Z]*\s*(?:\[[^\]]*\])*\s*\{([^}]*)\}", body):
        cited |= {k.strip() for k in m.group(1).split(",") if k.strip()}

    missing = sorted(cited - keys)
    if missing:
        rep.bad("citations resolve", f"cited but not in the .bib: {', '.join(missing)}")
    elif not cited:
        rep.unknown("citations resolve", "no \\cite commands found in the paper sources")
    else:
        rep.ok("citations resolve", f"{len(cited)} keys, all present in {bibs[0].name}")

    # An entry in the .bib that nothing cites is harmless to the PDF and useful to know about:
    # it is usually a citation site that was meant to exist and does not.
    unused = sorted(keys - cited)
    if unused and cited:
        rep.ok("unused bib entries", f"{len(unused)} not cited: {', '.join(unused[:6])}")


def count_pages(pdf: Path) -> int:
    """Page count, including PDFs whose page tree lives in a compressed object stream.

    A plain regex over the raw bytes finds `/Type /Page` only when the objects are uncompressed.
    Modern engines --- xdvipdfmx among them --- pack the page tree into object streams, and the
    regex then returns 0, which reads as `could not count` rather than as `wrong`. So: try the raw
    bytes, then try again over every inflated stream, and prefer the larger answer.
    """
    blob = pdf.read_bytes()
    direct = len(re.findall(rb"/Type\s*/Page[^s]", blob))

    inflated = 0
    for m in re.finditer(rb"stream\r?\n", blob):
        start = m.end()
        end = blob.find(b"endstream", start)
        if end == -1:
            continue
        try:
            data = zlib.decompress(blob[start:end])
        except zlib.error:
            continue
        inflated += len(re.findall(rb"/Type\s*/Page[^s]", data))

    # A /Count in the page-tree root is the authoritative answer when it survives compression.
    counts = [int(c) for c in re.findall(rb"/Type\s*/Pages.{0,200}?/Count\s+(\d+)", blob, re.S)]
    return max([direct, inflated] + counts) if (direct or inflated or counts) else 0


def check_page_count(rep: Report, paper: Path) -> None:
    pdfs = sorted(paper.glob("*.pdf"))
    if not pdfs:
        rep.unknown("page count", f"no built PDF in {paper.name}/ — nothing to count")
        return
    pdf = pdfs[0]
    pages = count_pages(pdf)
    if not pages:
        rep.unknown("page count", f"could not count pages in {pdf.name}")
    elif pages > PAGE_LIMIT:
        rep.bad("page count", f"{pdf.name} is {pages} pages; the limit is {PAGE_LIMIT} "
                              f"excluding references and appendices — check where the split falls")
    else:
        rep.ok("page count", f"{pages} pages in {pdf.name} (limit {PAGE_LIMIT})")


def check_keywords(rep: Report, tex: list[Path]) -> None:
    body = read_all(tex)
    if not body.strip():
        rep.unknown("keywords present", "no paper sources")
    elif re.search(r"\\begin\{keywords\}", body):
        rep.ok("keywords present")
    else:
        rep.bad("keywords present", "no \\begin{keywords} block in the paper sources")


def check_compiles(rep: Report) -> None:
    import shutil
    if shutil.which("pdflatex") or shutil.which("latexmk") or shutil.which("xelatex"):
        rep.unknown("bibliography compiles",
                    "a TeX toolchain is present — run latexmk and read the .blg yourself")
    else:
        rep.unknown("bibliography compiles",
                    "no TeX toolchain on this machine; must be checked where the paper is built")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper-dir", default="paper")
    args = ap.parse_args()
    paper = ROOT / args.paper_dir

    rep = Report()
    if not paper.exists():
        print(f"no {args.paper_dir}/ directory")
        sys.exit(1)
    tex = tex_sources(paper)
    if not any(re.search(r"\\begin\{document\}", p.read_text(errors="replace")) for p in tex):
        rep.unknown("main document", f"no \\begin{{document}} in {args.paper_dir}/ — the jmlr "
                                     f"skeleton has not been assembled yet")

    check_placeholders(rep, tex)
    check_watermark(rep, tex)
    check_anonymity(rep, paper, tex)
    check_figures(rep, tex)
    check_bibliography(rep, paper, tex)
    check_keywords(rep, tex)
    check_page_count(rep, paper)
    check_compiles(rep)

    print(f"\nsubmission check — {args.paper_dir}/\n")
    sys.exit(rep.render())


if __name__ == "__main__":
    main()
