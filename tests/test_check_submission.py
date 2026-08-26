"""The submission checklist, pinned against passing for the wrong reason.

A checklist is only worth running if a failure is louder than an omission. Every test here is a
case where the check could plausibly report `PASS` while the condition does not hold -- which has
already happened once: comment-stripping was added so that a file explaining it contains no
`\\begin{document}` would not count as containing one, and it silently blinded the bibliography
check, whose markers live behind a `%`. That bug passed a manual read and would have shipped.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "check_submission", ROOT / "scripts" / "check_submission.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


CS = _load()


def _states(rep):
    return {name: state for state, name, _detail in rep.rows}


# --- strip_comments, which is the load-bearing helper -------------------------------------------

def test_comments_are_stripped_so_prose_about_a_macro_is_not_read_as_the_macro():
    text = r"% there is no \begin{document} in this file" "\n" r"\section{Real}"
    assert r"\begin{document}" not in CS.strip_comments(text)
    assert r"\section{Real}" in CS.strip_comments(text)


def test_an_escaped_percent_is_not_a_comment():
    assert CS.strip_comments(r"varies by 1.5\% across widths") == r"varies by 1.5\% across widths"


def test_a_mid_line_comment_keeps_the_code_before_it():
    assert CS.strip_comments(r"\label{fig:a} % naming").rstrip() == r"\label{fig:a}"


# --- the bug this file exists for ---------------------------------------------------------------

def test_an_unverified_bib_field_fails_even_though_its_marker_is_behind_a_percent(tmp_path):
    (tmp_path / "refs.bib").write_text(
        "@article{x,\n  author = {A},\n  journal = {J},\n"
        "  % TODO-VERIFY: volume is not in the reference record\n}\n")
    (tmp_path / "body.tex").write_text(r"\cite{x}")
    rep = CS.Report()
    CS.check_bibliography(rep, tmp_path, CS.tex_sources(tmp_path))
    assert _states(rep)["bibliography fields verified"] == "FAIL"


def test_an_unverified_bib_field_fails_when_its_marker_is_a_note_field(tmp_path):
    """The marker moved out of a `%` comment and the check went green with four still in the file.

    `%` does not comment inside a BibTeX entry -- it was deleting whole entries -- so the markers
    became `note` fields. The check matched only indented `%` lines, so it passed by no longer
    looking. It must match the token wherever it is, not the syntax carrying it.
    """
    (tmp_path / "refs.bib").write_text(
        "@article{x,\n  author = {A},\n  journal = {J},\n"
        "  note    = {TODO-VERIFY: volume is not in the reference record},\n}\n")
    (tmp_path / "body.tex").write_text(r"\cite{x}")
    rep = CS.Report()
    CS.check_bibliography(rep, tmp_path, CS.tex_sources(tmp_path))
    assert _states(rep)["bibliography fields verified"] == "FAIL"


def test_the_bib_header_describing_the_marker_does_not_count_as_one(tmp_path):
    """A header explaining the convention names the token; that is not an unverified field."""
    (tmp_path / "refs.bib").write_text(
        "% Entries needing a lookup carry TODO-VERIFY in a note field.\n"
        "@article{x,\n  author = {A},\n  volume = {11},\n}\n")
    (tmp_path / "body.tex").write_text(r"\cite{x}")
    rep = CS.Report()
    CS.check_bibliography(rep, tmp_path, CS.tex_sources(tmp_path))
    assert _states(rep)["bibliography fields verified"] == "PASS"


def test_a_fully_specified_bib_passes(tmp_path):
    (tmp_path / "refs.bib").write_text("@article{x,\n  author = {A},\n  volume = {11},\n}\n")
    (tmp_path / "body.tex").write_text(r"\cite{x}")
    rep = CS.Report()
    CS.check_bibliography(rep, tmp_path, CS.tex_sources(tmp_path))
    assert _states(rep)["bibliography fields verified"] == "PASS"


def test_a_citation_with_no_bib_entry_fails(tmp_path):
    (tmp_path / "refs.bib").write_text("@article{present,\n  volume = {1},\n}\n")
    (tmp_path / "body.tex").write_text(r"as shown by \citet{absent}")
    rep = CS.Report()
    CS.check_bibliography(rep, tmp_path, CS.tex_sources(tmp_path))
    assert _states(rep)["citations resolve"] == "FAIL"


def test_no_citations_at_all_is_unknown_rather_than_pass(tmp_path):
    (tmp_path / "refs.bib").write_text("@article{x,\n  volume = {1},\n}\n")
    (tmp_path / "body.tex").write_text(r"\section{No citations here}")
    rep = CS.Report()
    CS.check_bibliography(rep, tmp_path, CS.tex_sources(tmp_path))
    assert _states(rep)["citations resolve"] == "UNKNOWN"


# --- anonymisation must read comments, unlike everything else ----------------------------------

def test_an_identifying_string_in_a_comment_still_fails(tmp_path):
    (tmp_path / "body.tex").write_text("% drawn from /home/someone/work\n\\section{A}")
    rep = CS.Report()
    CS.check_anonymity(rep, tmp_path, CS.tex_sources(tmp_path))
    assert _states(rep)["anonymisation"] == "FAIL"


def test_a_self_citation_by_arxiv_id_is_treated_as_deanonymising(tmp_path):
    (tmp_path / "body.tex").write_text(r"our earlier work (arXiv:2604.27656) showed")
    rep = CS.Report()
    CS.check_anonymity(rep, tmp_path, CS.tex_sources(tmp_path))
    assert _states(rep)["anonymisation"] == "FAIL"


# --- placeholders and keywords -----------------------------------------------------------------

def test_a_placeholder_macro_fails_but_prose_naming_it_does_not(tmp_path):
    (tmp_path / "a.tex").write_text(r"\todo{write this}")
    (tmp_path / "b.tex").write_text(r"% the \todo{} count must reach zero")
    rep = CS.Report()
    CS.check_placeholders(rep, [tmp_path / "a.tex"])
    CS.check_placeholders(rep, [tmp_path / "b.tex"])
    assert [s for s, _n, _d in rep.rows] == ["FAIL", "PASS"]


def test_missing_keywords_fails_and_present_keywords_passes(tmp_path):
    (tmp_path / "a.tex").write_text(r"\section{A}")
    rep = CS.Report()
    CS.check_keywords(rep, [tmp_path / "a.tex"])
    (tmp_path / "a.tex").write_text("\\begin{keywords}\ngeometry\n\\end{keywords}")
    CS.check_keywords(rep, [tmp_path / "a.tex"])
    assert [s for s, _n, _d in rep.rows] == ["FAIL", "PASS"]


# --- exit codes: UNKNOWN is not PASS -----------------------------------------------------------

@pytest.mark.parametrize("states,expected", [
    (["PASS", "PASS"], 0),
    (["PASS", "UNKNOWN"], 2),
    (["PASS", "FAIL"], 1),
    (["FAIL", "UNKNOWN"], 1),
])
def test_an_uncheckable_item_never_exits_clean(states, expected, capsys):
    rep = CS.Report()
    for i, s in enumerate(states):
        rep.add(s, f"check {i}", "")
    assert rep.render() == expected
    capsys.readouterr()


# --- the real paper directory -------------------------------------------------------------------

def test_the_real_figure_references_are_all_in_the_manifest():
    """The paper must not embed a graphic the manifest does not claim, or a superseded variant."""
    paper = ROOT / "paper"
    rep = CS.Report()
    CS.check_figures(rep, CS.tex_sources(paper))
    assert _states(rep)["figures traceable"] == "PASS"
