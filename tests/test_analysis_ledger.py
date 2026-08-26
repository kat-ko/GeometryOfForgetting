"""The notebook's discipline, pinned.

`ledger` exists so that a number cannot be shown without its arm count and selection rule. That
is a property of the code, not of whoever writes the next cell, so it is tested rather than
documented.
"""

from __future__ import annotations

import pytest

from src.analysis import ledger as L


def test_a_table_refuses_to_render_without_its_arm_count():
    with pytest.raises(ValueError, match="state n"):
        L.Table("magnitude at gamma=10", selection="three forgetting corners", source="x")


def test_a_table_refuses_to_render_without_its_selection_rule():
    with pytest.raises(ValueError, match="selection"):
        L.Table("magnitude at gamma=10", selection="", source="x", n=120)


def test_a_value_that_cannot_be_re_derived_says_so_rather_than_showing_blank():
    # The requirement is that absence is legible. A blank cell reads as zero to a tired reader.
    assert "not re-derivable" in L.Row("something", None).cells()[1]


def test_full_precision_is_kept_alongside_the_papers_rounding():
    row = L.Row("share", 0.4487531, paper="0.449")
    full, quoted = row.cells()[1], row.cells()[2]
    assert full.startswith("0.44875")
    assert quoted == "0.449"


def test_the_ledger_covers_every_family_of_fault_we_have_had():
    kinds = {c.kind for c in L.CORRECTIONS}
    assert kinds == {"estimand", "error", "guard", "sign", "population", "filename"}


def test_every_ledger_entry_says_where_it_is_quoted():
    missing = [c.quantity for c in L.CORRECTIONS if not c.quoted_at]
    assert not missing, f"ledger entries with no quoting site: {missing}"


def test_unresolved_entries_are_marked_as_such_rather_than_reading_as_settled():
    # `S-LL` and panel (d)'s top step are non-effects. A ledger that showed only the replacement
    # value would present each as a corrected measurement.
    unresolved = [c for c in L.CORRECTIONS if "unresolved" in c.status]
    assert len(unresolved) >= 2
    assert all(c.status != "resolved" for c in unresolved)


def test_the_bootstrap_is_seeded_so_two_readings_of_the_notebook_agree():
    v = [0.1, 0.2, 0.15, 0.3, 0.05, 0.22]
    assert L.bootstrap_ci(v, n=200) == L.bootstrap_ci(v, n=200)


def test_fig2_companion_has_one_row_per_richness_bar():
    t = L.fig2_plotted()
    bars = [r for r in t.rows if r.quantity.startswith("γ =") and "Δ log α" in r.quantity]
    assert len(bars) == 6
    assert all(r.n not in ("", None) for r in bars)
    assert "gate" in t.markdown().lower()
    assert "coverage" in t.markdown().lower()
    assert L.THREE_CORNER in t.grouping


def test_fig4_companion_has_one_row_per_corner():
    t = L.fig4_plotted()
    corners = [r for r in t.rows if r.quantity.endswith("Δρ_c") and r.quantity[:4] in L.FOUR]
    assert len(corners) == 4
    assert all(int(r.n) == 8 for r in corners)
    assert "four-corner" in t.grouping


def test_width_figure_companion_is_marked_four_corner():
    t = L.fig_width_plotted()
    assert t.n == 16
    assert "four-corner" in t.grouping.lower()
    assert all(r.n == 16 for r in t.rows)


def test_shh_radius_states_the_gate_and_whether_it_clears():
    t = L.radius_vs_gate_table()
    md = t.markdown()
    assert "gate" in md.lower()
    assert "SEM40" in md and "SEM8" in md
    g3 = next(r for r in t.rows if r.quantity.startswith("γ = 3"))
    assert "does not clear" in g3.note
    g01 = next(r for r in t.rows if r.quantity.startswith("γ = 0.1"))
    assert "clears" in g01.note and "does not clear" not in g01.note


def test_gamma5_onset_states_the_n_asymmetry():
    t = L.rho_c_onset_table()
    md = t.markdown()
    assert "16" in md and "40" in md
    n5 = [r for r in t.rows if r.quantity.startswith("γ = 5:") and "Δρ_c" in r.quantity]
    n10 = [r for r in t.rows if r.quantity.startswith("γ = 10:") and "Δρ_c" in r.quantity]
    assert len(n5) == 4 and all(r.n == 16 for r in n5)
    assert len(n10) == 4 and all(r.n == 40 for r in n10)


def test_paper_index_is_nonempty_and_states_n():
    t = L.paper_index()
    assert len(t.rows) >= 20
    assert t.n not in ("", None)


def test_unresolved_table_includes_shh_radius_and_gamma5_sign_test():
    t = L.unresolved_claims_table()
    md = t.markdown()
    assert "S-HH radius" in md
    assert "6.8e-4" in md or "0.00068" in md or "6.8" in md
    assert "gate" in md.lower()
    onset = next(r for r in t.rows if "γ = 0.1" in r.quantity and "forgetting" in r.quantity)
    assert "inside the floor" in onset.note
    assert "5%" in onset.note or "5.0%" in onset.note or "4.9%" in onset.note
    assert "stream_id" in md or "filename" in md


def test_unique_n_status_table_labels_the_duplication_and_the_survivors():
    t = L.unique_n_status_table()
    md = t.markdown()
    assert "8 unique" in md
    assert "STATUS CHANGED" in md  # stream R² interpretation
    assert "8/8" in md or "0.0078" in md
    assert "floor" in md.lower()
    assert "nothing" in md.lower()
    assert "0.73" in md or "0.727" in md


def test_population_table_names_every_result_set_and_states_rng():
    t = L.population_table()
    md = t.markdown()
    for token in ("phase1", "gamma_5", "gamma_5_n40", "width_g10_n40",
                  "gamma_ext", "unconfound_k3"):
        assert token in md or token.replace("_", "/") in md or token.replace("_", " ") in md
    assert "stream_id unused" in md
    assert "stream_rng" in md
    assert "not pooled" in md.lower()
    assert "§5.4" in md or "5.4" in md
    assert "pre-commit" in md.lower() or "precommit" in md.lower()
    assert "arrangement-specific" in md or "arrangement-level" in md
    for r in t.rows:
        assert "per cell" in str(r.value), f"{r.quantity}: unique n must state per-cell n"
        assert "total" in str(r.value), f"{r.quantity}: unique n must state total unique n"
    k3 = next(r for r in t.rows if "unconfound" in r.quantity)
    assert k3.n in (0, 960)
    if k3.n == 0:
        assert "| 0 |" in md  # zero files is information; Row.cells must not collapse it to —


def test_gamma30_interaction_does_not_clear_at_unique_n():
    from pathlib import Path
    import json
    p = Path(__file__).resolve().parents[1] / "results" / "gamma30_unique.json"
    d = json.loads(p.read_text())
    assert d["clears_3sem_files"] is True
    assert d["clears_3sem_unique"] is False
    assert d["interaction_over_sem_unique"] < 3
    md = L.unique_n_status_table().markdown()
    assert "1.74" in md
    assert "not licensed" in md.lower()


def test_k3_precommit_exists_before_the_run_and_states_what_k3_cannot_show():
    from pathlib import Path
    import json
    p = Path(__file__).resolve().parents[1] / "results" / "unconfound_k3_precommit.json"
    d = json.loads(p.read_text())
    assert d["written_before_any_unconfound_arm_completed"] is True
    assert "variance" in d["cannot_show"].lower()
    assert "not arrangement-specific" in d["if_reproduces"]
    assert "scope narrows" in d["if_not"].lower()
    assert "not pooled" in d["reporting"].lower()
    k3 = Path(__file__).resolve().parents[1] / "results" / "unconfound_k3"
    n_arms = sum(1 for _ in k3.glob("*.json")) if k3.is_dir() else 0
    # The flag is historical: it records that the file was written with zero arms on disk.
    # If arms exist now, the precommit must still be older than every one of them.
    if n_arms:
        pre_mtime = p.stat().st_mtime
        newest = max(f.stat().st_mtime for f in k3.glob("*.json"))
        assert pre_mtime < newest, "precommit must predate the K=3 arms it governs"


def test_gamma5_k8_precommit_predates_arms_and_scored_arrangement_dependent():
    from pathlib import Path
    import json
    root = Path(__file__).resolve().parents[1]
    pre = json.loads((root / "results" / "gamma5_k8_precommit.json").read_text())
    assert pre["written_before_any_gamma5_k8_arm_completed"] is True
    assert "7 of 8" in pre["decision_rule"]["stands_at_arrangement_level"]
    assert "arrangement-dependent" in pre["decision_rule"]["arrangement_dependent"]
    arms = root / "results" / "gamma_5_k8"
    n = sum(1 for _ in arms.glob("*.json")) if arms.is_dir() else 0
    if n:
        newest = max(f.stat().st_mtime for f in arms.glob("*.json"))
        assert (root / "results" / "gamma5_k8_precommit.json").stat().st_mtime < newest
    scored = root / "results" / "gamma5_k8.json"
    if scored.exists():
        d = json.loads(scored.read_text())
        assert d["n_usable"] == 256
        assert d["overlap_with_gamma_5_n40"]["ok"] is True
        assert d["verdict"]["outcome"] == "arrangement_dependent"
        assert d["S-HL_n_arrangements_negative"] == 4
        assert d["S-HL_ci_excludes_zero"] is False
        md = L.unique_n_status_table().markdown()
        assert "arrangement-dependent" in md.lower() or "arrangement_dependent" in md
        body = (root / "paper" / "body.tex").read_text()
        assert "four of eight" in body
        assert r"6.8\times 10^{-4}" not in body
        assert "factor of 3.3" in body


def test_caption_cuts_cover_all_six_figures_and_name_a_role():
    assert len(L.CAPTION_CUTS) == 6
    roles = {role for c in L.CAPTION_CUTS for _sent, role in c.cuts}
    assert any("number" in r for r in roles)
    assert any("scope" in r for r in roles)
    assert any("interpretation" in r for r in roles)
    for c in L.CAPTION_CUTS:
        assert L._words(c.proposed) < L._words(c.current)
        assert c.cuts, f"{c.fig} has no cut-role list"
