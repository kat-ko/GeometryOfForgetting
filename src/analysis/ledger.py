"""Support for `notebooks/results.ipynb`: provenance, tables that state their own n, the ledger.

The notebook is a diagnostic artifact, not a figure source. Two of this project's errors were
errors of *sourcing* — a right computation reading a wrong cell — so the notebook's job is to put
every number beside its provenance and its arm count, and the job of this module is to make that
the path of least resistance rather than a discipline the notebook has to remember.

Three rules are enforced here rather than asked for.

**Nothing is trained or measured.** Every function reads `results/` or `figures/` and derives.
Attribution is re-derived from stored geometry as everywhere else in the project, which is
derivation, not measurement; the distinction that matters is that no function here can produce a
number that did not already exist on disk.

**Every table carries `n` and its selection rule.** `Table` will not render without them. The
4-arm width row reached drafted text twice because its arm count was recoverable but not visible,
and a table whose n is optional is a table whose n will be omitted.

**Full precision is the notebook's job.** Each row carries the value at full precision and, where
the paper quotes it, the rounded form it appears as. The inventory compares at paper precision; the
notebook is where the unrounded value lives, so that the next person to quote it does not have to
choose a rounding.
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src import provenance
from src.analysis import grid as G
from src.analysis import timewarp
from src.analysis.attribution import (FACTORS, FLOOR_GATE_K, R_EFF_FLOOR_CV, GeometryPoint,
                                      attribute, attribution_table, min_dlog_R_for)

_SOURCE = provenance.register(__file__)

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
FIG = ROOT / "figures"

FOUR = ("S-HH", "S-HL", "S-LH", "S-LL")
THREE = ("S-HL", "S-LH", "S-LL")
BENIGN = "S-HH"
MIN_FLOORS = 2.0
BOOTSTRAP_SEED = 20260813
BOOTSTRAP_N = 2000
# W4: CV from n=4 measurement seeds. Used to mark the γ=0.1 onset as marginal
# (margin inside the floor's own uncertainty). Not a change to any gate.
FLOOR_N_SEEDS = 4
FLOOR_REL_SE = 1.0 / np.sqrt(2 * (FLOOR_N_SEEDS - 1))
# Diagnosed 2026-08-17: make_stream never consumed stream_id, so 5×8 files are
# 8 unique (seed-tied arrangement + init) draws, each written five times.
STREAM_ID_COPIES = 5
UNIQUE_SEEDS_PER_CELL = 8

# Grouping labels. After the regroup, a table that still reports four-corner values has to say so
# on the page; silent four-corner pooling is how 50.7 floors survived as a forgetting magnitude.
THREE_CORNER = "three-corner (the paper's forgetting pool: S-HL, S-LH, S-LL)"
FOUR_BY_DESIGN = ("four-corner by design — a 2×2 / per-corner split, not the forgetting pool")
FOUR_SUPERSEDED = ("four-corner (superseded by the regroup; shown so the movement is visible, "
                   "not so it can be quoted as forgetting)")
FOUR_FIGURE_WIDTH = ("four-corner (what this figure plots: n = 16 = 4 conditions × 2 streams × "
                     "2 seeds). The paper quotes genuine n=40 (0.62%) from width_table(); this "
                     "figure is the duplicate-stream matched subset.")
FOUR_POOLED = ("four-corner pooled — predates the regroup, or is the all-arm pool the paper "
               "quotes for a non-forgetting quantity; not the forgetting magnitude")


# --- rendering ----------------------------------------------------------------

def show(markdown: str) -> None:
    """Render in a notebook, print outside one, so every function works in both."""
    try:
        from IPython.display import Markdown, display
    except ImportError:
        print(markdown)
        return
    display(Markdown(markdown))


@dataclass
class Row:
    quantity: str
    value: float | str | None
    paper: str = ""
    n: int | str = ""
    note: str = ""

    def cells(self) -> list[str]:
        if self.value is None:
            v = "*not re-derivable from stored data*"
        elif isinstance(self.value, str):
            v = self.value
        elif abs(self.value) < 1e-6 and self.value != 0:
            v = f"{self.value:.3e}"
        else:
            v = f"{self.value:.6g}"
        n = "—" if self.n in ("", None) else str(self.n)
        return [self.quantity, v, self.paper or "—", n, self.note or ""]


@dataclass
class Table:
    """A table that cannot be rendered without saying what it is over.

    `n` and `selection` are required arguments rather than optional metadata. Their absence is
    how a 4-arm cell passed for a 40-arm one in two drafts.
    """

    title: str
    selection: str
    source: str
    rows: list[Row] = field(default_factory=list)
    n: int | str = ""
    grouping: str = ""

    def __post_init__(self) -> None:
        if not self.selection:
            raise ValueError(f"{self.title!r}: state the arm-selection rule, visibly")
        if self.n in ("", None):
            raise ValueError(f"{self.title!r}: state n")

    def markdown(self) -> str:
        banner = f"> **Grouping: {self.grouping}**\n\n" if self.grouping else ""
        head = (f"**{self.title}**  \n"
                f"*n = {self.n}; {self.selection}*  \n"
                f"*source: `{self.source}`*\n\n"
                "| quantity | full precision | as quoted | n | note |\n|---|---|---|---|---|\n")
        return banner + head + "\n".join("| " + " | ".join(r.cells()) + " |"
                                        for r in self.rows) + "\n"

    def display(self) -> None:
        show(self.markdown())


def header(section: str, sources: list[str], *, note: str = "") -> None:
    """Per-section provenance: what code, what commit, what result set."""
    lines = [f"### {section}", "",
             f"- **generating code**: `src/analysis/ledger.py` @ module hash "
             f"`{provenance.code_stamp()['modules'].get('ledger', 'unregistered')[:16]}`",
             f"- **git SHA**: `{git_sha()}`",
             f"- **result set**: `{result_set_sha()}` "
             f"({len(_grid()):,} arms, a = 0 only)",
             "- **reads**: " + ", ".join(f"`{s}`" for s in sources)]
    if note:
        lines += ["", note]
    show("\n".join(lines) + "\n")


def git_sha() -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip() or "n/a"
    except Exception:  # noqa: BLE001
        return "not a git repository"


# --- cached loads -------------------------------------------------------------

_CACHE: dict[str, object] = {}


def _grid() -> list[dict]:
    if "grid" not in _CACHE:
        _CACHE["grid"] = G.load()
    return _CACHE["grid"]  # type: ignore[return-value]


def _arms(name: str) -> list[dict]:
    if name not in _CACHE:
        _CACHE[name] = G.load(arms_dir=RES / name)
    return _CACHE[name]  # type: ignore[return-value]


def figjson(fig_id: str) -> dict:
    man = json.loads((FIG / "MANIFEST.json").read_text())
    name = next(n for n in man["figures"][fig_id]["outputs"] if n.endswith(".json"))
    return json.loads((FIG / name).read_text())


def _unique_n8() -> dict:
    """SEMs, CIs and sign tests reread at unique n=8. Missing file → empty dict."""
    if "unique_n8" not in _CACHE:
        p = RES / "unique_n8.json"
        _CACHE["unique_n8"] = json.loads(p.read_text()) if p.exists() else {}
    return _CACHE["unique_n8"]  # type: ignore[return-value]


def result_set_sha() -> str:
    return figjson("fig2")["result_set_sha"]


def observations(recs: list[dict], module: str = "A") -> list[dict]:
    """Retained-capacity comparisons with their design fields — the notebook's atom."""
    out = []
    for r in recs:
        pts = {(g["module"], g["task"], g["boundary"]): g
               for g in r["geometry"] if g["task"] is not None}
        s = r["spec"]
        for (mod, task, b), g in sorted(pts.items()):
            if mod != module or b <= task:
                continue
            origin = pts.get((mod, task, task))
            if origin is None:
                continue
            out.append({
                "gamma": s["gamma_0"], "condition": s["condition"], "N": s.get("N", 300),
                "seed": s["seed"], "stream_id": s.get("stream_id", 0), "task": task,
                "lag": b - task, "alpha_own": float(origin["alpha"]),
                "att": attribute(GeometryPoint.from_result(origin), GeometryPoint.from_result(g),
                                 strict=False, gamma=s["gamma_0"])})
    return out


def _obs(name: str = "grid") -> list[dict]:
    key = f"obs::{name}"
    if key not in _CACHE:
        _CACHE[key] = observations(_grid() if name == "grid" else _arms(name))
    return _CACHE[key]  # type: ignore[return-value]


def pick(rows, conditions=None, **kw) -> list[dict]:
    out = [r for r in rows if all(r[k] == v for k, v in kw.items())]
    return [r for r in out if conditions is None or r["condition"] in conditions]


def pool(rows: list[dict]) -> dict | None:
    if not rows:
        return None
    t = attribution_table([r["att"] for r in rows])
    return {**t, "floors": float(np.sign(t["dlog_alpha"]) * G.floors(t["dlog_alpha"], "alpha"))}


def bootstrap_ci(values, statistic=np.mean, *, n: int = BOOTSTRAP_N) -> tuple[float, float]:
    """Percentile CI over the sampling unit given, with the seed fixed and stated."""
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = [statistic(v[rng.integers(0, len(v), len(v))]) for _ in range(n)]
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


# --- 1. grid summary ----------------------------------------------------------

def grid_summary() -> Table:
    au = json.loads((RES / "audit_scope.json").read_text())
    rows_ = au.get("rows", au.get("checks", []))

    def says(fragment: str) -> str:
        for r in rows_:
            text = " ".join(str(x) for x in (r.values() if isinstance(r, dict) else r))
            if fragment in text:
                return text
        return "not found in audit"

    recs = _grid()
    resid = max(abs(g["identity_residual"]) for r in recs for g in r["geometry"])
    stamps = sorted({r["code"].get("git_sha", "?") for r in recs})
    n_fail = sum(1 for r in rows_ if isinstance(r, dict) and r.get("status") == "fail")
    t = Table("Grid completeness and provenance",
              "all a = 0 arms of the registered grid (γ ∈ {0.03…10}, N = 300)",
              "results/phase1/*.json, results/audit_scope.json", n=len(recs))
    t.rows = [
        Row("arms on disk (all a)", 1280, "1280", 1280),
        Row("arms in the headline set (a = 0)", len(recs), "960", len(recs),
            "a > 0 arms have bitwise-identical modules; see 06-scope §4"),
        Row("retained-capacity comparisons available", len(_obs()), "", len(_obs()),
            "module A only, task-own-boundary baseline"),
        Row("worst identity residual", resid, "4.4 × 10⁻¹⁶", "48,640 evaluations"),
        Row("git SHAs across the result set", ", ".join(stamps), "", len(stamps),
            "two commits: the solver change mid-grid, both re-derived identically"),
        Row("scope-audit checks failing", n_fail, "0", len(rows_),
            f"of {len(rows_)} checks; see `results/audit_scope.json` for each"),
        Row("arms non-converged", 0 if "0 non-converged" in says("completeness") else "see audit",
            "0", 1280, "every task in every arm reached the matched-loss target"),
    ]
    return t


# --- 2. noise floors ----------------------------------------------------------

def floors_table() -> Table:
    mn = json.loads((RES / "measurement_null.json").read_text())
    pg = mn["per_gamma"]

    def cv(gam: str, ch: str) -> float:
        a = pg[gam]["cells"]["A|retained"][ch]["cv"]
        b = pg[gam]["cells"]["B|retained"][ch]["cv"]
        return float(np.sqrt((a ** 2 + b ** 2) / 2))

    n_seeds = mn["n_seeds"]
    t = Table("Measurement floors: registered, and re-measured per richness",
              "one trained representation per γ, re-measured under several measurement seeds; "
              "modules A and B pooled in quadrature",
              "results/measurement_null.json, src/analysis/timewarp.py", n=f"{n_seeds} seeds per γ")
    t.rows = [Row(f"registered {k} floor (CV)", v, "", "", "used for cross-γ comparisons")
              for k, v in sorted(timewarp.NOISE_FLOOR_CV.items())]
    for gam in ("0.03", "0.1", "0.3", "1", "3", "10"):
        if gam in pg:
            t.rows.append(Row(f"measured R_eff floor at γ = {gam} (CV)", cv(gam, "R_eff"),
                              "0.28%–1.13% across γ", n_seeds,
                              "rises monotonically with richness — not a constant"))
    t.rows += [
        Row("α floor, measured range across γ (CV)",
            f"{min(cv(g, 'alpha') for g in pg):.4f}–{max(cv(g, 'alpha') for g in pg):.4f}",
            "1.49%–2.03%", n_seeds, "flat in γ; registered 1.87% stands"),
        Row("panel (d) gate", FLOOR_GATE_K, "3×", "",
            "3 floors, not 1: a denominator at 1σ carries ~100% relative error"),
        Row("resolution limit of this design",
            f"×/÷ {np.exp(1.96 / np.sqrt(2 * (n_seeds - 1))):.1f}", "~2×", n_seeds,
            "a CV estimated from 4 seeds has relative standard error "
            f"1/√(2(n−1)) = {1 / np.sqrt(2 * (n_seeds - 1)):.2f}, so the floors themselves are "
            "only good to about a factor of two — which is why a 2.3× error in one of them was "
            "detectable and a 20% one would not have been"),
    ]
    for r in t.rows:
        if r.quantity == "registered R_eff floor (CV)":
            r.note = "**superseded** by the per-γ values below; kept because cross-γ comparisons " \
                     "of other channels still use the registered registry"
    return t


def floor_denominated() -> str:
    return (
        "Quantities denominated in a measured floor: the capacity magnitudes of §5.1 and §5.3 "
        "(α floor), the geometric excursion progression (per channel), and panel (d)'s gate "
        "(`R_eff` floor, per γ, at 3σ). Quantities **not** floor-denominated, and therefore read "
        "against their own bootstrap or sign test instead: Δρ_c and its 2×2 decomposition, the "
        "probe correlations, and every R² in §B."
    )


# --- 3. Figure 2, both groupings ---------------------------------------------

def figure2_tables() -> list[Table]:
    out = []
    for name, conds in (("three forgetting corners (the paper's grouping)", THREE),
                        ("all four corners (superseded: mixes a gain with three losses)", FOUR)):
        sel = pick(_obs(), conditions=conds, lag=12)
        gs = sorted({r["gamma"] for r in sel})
        t = Table(f"Figure 2 — magnitude and decomposition, {name}",
                  f"lag 12, which exists only for task 0; module A; conditions {list(conds)}",
                  "results/phase1/*.json (attribution re-derived)",
                  n=f"{len(pick(sel, gamma=10.0))} comparisons per γ",
                  grouping=THREE_CORNER if conds == THREE else FOUR_SUPERSEDED)
        for g in gs:
            p = pool(pick(sel, gamma=g))
            res = "resolvable" if abs(p["floors"]) >= MIN_FLOORS else "**below floor**"
            t.rows.append(Row(f"γ = {g:g}: Δ log α", p["dlog_alpha"], "", p["n"],
                              f"{p['floors']:+.1f} floors — {res}"))
            for f in FACTORS:
                t.rows.append(Row(f"γ = {g:g}: {f} term (signed)", p["terms"][f], "",
                                  p["n"], f"share {p['shares'][f]:.3f}"))
        out.append(t)
    return out


def panel_d_table() -> Table:
    sel = pick(_obs(), conditions=THREE, lag=12)
    t = Table("Panel (d) — center-collapse share, with each bar's own gate",
              "lag 12, task 0, three forgetting corners; a cell is reported only if "
              "|Δ log R_eff| clears 3× the R_eff floor measured at that γ",
              "results/phase1/*.json, results/measurement_null.json",
              n=f"{len(pick(sel, gamma=10.0))} comparisons per γ",
              grouping=THREE_CORNER)
    for g in sorted({r["gamma"] for r in sel}):
        rows = pick(sel, gamma=g)
        frac = [r["att"].center["attributable_fraction"] for r in rows
                if r["att"].center.get("attributable_fraction") is not None]
        gate = FLOOR_GATE_K * float(np.log1p(R_EFF_FLOOR_CV[g]))
        t.rows.append(Row(
            f"γ = {g:g}: median share", float(np.median(frac)) if frac else None, "",
            f"{len(frac)}/{len(rows)}",
            f"coverage {100 * len(frac) / len(rows):.0f}%, gate {gate:.4f} log units "
            f"— coverage is **not** comparable across rows"))
    return t


# --- 4. the benign corner (Tier 1 part; Tier 2 fills the rest) ----------------

def benign_table() -> Table:
    sel = pick(_obs(), conditions=(BENIGN,), lag=12)
    los = pick(_obs(), conditions=THREE, lag=12)
    t = Table("S-HH backward transfer — magnitude and channel signature",
              "lag 12 (task 0), module A, the benign corner alone; the loss corners are shown "
              "beside it at matched γ",
              "results/phase1/*.json (attribution re-derived)",
              n=f"{len(pick(sel, gamma=10.0))} comparisons per γ",
              grouping="S-HH alone (not a pool)")
    for g in sorted({r["gamma"] for r in sel}):
        p, q = pool(pick(sel, gamma=g)), pool(pick(los, gamma=g))
        signs = " ".join(f"{f[:3]} {p['terms'][f]:+.4f}" for f in FACTORS)
        opposes = np.sign(p["terms"]["radius"]) != np.sign(p["terms"]["utility"])
        t.rows.append(Row(f"γ = {g:g}: Δ log α", p["dlog_alpha"], "", p["n"],
                          f"{p['floors']:+.1f} floors against {q['floors']:+.1f} for the loss "
                          f"corners; {signs}"
                          + ("; **radius opposes** the other two" if opposes else
                             "; all three terms agree in sign")))
    return t


def _tier2() -> dict:
    return json.loads((RES / "tier2_backward_transfer.json").read_text())


def channel_table() -> Table:
    """2.1 — signed, because this is the one place in the paper where the terms disagree."""
    t2 = _tier2()["2.1_per_corner_terms"]
    t = Table("2.1 — the gain and the losses in the same three-term currency",
              "lag 12 (task 0), module A, each corner separately; only richnesses where the "
              "corner clears ±2 floors are shown, since a term of an unresolvable total is not "
              "interpretable. Signed-term agreement is not a gate — S-HH radius vs the analysis's "
              "own 3σ R_eff gate is radius_vs_gate_table(), and at γ ≥ 1 it does not clear.",
              "results/tier2_backward_transfer.json (scripts/tier2_backward_transfer.py)", n=40,
              grouping=FOUR_BY_DESIGN)
    for g, d in t2.items():
        if not any(v["resolved"] for v in d["corners"].values()):
            continue
        for c, v in d["corners"].items():
            if not v["resolved"]:
                continue
            net = v["dlog_alpha"]
            sh = " ".join(f"{f[:3]} {v['terms'][f] / net:+.3f}" for f in FACTORS)
            t.rows.append(Row(f"γ = {g}: {c}", v["dlog_alpha"], "", 40,
                              f"{v['floors']:+.1f} floors; signed shares of the net — {sh}"))
            t.rows.append(Row(f"γ = {g}: radius term negative in every resolvable corner",
                          str(d["radius_negative_everywhere"]), "", "",
                          "signed-term agreement is **not** a shared-channel claim. At γ ≥ 1 "
                          "S-HH's radius term is negative but below the 3σ R_eff gate — see "
                          "radius_vs_gate_table(). The paper claims a bound, not a sign."))
    return t


def generality_table() -> Table:
    """2.2 — two matched series, never pooled, against a framing fixed before the run."""
    t2 = _tier2()["2.2_generality"]
    t = Table("2.2 — is the gain general, or a property of task 0 at lag 12?",
              "the benign corner only; position read at matched lag 4 (tasks 0, 4, 8) and lag "
              "read within task 0 (lags 4, 8, 12, 15). Lag and position are confounded by "
              "construction, so the two series are never pooled",
              "results/tier2_backward_transfer.json (scripts/tier2_backward_transfer.py)", n=40,
              grouping="S-HH alone (not a pool)")
    for g, d in t2.items():
        t.rows.append(Row(f"γ = {g}: verdict", d["verdict"], "general", d["n_cells"] * 40,
                          f"{d['n_positions_resolved']}/{d['n_positions']} positions, "
                          f"{d['n_lags_resolved']}/{d['n_lags']} lags, "
                          f"{d['n_cells_resolved']}/{d['n_cells']} cells resolve as gains"))
        for k, series in (("task", "position_at_matched_lag_4"), ("lag", "lag_within_task_0")):
            for key, v in d[series].items():
                t.rows.append(Row(
                    f"γ = {g}: {k} {key}"
                    + (" (at lag 4)" if k == "task" else " (within task 0)"),
                    v["dlog_alpha"], "", v["n"],
                    f"{v['floors']:+.1f} floors, CI [{v['ci_floors'][0]:+.1f}, "
                    f"{v['ci_floors'][1]:+.1f}]"))
    t.rows.append(Row("pre-committed framing", "outcome (1), 'general'", "", "",
                      "The alternative outcomes and the sentence each would license were fixed "
                      "in `results/LOG.md` before this ran. Had the gain been confined to task 0 "
                      "the claim would have been about the first task's position, not about "
                      "backward transfer, and §5.1's categorical sentence would have narrowed."))
    return t


def peak_table() -> Table:
    """2.3 — the CI, and the three things that have to be true for it to mean a location."""
    p = _tier2()["2.3_peak_location"]
    ck, sens = p["checks"], p["model_sensitivity"]
    pk = _unique_n8().get("peak_lag12", {})
    t = Table("2.3 — where the gain is largest, and why we do not say",
              "the benign corner at lag 12 unless stated; 4,000 bootstrap draws, seed 20260813; "
              "file CI is over 40 copy-inflated (seed, stream) units; unique CI over 8 seeds",
              "results/tier2_backward_transfer.json, results/unique_n8.json", n=8,
              grouping="S-HH alone (not a pool)")
    t.rows = [
        Row("highest of the six sampled richnesses", p["on_grid_argmax"], "γ = 1", 8,
            f"{p['modal_share']:.0%} of bootstrap draws — but this only ranks six points; the "
            f"grid steps by ×{p['grid_step_factor']:.2f}, so it brackets the peak no more "
            f"tightly than {p['neighbour_bracket']}"),
        Row("interpolated peak (parabola in log γ)", p["interpolated_peak"], "1.23", 8,
            f"95% CI files {[round(x, 3) for x in p['interpolated_peak_ci']]}; "
            f"unique CI {([round(x, 3) for x in pk['ci_unique']] if pk.get('ci_unique') else '—')}; "
            "the point is a mean and is unchanged"),
        Row("paired bootstrap over unique seeds",
            (f"CI [{pk['ci_unique'][0]:.3f}, {pk['ci_unique'][1]:.3f}]"
             if pk.get("ci_unique") else "not yet in unique_n8.json"),
            "[1.12, 1.35]", pk.get("n_units_unique", 8),
            f"{pk.get('ci_width_in_steps_unique', '—')} grid steps unique "
            f"(files: {ck['paired_bootstrap']['ci_width_in_steps']:.2f} over "
            f"{ck['paired_bootstrap']['n_units']} copy-inflated units). "
            "The location claim is already declined: the peak moves with lag"),
        Row("spread across defensible functional forms", sens["spread_in_grid_steps"], "0.28", 5,
            f"vertices from {[round(v, 2) for v in sens['vertex_by_model'].values()]}; "
            "**larger than the sampling CI**, so the interval above is reporting the precision "
            "of an assumption more than of a measurement"),
    ]
    for lag, v in ck["per_lag"].items():
        t.rows.append(Row(f"peak within task 0 at lag {lag}", v["interpolated_peak"], "", 40,
                          f"on-grid argmax γ = {v['on_grid_argmax']:g}"))
    t.rows += [
        Row("peak moves monotonically with lag", str(ck["peak_moves_monotonically_with_lag"]),
            "yes", 160,
            f"spanning {ck['peak_span_in_grid_steps']:.2f} grid steps over "
            f"[{ck['peak_range_across_lags'][0]:.2f}, {ck['peak_range_across_lags'][1]:.2f}] — "
            "**this is what decides it**: a location that slides with the lag it is read at is "
            "not a location, however tight the interval at any one lag"),
        Row("location identified at lag 12", str(p["location_identified_at_lag_12"]), "", 40, ""),
        Row("location identified for the corner", str(p["location_identified"]), "no", 160,
            "so §5.1 reports the shape and makes no location claim"),
        Row("text rule", "unconditional, fixed before the run", "", "", p["text_rule"]),
    ]
    return t


def sign_audit_table() -> Table:
    ap = json.loads((RES / "audit_propagation.json").read_text())
    s = ap["sign_audit"]
    t = Table("Is the gain unique to that corner?",
              "every (γ, condition, N, task, lag) cell of the registered grid, the width arms "
              "and the γ = 30 probe; a cell counts as a gain if its mean Δ log α is positive",
              "results/audit_propagation.json (scripts/audit_propagation.py)", n=s["n_cells"],
              grouping=FOUR_BY_DESIGN)
    outside = s["resolved_gains_outside_benign"]
    t.rows = [
        Row("cells examined", s["n_cells"], "520", s["n_cells"]),
        Row("cells with a positive mean", len(s["gains"]), "", s["n_cells"]),
        Row("of those, clearing ±2 floors", len(s["resolved_gains"]), "98", s["n_cells"]),
        Row("resolvable gains outside S-HH", len(outside), "0", s["n_cells"],
            "so backward transfer is a property of that corner, not a tendency present "
            "elsewhere in weaker form"),
        Row("largest positive cell outside S-HH",
            max((r["floors"] for r in s["gains"] if r["condition"] != BENIGN), default=0.0),
            "0.74 floors", "", "a third of the resolution gate"),
    ]
    return t


# --- 5. the capacity pair -----------------------------------------------------

def capacity_table() -> Table:

    def curves(recs):
        gen, ret = {}, {}
        for r in recs:
            g = r["spec"]["gamma_0"]
            for rec in r["geometry"]:
                (gen if rec["ensemble"] == "generic" else ret).setdefault(g, []).append(
                    rec["alpha"])
        return gen, ret

    gen, ret = curves(_grid())
    egen, eret = curves(_arms("gamma_ext"))
    t = Table("Generic and retained capacity across richness",
              "all a = 0 arms, all boundaries, both modules; γ = 30 rows are the fenced "
              "appendix probe and are not part of the registered sweep",
              "results/phase1/*.json, results/gamma_ext/*.json", n="see per-row n",
              grouping=FOUR_POOLED)
    for g in sorted(gen) + sorted(egen):
        src = (gen, ret) if g in gen else (egen, eret)
        a, b = float(np.mean(src[0][g])), float(np.mean(src[1][g]))
        t.rows += [
            Row(f"γ = {g:g}: generic α", a, "", len(src[0][g]),
                "label-agnostic" + ("" if g in gen else " — γ = 30 probe")),
            Row(f"γ = {g:g}: retained α", b, "", len(src[1][g]), ""),
            Row(f"γ = {g:g}: retained − generic", b - a, "", "",
                f"ratio {b / a:.3f}; retained exceeds generic at every γ"),
        ]
    return t


def capacity_per_corner() -> Table:
    t = Table("Generic capacity per corner — the pooled rise is not uniform",
              "a = 0 arms at γ = 0.03 and γ = 10, generic ensemble, both modules",
              "results/phase1/*.json", n="see per-row n",
              grouping=FOUR_BY_DESIGN)
    for c in FOUR:
        vals = {}
        for g in (0.03, 10.0):
            v = [rec["alpha"] for r in _grid()
                 if r["spec"]["condition"] == c and r["spec"]["gamma_0"] == g
                 for rec in r["geometry"] if rec["ensemble"] == "generic"]
            vals[g] = (float(np.mean(v)), len(v))
        t.rows.append(Row(f"{c}: generic α, γ = 0.03 → 10",
                          f"{vals[0.03][0]:.4f} → {vals[10.0][0]:.4f}", "",
                          vals[10.0][1],
                          f"ratio {vals[10.0][0] / vals[0.03][0]:.3f}"
                          + ("  — **the only corner that falls**"
                             if vals[10.0][0] < vals[0.03][0] else "")))
    return t


# --- 6. the 2x2 ---------------------------------------------------------------

def corners_table() -> Table:
    f4, f4g3 = figjson("fig4"), figjson("fig4_gamma3")
    u10 = _unique_n8().get("fig4", {}).get("10", {}).get("cells", {})
    t = Table("Center correlation across the 2×2, and the additive decomposition",
              "8 unique seeds per corner at N = 300, γ = 10 (40 files are five copies); "
              "first and last stream boundary. SEMs and sign tests at unique n.",
              "figures/fig4_corners_rho_c__gamma10*.json, results/unique_n8.json", n=8,
              grouping=FOUR_BY_DESIGN)
    for c in FOUR:
        d = f4["rich"][c]
        cu = u10.get(c, {})
        n_u = cu.get("n_unique", 8)
        sem_u = cu.get("sem_unique", d["delta_sem"])
        p_u = cu.get("p_unique", d["sign_test_p"])
        n_dec = cu.get("n_declining_unique",
                       int(round(d["fraction_declining"] * d["n_arms"])))
        t.rows.append(Row(f"{c}: Δρ_c over the stream", d["delta_mean"], "", n_u,
                          f"±{sem_u:.4f} SEM unique (file SEM {d['delta_sem']:.4f}); "
                          f"{n_dec}/{n_u} unique declining; sign-test p = {p_u:.4g} "
                          f"(file p = {d['sign_test_p']:.4g} on {d['n_arms']} copies)"))
    eff = f4["effects_by_gamma"]["10.0"]
    band = f4["baseline"]["band"]
    t.rows += [
        Row("readout-similarity main effect", eff["readout"], "+0.104", 80, "drives convergence"),
        Row("feature-similarity main effect", eff["feature"], "−0.061", 80, "drives decorrelation"),
        Row("interaction", eff["interaction"], "+0.006", "8 unique × 4",
            "at the resolution limit: "
            f"{_unique_n8().get('fig4', {}).get('10', {}).get('interaction_over_sem_unique', 0):.2f} "
            "unique SEM (file SEM ratio was 1.94)"),
        Row("additive prediction error on the held-out corner",
            abs(eff["additive_prediction_SHH"] - eff["measured_SHH"]), "0.012", 160, ""),
        Row("lazy-arm drift band", f"[{band[0]:.4f}, {band[1]:.4f}]", "[+0.005, +0.010]", "",
            "γ = 0.03, where the representation barely moves"),
        Row("S-HL Δρ_c at γ = 3 (the limit on the claim)",
            f4g3["rich"]["S-HL"]["delta_mean"], "+0.003", f4g3["rich"]["S-HL"]["n_arms"],
            "absent at γ = 3, present at γ = 10 — the effect rests on one richness level"),
    ]
    return t


def corners_gamma30() -> Table:
    u = json.loads((RES / "gamma30_unique.json").read_text())
    t = Table("γ = 30 extension — what saturates and what does not",
              "4 unique seeds per corner at γ = 30, N = 300 (16 files are 4 unused stream_id "
              "copies); fenced appendix probe, not promotable, not pooled with the grid",
              "results/gamma30_unique.json (scripts/audit_gamma30_unique.py)",
              n="4 unique seeds per corner",
              grouping=FOUR_BY_DESIGN)
    for c in FOUR:
        cell = u["cells"][c]
        t.rows.append(Row(
            f"{c}: Δρ_c at γ = 30", cell["delta_mean"], "", cell["n_unique"],
            f"unique SEM {cell['sem_unique']:.4f}; "
            f"{cell['n_declining_unique']}/{cell['n_unique']} declining, "
            f"p = {cell['p_unique']:.3g} (files were {cell['n_declining_files']}/"
            f"{cell['n_files']}, p = {cell['p_files']:.3g})"))
    t.rows.append(Row(
        "interaction at γ = 30", u["effects"]["interaction"], "+0.016", 4,
        f"{u['interaction_over_sem_unique']:.2f} unique SEM "
        f"(files: {u['interaction_over_sem_files']:.2f} SEM on 16/corner). "
        "Does not clear 3 SEM at unique n=4. Paired-seed "
        f"{u['paired_seed_over_sem']:.2f} SEM, bootstrap CI includes 0. "
        "The additivity-failure claim is not licensed at unique n."))
    t.rows.append(Row(
        "readout main effect at γ = 30", u["effects"]["readout"], "", 4,
        "point estimates: both main effects flatten above γ = 10 while the interaction "
        "grows; that growth is not resolved at unique n=4"))
    return t


# --- 7. width -----------------------------------------------------------------

def width_table() -> Table:
    wn = json.loads((RES / "width_g10_n40.json").read_text())
    wid = _obs("width")
    matched = [r for r in _obs() if r["stream_id"] < 2 and r["seed"] < 2]
    t = Table("Width invariance at matched lag — genuine n=40 is what §5.3 quotes",
              "lag 12 (task 0), module A, three forgetting corners. N = 150 vs 600 at genuine "
              "n=40 (stream_rng). N = 300 is the registered grid at unique n=8 and is not in "
              "the contrast. Duplicate-stream n=12 rows below are superseded.",
              "results/width_g10_n40.json; results/width/*.json, results/phase1/*.json (superseded)",
              n="120 comparisons per genuine-n cell",
              grouping=THREE_CORNER)
    for N in (150, 600):
        p = wn["by_N"][str(N)]["three"]
        sh = p["shares"]
        t.rows.append(Row(
            f"N = {N} genuine n=40: Δ log α at γ = 10", p["dlog_alpha"], "", p["n"],
            f"{p['floors']:+.1f} floors; utility {sh['utility']:.3f}, "
            f"radius {sh['radius']:.3f}, dimension {sh['dimension']:.3f}"))
    t.rows.append(Row(
        "spread N=150 vs 600 at γ = 10 (genuine n=40)", wn["three_corner_spread_pct"],
        "0.62%", 120,
        "what §5.3 quotes. Alignment share Δ="
        f"{wn['utility_share_delta']:.3f}; radius {wn['radius_share_150']:.3f} → "
        f"{wn['radius_share_600']:.3f}, dimension {wn['dimension_share_150']:.3f} → "
        f"{wn['dimension_share_600']:.3f}, sum {wn['radius_plus_dimension_150']:.3f} → "
        f"{wn['radius_plus_dimension_600']:.3f}. Two widths, not a located dependence."))
    for label, src, N in (("150 duplicate-stream", wid, 150),
                          ("300 (matched, 12)", matched, 300),
                          ("300 (full grid, 120)", _obs(), 300),
                          ("600 duplicate-stream", wid, 600)):
        p = pool(pick(src, conditions=THREE, gamma=10.0, N=N, lag=12))
        if p is None:
            continue
        t.rows.append(Row(f"N = {label}: Δ log α at γ = 10 (superseded)", p["dlog_alpha"], "",
                          p["n"],
                          f"{p['floors']:+.1f} floors; duplicate-stream. Paper quotes genuine n=40."))
    a = pool(pick(matched, conditions=THREE, gamma=10.0, N=300, lag=12))
    b = pool(pick(_obs(), conditions=THREE, gamma=10.0, N=300, lag=12))
    fl = [abs(pool(pick(src, conditions=THREE, gamma=10.0, N=N, lag=12))["floors"])
          for src, N in ((wid, 150), (matched, 300), (wid, 600))]
    spread = 100 * (max(fl) - min(fl)) / float(np.mean(fl))
    sampling = 100 * abs(a["floors"] / b["floors"] - 1)
    t.rows += [
        Row("spread across widths at γ = 10 (duplicate-stream; superseded)", spread, "1.5%",
            "12 per cell",
            "superseded. Paper quotes 0.62% at genuine n=40."),
        Row("matched-subset vs full grid at N = 300 (duplicate-stream; superseded)",
            sampling, "4.5%", "12 vs 120",
            f"**{sampling / spread:.1f}× the old spread** — why 1.5% was a bound. "
            "Not what §5.3 quotes."),
    ]
    return t


# --- 8. lag and task position -------------------------------------------------

def lag_task_table() -> Table:
    sel = pick(_obs(), conditions=THREE)
    t = Table("Lag and task position, measured separately",
              "module A, three forgetting corners, γ = 10; the lag series is within task 0, "
              "the position series at lag 4 held fixed",
              "results/phase1/*.json", n="360 comparisons per cell",
              grouping=THREE_CORNER)
    lag = {}
    for lg in sorted({r["lag"] for r in sel if r["task"] == 0}):
        p = pool(pick(sel, gamma=10.0, task=0, lag=lg))
        lag[lg] = p
        t.rows.append(Row(f"task 0, lag {lg}", p["dlog_alpha"], "", p["n"],
                          f"{p['floors']:+.1f} floors"))
    ls = sorted(lag)
    t.rows.append(Row("lag effect within task 0 (lag 4 → 15)",
                      abs(lag[ls[-1]]["floors"] / lag[ls[0]]["floors"]), "1.29×", "", "ratio"))
    pos = {}
    for tk in sorted({r["task"] for r in sel if r["lag"] == 4}):
        p = pool(pick(sel, gamma=10.0, task=tk, lag=4))
        pos[tk] = p
        t.rows.append(Row(f"lag 4, task {tk}", p["dlog_alpha"], "", p["n"],
                          f"{p['floors']:+.1f} floors"))
    ts = sorted(pos)
    t.rows.append(Row("position effect at lag 4 (task 8 → 0)",
                      abs(pos[ts[0]]["floors"] / pos[ts[-1]]["floors"]), "1.56×", "",
                      "**larger than the lag effect** — pooling by lag alone overstates lag"))
    return t


def variance_table() -> Table:
    ap = json.loads((RES / "audit_propagation.json").read_text())
    t = Table("What explains per-arm forgetting at γ = 10",
              "three forgetting corners, module A, all (task, lag) comparisons; OLS R², "
              "categorical predictors one-hot with the first level dropped",
              "results/audit_propagation.json (scripts/audit_propagation.py)",
              n=ap["variance"]["3-corner"]["variance"]["n"],
              grouping=THREE_CORNER)
    v3 = ap["variance"]["3-corner"]["variance"]
    v4 = ap["variance"]["4-corner"]["variance"]
    for k, paper in (("stream alone", "0.000"), ("seed alone", "0.007"),
                     ("distance + lag", "0.589"), ("condition alone", "0.523"),
                     ("condition + distance + lag", "0.882"), ("+ stream + seed", "0.892")):
        extra = f"four-corner value was {v4[k]:.3f}"
        if k == "stream alone":
            extra += ("; tautological on the registered grid: stream_id was unused, so the "
                      "dummy is a label on identical copies. Unique-n R² is also 0.000 "
                      "(n=240 observations, still one dummy per copy-group)")
        t.rows.append(Row(k, v3[k], paper, v3["n"], extra))
    k3p = RES / "unconfound_k3.json"
    if k3p.exists():
        k3 = json.loads(k3p.read_text())
        t.rows.append(Row(
            "stream alone (crossed arrangement×init, γ = 10)", k3["stream_R2_gamma10"],
            "0.001", k3["variance_three_corner"]["10"]["n"],
            "streams actually vary; this is the measurement. Grid 0.000 was tautological. "
            f"Seed R² {k3['seed_R2_gamma10']:.4f}."))
    t.rows += [
        Row("residual sd after all five predictors", v3["residual_sd"], "0.143", v3["n"],
            f"against a total of {v3['total_sd']:.3f}; "
            f"{v3['residual_sd'] / np.log1p(timewarp.NOISE_FLOOR_CV['alpha']):.1f} α floors"),
        Row("implied common asymptote α_∞", v3["alpha_asymptote"], "0.526", v3["n"],
            f"four-corner value was {v4['alpha_asymptote']:.3f}"),
    ]
    return t


# --- 9. stratification --------------------------------------------------------

def _probe_pairs() -> list[dict]:
    """(generic α, probe margin) paired at the same boundary and module."""
    if "probe" in _CACHE:
        return _CACHE["probe"]  # type: ignore[return-value]
    out = []
    for r in _grid():
        gen = {(g["module"], g["boundary"]): g["alpha"] for g in r["geometry"]
               if g["ensemble"] == "generic"}
        for mc in r["manipulation_checks"]:
            for mod, d in mc["probe_decodability"].items():
                a = gen.get((mod, mc["boundary"]))
                if a is not None and "margin" in d:
                    out.append({"gamma": r["spec"]["gamma_0"],
                                "condition": r["spec"]["condition"],
                                "boundary": mc["boundary"], "module": mod,
                                "alpha": a, "margin": d["margin"]})
    _CACHE["probe"] = out
    return out


def _spearman(x, y) -> float:
    from scipy.stats import rankdata
    x, y = rankdata(x), rankdata(y)
    if x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def stratification_table() -> Table:
    p = _probe_pairs()
    t = Table("H2d at three levels of pooling — the reversal, and where the effect lives",
              "all a = 0 arms, generic α against refit-readout margin, paired at the same "
              "boundary and module; Spearman, 2,000-sample bootstrap over pairs, seed "
              f"{BOOTSTRAP_SEED}",
              "results/phase1/*.json", n=len(p),
              grouping=FOUR_POOLED)
    a = np.array([r["alpha"] for r in p])
    m = np.array([r["margin"] for r in p])
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = [_spearman(a[s], m[s]) for s in
             (rng.integers(0, len(p), len(p)) for _ in range(400))]
    t.rows.append(Row("pooled across richness", _spearman(a, m), "negative", len(p),
                      f"95% CI [{np.percentile(draws, 2.5):+.3f}, "
                      f"{np.percentile(draws, 97.5):+.3f}] — **the reversal**: rich training "
                      "raises α and lowers the margin, a between-γ opposition"))
    for g in sorted({r["gamma"] for r in p}):
        sub = [r for r in p if r["gamma"] == g]
        x = np.array([r["alpha"] for r in sub])
        y = np.array([r["margin"] for r in sub])
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        d = [_spearman(x[s], y[s]) for s in
             (rng.integers(0, len(sub), len(sub)) for _ in range(400))]
        lo, hi = np.percentile(d, 2.5), np.percentile(d, 97.5)
        t.rows.append(Row(f"within γ = {g:g}", _spearman(x, y), "", len(sub),
                          f"95% CI [{lo:+.3f}, {hi:+.3f}] — "
                          f"{'excludes 0' if lo > 0 or hi < 0 else 'includes 0'}"))
    cells = {}
    for r in p:
        cells.setdefault((r["gamma"], r["condition"], r["boundary"]), []).append(r)
    for g in sorted({r["gamma"] for r in p}):
        rs = [_spearman([x["alpha"] for x in v], [x["margin"] for x in v])
              for k, v in cells.items() if k[0] == g and len(v) > 3]
        rs = [x for x in rs if not np.isnan(x)]
        mean = float(np.mean(rs))
        t.rows.append(Row(f"within (γ = {g:g}, condition, boundary)", mean, "",
                          f"{len(rs)} cells",
                          "survives the tightest stratification" if abs(mean) > 0.1 else
                          "near zero once boundary and condition are controlled"))
    t.rows.append(Row("what the three levels say together", "see note", "", len(p),
                      "the pooled statistic is negative, the within-γ statistic is positive at "
                      "γ ≥ 1, and only γ = 10 survives full stratification. The pooled sign is a "
                      "between-γ opposition — rich training raises α and lowers the margin — so "
                      "it answers a different question than H2d asks, and the support is "
                      "concentrated in the rich regime rather than general."))
    return t


# --- 10. corrections ledger ---------------------------------------------------

@dataclass(frozen=True)
class Correction:
    quantity: str
    was: str
    now: str
    why: str
    kind: str  # "estimand" | "error" | "guard" | "sign" | "population" | "filename"
    status: str = "resolved"
    quoted_at: str = ""


CORRECTIONS: tuple[Correction, ...] = (
    Correction("Figure 2 magnitude at γ = 10", "50.7 floors", "69.7 floors",
               "pooled a corner that gains capacity with three that lose it; the population "
               "changed, not the computation", "estimand", quoted_at="07-writeup.md §5.1, §D"),
    Correction("Figure 2 magnitude at γ = 0.1", "0.4 floors (unresolvable)",
               "2.1 floors (resolvable)", "same regroup; one further richness level clears the "
               "two-floor gate, which moved the left endpoint of every share range",
               "estimand", quoted_at="07-writeup.md §5.1"),
    Correction("utility share range", "0.19 → 0.45 (from γ = 0.3)", "0.08 → 0.45 (from γ = 0.1)",
               "the range starts where the magnitude first clears the gate, which moved",
               "estimand", quoted_at="07-writeup.md §5.1, §D"),
    Correction("dimension share", "near 0.44 throughout",
               "falls 0.53 → 0.44 across the first step, flat after",
               "same cause; the constancy was true only over the four-corner resolvable range",
               "estimand", quoted_at="07-writeup.md §5.1, §D"),
    Correction("width magnitudes at γ = 10", "47.9 / 48.2 / 47.5 floors",
               "65.7 / 66.6 / 65.6 floors",
               "three-corner levels; the four-corner values remain correct for the "
               "like-for-like width contrast and are reported alongside", "estimand",
               quoted_at="07-writeup.md §5.3"),
    Correction("width precision claim", "varies by 1.5% (duplicate-stream n=12)",
               "0.62% at genuine n=40 (N=150 vs 600); still a bound (two widths)",
               "the 1.5%/4.5% pair was the duplicate-stream matched-subset contrast; "
               "§5.3 now quotes the RNG-fixed width arm. N=300 is unique n=8 and is not "
               "in the contrast", "estimand", quoted_at="paper/body.tex §5.3; 07-writeup.md §5.3"),
    Correction("lag effect within task 0", "1.34×", "1.29×",
               "three-corner regroup", "estimand", quoted_at="07-writeup.md §5.3, §A.1"),
    Correction("task-position effect at lag 4", "1.7×", "1.56×",
               "three-corner regroup", "estimand", quoted_at="07-writeup.md §5.3, §A.1"),
    Correction("γ = 30 magnitude at lag 12", "71.1 floors", "94.5 floors",
               "three-corner regroup", "estimand", quoted_at="results/LOG.md γ = 30 probe"),
    Correction("variance explained by condition alone", "0.814", "0.523",
               "with one corner gaining, the condition variable was largely encoding the "
               "*direction* of the change rather than its size", "estimand",
               quoted_at="07-writeup.md §5.3, §B"),
    Correction("variance explained by distance + lag", "0.099", "0.589",
               "same cause, in the other direction: no distance-above-asymptote predicts a "
               "corner moving away from the asymptote", "estimand",
               quoted_at="07-writeup.md §5.3, §B"),
    Correction("cell-mean inflation pair", "0.889 → 0.099 (factor 9)",
               "0.878 → 0.589 (factor 1.5)",
               "the appendix documenting the pooling fault had committed it in sizing its own "
               "example", "estimand", quoted_at="07-writeup.md §A.1"),
    Correction("S-LH Δρ_c at γ = 10", "+0.0922", "+0.1098",
               "read from a 4-arm width cell rather than the 40-arm grid; the row is now "
               "labelled by arm count", "error", quoted_at="07-writeup.md §5.4, §E"),
    Correction("S-LL Δρ_c at γ = 10", "+0.0246, then +0.0210", "+0.0114",
               "pooled across γ, then read from the same 4-arm row; against a drift band of "
               "[+0.005, +0.010] this is 1.1× baseline", "error",
               status="**unresolved non-effect** (unique p = 0.73; nothing, not the file-level p = 0.15)",
               quoted_at="07-writeup.md §5.4, §A.1"),
    Correction("panel (d) γ = 3 → 10 step", "+0.004", "−0.010",
               "per-γ floor refit changed the gate at each richness", "guard",
               status="**both values unresolved** (CI [−0.038, +0.070])",
               quoted_at="07-writeup.md §5.1, §D"),
    Correction("R_eff noise floor", "one constant, CV 0.50%", "per γ, CV 0.28%–1.13%",
               "measured at one setting and applied everywhere; 2.3× too permissive at the "
               "richness where the guard is used most", "guard",
               quoted_at="07-writeup.md §A.2"),
    Correction("panel (d) gate", "1× the floor", "3× the floor",
               "a denominator at 1σ of its own noise carries ~100% relative error, so the "
               "ratio built on it is noise over noise even though the guard passed it", "guard",
               quoted_at="07-writeup.md §A.2"),
    Correction("ρ_c → R_eff calibration input", "rho_c_glue (unnormalized)",
               "rho_c_signed (normalized)",
               "the unnormalized convention reaches 1.49 on real representations, where the "
               "fitted form is undefined; out-of-domain inputs were clipped rather than "
               "refused", "guard", quoted_at="07-writeup.md §A.2"),
    Correction("Figure 2 panel (a)", "|Δ log α|", "signed Δ log α",
               "a magnitude under an axis that named a direction, so a gain of 6.2 floors was "
               "drawn at the height of a loss of 6.2", "sign",
               quoted_at="07-writeup.md §A.3"),
    Correction("center correlation convention", "|ρ_c|", "signed ρ_c",
               "H1d and H6 are claims about direction, which an absolute value cannot "
               "represent", "sign", quoted_at="07-writeup.md §A.3"),
    Correction("channel shares in the benign corner", "|term| / Σ|term|",
               "term / net, keeping the sign",
               "the magnitude share reported the radius term as contributing 11% of the gain "
               "where it subtracts 11%; a normalisation by a sum of magnitudes takes an absolute "
               "value without spelling it `abs`", "sign", quoted_at="07-writeup.md §A.3, §5.1"),
    Correction("matched-subset gap vs width spread", "5.1% against 1.5%, at different groupings",
               "4.5% against 1.52%, both at three corners",
               "each percentage was right about its own population, so the comparison between "
               "them had no defined referent; the conclusion survives but was accidentally true",
               "population", quoted_at="07-writeup.md §5.3, §A.4"),
    Correction("\u00a75.3 magnitude vs lag and position ratios",
               "decomposition at three corners, ratios at four",
               "all at three corners",
               "adjacent sentences describing different populations, which the inventory cannot "
               "catch because both values were correct", "population",
               quoted_at="07-writeup.md §5.3, §A.4"),
    Correction("peak location of the benign-corner gain",
               "largest at γ₀ = 1 of six sampled richnesses",
               "no location claim; the peak moves with lag",
               "at lag 12 the peak is at γ₀ = 1.23 with a [1.18, 1.28] interval, but it slides "
               "monotonically to 2.86 at lag 4 and 0.96 at lag 15, so there is one peak per lag "
               "rather than one for the corner", "estimand",
               quoted_at="07-writeup.md §5.1"),
    Correction("stability check on the peak across lags", "range test: span 0.90 of a grid step",
               "monotonicity test: peak ordered by lag",
               "the four peaks are 2.86, 1.75, 1.23, 0.96 — a monotone slide, which is the "
               "dependence the check existed to detect, and which a range test cannot see because "
               "it collapses an ordered sequence to its two extremes", "guard",
               quoted_at="07-writeup.md §A.2"),
    Correction("registered-grid n per (γ, condition) cell",
               "n = 40 independent arms",
               "n = 8 unique seeds (40 files are five copies)",
               "stream_id was written into the filename and never read by make_stream; seed "
               "keyed both initialisation and arrangement, so the two are confounded. Point "
               "estimates are invariant; every SEM, CI and sign-test p is not. A parameter in "
               "a filename is not evidence it was used",
               "filename", quoted_at="paper/body.tex Discussion; 07-writeup.md §A.6"),
    Correction("S-HL sign-test p at γ = 10",
               "40/40, p = 1.8×10⁻¹²",
               "8/8 unique, p = 0.0078",
               "copies treated as independent observations",
               "filename", quoted_at="paper/body.tex §5.4"),
    Correction("S-LL sign-test p at γ = 10",
               "15/40, p = 0.15",
               "3/8 unique, p = 0.73",
               "copies treated as independent observations; still unresolved",
               "filename", status="**unresolved non-effect** (unique p = 0.73; nothing, not marginal)",
               quoted_at="07-writeup.md §5.4, §A.6"),
    Correction("S-HL sign-test p at γ = 5",
               "12/16, p = 0.077; then 3/4 unique, p = 0.625; then 31/40, p = 6.8×10⁻⁴",
               "4/8 arrangements, interval [−0.023, +0.010] includes zero (8-arrangement pre-commit)",
               "the n=16 set was 4 unique seeds × 4 unused copies, and the RNG-fixed 5×8 run "
               "does give 40 distinct arms — but they are crossed, carrying only five "
               "arrangement draws. The 8-arrangement pre-commit (256 arms, written before those "
               "arms existed) scored 4/8 negative. Fixing the count did not fix the unit. "
               "By initialisation the 5×8 effect is 8/8 (p = 0.0078)",
               "population",
               status="**unresolved at the arrangement unit** (arm-level p = 6.8×10⁻⁴ is an "
                      "inference about new arms; §5.4 quotes 4/8 and the grid's 3 → 10 bracket)",
               quoted_at="paper/body.tex §5.4; docs/14-claim-security.md §6"),
    Correction("Δρ_c magnitudes quoted with a within-arm SEM",
               "−0.055 at γ = 10 with ±0.0078",
               "sign and ordering claimed; magnitude −0.026 to −0.051 across K=3 arrangements",
               "the between-arrangement SEM is 5.5–10× the within-arrangement SEM on all four "
               "corners, and the grid's −0.055 falls outside the three-arrangement range. The "
               "grid's own SEM8 is correctly scaled (0.55–1.77× the between-arrangement SEM) "
               "because its 8 unique arms each carry their own arrangement",
               "population", quoted_at="paper/body.tex §5.4; docs/14-claim-security.md §4"),
    Correction("γ = 30 Figure 4 interaction",
               "+0.016, 3.9 SEM (16 files/corner)",
               "+0.016, 1.74 unique SEM (4 unique/corner); does not clear 3 SEM",
               "duplicate-stream copies treated as independent. Paired-seed reading 1.53 SEM, "
               "bootstrap CI includes 0, one of four seeds opposite sign. The additivity-failure "
               "claim is not licensed at unique n",
               "filename", status="**unresolved at unique n=4** (was resolved on 16 files)",
               quoted_at="07-writeup.md §5.4; results/gamma30_unique.json"),
)


def ledger_table() -> Table:
    t = Table("Corrections ledger — read the draft against this",
              "every number in the paper that has been superseded, with the reason and the "
              "family of fault; `kind` is estimand (population changed), error (wrong cell), "
              "guard (failed open), sign (transformation destroyed direction), population (two "
              "correct values compared across different populations), filename (a field stored "
              "in a path and never read, so n was wrong while the files were correct)",
              "src/analysis/ledger.py::CORRECTIONS", n=len(CORRECTIONS),
              grouping="not a corner pool")
    t.rows = [Row(c.quantity, f"{c.was} → **{c.now}**", "", c.kind,
                  f"{c.why}. Status: {c.status}. Quoted at: {c.quoted_at or '—'}")
              for c in CORRECTIONS]
    return t


# --- figures: plotted values beside the rendering --------------------------------

def figure_png(fig_id: str) -> Path:
    man = json.loads((FIG / "MANIFEST.json").read_text())
    name = next(n for n in man["figures"][fig_id]["outputs"] if n.endswith(".png"))
    return FIG / name


def show_figure(fig_id: str) -> None:
    """Render the PNG the paper includes, so the table above it is the same numbers."""
    path = figure_png(fig_id)
    try:
        from IPython.display import Image, display
    except ImportError:
        show(f"*(figure `{path.name}` — open `{path}` outside a notebook)*")
        return
    display(Image(filename=str(path)))


def _gate_for(gamma: float, cv: float | None = None) -> float:
    if cv is not None:
        return FLOOR_GATE_K * float(np.log1p(cv))
    return min_dlog_R_for(gamma)


def _fig2_gamma_row(cell: dict, gamma: float, paper_floors: str = "") -> Row:
    """One richness level of a fig2-family plot: magnitude, shares, gate, coverage."""
    n = cell["n"]
    gate = _gate_for(gamma, cell.get("R_eff_floor_cv"))
    cov = cell.get("center_coverage")
    share = cell.get("center_share_median")
    sh = cell["shares"]
    floors = cell.get("floors_signed", cell.get("floors"))
    resolved = abs(floors) >= MIN_FLOORS
    note = (f"utility {sh['utility']:.6g}, radius {sh['radius']:.6g}, "
            f"dimension {sh['dimension']:.6g}")
    if not resolved:
        note += " — **below the ±2-floor band, shares not drawn**"
    if cov is not None:
        note += (f"; center-collapse share {share if share is not None else 'n/a'}, "
                 f"coverage {100 * cov:.0f}%, gate {gate:.4f} log units "
                 f"(3 × R_eff floor at this γ)")
    return Row(f"γ = {gamma:g}: Δ log α", cell["dlog_alpha"], paper_floors, n,
               f"{floors:+.4g} floors; {note}")


def fig2_plotted() -> Table:
    """The six γ bars of Figure 2, with the numbers the PNG was drawn from."""
    raw = figjson("fig2")
    d = raw["data"]
    paper = {"0.03": "0.2 floors", "0.1": "2.1 floors", "10.0": "69.7 floors"}
    t = Table("Figure 2 — plotted values (one row per richness bar)",
              "lag 12, task 0, three forgetting corners; n is comparisons per γ, matching "
              "the figure sidecar",
              "figures/fig2_gamma_sweep__forgetting__lag12__*.json",
              n=d["10.0"]["n"], grouping=THREE_CORNER)
    t.rows = [_fig2_gamma_row(d[k], float(k), paper.get(k, ""))
              for k in ("0.03", "0.1", "0.3", "1.0", "3.0", "10.0")]
    t.rows += [
        Row("utility share at first resolvable γ", d["0.1"]["shares"]["utility"], "0.08",
            d["0.1"]["n"], "panel (b) left endpoint; hatched as marginal — the 0.10-floor "
            f"margin is {FLOOR_REL_SE / ((d['0.1']['floors'] - MIN_FLOORS) / d['0.1']['floors']):.1f}× "
            "smaller than the floor's own relative SE"),
        Row("utility share at γ = 10", d["10.0"]["shares"]["utility"], "0.45",
            d["10.0"]["n"], "panel (b) right endpoint"),
        Row("radius share at first resolvable γ", d["0.1"]["shares"]["radius"], "0.39",
            d["0.1"]["n"], ""),
        Row("radius share at γ = 10", d["10.0"]["shares"]["radius"], "0.11",
            d["10.0"]["n"], ""),
        Row("dimension share at first resolvable γ", d["0.1"]["shares"]["dimension"], "0.53",
            d["0.1"]["n"], ""),
        Row("dimension share at γ = 10", d["10.0"]["shares"]["dimension"], "0.44",
            d["10.0"]["n"], "flat after the first resolvable step"),
        Row("center-collapse share at γ = 3", d["3.0"]["center_share_median"], "0.44",
            d["3.0"]["n"], "the through-γ=3 rise the caption quotes"),
        Row("center-collapse share at γ = 10", d["10.0"]["center_share_median"], "0.45",
            d["10.0"]["n"], "sidecar median; **not quoted in the paper as a measured step**. "
            "The paper quotes 0.44 as the γ=3 end of the rise and states the γ=3→10 step is "
            "unresolved (CI includes 0). 0.45 in the paper is the utility share, not this."),
        Row("panel (d) coverage at γ = 0.03", d["0.03"]["center_coverage"], "0%",
            d["0.03"]["n"], "**empty bar** — nothing clears the 3σ gate"),
        Row("utility sign-positive fraction, γ ≥ 1",
            max(d[k]["sign_positive_fraction"]["utility"] for k in ("1.0", "3.0", "10.0")),
            "0% / 100% negative", d["10.0"]["n"], "panel (c)"),
    ]
    return t


def fig2_benign_plotted() -> Table:
    d = figjson("fig2_benign")["data"]
    paper = {"1.0": "11.8 floors", "10.0": "6.2 floors"}
    t = Table("Figure S (benign) — plotted values (one row per richness bar)",
              "lag 12, task 0, S-HH alone",
              "figures/fig2_gamma_sweep__S-HH__lag12__*.json",
              n=d["10.0"]["n"], grouping="S-HH alone (not a pool)")
    t.rows = [_fig2_gamma_row(d[k], float(k), paper.get(k, ""))
              for k in ("0.03", "0.1", "0.3", "1.0", "3.0", "10.0")]
    return t


def fig2_lag4_plotted() -> Table:
    d12 = figjson("fig2")["data"]
    d4 = figjson("fig2_lag4")["data"]
    t = Table("Figure S (lag 4) — plotted values, against the lag-12 shares they are compared to",
              "lag 4, task 0, three forgetting corners",
              "figures/fig2_gamma_sweep__forgetting__lag4__task0__*.json",
              n=d4["10.0"]["n"], grouping=THREE_CORNER)
    t.rows = [_fig2_gamma_row(d4[k], float(k))
              for k in ("0.03", "0.1", "0.3", "1.0", "3.0", "10.0")]
    diffs = [abs(d12[k]["shares"][f] - d4[k]["shares"][f])
             for k in ("0.3", "1.0", "3.0", "10.0")
             for f in FACTORS]
    t.rows += [
        Row("largest share difference vs lag 12", max(diffs), "0.032", d4["10.0"]["n"],
            "the caption's composition-holds number"),
        Row("magnitude ratio lag 12 : lag 4 at γ = 10",
            d12["10.0"]["dlog_alpha"] / d4["10.0"]["dlog_alpha"], "1.27×",
            d4["10.0"]["n"], ""),
    ]
    return t


def _fig4_plotted(fig_id: str, title: str, paper_gamma: str) -> Table:
    raw = figjson(fig_id)
    rich = raw["rich"]
    band = raw["baseline"]["band"]
    ug = _unique_n8().get("fig4", {}).get(f"{raw['gamma']:g}", {})
    cells_u = ug.get("cells", {})
    t = Table(title,
              f"8 unique seeds per corner at N = 300, γ = {raw['gamma']:g} "
              f"(40 files are five unused stream_id copies); first vs last stream "
              "boundary, generic ensemble, signed ρ_c. SEMs and sign tests at unique n. "
              "The lazy-arm band is the Δρ_c gate.",
              f"figures/fig4_corners_rho_c__gamma{paper_gamma}*.json, results/unique_n8.json",
              n=8, grouping=FOUR_BY_DESIGN)
    paper_delta = {"S-HL": ("−0.055" if raw["gamma"] == 10 else "+0.003"),
                   "S-LH": ("+0.110" if raw["gamma"] == 10 else ""),
                   "S-LL": ("+0.011" if raw["gamma"] == 10 else "")}
    for c in FOUR:
        d = rich[c]
        cu = cells_u.get(c, {})
        n_u = cu.get("n_unique", 8)
        sem_u = cu.get("sem_unique", d["delta_sem"])
        p_u = cu.get("p_unique", d["sign_test_p"])
        n_dec = cu.get("n_declining_unique",
                       int(round(d["fraction_declining"] * d["n_arms"])))
        in_band = band[0] <= d["delta_mean"] <= band[1]
        t.rows.append(Row(
            f"{c}: Δρ_c", d["delta_mean"], paper_delta.get(c, ""), n_u,
            f"±{sem_u:.4f} SEM unique (file SEM {d['delta_sem']:.4f}); "
            f"{n_dec}/{n_u} unique declining; sign-test p = {p_u:.4g} "
            f"(file p = {d['sign_test_p']:.4g} on {d['n_arms']} copies); "
            f"Spearman median {d['spearman_median']:+.3f}; "
            f"lazy-band gate [{band[0]:.4f}, {band[1]:.4f}] "
            + ("**— inside the band**" if in_band else "— outside the band")))
    gkey = f"{raw['gamma']:.1f}"
    eff = raw["effects_by_gamma"][gkey]
    z_u = ug.get("interaction_over_sem_unique")
    t.rows += [
        Row("readout-similarity main effect", eff["readout"],
            "+0.104" if raw["gamma"] == 10 else "", 8, ""),
        Row("feature-similarity main effect", eff["feature"],
            "−0.061" if raw["gamma"] == 10 else "", 8, ""),
        Row("interaction", eff["interaction"],
            "+0.006" if raw["gamma"] == 10 else "", 8,
            (("at the resolution limit: "
              f"{z_u:.2f} unique SEM" if z_u is not None else
              "at the resolution limit within the registered range")
             if raw["gamma"] == 10 else "")),
        Row("additive prediction error on S-HH",
            abs(eff["additive_prediction_SHH"] - eff["measured_SHH"]),
            "0.012" if raw["gamma"] == 10 else "", 8, ""),
        Row("lazy-arm drift band (Δρ_c gate)", f"[{band[0]:.4f}, {band[1]:.4f}]",
            "[+0.005, +0.010]", "", "γ = 0.03, where the representation barely moves"),
    ]
    return t


def fig4_plotted() -> Table:
    return _fig4_plotted("fig4", "Figure 4 — plotted values at γ = 10 (one row per corner)", "10")


def fig4_gamma3_plotted() -> Table:
    return _fig4_plotted("fig4_gamma3",
                         "Figure S (γ = 3) — plotted values (one row per corner)", "3")


def fig_width_plotted() -> Table:
    raw = figjson("fig_width")
    t = Table("Figure S (width) — plotted values. FOUR-CORNER n = 16, not the paper's 0.62% row",
              "lag 12, task 0; duplicate-stream matched subset streams 0–1, seeds 0–1; the "
              "figure pools all four corners, so each cell is 16 comparisons. The paper quotes "
              "genuine n=40, N=150 vs 600, 0.62% from width_table(), not these cells.",
              "figures/fig_width_invariance__lag12__*.json",
              n=16, grouping=FOUR_FIGURE_WIDTH)
    paper = {"N150_g10": "47.9 floors (four-corner, not quoted as forgetting)",
             "N600_g10": "47.5 floors (four-corner, not quoted as forgetting)"}
    for key, cell in raw["cells"].items():
        t.rows.append(Row(
            key, cell["dlog_alpha"], paper.get(key, ""), cell["n"],
            f"{cell['floors']:+.4g} floors; utility {cell['shares']['utility']:.4g}, "
            f"radius {cell['shares']['radius']:.4g}, "
            f"dimension {cell['shares']['dimension']:.4g}"))
    return t


def show_figure_with_table(fig_id: str) -> None:
    """Table of plotted values, then the PNG. A figure whose numbers cannot be read off
    the table next to it is a figure that cannot be checked."""
    builders = {
        "fig2": fig2_plotted, "fig2_benign": fig2_benign_plotted,
        "fig2_lag4": fig2_lag4_plotted, "fig4": fig4_plotted,
        "fig4_gamma3": fig4_gamma3_plotted, "fig_width": fig_width_plotted,
    }
    show(f"**`{fig_id}` — plotted values, then the rendering. Same numbers, same n, same gates.**")
    builders[fig_id]().display()
    show_figure(fig_id)


# --- S-HH radius against the analysis's own gate --------------------------------

def radius_vs_gate_table() -> Table:
    """The radius term in the benign corner, against the 3σ R_eff gate used in panel (d).

    The paper claims no sign for this series at γ ≥ 1, only a bound. This table is that
    bound, not an assertion that the term is the channel all four corners share.

    SEM is reported beside the floor so both readings are visible: the 3σ gate asks
    whether one measurement would resolve the term; the SEM of the mean asks whether
    the mean is distinguishable from zero. The gate is not moved.
    """
    d = figjson("fig2_benign")["data"]
    t = Table("S-HH radius term vs the analysis's own 3σ R_eff gate",
              "lag 12, task 0, S-HH alone; gate = 3 × log(1 + R_eff CV) at that γ, the same "
              "threshold panel (d) uses as a denominator. The two-floor convention on α is "
              "a different ruler and is shown beside it. n=40 files are 8 unique seeds × 5 "
              "unused stream_id copies — SEM40 treats copies as independent; SEM8 does not.",
              "figures/fig2_gamma_sweep__S-HH__lag12__*.json, src/analysis/attribution.py",
              n=40, grouping="S-HH alone (not a pool); 8 unique seeds behind n=40 files")
    paper_floors = {"1.0": "1.9 floors", "3.0": "0.2 floors", "10.0": "1.1 floors"}
    for k in ("0.03", "0.1", "0.3", "1.0", "3.0", "10.0"):
        g = float(k)
        cell = d[k]
        term = cell["terms"]["radius"]
        gate = min_dlog_R_for(g)
        fl = float(np.sign(term) * G.floors(term, "R_eff", gamma=g))
        clears_gate = abs(term) >= gate
        clears_two = abs(fl) >= MIN_FLOORS
        status = "clears" if clears_gate else "does not clear"
        n_files = int(cell["n"])
        sem40 = float(cell["term_sem"]["radius"])
        # Exact copies: SEM of 8 unique seeds from the 40-file SEM.
        # var_40 = (STREAM_ID_COPIES * (n_unique-1) / (n_files-1)) * var_unique
        n_unique = UNIQUE_SEEDS_PER_CELL if n_files == STREAM_ID_COPIES * UNIQUE_SEEDS_PER_CELL \
            else n_files
        if n_unique < n_files:
            sem8 = sem40 * np.sqrt(n_files / n_unique) * np.sqrt(
                (n_files - 1) / (STREAM_ID_COPIES * (n_unique - 1)))
        else:
            sem8 = sem40
        t.rows.append(Row(
            f"γ = {g:g}: radius term", term, paper_floors.get(k, ""), cell["n"],
            f"gate {gate:.4f} log units (3σ single-measurement); {abs(fl):.2f} R_eff floors; "
            f"**{status}** the 3σ gate. "
            f"SEM40={sem40:.5f} (|mean|/SEM40={abs(term) / sem40:.2f}); "
            f"SEM8={sem8:.5f} (|mean|/SEM8={abs(term) / sem8:.2f}). "
            "Gate unchanged — SEM is the other reading, not a replacement."
            + ("; also below the two-floor convention" if not clears_two else
               "; two-floor convention would pass" if not clears_gate else "")
            + ("; this is the unresolved series the paper quotes as a bound"
               if g >= 1 and not clears_gate else "")))
    return t


def capacity_table_three_corner() -> Table:
    """Generic and retained capacity on the paper's forgetting pool, so §5 is not four-corner-only."""

    def curves(recs, conditions):
        gen, ret = {}, {}
        for r in recs:
            if r["spec"]["condition"] not in conditions:
                continue
            g = r["spec"]["gamma_0"]
            for rec in r["geometry"]:
                (gen if rec["ensemble"] == "generic" else ret).setdefault(g, []).append(
                    rec["alpha"])
        return gen, ret

    gen, ret = curves(_grid(), THREE)
    t = Table("Generic and retained capacity, three forgetting corners",
              "a = 0 arms in S-HL, S-LH, S-LL, all boundaries, both modules",
              "results/phase1/*.json", n="see per-row n", grouping=THREE_CORNER)
    for g in sorted(gen):
        a, b = float(np.mean(gen[g])), float(np.mean(ret[g]))
        t.rows += [
            Row(f"γ = {g:g}: generic α", a, "", len(gen[g]), ""),
            Row(f"γ = {g:g}: retained α", b, "", len(ret[g]), ""),
            Row(f"γ = {g:g}: retained − generic", b - a, "", "",
                f"ratio {b / a:.3f}"),
        ]
    return t


# --- γ = 5 probe in the ρ_c section ---------------------------------------------

def sampling_unit_table() -> Table:
    """Which unit each interval is an inference about — the question unique-n did not settle.

    `audit_unique_n.py` fixed the *count*; `audit_seed_security.py` asks what the sampling unit
    is. A SEM over arms is an inference about new arms, and only about new arrangements if the
    arms sample arrangements independently. Three populations differ exactly there.
    """
    ss = json.loads((RES / "seed_security.json").read_text())
    k3 = ss["k3_arrangement_component"]
    w = ss["width"]
    shl5 = ss["gamma5_onset"]["corners"]["S-HL"]
    t = Table("Sampling unit — is this an inference about arms, initialisations, or arrangements?",
              "the registered grid confounds arrangement with initialisation, which costs "
              "attribution but leaves its interval correctly scaled. The RNG-fixed 5×8 arms "
              "separate the two but sample only five arrangements. K=3 is 40 inits × 3 "
              "arrangements. Sign-test floors: n=4 → 0.125, n=5 → 0.0625 (neither can reach "
              "α=0.05), n=8 → 0.0078 unanimous but 0.0703 after one flip.",
              "results/seed_security.json (scripts/audit_seed_security.py)",
              n="see per-row n",
              grouping="not a corner pool — this is a sampling-unit audit")
    g = ss["grid_duplication"]
    t.rows.append(Row(
        "grid copies are identical, not merely similar", g["worst_spread_across_copies"], "0.000",
        f"{g['arms_on_disk_a0']} arms at a=0",
        "largest Δρ_c spread across the five stream_id copies of a seed at γ=10. The copies "
        "carry no information; the unique-n relabelling was not conservatism."))
    for c in FOUR:
        v = k3["corners"][c]
        sp = v["spread"]
        t.rows.append(Row(
            f"{c} Δρ_c at γ=10: between- vs within-arrangement SEM",
            v["between_over_within_sem"], "", 3,
            f"arrangement means "
            + ", ".join(f"{x:+.4f}" for x in v["per_arrangement_mean"].values())
            + f"; between-arrangement SEM {sp['sem']:.4f} against within-arrangement "
            f"{v['median_within_arrangement_sem']:.4f}. Sign-stable: {sp['sign_stable']}. "
            f"Grid SEM8 {v['grid_sem_unique_n8']:.4f} is "
            f"{v['grid_sem_over_between_arrangement_sem']:.2f}× the between-arrangement SEM — "
            f"the grid's interval is the right size. Grid mean "
            f"{v['grid_value_at_unique_n8']:+.4f} inside the arrangement range: "
            f"{v['grid_value_inside_arrangement_range']}."))
    for name, v in k3["effects"].items():
        sp = v["spread"]
        t.rows.append(Row(
            f"{name} effect at γ=10 across K=3 arrangements", sp["mean"], "", 3,
            ", ".join(f"{x:+.4f}" for x in v["per_arrangement"].values())
            + f"; sign-stable {sp['sign_stable']}; grid {v['grid_value_at_unique_n8']:+.4f}. "
            + ("robust in sign and roughly in size" if name == "readout" else
               "robust in sign, magnitude varies" if name == "feature" else
               "**sign flips across arrangements** — this is why the interaction is unresolved")))
    t.rows.append(Row(
        "S-HL γ=10: inits declining in every arrangement",
        k3["S-HL_inits_declining_in_all_arrangements"], "40 of 40", 120,
        "the best-supported claim in the paper: unanimous on both units, on top of 8/8 unique "
        "seeds on the grid."))
    au = shl5["arrangement_unit"]
    ci = au["spread"]["t_ci_95"]
    t.rows.append(Row(
        "γ=5 onset: arms vs arrangements", au["sign_test"]["n_declining"], "3 of 5", 5,
        f"arm unit {shl5['arm_unit']['sign_test_on_spearman']['n_declining']}/40, "
        f"p = {shl5['arm_unit']['sign_test_on_spearman']['p']:.2g}, SEM "
        f"{shl5['arm_unit']['sem_assuming_independent_arms']:.4f}. Arrangement unit "
        f"{au['sign_test']['n_declining']}/5, p = {au['sign_test']['p']:.3g}, SEM "
        f"{au['spread']['sem']:.4f}, 95% CI [{ci[0]:+.4f}, {ci[1]:+.4f}] **includes zero**. "
        f"Init unit {shl5['init_unit']['sign_test']['n_declining']}/8, "
        f"p = {shl5['init_unit']['sign_test']['p']:.4g}."))
    k8p = RES / "gamma5_k8.json"
    if k8p.exists():
        k8 = json.loads(k8p.read_text())
        au8 = k8["corners"]["S-HL"]["arrangement_unit"]
        ci8 = k8["S-HL_t_ci_95"]
        t.rows.append(Row(
            "γ=5 onset: 8-arrangement pre-commit (what §5.4 quotes)",
            k8["S-HL_n_arrangements_negative"], "4 of 8", 8,
            f"verdict {k8['verdict']['outcome']}. "
            f"{k8['S-HL_n_arrangements_negative']}/8 negative, p = {au8['sign_test']['p']:.3g} "
            f"(ceiling {au8['sign_test']['p_floor']:.4g}); 95% CI "
            f"[{ci8[0]:+.4f}, {ci8[1]:+.4f}] **includes zero**. The three new arrangements "
            "were +0.0015, +0.0166, −0.0221 — two more non-negative. Power was enough to "
            "clear α=0.05 if the effect were arrangement-stable; it is not."))
    pw = w["paired_within_arrangement"]
    t.rows.append(Row(
        "width: arrangement spread against the width gap",
        w["arrangement_spread_over_width_gap"], "17×", 5,
        f"the {w['width_gap_pct']:.2f}% width gap against "
        f"{w['worst_within_width_arrangement_spread_pct']:.2f}% across arrangements at fixed "
        f"width. Paired within arrangement (600−150): "
        + ", ".join(f"{v:+.2f}" for v in
                    pw["differences_floors_600_minus_150"].values())
        + f" floors, mean {pw['spread']['mean']:+.2f}, 95% CI "
        f"[{pw['spread']['t_ci_95'][0]:+.2f}, {pw['spread']['t_ci_95'][1]:+.2f}], "
        f"{pw['n_arrangements_with_600_larger_loss']}/5 same direction "
        f"(p = {pw['sign_test_p']:.3g}, ceiling {pw['sign_test_p_floor']:.3g}). Direction "
        "consistent, magnitude under 1%, still a bound."))
    f = ss["floors"]
    t.rows.append(Row(
        "the ruler itself", f["relative_se_of_a_cv"], "41%", f["n_measurement_seeds"],
        f"a CV from {f['n_measurement_seeds']} measurement seeds carries "
        f"{100 * f['relative_se_of_a_cv']:.0f}% relative SE, so a floor is good to "
        f"×/÷{f['factor_uncertainty_on_a_floor']:.2f} "
        f"(ratio {f['ratio_between_plus_and_minus_1se_ends']:.1f} between the ±1 SE ends). "
        "Every gate statement inherits it."))
    t.rows.append(Row(
        "measurement noise is on the init axis, not the arrangement axis", "by construction", "",
        "", "the measurement seed is derived from `seed`, so each arrangement group contains the "
        "same measurement seeds. The between-arrangement numbers above are not inflated by "
        "measurement noise."))
    return t


def rho_c_onset_table() -> Table:
    """S-HL Δρ_c at γ = 3, 5, 10, 30, with arm counts so the n = 16 vs n = 40 asymmetry is visible.

    γ = 5 is loaded from `results/gamma_5/` via `G.load(arms_dir=...)` and is never adopted by
    `G.load()` of the registered grid. Compared to γ = 3/10 at n = 40, this is **not** a
    matched 16-arm subset.
    """
    g5 = json.loads((RES / "gamma5_onset.json").read_text())
    n5_disk = len(_arms("gamma_5"))
    t = Table("S-HL Δρ_c onset, including the γ = 5 probe — four richnesses, with n",
              "generic ensemble, first vs last stream boundary; γ = 3 and 10 are the registered "
              "grid (40 arms/corner); γ = 5 is results/gamma_5/ loaded with arms_dir (64 usable "
              f"arms on disk, {g5['by_gamma']['5']['S-HL']['n_arms']} per corner); γ = 30 is "
              "results/gamma_ext/. The 16-arm and 40-arm columns are different designs.",
              "results/gamma5_onset.json, results/gamma_5/*.json (arms_dir, not G.load())",
              n=f"40 at γ=3,10; 16 at γ=5,30; {n5_disk} γ=5 arms on disk",
              grouping=FOUR_BY_DESIGN)
    paper = {"3": "+0.003", "5": "−0.0064 (n=16 files)", "10": "−0.055", "30": ""}
    paper_p = {"3": "unique p = 1.0",
               "5": "n=16 files / unique p = 0.625; superseded by the 5×8 set below",
               "10": "unique p = 0.0078", "30": ""}
    for g in ("3", "5", "10", "30"):
        block = g5["by_gamma"][g]
        # Full four-row block so every corner's n sits on the page, not just S-HL.
        for c in FOUR:
            dlt = block["delta_rho_c"][c]
            sem = block["delta_rho_c_sem"][c]
            n = block["n_arms_per_corner"][c]
            extra = ""
            if c == "S-HL":
                shl = block["S-HL"]
                extra = (f"; {int(round(shl['fraction_declining'] * shl['n_arms']))}/"
                         f"{shl['n_arms']} declining; sign-test p = {shl['sign_test_p']:.4g}"
                         f" {paper_p[g]}")
            t.rows.append(Row(
                f"γ = {g}: {c} Δρ_c", dlt, paper[g] if c == "S-HL" else "", n,
                f"±{sem:.4f} SEM; n = {n} arms{extra}"))
        add = block["effects"]
        t.rows.append(Row(
            f"γ = {g}: additive error on S-HH", abs(add["additive_prediction_SHH"]
                                                   - add["measured_SHH"]), "",
            sum(block["n_arms_per_corner"].values()),
            f"interaction {add['interaction']:+.4g}"))
    onset = g5["onset"]
    n40p = RES / "gamma5_n40_onset.json"
    if n40p.exists():
        r40 = json.loads(n40p.read_text())["by_gamma"]["5"]["S-HL"]
        t.rows.append(Row(
            "γ = 5, 5×8 set, arm unit (not the onset evidence)",
            r40["delta_mean"], "−0.0102 (draft only)", 40,
            f"±{r40['delta_sem']:.4f}; "
            f"{int(round(r40['fraction_declining'] * r40['n_arms']))}/40 declining; "
            f"sign-test p = {r40['sign_test_p']:.4g}. An inference about new *arms*; the "
            "40 are crossed 5 arrangements × 8 inits. See the arrangement row below."))
    ssp = RES / "seed_security.json"
    if ssp.exists():
        shl5 = json.loads(ssp.read_text())["gamma5_onset"]["corners"]["S-HL"]
        au = shl5["arrangement_unit"]
        ci = au["spread"]["t_ci_95"]
        counts = "  ".join(f"{k}:{v['n_declining']}/{v['n']}"
                           for k, v in au["per_arrangement_arm_signs"].items())
        t.rows.append(Row(
            "γ = 5, 5×8 arrangement unit (superseded as onset evidence)",
            au["spread"]["mean"], "3 of 5 arrangements", 5,
            f"per-arrangement means "
            + ", ".join(f"{v:+.4f}" for v in au["means"].values())
            + f" (arm signs {counts}); {au['sign_test']['n_declining']}/5 declining, "
            f"p = {au['sign_test']['p']:.3g} against a ceiling of "
            f"{au['sign_test']['p_floor']:.4g}; SEM {au['spread']['sem']:.4f} vs "
            f"{shl5['arm_unit']['sem_assuming_independent_arms']:.4f} over arms; "
            f"95% t CI [{ci[0]:+.4f}, {ci[1]:+.4f}] **includes zero**. The two positive "
            "arrangements land on the lazy-arm drift band."))
        iu = shl5["init_unit"]
        t.rows.append(Row(
            "γ = 5, initialisation unit", iu["spread"]["mean"], "", iu["n"],
            f"{iu['sign_test']['n_declining']}/{iu['n']} inits declining, "
            f"p = {iu['sign_test']['p']:.4g}. Secure against W(0); the arrangement axis is "
            "the one that does not resolve."))
    k8p = RES / "gamma5_k8.json"
    if k8p.exists():
        k8 = json.loads(k8p.read_text())
        au8 = k8["corners"]["S-HL"]["arrangement_unit"]
        ci8 = k8["S-HL_t_ci_95"]
        counts8 = "  ".join(f"{k}:{v['n_declining']}/{v['n']}"
                            for k, v in au8["per_arrangement_arm_signs"].items())
        t.rows.append(Row(
            "γ = 5, 8-arrangement pre-commit — what §5.4 quotes",
            k8["S-HL_n_arrangements_negative"], "4 of 8 arrangements", 8,
            f"verdict {k8['verdict']['outcome']}. Means "
            + ", ".join(f"{v:+.4f}" for v in k8["S-HL_arrangement_means"].values())
            + f" (arm signs {counts8}); {k8['S-HL_n_arrangements_negative']}/8 declining, "
            f"p = {au8['sign_test']['p']:.3g} against a ceiling of "
            f"{au8['sign_test']['p_floor']:.4g}; 95% t CI "
            f"[{ci8[0]:+.4f}, {ci8[1]:+.4f}] **includes zero**. "
            f"Overlap with gamma_5_n40: {k8['overlap_with_gamma_5_n40']['n_identical']}/160 "
            "identical. Arrangements 5–7 added two more non-negative means."))
    t.rows += [
        Row("last non-negative γ on the registered grid", onset["last_non_negative"], "3", 8,
            "4/8 unique declining, p = 1.0 — absent"),
        Row("first γ that resolves on the registered grid", 10.0, "10", 8,
            "8/8 unique, p = 0.0078. γ = 5 is not on the grid and does not resolve at the "
            "arrangement unit, so it does not move this endpoint"),
        Row("onset bracket factor — what §5.4 quotes",
            onset["previous_bracket_factor"], "3.3", 8,
            f"the grid's 3 → 10 bracket. The 1.67 bracket "
            f"({onset['bracket_factor']:.2f}) assumed the γ = 5 arm-level reading and is "
            "withdrawn; see the Corrections table"),
        Row("n asymmetry", "n=16 files vs 5×8 vs 8×8 vs grid unique n=8", "on the page",
            "16 vs 40 vs 64 vs 8",
            "the n=16 directory is 4 unique × 4 copies; the 5×8 set is 40 distinct arms over "
            "5 arrangements; the 8×8 set is what §5.4 quotes (64 arms/corner, 8 arrangements); "
            "γ = 3,10 on the grid are 8 unique × 5 copies, and their 8 arms "
            "each carry their own arrangement."),
    ]
    return t


# --- paper-number index and unresolved claims -----------------------------------

PAPER_FILES = ("paper/body.tex", "paper/figures.tex", "paper/figures-appendix.tex")


def _inventory_mod():
    if "inv" not in _CACHE:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "make_inventory", ROOT / "scripts" / "make_inventory.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _CACHE["inv"] = mod
    return _CACHE["inv"]


def _resolution_status(name: str, expected, got) -> str:
    """Clears its gate, is a bound, or is unresolved — the column the draft is read against."""
    n = name.lower()
    if any(s in n for s in ("s-ll", "sign-test p at γ₀=10", "sign-test p at γ₀=5",
                            "interaction at γ", "3 → 10", "3 -> 10",
                            "center-collapse share at γ₀=10",
                            "matched-subset", "1.5", "width")):
        pass
    unresolved_names = (
        "s-ll δρ", "s-ll sign-test", "interaction at γ₀=10",
        "center-collapse share at γ₀=10", "panel (d)",
        "s-hl δρ_c at γ₀=5", "s-hl sign-test p at γ₀=5",
        "matched-subset vs full-grid", "forgetting at γ₀=0.1",
    )
    if any(u in n for u in unresolved_names) or "γ₀=5" in n or ("gamma" in n and "5" in n):
        if "duplicate" in n or "n=16" in n or "4 unique" in n:
            return ("unresolved on the duplicate-stream unique n=4 set; "
                    "the 8-arrangement pre-commit is what §5.4 quotes")
        if "γ₀=5" in n or ("gamma" in n and "5" in n):
            return ("unresolved at the arrangement unit (4/8; 8-arrangement pre-commit; "
                    "arm-level p = 6.8e-4 is a different unit)")
    if "s-ll" in n:
        return "unresolved (unique sign-test p = 0.73; nothing, not marginal)"
    if "interaction at γ₀=10" in n:
        return "unresolved (at the resolution limit)"
    if "center-collapse share at γ₀=10" in n:
        return "unresolved (γ=3→10 step CI includes 0)"
    if "genuine n=40" in n and ("spread" in n or "width" in n or "alignment-share" in n
                                or "radius" in n or "dimension" in n):
        return "quoted (genuine n=40; still a bound: two widths, N=300 not in the contrast)"
    if "matched-subset" in n or "duplicate-stream" in n:
        return "superseded (duplicate-stream n=12; paper quotes genuine n=40 0.62%)"
    if "forgetting at γ₀=0.1" in n:
        return ("marginal (0.10-floor / 5% margin sits inside the floor's ~41% "
                "relative SE; body keeps 'between 0.1 and 0.3')")
    if "s-hh capacity" in n or (isinstance(expected, (int, float)) and abs(expected) >= 2
                                and "floors" in str(name).lower()):
        return "resolved"
    if got is None:
        return "not re-derivable"
    return "see unresolved_claims_table() if the paper flags it; else quoted"


def paper_index() -> Table:
    """Every inventoried value that appears in the paper files, in order of first appearance.

    This is the list the draft is read against. n and resolution sit on the same row as the
    number, so a claim whose support is a bound or a failed gate cannot be misread as settled.
    """
    inv = _inventory_mod()
    sources = []
    for rel in PAPER_FILES:
        raw = (ROOT / rel).read_text()
        sources.append((rel, raw, inv.normalise(raw)))

    located = []
    for row in inv.rows():
        got = None
        if row["getter"] is not None:
            try:
                got = row["getter"]()
            except Exception as e:  # noqa: BLE001
                got = f"getter failed: {e}"
        pos = None
        where = None
        if row.get("quote"):
            needle = inv.normalise(row["quote"]).strip()
            for rel, raw, norm in sources:
                idx = norm.find(needle)
                if idx >= 0:
                    pos = (PAPER_FILES.index(rel), idx)
                    ln = inv.line_of(raw, needle)
                    where = f"{rel}:{ln}" if ln else rel
                    break
        if pos is None:
            # Distinctive rendering of the expected value, first hit in paper order.
            token = None
            exp = row["expected"]
            if isinstance(exp, float):
                token = f"{exp:g}" if abs(exp) >= 1e-4 else f"{exp:.2e}"
            elif isinstance(exp, int):
                token = str(exp)
            if token and token not in {"0", "1", "3", "4", "5", "6", "8", "10", "16"}:
                for rel, raw, norm in sources:
                    idx = norm.find(token)
                    if idx >= 0:
                        pos = (PAPER_FILES.index(rel), idx)
                        where = rel
                        break
        if pos is None:
            continue
        located.append((pos, row, got, where))

    located.sort(key=lambda x: x[0])
    t = Table("Paper-number index — every inventoried value in the paper, in appearance order",
              "values whose quote or distinctive rendering occurs in paper/body.tex, "
              "paper/figures.tex, or paper/figures-appendix.tex; n as inventoried",
              "scripts/make_inventory.py::rows() against the three paper files",
              n=len(located), grouping="mixed; per-row n is the claim's own")
    for _pos, row, got, where in located:
        full = got if isinstance(got, (int, float, str)) or got is None else str(got)
        t.rows.append(Row(
            f"{row['section']} {row['name']}", full, str(row["expected"]),
            row.get("n") or "",
            f"source `{row['source']}`; appears {where}; "
            f"{_resolution_status(row['name'], row['expected'], got)}"))
    return t


def unique_n_status_table() -> Table:
    """Every SEM, sign-test p and CI reread at unique n=8, with whether status flips.

    This is the §12 companion to the duplication: the registered grid is labelled n=8 unique;
    new arms after the RNG fix are genuine n=40.
    """
    u = _unique_n8()
    t = Table("Unique-n status — SEMs, CIs and sign tests at n=8 unique, not n=40 files",
              "registered grid: 40 files per (γ, condition) cell are 8 unique seeds × 5 unused "
              "stream_id copies. Point estimates are invariant. Inference is not. New arms "
              "(γ=5 n40, width γ=10 n40, K=3, γ=5 k8) are genuine after the RNG fix. "
              "The k8 set is 8 arrangements × 8 inits — the unit, not the file count, is the claim.",
              "results/unique_n8.json (scripts/audit_unique_n.py)",
              n="8 unique seeds per cell (40 files)",
              grouping="not a corner pool — this is an n-audit")
    t.rows.append(Row(
        "duplication (filename field never read)",
        "n=8 unique, arrangement confounded with init",
        "n=40 files",
        8,
        "make_stream never consumed stream_id; the field was stored in the filename. "
        "A parameter written into a filename is not evidence it was used. "
        "The registered grid is labelled n=8 unique; new arms after the RNG fix are "
        "genuine n=40."))
    for c in u.get("status_changes", []):
        n_row = 4 if "γ=5" in c["quantity"] else 8
        extra = ""
        if "S-LL" in c["quantity"] and "γ=10" in c["quantity"] and "sign" in c["quantity"]:
            extra = (". Nothing, not marginal: p=0.73 is a coin-flip at n=8; "
                     "p=0.15 on 40 files was copy-inflation")
        if "S-HL" in c["quantity"] and "γ=10" in c["quantity"] and "sign" in c["quantity"]:
            extra = ". p=0.0078 is the floor of a two-sided sign test at n=8 (8/8)"
        t.rows.append(Row(
            c["quantity"], c["unique"], c.get("files", ""), n_row,
            f"files: {c['files']}. unique: {c['unique']}. "
            f"{'STATUS CHANGED' if c.get('changed') else 'status unchanged at α=0.05'}"
            + (f". {c['note']}" if c.get("note") else "")
            + extra))
    g30p = RES / "gamma30_unique.json"
    if g30p.exists():
        g30 = json.loads(g30p.read_text())
        t.rows.append(Row(
            "γ=30 Figure 4 interaction / SEM",
            f"{g30['interaction_over_sem_unique']:.2f} SEM (4 unique/corner)",
            f"{g30['interaction_over_sem_files']:.2f} SEM (16 files/corner)",
            4,
            "STATUS CHANGED. 3.9 SEM on 16 files was copy-inflation; at unique n=4 "
            f"the interaction is {g30['interaction_over_sem_unique']:.2f} SEM and does "
            "not clear. Paired-seed "
            f"{g30['paired_seed_over_sem']:.2f} SEM, bootstrap CI includes 0. "
            "The additivity-failure claim is not licensed at unique n."))
    k3p = RES / "unconfound_k3.json"
    if k3p.exists():
        k3 = json.loads(k3p.read_text())
        t.rows.append(Row(
            "K=3 unconfound vs precommit",
            f"{k3['n_arrangements_reproducing']}/3 leading; "
            f"{k3['n_arrangements_S-HL_strictly_only']}/3 S-HL-only",
            "not arrangement-specific",
            k3["n_usable"],
            "STATUS: leading pattern 3/3 (channel reorg; S-LH largest; S-HL decorrelates; "
            "S-HH a gain). Strict 'S-HL the only decorrelating corner' is 2/3: arrangement 2 "
            "has S-LL also decorrelating — the unresolved S-LL corner, not a scope narrowing. "
            f"Stream R² at γ=10 (streams vary) = {k3['stream_R2_gamma10']:.3f}. "
            "K=5 not run: detection, not estimation; fishing after seeing arr-2 S-LL."))
    k8p = RES / "gamma5_k8.json"
    if k8p.exists():
        k8 = json.loads(k8p.read_text())
        t.rows.append(Row(
            "γ=5 8-arrangement pre-commit",
            f"{k8['S-HL_n_arrangements_negative']}/8 negative; "
            f"verdict {k8['verdict']['outcome']}",
            "4 of 8; keep 3→10 bracket",
            k8["n_usable"],
            "STATUS: arrangement-dependent. Pre-commit required ≥7/8 negative *and* a t "
            "interval excluding zero to locate the onset at 5. Interval "
            f"[{k8['S-HL_t_ci_95'][0]:+.3f}, {k8['S-HL_t_ci_95'][1]:+.3f}] includes zero. "
            f"Overlap with gamma_5_n40: {k8['overlap_with_gamma_5_n40']['n_identical']}/160 "
            "identical. Arm-level p = 6.8e-4 is not the onset evidence."))
    return t


def _arm_count(folder: str) -> int:
    d = RES / folder
    if not d.is_dir():
        return 0
    return sum(1 for _ in d.glob("*.json"))


def population_table() -> Table:
    """Every result set the paper can quote: file n, unique n, RNG, and which claims draw on it.

    A sentence that compares a genuine-n number to a unique-n=8 number without saying so is
    the §A.4 mismatched-population fault. This table is what the draft is checked against.

    The unique-n column is always `N per cell; T total`. N is the number that enters a sign
    test (or a cell SEM) on one inference cell; T is unique files in the set. Mixing N from
    one row with T from another is the same fault one level up.
    """
    k3 = _arm_count("unconfound_k3")
    w40 = _arm_count("width_g10_n40")
    g5n40 = _arm_count("gamma_5_n40")
    w_old = _arm_count("width")
    g5old = _arm_count("gamma_5")
    g5k8 = _arm_count("gamma_5_k8")
    g30 = _arm_count("gamma_ext")
    t = Table(
        "Result-set populations — check the draft against this before comparing numbers",
        "one row per directory. Unique n is always 'N per cell; T total': N is the sign-test "
        "n on one cell, T is unique files in the set. RNG says whether stream_id keyed the "
        "arrangement. Do not compare N from one row to T from another, and do not compare "
        "across rows without naming both populations.",
        "results/*/ listings; results/unique_n8.json; results/gamma30_unique.json; "
        "results/unconfound_k3_precommit.json",
        n="per-row files on disk",
        grouping="not a corner pool — this is a population audit")
    specs = [
        ("registered grid (phase1)",
         "8 per cell; 192 total",
         "stream_id unused; seed keys init+arrangement+dichotomy",
         960,
         "cell = (γ, condition). Fig 2, Fig 4, §5.1–5.3, §B; γ=3 and γ=10 onset endpoints. "
         "Sign-test n = 8."),
        ("γ=5 duplicate-stream (gamma_5)",
         "4 per cell; 16 total",
         "stream_id unused (same as grid)",
         g5old,
         "cell = corner at γ=5. Kept as the before; not what §5.4 quotes. Sign-test n = 4."),
        ("γ=5 crossed 5×8 (gamma_5_n40)",
         "40 per cell = 5 arrangements × 8 inits; 160 total",
         "stream_rng(stream_id); paired_init(seed)",
         g5n40,
         "cell = corner at γ=5. 40 distinct arms, but **five arrangement draws**: the arm-level "
         "sign test (31/40, p=6.8e-4) is an inference about new arms, and by arrangement it is "
         "3/5 with a CI including zero. §5.4 no longer offers this as the onset. "
         "Sign-test n = 40 arms / 5 arrangements / 8 inits — name which."),
        ("width duplicate-stream (width/)",
         "2 per cell; 48 total",
         "stream_id unused (same as grid)",
         w_old,
         "cell = (N, γ, condition). Superseded for §5.3; the figure still shows this set."),
        ("width crossed 5×8, γ=10 (width_g10_n40)",
         "40 per cell = 5 arrangements × 8 inits; 320 total",
         "stream_rng(stream_id); paired_init(seed)",
         w40,
         "cell = (N, condition) at γ=10. What §5.3 quotes: 0.62% spread, radius/dimension "
         "trade. Five arrangement draws, but the width contrast is **paired** across them, so "
         "the arrangement offsets cancel: 5/5 same direction, CI [−0.34, +1.23] floors. "
         "Appendix; not pooled with the grid."),
        ("γ=30 probe (gamma_ext)",
         "4 per cell; 16 total",
         "stream_id unused (same as grid)",
         g30,
         "cell = corner at γ=30. Appendix robustness; fenced. Interaction +0.016 is "
         "1.74 unique SEM (was 3.9 SEM on 16 files) and does not clear. Sign-test n = 4."),
        ("K=3 unconfound (unconfound_k3)",
         "120 per cell; 960 total",
         "stream_rng(stream_id)×3 arrangements; paired_init×40; not pooled",
         k3,
         "cell = (γ, condition): 40 inits × 3 arrangements. Appendix robustness. "
         "Pre-commit in results/unconfound_k3_precommit.json (written before any arm "
         "completed): leading pattern 3/3 ⇒ not arrangement-specific. "
         "'S-HL the only decorrelating corner' is 2/3 (arrangement 2: S-LL also "
         "decorrelates). Cannot estimate arrangement-level variance."),
        ("γ=5 at 8 arrangements (gamma_5_k8)",
         "64 per cell = 8 arrangements × 8 inits; 256 total",
         "stream_rng(stream_id)×8; paired_init(seed)×8",
         g5k8,
         "cell = corner at γ=5. The power fix for the onset: n=8 arrangements is the smallest "
         "count whose sign-test floor (0.0078) clears α=0.05. Pre-commit in "
         "results/gamma5_k8_precommit.json, written before any arm existed. Arrangements 0–4 "
         "reproduce gamma_5_n40 bitwise by construction, which is the built-in check."
         + ("" if g5k8 >= 256 else f"  **RUNNING — {g5k8}/256 arms on disk; not yet analysed.**")),
    ]
    t.rows = [
        Row(name, unique, rng, files, claims)
        for name, unique, rng, files, claims in specs
    ]
    return t


def unresolved_claims_table() -> Table:
    """Every quantity the paper mentions that does not clear its gate, with the gate and the margin.

    If the paper says something is unresolved, this is the number that makes it so.
    """
    d2 = figjson("fig2")["data"]
    d2b = figjson("fig2_benign")["data"]
    f4 = figjson("fig4")
    g5 = json.loads((RES / "gamma5_onset.json").read_text())
    ap = json.loads((RES / "audit_propagation.json").read_text())
    u = _unique_n8()
    t = Table("Unresolved claims — the number that makes each one so",
              "every quantity the paper mentions whose gate has not been cleared, including "
              "bounds the paper states as bounds. Sign tests and SEMs on the registered grid "
              "are at unique n=8 (40 files are copies).",
              "figure sidecars, results/gamma5_onset.json, results/audit_propagation.json, "
              "results/unique_n8.json, results/LOG.md (panel (d) CI)",
              n="see per-row n")
    t.rows.append(Row(
        "registered-grid n (stream_id stored, never read)",
        "8 unique seeds per cell", "n = 40 files",
        "8 unique (40 files)",
        "make_stream never consumed stream_id. 40 files are five copies of 8 "
        "(init, arrangement, dichotomy) triples keyed by seed. Arrangement and "
        "initialisation are confounded. Point estimates invariant; SEMs, CIs and "
        "sign tests are not. A parameter written into a filename is not evidence "
        "it was used. New arms after the RNG fix are genuine n=40."))
    # S-HH radius series
    rad = u.get("shh_radius", {})
    for k in ("0.03", "0.1", "0.3", "1.0", "3.0", "10.0"):
        g = float(k)
        term = d2b[k]["terms"]["radius"]
        gate = min_dlog_R_for(g)
        fl = float(np.sign(term) * G.floors(term, "R_eff", gamma=g))
        margin = abs(term) - gate
        if abs(term) >= gate:
            continue
        rr = rad.get(f"{g:g}", {})
        mean_note = ""
        if rr:
            mean_note = (
                f" Mean {rr['mean_over_sem']:.1f} SEM8 from zero"
                f"{' — resolvable as a mean' if rr.get('resolvable_as_mean_3sem') else ''}.")
        paper_mean = ""
        if k in ("1.0", "10.0"):
            paper_mean = (" Paper: distinguishable as a population mean "
                          "(3.9 SEM8 at γ=1, 2.5 at γ=10), not as a single measurement; "
                          "3σ gate not moved.")
        t.rows.append(Row(
            f"S-HH radius term at γ = {g:g}", term,
            {"1.0": "1.9 floors", "3.0": "0.2 floors", "10.0": "1.1 floors"}.get(k, ""),
            rr.get("n_unique", 8),
            f"gate {gate:.4f} log units ({FLOOR_GATE_K:.0f}σ R_eff); "
            f"{abs(fl):.2f} R_eff floors; margin {margin:+.4f} (negative = short of the gate). "
            f"No sign for a typical arm, only a bound under 2 floors.{paper_mean}{mean_note}"))
    # γ=3→10 center-collapse step
    step = d2["10.0"]["center_share_median"] - d2["3.0"]["center_share_median"]
    t.rows.append(Row(
        "γ = 3 → 10 center-collapse step (difference of sidecar medians)", step,
        "−0.010 (LOG.md refit); CI [−0.038, +0.070]", "24 unique (120 files)",
        "gate: CI must exclude 0. The quoted CI includes 0 (margin: 0 is 0.038 from the "
        "lower end, 0.070 from the upper). Sidecar difference of medians is shown at full "
        "precision; the paper quotes the per-arm-refit step from results/LOG.md, which is "
        "the same unresolved verdict."))
    t.rows.append(Row(
        "panel (d) coverage at γ = 0.03", d2["0.03"]["center_coverage"], "0%",
        "24 unique (120 files)",
        f"gate {min_dlog_R_for(0.03):.4f} log units; nothing clears, so the bar is empty "
        "rather than a measured share"))
    # γ=5: the onset is a claim about arrangements. The 8-arrangement pre-commit is what §5.4 quotes.
    g5n40_path = RES / "gamma5_n40_onset.json"
    k8_path = RES / "gamma5_k8.json"
    if g5n40_path.exists():
        shl40 = json.loads(g5n40_path.read_text())["by_gamma"]["5"]["S-HL"]
        extra = (f"{int(round(shl40['fraction_declining'] * shl40['n_arms']))}/40 arms on the "
                 f"5×8 set decline (arm-level p = 6.8e-4); those 40 are crossed, not 40 "
                 f"arrangement draws.")
        n_arr, paper_n = 5, "40 arms / 5 arrangements"
        if k8_path.exists():
            k8 = json.loads(k8_path.read_text())
            ci8 = k8["S-HL_t_ci_95"]
            extra += (f" 8-arrangement pre-commit: {k8['S-HL_n_arrangements_negative']}/8 "
                      f"negative, 95% CI [{ci8[0]:+.3f}, {ci8[1]:+.3f}] **includes zero**, "
                      f"verdict {k8['verdict']['outcome']}. §5.4 quotes 4 of 8 and the "
                      f"grid's 3 → 10 bracket.")
            n_arr, paper_n = 8, "64 arms / 8 arrangements"
        t.rows.append(Row(
            "γ = 5 S-HL — unresolved at the arrangement unit", shl40["sign_test_p"],
            "4 of 8 arrangements", paper_n,
            "gate: the onset is a claim about arrangements. " + extra))
    shl3u = u.get("fig4", {}).get("3", {}).get("cells", {}).get("S-HL", {})
    t.rows.append(Row(
        "γ = 3 S-HL (the other end of the onset bracket)",
        g5["by_gamma"]["3"]["S-HL"]["delta_mean"], "+0.003, p = 1.0",
        shl3u.get("n_unique", 8),
        f"unique sign test p = {shl3u.get('p_unique', 1.0):.4g}; "
        f"{shl3u.get('n_declining_unique', 4)}/{shl3u.get('n_unique', 8)} unique declining "
        "(files: 20/40). Absent, not a weak version of γ = 10."))
    # S-LL
    sll = f4["rich"]["S-LL"]
    sll_u = u.get("fig4", {}).get("10", {}).get("cells", {}).get("S-LL", {})
    band = f4["baseline"]["band"]
    t.rows.append(Row(
        "S-LL Δρ_c at γ = 10", sll["delta_mean"], "+0.0114, unique p = 0.73",
        sll_u.get("n_unique", 8),
        f"gate: sign test 0.05 and the lazy-arm band [{band[0]:.4f}, {band[1]:.4f}]. "
        f"Unique n=8: {sll_u.get('n_declining_unique', 3)}/{sll_u.get('n_unique', 8)} declining, "
        f"p = {sll_u.get('p_unique', 0.73):.4g} — nothing, not a marginal non-effect "
        f"(files were 15/40, p = {sll['sign_test_p']:.4g}, which read as marginal only because "
        "copies were treated as independent). Inside/at the lazy-arm band. Rank in any corner "
        "ordering carries no information."))
    # interaction
    eff = f4["effects_by_gamma"]["10.0"]
    z_u = u.get("fig4", {}).get("10", {}).get("interaction_over_sem_unique")
    z_f = u.get("fig4", {}).get("10", {}).get("interaction_over_sem_files")
    t.rows.append(Row(
        "Figure 4 interaction at γ = 10", eff["interaction"], "+0.006", 8,
        (f"unique SEM ratio {z_u:.2f}" + (f" (files: {z_f:.2f})" if z_f is not None else "")
         if z_u is not None else "at the resolution limit within the registered range")
        + ". Still unresolved. Not a detected non-additivity; the additive model still "
          "predicts S-HH to 0.012."))
    # width 1.5%
    w = ap["width"]
    fls = [abs(w[f"10|{n}"]["three"]["floors_signed"])
           for n in ("150", "300 (matched)", "600")]
    spread = 100 * (max(fls) - min(fls)) / float(np.mean(fls))
    sampling = 100 * abs(w["10|300 (matched)"]["three"]["floors_signed"]
                         / w["10|300 (full grid)"]["three"]["floors_signed"] - 1)
    wn_path = RES / "width_g10_n40.json"
    if wn_path.exists():
        wn = json.loads(wn_path.read_text())
        t.rows.append(Row(
            "width spread at γ = 10 (genuine n=40, N=150 vs 600)",
            wn["three_corner_spread_pct"], "0.62%", 120,
            "what §5.3 quotes. Still a bound: two widths, not a located dependence. "
            "N=300 is the registered grid at unique n=8 and is not in this contrast. "
            "Alignment/non-alignment split stable (Δ=0.020); radius/dimension boundary moves."))
    t.rows.append(Row(
        "width spread at γ = 10 (duplicate-stream three-corner; superseded)", spread, "1.5%",
        "12 per cell",
        f"superseded by genuine n=40. Sampling gap was {sampling:.2f}% "
        f"({sampling / spread:.1f}× this spread) — why 1.5% was a bound."))
    t.rows.append(Row(
        "matched-subset vs full-grid gap at N = 300, γ = 10 (superseded)", sampling, "4.5%",
        "12 vs 120",
        "the number that made the 1.5% a bound. Not what §5.3 quotes."))
    # γ=0.1 onset of forgetting
    obs = d2["0.1"]["floors"]
    margin = obs - MIN_FLOORS
    margin_frac = margin / obs
    t.rows.append(Row(
        "forgetting at γ = 0.1 (first resolvable)", obs, "2.1 floors",
        d2["0.1"]["n"],
        f"gate: ±2 α floors. Observed {obs:.4g}; margin {margin:+.2f} floors "
        f"({100 * margin_frac:.1f}% of the observed value — the binding comparison). "
        f"The floor itself is known to relative SE {FLOOR_REL_SE:.2f} "
        f"(~{100 * FLOOR_REL_SE:.0f}%, W4, n={FLOOR_N_SEEDS} measurement seeds), "
        f"{FLOOR_REL_SE / margin_frac:.1f}× the margin. The margin sits inside the "
        f"floor's own uncertainty. Body keeps 'between γ₀ = 0.1 and 0.3'. Panel (b) "
        f"hatches γ=0.1 as marginal. A 20% floor error would also un-resolve it; "
        f"that counterfactual is weaker than the floor's own ~41% uncertainty, so 5% "
        f"is the binding statement."))
    return t


# --- caption cuts: propose, do not apply from here ------------------------------

def _words(text: str) -> int:
    return len(text.split())


@dataclass(frozen=True)
class CaptionCut:
    fig: str
    current: str
    proposed: str
    cuts: tuple[tuple[str, str], ...]  # (sentence, role) role in {number, scope, interpretation}


CAPTION_CUTS: tuple[CaptionCut, ...] = (
    CaptionCut(
        "fig:decomposition (Figure 2)",
        "Forgetting decomposes exactly, and the channel that carries it shifts with richness. "
        "Retained capacity α(·; y_j) for task 0, measured at its own task boundary and again "
        "twelve boundaries later, decomposed by the exact identity Δ log α = Δ log Ψ_eff + "
        "Δ log(1+R_eff^{-2}) − Δ log D_eff of Chou 2026. Pooled over the three stream conditions "
        "that lose capacity (S-HL, S-LH, S-LL; 120 arms per richness level); the benign corner "
        "S-HH, whose retained capacity rises, is shown separately in Figure fig:benign. "
        "(a) Magnitude in units of the estimator's measured Monte-Carlo noise floor, signed: "
        "0.2 floors at γ0=0.03, rising to 69.7 at γ0=10. The grey band marks ±2 floors, below "
        "which nothing is resolvable. "
        "(b) Channel composition, |term|/Σ|term|, drawn only where the total clears the band — "
        "γ0=0.1 and above; the utility channel grows from 0.08 to 0.45 of total motion while "
        "the radius channel falls from 0.39 to 0.11, and dimension falls from 0.53 to 0.44 "
        "across the first step and is flat thereafter. "
        "(c) The same terms with their signs, hollow markers where a term's sign is not resolved "
        "across arms; the utility term is negative on 100% of arms for γ0 ≥ 1. "
        "(d) How much of the radius channel is center collapse: the change in ρ_c converted "
        "through a synthetic-manifold calibration, as a fraction of the observed radius change. "
        "Rising from 0.05 to 0.44 through γ0=3 and then flat — the γ0=3→10 step is not resolved "
        "(bootstrap CI [−0.038,+0.070]). Each bar is gated at three times the R_eff noise floor "
        "measured at its own richness, printed on the bar with its coverage; because the "
        "threshold differs per bar the coverage percentages are not comparable across bars. "
        "Lag 12 exists only for task 0 in a 16-task stream with boundaries at 0, 4, 8, 12, 15, "
        "so this is a task-0 figure by construction, and task 0 is the most-forgotten task — it "
        "reports the largest forgetting in the grid rather than its average.",
        "Forgetting decomposes exactly, and the channel that carries it shifts with richness. "
        "Retained capacity for task 0 at lag 12, decomposed by the exact identity of Chou 2026. "
        "Pooled over the three conditions that lose capacity (24 unique seeds per γ; "
        "120 files are copies); S-HH is Figure "
        "fig:benign. "
        "(a) Magnitude in noise floors, signed: 0.2 at γ0=0.03 to 69.7 at γ0=10. Grey band: ±2 "
        "floors. "
        "(b) Channel shares, |term|/Σ|term|, drawn where the total clears the band. "
        "γ0=0.1 hatched (margin inside the floor's uncertainty); γ0=0.3 and above solid: "
        "utility 0.08→0.45, radius 0.39→0.11, dimension 0.53→0.44 then flat. "
        "(c) Signed terms; utility negative on 100% of arms for γ0≥1. "
        "(d) Center-collapse share of the radius channel, 0.05→0.44 through γ0=3 then flat "
        "(γ0=3→10 step unresolved, CI [−0.038,+0.070]). Each bar gated at 3× its own R_eff "
        "floor; coverage is not comparable across bars. "
        "Lag 12 exists only for task 0, the most-forgotten task — the largest forgetting in the "
        "grid, not its average.",
        (
            ("Retained capacity α(·; y_j) for task 0, measured at its own task boundary and again "
             "twelve boundaries later, decomposed by the exact identity … of Chou 2026.",
             "scope — names the estimand, the lag, and the identity"),
            ("Pooled over the three stream conditions that lose capacity (… 120 arms per richness "
             "level); the benign corner S-HH … is shown separately in Figure fig:benign.",
             "scope — grouping and n; pointer to the other figure"),
            ("(a) Magnitude in units of the estimator's measured Monte-Carlo noise floor, signed: "
             "0.2 floors at γ0=0.03, rising to 69.7 at γ0=10.",
             "number — kept, compressed"),
            ("The grey band marks ±2 floors, below which nothing is resolvable.",
             "scope — the resolution gate drawn on the panel"),
            ("(b) Channel composition, |term|/Σ|term|, drawn only where the total clears the band "
             "— γ0=0.1 and above; the utility channel grows from 0.08 to 0.45 … dimension … flat "
             "thereafter.",
             "number — kept, compressed; the |term|/Σ|term| definition is the cut"),
            ("(c) The same terms with their signs, hollow markers where a term's sign is not "
             "resolved across arms; the utility term is negative on 100% of arms for γ0 ≥ 1.",
             "number + scope — hollow-marker convention cut; 100% kept"),
            ("(d) How much of the radius channel is center collapse: the change in ρ_c converted "
             "through a synthetic-manifold calibration, as a fraction of the observed radius "
             "change.",
             "scope — names the calibration; cut if the panel axis already says it"),
            ("Rising from 0.05 to 0.44 through γ0=3 and then flat — the γ0=3→10 step is not "
             "resolved (bootstrap CI [−0.038,+0.070]).",
             "number — kept"),
            ("Each bar is gated at three times the R_eff noise floor measured at its own "
             "richness, printed on the bar with its coverage; because the threshold differs per "
             "bar the coverage percentages are not comparable across bars.",
             "scope — the incomparable-coverage warning; kept compressed"),
            ("Lag 12 exists only for task 0 in a 16-task stream with boundaries at 0, 4, 8, 12, "
             "15, so this is a task-0 figure by construction, and task 0 is the most-forgotten "
             "task — it reports the largest forgetting in the grid rather than its average.",
             "scope — lag/task confound; interpretation — 'largest not average'. Boundaries "
             "list cut; the rest compressed"),
        ),
    ),
    CaptionCut(
        "fig:corners (Figure 4)",
        "Center correlation moves in opposite directions across the similarity design, and the "
        "two axes act almost separately. "
        "(a) Signed center correlation ρ_c over the task stream at γ0=10, per corner of the "
        "feature × readout similarity design of Hiratani 2024; all four corners begin together "
        "(ρ_c = 0.391–0.392), so the split is produced by training rather than by initialization. "
        "Three corners converge; S-HL — high feature similarity, low readout similarity, the same "
        "stimuli under different rules — decorrelates. "
        "(b) The same endpoint difference across the richness sweep, with the grey band showing "
        "the drift measured in lazy arms (γ0=0.03), where the representation barely moves; the "
        "corner ordering is a rich-regime effect, and S-LL never leaves the band. "
        "(c) Per-arm Spearman correlation between ρ_c and block index: the S-HL decline is "
        "progressive rather than an endpoint difference, negative on 40 of 40 arms (median −0.79), "
        "which is the form the practice result of Menghi 2025 takes. "
        "(d) Both pre-registered predictions fail. H1d predicted decorrelation with richness; "
        "three of four corners converge. H6 predicted the largest |Δρ_c| in the catastrophic "
        "corner S-HL; the largest is S-LH, twice the size. "
        "(e) Read as two main effects, readout similarity drives convergence (+0.104) and feature "
        "similarity drives decorrelation (−0.061), with an interaction at the resolution limit "
        "(+0.006) and an additive model predicting S-HH to within 0.012; across the registered "
        "range both effects grow with richness at a roughly constant ratio, so richness sets the "
        "gain and the design sets the sign. "
        "(f) Both main effects survive a 4× change in width, with the corner ordering identical "
        "at all three widths.",
        "Center correlation moves in opposite directions across the 2×2, and the two axes act "
        "almost separately. "
        "(a) Signed ρ_c at γ0=10, Hiratani 2024 design. All four corners start together "
        "(ρ_c = 0.391–0.392); three converge, S-HL (same stimuli, different rules) decorrelates. "
        "(b) Grey band: lazy-arm drift (γ0=0.03). S-LL never leaves it. "
        "(c) S-HL decline is progressive, negative on 8 of 8 unique seeds (p=0.0078; "
        "40 files are copies), median Spearman −0.79. "
        "(d) H1d and H6 fail in sign: three of four corners converge; largest |Δρ_c| is S-LH, "
        "twice S-HL. "
        "(e) Readout +0.104, feature −0.061, interaction +0.006; additive error 0.012. "
        "(f) Both main effects survive a 4× width change, ordering unchanged.",
        (
            ("(a) … per corner of the feature × readout similarity design of Hiratani 2024",
             "scope — citation and design name; cut if the body already carries it"),
            ("all four corners begin together (ρ_c = 0.391–0.392), so the split is produced by "
             "training rather than by initialization.",
             "number + interpretation — numbers kept; 'training not init' compressed into 'start "
             "together'"),
            ("Three corners converge; S-HL — high feature similarity, low readout similarity, the "
             "same stimuli under different rules — decorrelates.",
             "scope — the parenthetical gloss of S-HL; cut. Direction kept."),
            ("(b) The same endpoint difference across the richness sweep, with the grey band "
             "showing the drift measured in lazy arms (γ0=0.03), where the representation barely "
             "moves; the corner ordering is a rich-regime effect, and S-LL never leaves the band.",
             "scope + interpretation — 'rich-regime effect' cut; band and S-LL kept"),
            ("(c) … which is the form the practice result of Menghi 2025 takes.",
             "interpretation — correspondence with the human result; cut from the caption"),
            ("(d) Both pre-registered predictions fail. H1d … H6 … twice the size.",
             "interpretation + number — kept compressed"),
            ("(e) … across the registered range both effects grow with richness at a roughly "
             "constant ratio, so richness sets the gain and the design sets the sign.",
             "interpretation — the gain/sign slogan; cut from the caption (body already has it)"),
            ("(f) Both main effects survive a 4× change in width, with the corner ordering "
             "identical at all three widths.",
             "number/scope — kept compressed"),
        ),
    ),
    CaptionCut(
        "fig:benign",
        "The benign corner gains retained capacity. The same decomposition as Figure "
        "fig:decomposition, for S-HH alone (40 arms per richness level). Δ log α is positive at "
        "every richness, largest at γ0=1 with 11.8 floors and falling to 6.2 by γ0=10. Utility and "
        "dimension move in the gaining direction. The radius term is not resolved at any γ0≥1 "
        "(1.9, 0.2 and 1.1 floors), so this panel claims no sign for it — only a bound, under 2 "
        "floors against 11.7 to 14.1 in the forgetting corners.",
        "The benign corner gains retained capacity. S-HH alone, 40 arms per γ, same decomposition "
        "as Figure fig:decomposition. Largest at γ0=1 (11.8 floors), 6.2 by γ0=10. Radius term not "
        "resolved at γ0≥1 (1.9, 0.2, 1.1 floors) — a bound under 2 floors, not a sign.",
        (
            ("The same decomposition as Figure fig:decomposition, for S-HH alone (40 arms per "
             "richness level).",
             "scope — kept compressed"),
            ("Δ log α is positive at every richness, largest at γ0=1 with 11.8 floors and falling "
             "to 6.2 by γ0=10.",
             "number — kept"),
            ("Utility and dimension move in the gaining direction.",
             "interpretation — cut; the bound on radius is the claim this panel has to carry"),
            ("The radius term is not resolved at any γ0≥1 (1.9, 0.2 and 1.1 floors), so this panel "
             "claims no sign for it — only a bound, under 2 floors against 11.7 to 14.1 in the "
             "forgetting corners.",
             "number + scope — floors kept; the 11.7–14.1 comparison is in the body and can leave "
             "the caption"),
        ),
    ),
    CaptionCut(
        "fig:lag4",
        "The composition holds at a shorter lag. Figure fig:decomposition redrawn at lag 4 within "
        "task 0. The channel shares agree with lag 12 to within 0.032 at every richness while the "
        "magnitude changes by 1.27×, so the composition is a property of the richness rather than "
        "of the particular cell the main figure is drawn at.",
        "The composition holds at a shorter lag. Figure fig:decomposition at lag 4, task 0. Shares "
        "agree with lag 12 to within 0.032; magnitude changes by 1.27×.",
        (
            ("Figure fig:decomposition redrawn at lag 4 within task 0.",
             "scope — kept compressed"),
            ("The channel shares agree with lag 12 to within 0.032 at every richness while the "
             "magnitude changes by 1.27×",
             "number — kept"),
            ("so the composition is a property of the richness rather than of the particular cell "
             "the main figure is drawn at.",
             "interpretation — cut; the two numbers already say it"),
        ),
    ),
    CaptionCut(
        "fig:width",
        "Width robustness, in the main figure's form. Duplicate-stream matched subset "
        "(streams 0–1, seeds 0–1). The γ0=10 contrast quoted in the text is genuine n=40 "
        "(N=150 vs 600; magnitude varies by 0.62%, utility share by 0.020).",
        "Width robustness. Figure is the duplicate-stream subset. Text quotes genuine n=40 "
        "at γ0=10: 0.62%, utility 0.020.",
        (
            ("Duplicate-stream matched subset (streams 0–1, seeds 0–1).",
             "scope — kept compressed"),
            ("The γ0=10 contrast quoted in the text is genuine n=40 (N=150 vs 600; "
             "magnitude varies by 0.62%, utility share by 0.020).",
             "number — kept"),
        ),
    ),
    CaptionCut(
        "fig:corners-gamma3",
        "The corner structure at γ0=3. Figure fig:corners redrawn one richness step lower, where "
        "the S-HL decorrelation has not yet appeared (+0.003, p = 1.0). A probe at γ0=5 finds the "
        "mean first negative (−0.0064±0.0021, declining on 12 of 16 arms, sign-test p=0.077); at "
        "γ0=10 it is unanimous on 40 of 40. The onset sits between 3 and 5 (bracket factor 1.67; "
        "it was 3.33).",
        "The corner structure at γ0=3. S-HL decorrelation absent (+0.003, 4 of 8 unique, p=1.0); "
        "present at γ0=10 (8 of 8 unique). A γ0=5 probe is negative in four of eight "
        "arrangements, so the onset stays bracketed by the grid at a factor of 3.3.",
        (
            ("Figure fig:corners redrawn one richness step lower, where the S-HL decorrelation "
             "has not yet appeared (+0.003, p = 1.0).",
             "number + scope — kept compressed"),
            ("A probe at γ0=5 finds the mean first negative (−0.0064±0.0021, declining on 12 of "
             "16 arms, sign-test p=0.077); at γ0=10 it is unanimous on 40 of 40.",
             "number — superseded twice: the n=16 set was 4 unique copies, and the 5×8 "
             "replacement resolves at the arm unit but not at the arrangement unit. The caption "
             "now carries the arrangement count, which is the unit the onset needs"),
            ("The onset sits between 3 and 5 (bracket factor 1.67; it was 3.33).",
             "number — withdrawn; 1.67 assumed the arm-level γ0=5 reading, so the caption "
             "reports the grid's 3 → 10 bracket at 3.3"),
        ),
    ),
)


def caption_cut_proposal() -> None:
    """Half-length captions, with a role list for every cut sentence. Nothing is applied here."""
    lines = [
        "## Caption cuts — proposed, not applied",
        "",
        "Each current caption, a version at roughly half the word count, and what every cut "
        "sentence was doing (number, scope, or interpretation). The proposed halves are what "
        "landed in `paper/figures.tex` and `paper/figures-appendix.tex`. Restore any sentence "
        "the figure cannot carry alone.",
        "",
    ]
    for c in CAPTION_CUTS:
        lines += [
            f"### {c.fig}",
            "",
            f"- current: **{_words(c.current)} words**",
            f"- proposed: **{_words(c.proposed)} words** "
            f"({100 * _words(c.proposed) / _words(c.current):.0f}% of current)",
            "",
            "**Proposed caption**",
            "",
            c.proposed,
            "",
            "**What each cut (or compressed) sentence was doing**",
            "",
        ]
        for sent, role in c.cuts:
            lines.append(f"- *{role}.* {sent}")
        lines.append("")
    show("\n".join(lines))


__all__ = ["show", "header", "Row", "Table", "grid_summary", "floors_table",
           "floor_denominated", "figure2_tables", "panel_d_table", "benign_table",
           "sign_audit_table", "channel_table", "generality_table", "peak_table",
           "capacity_table", "capacity_table_three_corner",
           "capacity_per_corner", "corners_table", "corners_gamma30", "rho_c_onset_table",
           "width_table",
           "lag_task_table", "variance_table", "stratification_table", "ledger_table",
           "fig2_plotted", "fig2_benign_plotted", "fig2_lag4_plotted", "fig4_plotted",
           "fig4_gamma3_plotted", "fig_width_plotted", "show_figure", "show_figure_with_table",
           "radius_vs_gate_table", "paper_index", "unresolved_claims_table",
           "unique_n_status_table", "population_table", "sampling_unit_table",
           "caption_cut_proposal", "CAPTION_CUTS",
           "CORRECTIONS", "bootstrap_ci", "observations", "pick", "pool",
           "result_set_sha", "git_sha", "figjson",
           "THREE_CORNER", "FOUR_BY_DESIGN", "FOUR_SUPERSEDED", "FOUR_FIGURE_WIDTH", "FOUR_POOLED"]
