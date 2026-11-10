"""docs/34 similarity map: analysis of the Phase 1 checkpoint (load only; no training, not hashed).

Used by notebooks/core/aba_similarity_map.ipynb (built by _build_aba_similarity_map.py) and by
docs/35. EXPLORATORY throughout (docs/34): 95% t-intervals over seeds for description, no family.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

import aba_lib as L

T95 = stats.t.ppf(0.975, 7)                     # n = 8 seeds, description only
GAMMAS = (0.1, 1.0, 10.0)
H_SWEEP_SF = (1.0, 0.9)                          # the H sweep runs at these s_f
SF_SWEEP_H = (0, 4, 8)                           # the s_f sweep runs at these H (amended 2026-11-10)
BASE_FT = (8, 0.1)                               # FT_beh specific: relative to H8 × s_f 0.1, same seed and γ0
EXPOSURE_NOTE = ("Under matched-loss stopping a similar B needs fewer updates, so similarity effects include "
                 "their effect on exposure (steps on B are plotted beside every map). EXPLORATORY.")


# ------------------------------------------------------------------ loading

def map_cfg() -> L.ABAConfig:
    return L.map_config(L.ABAConfig.make(False))


def load_map(path: Path, cfg: L.ABAConfig | None = None) -> tuple[dict, dict, list[dict]]:
    """(design rows {(seed, γ0, H, s_f): row}, fresh rows {γ0: row}, all rows) for the map config."""
    cfg = cfg or map_cfg()
    want = json.dumps(cfg.key_fields(), sort_keys=True)
    keys = {s.key for s in L.phase1_specs(cfg)}
    rows = [r for r in map(json.loads, Path(path).read_text(encoding="utf-8").splitlines())
            if r["key"] in keys and json.dumps(r["config"], sort_keys=True) == want]
    D, F = {}, {}
    for r in rows:
        s = r["spec"]
        if s.get("noise", "copy") == "fresh":
            F[s["gamma"]] = r
        else:
            D[s["seed"], s["gamma"], s["hamming"], s["s_f"]] = r
    return D, F, rows


def phase1_cells(cfg: L.ABAConfig | None = None) -> list[tuple[int, float]]:
    cfg = cfg or map_cfg()
    return sorted({(s.hamming, s.s_f) for s in L.phase1_specs(cfg) if s.noise == "copy"})


# ------------------------------------------------------------------ quantities (per arm)

def frac_removed(r):
    return -r["backward"]["A1_to_B"]["total"] / r["backward"]["init_to_A1"]["total"]


def restoration(r):
    la = lambda b: math.log(r["geometry"][b]["alpha"])
    return (la("A@A2") - la("A@init")) / (la("A@A1") - la("A@init"))


def share(r, c):
    b = r["backward"]["A1_to_B"]
    return b[c] / b["total"] if b["total"] != 0 else float("nan")   # identity cell: no change to share


def acc_A_after_B(r):
    """Network sign accuracy on A's test points at the end of B (behaviour; chance = 0.5)."""
    return r["trajectory"]["B"][-1]["A_test_acc"]


QUANTS = {   # name -> (label, function of (row, D) )
    "f": ("fraction of A's gain removed by B", lambda r, D: frac_removed(r)),
    "dA": ("Δlog α_A, after A → after B", lambda r, D: r["backward"]["A1_to_B"]["total"]),
    "rho": ("restoration ρ after relearning A", lambda r, D: restoration(r)),
    "u": ("utility share of Δlog α_A(A → B)", lambda r, D: share(r, "utility")),
    "r": ("radius share", lambda r, D: share(r, "radius")),
    "d": ("dimension share", lambda r, D: share(r, "dimension")),
    "ftgeo": ("FT_geo specific (neutral baseline)", lambda r, D: r["forward"]["specific"]),
    "ftbeh": ("FT_beh", lambda r, D: r["behaviour"]["FT_beh"]),
    "ftbeh_spec": ("FT_beh specific (− FT_beh at H8 × s_f 0.1)",
                   lambda r, D: r["behaviour"]["FT_beh"]
                   - D[r["spec"]["seed"], r["spec"]["gamma"], *BASE_FT]["behaviour"]["FT_beh"]),
    "bott": ("bottleneck-probe forgetting: acc(after A) − acc(after B)",
             lambda r, D: r["bottleneck"]["A1"]["acc"] - r["bottleneck"]["B"]["acc"]),
    "accA": ("network accuracy on A at the end of B", lambda r, D: acc_A_after_B(r)),
    "drhoc": ("Δρ_c (signed, normalised), after A → after B",
              lambda r, D: r["geometry"]["A@B"]["rho_c_signed"] - r["geometry"]["A@A1"]["rho_c_signed"]),
    "savings": ("savings log((1+steps A1)/(1+steps A2))", lambda r, D: r["behaviour"]["savings"]),
    "stepsB": ("updates on B (exposure)", lambda r, D: r["behaviour"]["steps"]["B"]),
}
MEASURED = {"taskvec_cos_input": "task-vector cos (inputs)", "taskvec_cos_h0": "task-vector cos (h₀)",
            "rho_in_AB": "realised ρ_in(A, B)"}


def vals(D, q, g, h, sf, seeds) -> np.ndarray:
    return np.array([QUANTS[q][1](D[s, g, h, sf], D) for s in seeds], float)


def ci(x) -> tuple[float, float, float]:
    x = np.asarray(x, float)
    t = T95 if len(x) == 8 else stats.t.ppf(0.975, len(x) - 1)
    m, se = x.mean(), x.std(ddof=1) / math.sqrt(len(x))
    return m, m - t * se, m + t * se


def steps_range(D, g, h, sf, seeds) -> tuple[float, float, float]:
    """Updates on B: median [min, max] over seeds (exposure; a t-interval on counts is not used)."""
    v = vals(D, "stepsB", g, h, sf, seeds)
    return float(np.median(v)), float(v.min()), float(v.max())


def table(D, seeds, cfg=None) -> dict:
    """{(q, γ0, H, s_f): (mean, lo, hi)} for every quantity and Phase 1 cell."""
    out = {}
    for h, sf in phase1_cells(cfg):
        for g in GAMMAS:
            for q in QUANTS:
                out[q, g, h, sf] = ci(vals(D, q, g, h, sf, seeds))
    return out


# ------------------------------------------------------------------ validation

def validate(D, F, rows, seeds, cfg=None) -> list[tuple[str, bool, str]]:
    cfg = cfg or map_cfg()
    chk = []
    n_exp = len(L.phase1_specs(cfg))
    chk.append(("all Phase 1 arms present", len(rows) == n_exp, f"{len(rows)}/{n_exp}"))
    chk.append(("every arm usable (every phase converged)", all(r["usable"] for r in rows),
                f"{sum(r['usable'] for r in rows)} usable"))
    bad = {r["key"]: L.check_schema(r) for r in rows}
    bad = {k: v for k, v in bad.items() if v}
    chk.append(("schema: every field present and finite", not bad, f"{len(bad)} arms with gaps"))
    stamps = {json.dumps(r["stamp"], sort_keys=True) for r in rows}
    clean = all(not r["stamp"].get("dirty", True) for r in rows)
    chk.append(("one commit, clean tree", len(stamps) == 1 and clean, "; ".join(sorted(stamps))[:120]))
    hashes = {r["lib_hash"] for r in rows}
    chk.append(("one lib hash", len(hashes) == 1, ", ".join(sorted(hashes))))
    sr_ok = all(r["relations"]["s_r_AB"] == 1 - r["spec"]["hamming"] / 8 for r in rows)
    chk.append(("realised s_r = 1 − H/8 exactly", sr_ok, ""))
    rho1 = [D[k]["relations"]["rho_in_AB"] for k in D if k[3] == 1.0]
    chk.append(("ρ_in = 1 at s_f 1 (exact copy)", all(abs(v - 1) < 1e-12 for v in rho1), f"{len(rho1)} arms"))
    h0 = all(len({D[s, g, h, sf]["relations"]["taskvec_cos_h0"] for g in GAMMAS}) == 1
             for s in seeds for h, sf in phase1_cells(cfg))
    chk.append(("task-vector cos on h₀ identical across γ0 (I6)", h0, ""))
    ident = [D[s, g, 0, 1.0] for s in seeds for g in GAMMAS]
    ok = all(r["behaviour"]["steps"]["B"] == 0 and r["backward"]["A1_to_B"]["total"] == 0.0 for r in ident)
    chk.append(("identity cell (H0, s_f 1, copy): 0 updates on B, Δlog α_A = 0", ok, f"{len(ident)} arms"))
    fr = {g: F[g]["backward"]["A1_to_B"]["total"] for g in F}
    chk.append(("fresh-noise identity arm per γ0 present (Δlog α_A reported)", len(F) == 3,
                ", ".join(f"γ0 {g:g}: {v:+.4f}" for g, v in sorted(fr.items()))))
    return chk


# ------------------------------------------------------------------ expectation readings (rules fixed 2026-11-10)

READING_RULES = """\
Reading rules (operationalised by Claude on 2026-11-10, before any map arm; 95% t-intervals over the
8 seeds, description only, no family). Each expectation is read as held / not held / unresolved.
- E1: at each γ0, among the H4/H8 cells of the s_f sweep, the cell with the largest mean f AND the
  largest mean bottleneck forgetting is H8 × s_f ∈ {1, 0.95, 0.9}, and the cell with the smallest mean f
  has s_f ≤ 0.2. Held if both at every γ0. Not held if, at any γ0, another cell's interval lies wholly
  above the best H8 × high-similarity cell's interval, or the smallest-f cell has s_f > 0.2 and its
  interval lies wholly below every s_f ≤ 0.2 cell's interval. Otherwise unresolved.
- E2: per γ0 and H ∈ {4, 8}, across s_f: "peaked" if some interior s_f (0.2–0.95) has an interval
  wholly above both endpoint intervals (s_f 0.1 and 1); "monotone" if no peak and the Spearman
  correlation of the means with s_f is ≥ 0.8; "decreasing" if ≤ −0.8; otherwise unresolved. Recorded
  per (γ0, H). Held if every (γ0, H) is monotone or peaked; not held if any is decreasing.
- E3: at s_f ∈ {1, 0.9}: (i) f(H8) − f(H0), paired, has its interval above 0 and the means rise
  over H 0, 2, 4, 6, 8 with at most one inversion; (ii) the network's accuracy on A at the end of B
  has its interval below 0.5 at H12 and H16; (iii) Δlog α_A at H16 minus at H8, paired, has its
  interval above 0. Held if (i)–(iii) at both s_f and every γ0. Not held if any part has its interval
  wholly on the wrong side. Otherwise unresolved.
- E4: γ0 0.1, H16 × s_f 1: Δlog α_A(A → B). Held if the interval lies within ±0.05 (the docs/29
  SESOI for log α); not held if it excludes 0; otherwise unresolved. At γ0 10 it is recorded.
- E5: H0 × s_f ∈ {0.4, 0.6}: FT_beh. Held if its interval is above 0 at both s_f and every γ0; not
  held (matches Menghi) if it is below 0 at any; otherwise unresolved. FT_geo specific is reported.
- E6: Spearman correlation of the cell means of f across the 32 Phase 1 cells, for each γ0 pair.
  Held if every pair is ≥ 0.8; not held if any is < 0.5; otherwise unresolved.
- E7: γ0 1 and 10, H4 and H8 s_f slices: held if mean ρ ≥ 0.9 at every s_f ∈ {1, 0.95, 0.9} and the
  interval of ρ is below 0.9 at s_f 0.1 or 0.2; not held if the interval lies wholly below 0.9 at
  some s_f ≥ 0.9, or wholly at or above 0.9 at both s_f 0.1 and 0.2; otherwise unresolved. Per
  (γ0, H), then all four combined.
"""


def _below(a, b):   # interval a wholly below interval b
    return a[2] < b[1]


def read_expectations(D, seeds, cfg=None) -> dict:
    cfg = cfg or map_cfg()
    T = table(D, seeds, cfg)
    sfs = cfg.s_fs
    out = {}

    # E1
    res, det = [], []
    for g in GAMMAS:
        cells = [(h, sf) for h in (4, 8) for sf in sfs]
        best_f = max(cells, key=lambda c: T["f", g, *c][0])
        best_b = max(cells, key=lambda c: T["bott", g, *c][0])
        low = min(cells, key=lambda c: T["f", g, *c][0])
        hi_cells = [(8, sf) for sf in (1.0, 0.95, 0.9)]
        ok = best_f in hi_cells and best_b in hi_cells and low[1] <= 0.2
        top = max(hi_cells, key=lambda c: T["f", g, *c][0])
        worse = any(_below(T["f", g, *top], T["f", g, *c]) for c in cells if c not in hi_cells)
        lowc = [c for c in cells if c[1] <= 0.2]
        worse |= low[1] > 0.2 and all(_below(T["f", g, *low], T["f", g, *lc]) for lc in lowc)
        res.append("held" if ok else ("not held" if worse else "unresolved"))
        det.append(f"γ0 {g:g}: max f at H{best_f[0]} s_f {best_f[1]:g}, max bottleneck at H{best_b[0]} "
                   f"s_f {best_b[1]:g}, min f at H{low[0]} s_f {low[1]:g}")
    out["E1"] = (_combine(res), det)

    # E2
    res, det = [], []
    for g in GAMMAS:
        for h in (4, 8):
            ms = [T["f", g, h, sf] for sf in sfs]
            ends = [T["f", g, h, 1.0], T["f", g, h, 0.1]]
            peak = any(all(_below(e, T["f", g, h, sf]) for e in ends) for sf in sfs if 0.2 <= sf <= 0.95)
            rho = stats.spearmanr(sfs, [m[0] for m in ms]).statistic
            shape = "peaked" if peak else ("monotone" if rho >= 0.8 else ("decreasing" if rho <= -0.8
                                                                          else "unresolved"))
            res.append({"peaked": "held", "monotone": "held", "decreasing": "not held"}.get(shape, "unresolved"))
            det.append(f"γ0 {g:g}, H{h}: {shape} (Spearman of means with s_f {rho:+.2f})")
    out["E2"] = (_combine(res), det)

    # E3
    res, det = [], []
    for sf in H_SWEEP_SF:
        for g in GAMMAS:
            d80 = ci(vals(D, "f", g, 8, sf, seeds) - vals(D, "f", g, 0, sf, seeds))
            means = [T["f", g, h, sf][0] for h in (0, 2, 4, 6, 8)]
            inv = sum(b < a for a, b in zip(means, means[1:]))
            i_ok, i_bad = d80[1] > 0 and inv <= 1, d80[2] < 0
            accs = [T["accA", g, h, sf] for h in (12, 16)]
            ii_ok, ii_bad = all(a[2] < 0.5 for a in accs), any(a[1] > 0.5 for a in accs)
            g16 = ci(vals(D, "dA", g, 16, sf, seeds) - vals(D, "dA", g, 8, sf, seeds))
            iii_ok, iii_bad = g16[1] > 0, g16[2] < 0
            res.append("held" if (i_ok and ii_ok and iii_ok) else
                       ("not held" if (i_bad or ii_bad or iii_bad) else "unresolved"))
            det.append(f"s_f {sf:g}, γ0 {g:g}: f(H8)−f(H0) {_f(d80)}, inversions {inv}; acc on A at end of B "
                       f"H12 {_f(accs[0])}, H16 {_f(accs[1])}; Δlog α_A(H16) − (H8) {_f(g16)}")
    out["E3"] = (_combine(res), det)

    # E4
    c01, c10 = T["dA", 0.1, 16, 1.0], T["dA", 10.0, 16, 1.0]
    e4 = "held" if (-0.05 < c01[1] and c01[2] < 0.05) else ("not held" if (c01[2] < 0 or c01[1] > 0) else "unresolved")
    out["E4"] = (e4, [f"γ0 0.1: Δlog α_A(A → B) at H16 × s_f 1 {_f(c01)}",
                      f"γ0 10 (recorded): {_f(c10)}; γ0 1 (recorded): {_f(T['dA', 1.0, 16, 1.0])}"])

    # E5
    res, det = [], []
    for g in GAMMAS:
        for sf in (0.4, 0.6):
            c = T["ftbeh", g, 0, sf]
            res.append("held" if c[1] > 0 else ("not held" if c[2] < 0 else "unresolved"))
            det.append(f"γ0 {g:g}, s_f {sf:g}: FT_beh {_f(c)}; FT_geo specific {_f(T['ftgeo', g, 0, sf])}")
    out["E5"] = (_combine(res), det)

    # E6
    cells = phase1_cells(cfg)
    res, det = [], []
    for a, b in ((0.1, 1.0), (0.1, 10.0), (1.0, 10.0)):
        rho = stats.spearmanr([T["f", a, *c][0] for c in cells], [T["f", b, *c][0] for c in cells]).statistic
        res.append("held" if rho >= 0.8 else ("not held" if rho < 0.5 else "unresolved"))
        det.append(f"γ0 {a:g} vs {b:g}: Spearman {rho:+.3f} over {len(cells)} cells")
    out["E6"] = (_combine(res), det)

    # E7
    res, det = [], []
    for g in (1.0, 10.0):
        for h in (4, 8):
            hi = [T["rho", g, h, sf] for sf in (1.0, 0.95, 0.9)]
            lo = [T["rho", g, h, sf] for sf in (0.1, 0.2)]
            ok = all(c[0] >= 0.9 for c in hi) and any(c[2] < 0.9 for c in lo)
            bad = any(c[2] < 0.9 for c in hi) or all(c[1] >= 0.9 for c in lo)
            res.append("held" if ok else ("not held" if bad else "unresolved"))
            det.append(f"γ0 {g:g}, H{h}: ρ at s_f 1/0.95/0.9 " + ", ".join(f"{c[0]:.3f}" for c in hi)
                       + f"; at s_f 0.2 {_f(lo[1])}, 0.1 {_f(lo[0])}")
    out["E7"] = (_combine(res), det)
    return out


def _combine(res: list[str]) -> str:
    if all(r == "held" for r in res):
        return "held"
    if any(r == "not held" for r in res):
        return "not held"
    return "unresolved"


def _f(c) -> str:
    return f"{c[0]:+.3f} [{c[1]:+.3f}, {c[2]:+.3f}]"
