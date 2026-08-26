"""Tier 1 propagation audit — every pooled number, before and after the three-corner regroup.

`S-HH` gains retained capacity where the other three corners lose it, so any quantity that
averaged over the 2×2 averaged a gain against three losses. This recomputes each such quantity
both ways and reports the pair.

**Two kinds of row, and they must not be presented as one kind.** Where the regroup changed
*which population* a number describes, both values are correct and only one answers the paper's
question — the number did not become right, the question became explicit. Where a number was read
off the wrong cell, it was simply wrong. The table marks every row `estimand` or `error`, because
a reviewer meeting a 37% movement in a headline figure will otherwise ask which value to believe,
and the honest answer differs by row.

**Everything derives from one observation table**, so the two groupings cannot drift apart through
two implementations of the same pooling. Attribution is re-derived from stored geometry as
everywhere else; nothing here reads a stored `attribution` field and nothing is trained.

    python scripts/audit_propagation.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis import grid as G  # noqa: E402
from src.analysis.attribution import (FACTORS, FLOOR_GATE_K,  # noqa: E402
                                      R_EFF_FLOOR_CV, GeometryPoint, attribute,
                                      attribution_table, center_collapse_share)

FOUR = ("S-HH", "S-HL", "S-LH", "S-LL")
THREE = ("S-HL", "S-LH", "S-LL")
BENIGN = "S-HH"
MIN_FLOORS = 2.0
OUT = ROOT / "results" / "audit_propagation.json"


# --- the observation table ----------------------------------------------------

def observations(recs: list[dict], module: str = "A") -> list[dict]:
    """One row per (arm, past task, later boundary) retained-capacity comparison.

    Carries the arm's design fields, the comparison's position in the stream, the re-derived
    attribution, and the capacity the task was *left* at — the last of which is the regression
    predictor in §5.3 and is not recoverable from an `Attribution` alone.
    """
    rows = []
    for r in recs:
        pts = {(g["module"], g["task"], g["boundary"]): g
               for g in r["geometry"] if g["task"] is not None}
        spec = r["spec"]
        for (mod, task, b), g in sorted(pts.items()):
            if mod != module or b <= task:
                continue
            origin = pts.get((mod, task, task))
            if origin is None:
                continue
            before, after = GeometryPoint.from_result(origin), GeometryPoint.from_result(g)
            att = attribute(before, after, strict=False, gamma=spec["gamma_0"])
            rows.append({
                "before_pt": before, "after_pt": after,
                "gamma": spec["gamma_0"], "condition": spec["condition"],
                "N": spec.get("N", 300), "seed": spec["seed"],
                "stream_id": spec.get("stream_id", 0),
                "task": task, "lag": b - task, "boundary": b,
                "alpha_own": float(origin["alpha"]), "alpha_after": float(g["alpha"]),
                "dlog_alpha": att.dlog_alpha, "terms": att.terms,
                "center": att.center.get("attributable_fraction"),
                "att": att,
            })
    return rows


def sel(rows, **kw) -> list[dict]:
    """Filter the observation table; `conditions` takes a tuple, everything else a value."""
    conds = kw.pop("conditions", None)
    out = [r for r in rows if all(r[k] == v for k, v in kw.items())]
    return [r for r in out if conds is None or r["condition"] in conds]


def pooled(rows: list[dict]) -> dict | None:
    """Pooled attribution for a set of observations, in the same convention as Figure 2."""
    if not rows:
        return None
    t = attribution_table([r["att"] for r in rows])
    fl = G.floors(t["dlog_alpha"], "alpha")
    return {"n": t["n"], "dlog_alpha": t["dlog_alpha"],
            "floors_signed": float(np.sign(t["dlog_alpha"]) * fl),
            "shares": t["shares"], "terms": t["terms"],
            "center_median": t["center_attributable_median"],
            "center_coverage": t["center_attributable_n"] / t["n"]}


# --- regressions --------------------------------------------------------------

def r2(y: np.ndarray, X: np.ndarray) -> float:
    """OLS R² with an intercept prepended."""
    A = np.column_stack([np.ones(len(y)), X]) if X.size else np.ones((len(y), 1))
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ beta
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return float(1.0 - (resid ** 2).sum() / ss_tot) if ss_tot > 0 else 0.0


def dummies(labels: list) -> np.ndarray:
    """One-hot with the first level dropped, so the intercept is not collinear."""
    levels = sorted(set(labels))[1:]
    return np.array([[1.0 if x == lv else 0.0 for lv in levels] for x in labels]) \
        if levels else np.zeros((len(labels), 0))


def asymptote(rows: list[dict]) -> float:
    """Common asymptote implied by regressing cell-mean final capacity on cell-mean initial.

    `α_after = a + b·α_own` has fixed point `a/(1−b)`: the capacity a task would be left at if
    it started there, i.e. the level the stream pulls every task toward.
    """
    cells = defaultdict(list)
    for r in rows:
        cells[(r["task"], r["lag"])].append(r)
    own = np.array([np.mean([x["alpha_own"] for x in v]) for v in cells.values()])
    aft = np.array([np.mean([x["alpha_after"] for x in v]) for v in cells.values()])
    b, a = np.polyfit(own, aft, 1)
    return float(a / (1.0 - b))


def variance_decomposition(rows: list[dict]) -> dict:
    """§B's table: how much of per-arm forgetting each candidate source explains."""
    a_inf = asymptote(rows)
    y = np.array([r["dlog_alpha"] for r in rows])
    dist = np.array([[r["alpha_own"] - a_inf] for r in rows])
    lag = np.array([[float(r["lag"])] for r in rows])
    cond = dummies([r["condition"] for r in rows])
    stream = dummies([r["stream_id"] for r in rows])
    seed = dummies([r["seed"] for r in rows])
    out = {
        "n": len(rows), "alpha_asymptote": a_inf,
        "distance + lag": r2(y, np.hstack([dist, lag])),
        "condition alone": r2(y, cond),
        "stream alone": r2(y, stream),
        "seed alone": r2(y, seed),
        "condition + distance + lag": r2(y, np.hstack([cond, dist, lag])),
        "+ stream + seed": r2(y, np.hstack([cond, dist, lag, stream, seed])),
        "total_sd": float(y.std(ddof=1)),
    }
    A = np.column_stack([np.ones(len(y)), cond, dist, lag, stream, seed])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    out["residual_sd"] = float((y - A @ beta).std(ddof=1))
    return out


def cell_mean_r2(rows: list[dict]) -> dict:
    """The inflation pair: same regression at cell level and at arm level."""
    a_inf = asymptote(rows)
    cells = defaultdict(list)
    for r in rows:
        cells[(r["task"], r["lag"])].append(r)
    cy = np.array([np.mean([x["dlog_alpha"] for x in v]) for v in cells.values()])
    cd = np.array([[np.mean([x["alpha_own"] for x in v]) - a_inf,
                    float(k[1])] for k, v in cells.items()])
    y = np.array([r["dlog_alpha"] for r in rows])
    ad = np.array([[r["alpha_own"] - a_inf, float(r["lag"])] for r in rows])
    return {"n_cells": len(cells), "n_arms": len(rows),
            "cell_mean_r2_distance_only": r2(cy, cd[:, :1]),
            "cell_mean_r2": r2(cy, cd),
            "arm_level_R2_distance_only": r2(y, ad[:, :1]),
            "arm_level_R2": r2(y, ad)}


# --- report -------------------------------------------------------------------

def pair(rows, label, **kw) -> dict:
    """Same quantity under both groupings."""
    a = pooled(sel(rows, conditions=FOUR, **kw))
    b = pooled(sel(rows, conditions=THREE, **kw))
    return {"label": label, "four": a, "three": b}


def fmt(p: dict | None, key: str = "floors_signed") -> str:
    return "     —" if p is None else f"{p[key]:+6.1f}"


def main() -> None:
    print("Loading the registered grid, the width arms and the γ=30 probe.\n")
    main_rows = observations(G.load())
    width_rows = observations(G.load(arms_dir=ROOT / "results" / "width"))
    ext_rows = observations(G.load(arms_dir=ROOT / "results" / "gamma_ext"))
    report: dict = {}

    gs = sorted({r["gamma"] for r in main_rows})

    # 1. Figure 2 magnitudes and shares, at the figure's own cell (lag 12 = task 0).
    print("=" * 92)
    print("1. FIGURE 2 — magnitude and composition at lag 12 (task 0 by construction)")
    print("=" * 92)
    print(f"{'γ':>6}  {'4-corner':>9} {'3-corner':>9}  {'Δfloors':>8}   "
          f"{'uti 4→3':>13}  {'rad 4→3':>13}  {'dim 4→3':>13}")
    fig2 = {}
    for g in gs:
        p = pair(main_rows, f"fig2 γ={g:g}", gamma=g, lag=12)
        fig2[g] = p
        a, b = p["four"], p["three"]
        print(f"{g:>6g}  {fmt(a)} {fmt(b)}  {b['floors_signed'] - a['floors_signed']:>+8.1f}   "
              + "  ".join(f"{a['shares'][f]:.3f}→{b['shares'][f]:.3f}" for f in FACTORS))
    report["fig2"] = fig2

    # Which γ the figure actually draws shares for, and therefore which share the caption's
    # range should start from. The regroup made one more level resolvable, which moves the
    # start of the range even though no share at any given γ moved much.
    res = {k: [g for g in gs if abs(fig2[g][k]["floors_signed"]) >= MIN_FLOORS]
           for k in ("four", "three")}
    print(f"\n  resolvable at ±{MIN_FLOORS:g} floors — 4-corner: "
          f"{[f'{g:g}' for g in res['four']]}   3-corner: {[f'{g:g}' for g in res['three']]}")
    for k, name in (("four", "4-corner"), ("three", "3-corner")):
        lo, hi = res[k][0], res[k][-1]
        print(f"  {name} utility share over its resolvable range: "
              f"{fig2[lo][k]['shares']['utility']:.3f} (γ={lo:g}) → "
              f"{fig2[hi][k]['shares']['utility']:.3f} (γ={hi:g})")
    report["resolvable"] = res

    # 2. The benign corner on its own, and the sign of its terms.
    print("\n" + "=" * 92)
    print("2. S-HH — the corner that gains, and how its three terms are signed")
    print("=" * 92)
    print(f"{'γ':>6}  {'floors':>7}  {'utility':>9} {'radius':>9} {'dimension':>9}  "
          f"{'radius sign vs others':>22}")
    shh = {}
    for g in gs:
        p = pooled(sel(main_rows, conditions=(BENIGN,), gamma=g, lag=12))
        q = pooled(sel(main_rows, conditions=THREE, gamma=g, lag=12))
        opp = np.sign(p["terms"]["radius"]) != np.sign(p["terms"]["utility"])
        shh[g] = {"benign": p, "forgetting": q, "radius_opposes_utility": bool(opp)}
        print(f"{g:>6g}  {p['floors_signed']:>+7.1f}  "
              + " ".join(f"{p['terms'][f]:>+9.4f}" for f in FACTORS)
              + f"  {'opposes' if opp else 'agrees':>22}")
    report["benign_corner"] = shh

    # 3. Width — the like-for-like contrast, both ways.
    #
    # `fig_width_invariance.py` draws N=300 from a *matched subset* of the registered grid
    # (streams 0–1, seeds 0–1) so that the middle line is not an average over ten times more
    # arms than its neighbours. That subset is also the 4-arm row that produced two wrong
    # numbers in drafted text, so both it and the full-grid value are reported here.
    print("\n" + "=" * 92)
    print("3. WIDTH — §5.3's invariance claim, four-corner (as drafted) and three-corner")
    print("=" * 92)
    wg = sorted({r["gamma"] for r in width_rows})
    matched = [r for r in main_rows if r["stream_id"] < 2 and r["seed"] < 2]
    print(f"width arms: γ ∈ {[f'{g:g}' for g in wg]}, N ∈ "
          f"{sorted({r['N'] for r in width_rows})}, lag 12 (task 0), "
          f"N=300 from the grid's matched subset (streams 0–1, seeds 0–1)")
    width = {}
    for g in wg:
        print(f"\n  γ = {g:g}")
        print(f"  {'N':>16}  {'4-corner':>9} {'3-corner':>9}   "
              f"{'uti 4→3':>13}  {'rad 4→3':>13}  {'dim 4→3':>13}   {'n 4c':>5}")
        for label, src, n in (("150", width_rows, 150), ("300 (matched)", matched, 300),
                              ("300 (full grid)", main_rows, 300), ("600", width_rows, 600)):
            p = pair(src, f"width N={label} γ={g:g}", gamma=g, N=n, lag=12)
            width[f"{g:g}|{label}"] = p
            a, b = p["four"], p["three"]
            if a is None or b is None:
                continue
            print(f"  {label:>16}  {fmt(a)} {fmt(b)}   "
                  + "  ".join(f"{a['shares'][f]:.3f}→{b['shares'][f]:.3f}" for f in FACTORS)
                  + f"   {a['n']:>5}")
    report["width"] = width

    # 4. W5a — lag series within task 0, and W5b — task series at fixed lag.
    print("\n" + "=" * 92)
    print("4. W5 — lag series (task 0) and task series (lag 4), both groupings")
    print("=" * 92)
    lags = sorted({r["lag"] for r in main_rows if r["task"] == 0})
    w5 = {"lag_series": {}, "task_series": {}}
    for g in (1.0, 3.0, 10.0):
        print(f"\n  γ = {g:g}   lag series within task 0")
        print(f"  {'':>10}" + "".join(f"{l:>10}" for l in lags) + f"{'ratio':>9}")
        for name, conds in (("4-corner", FOUR), ("3-corner", THREE)):
            v = [pooled(sel(main_rows, conditions=conds, gamma=g, task=0, lag=l))
                 for l in lags]
            w5["lag_series"][f"{g:g}|{name}"] = {
                str(l): (x["floors_signed"] if x else None) for l, x in zip(lags, v)}
            ratio = abs(v[-1]["floors_signed"] / v[0]["floors_signed"])
            w5["lag_series"][f"{g:g}|{name}|ratio"] = ratio
            print(f"  {name:>10}" + "".join(f"{x['floors_signed']:>10.1f}" for x in v)
                  + f"{ratio:>8.2f}×")

    tasks = sorted({r["task"] for r in main_rows if r["lag"] == 4})
    for g in (1.0, 10.0):
        print(f"\n  γ = {g:g}   task series at fixed lag 4")
        print(f"  {'':>10}" + "".join(f"{'task ' + str(t):>10}" for t in tasks) + f"{'ratio':>9}")
        for name, conds in (("4-corner", FOUR), ("3-corner", THREE)):
            v = [pooled(sel(main_rows, conditions=conds, gamma=g, lag=4, task=t))
                 for t in tasks]
            w5["task_series"][f"{g:g}|{name}"] = {
                str(t): (x["floors_signed"] if x else None) for t, x in zip(tasks, v)}
            ratio = abs(v[0]["floors_signed"] / v[-1]["floors_signed"])
            w5["task_series"][f"{g:g}|{name}|ratio"] = ratio
            print(f"  {name:>10}" + "".join(f"{x['floors_signed']:>10.1f}" for x in v)
                  + f"{ratio:>8.2f}×")
    report["w5"] = w5

    # 5. Cell-mean inflation and the §B variance table.
    print("\n" + "=" * 92)
    print("5. §A.1 CELL-MEAN INFLATION and §B VARIANCE DECOMPOSITION (γ = 10)")
    print("=" * 92)
    vd = {}
    for name, conds in (("4-corner", FOUR), ("3-corner", THREE)):
        rows = sel(main_rows, conditions=conds, gamma=10.0)
        vd[name] = {"cell_mean": cell_mean_r2(rows), "variance": variance_decomposition(rows)}
    c4, c3 = vd["4-corner"]["cell_mean"], vd["3-corner"]["cell_mean"]
    print(f"  cell-mean r² (distance+lag, {c4['n_cells']} cells): "
          f"{c4['cell_mean_r2']:.3f} → {c3['cell_mean_r2']:.3f}")
    print(f"  cell-mean r² (distance only)                    : "
          f"{c4['cell_mean_r2_distance_only']:.3f} → "
          f"{c3['cell_mean_r2_distance_only']:.3f}")
    print(f"  arm-level R² (distance+lag)                     : "
          f"{c4['arm_level_R2']:.3f} → {c3['arm_level_R2']:.3f}  "
          f"(n {c4['n_arms']} → {c3['n_arms']})")
    print(f"\n  {'predictors':<28}{'4-corner':>10}{'3-corner':>10}")
    for k in ("distance + lag", "condition alone", "stream alone", "seed alone",
              "condition + distance + lag", "+ stream + seed"):
        print(f"  {k:<28}{vd['4-corner']['variance'][k]:>10.3f}"
              f"{vd['3-corner']['variance'][k]:>10.3f}")
    for k in ("total_sd", "residual_sd", "alpha_asymptote", "n"):
        print(f"  {k:<28}{vd['4-corner']['variance'][k]:>10.3f}"
              f"{vd['3-corner']['variance'][k]:>10.3f}")
    report["variance"] = vd

    # 6. γ = 30.
    print("\n" + "=" * 92)
    print("6. γ = 30 PROBE — magnitude series")
    print("=" * 92)
    ext = {}
    for lg in sorted({r["lag"] for r in ext_rows}):
        p = pair(ext_rows, f"gamma_ext lag {lg}", gamma=30.0, lag=lg)
        ext[str(lg)] = p
        mark = "   <- the series' own cell" if lg == 12 else ""
        print(f"  lag {lg:>2}: 4-corner {fmt(p['four'])}   3-corner {fmt(p['three'])}{mark}")
    report["gamma30"] = ext
    for name, key in (("4-corner", "four"), ("3-corner", "three")):
        print(f"  γ = 3 → 10 → 30 at lag 12 ({name}): "
              + " → ".join(f"{fig2[g][key]['floors_signed']:.1f}" for g in (3.0, 10.0))
              + f" → {ext['12'][key]['floors_signed']:.1f}")

    # 6b. Center-collapse share — panel (d), whose gate is per-γ.
    print("\n  panel (d) center-collapse share, four-corner vs three-corner (lag 12):")
    print(f"  {'γ':>6}  {'4c median':>10} {'4c cover':>9}   {'3c median':>10} {'3c cover':>9}")
    for g in gs:
        a, b = fig2[g]["four"], fig2[g]["three"]
        f4 = "      n/a" if a["center_median"] is None else f"{a['center_median']:>+9.3f}"
        f3 = "      n/a" if b["center_median"] is None else f"{b['center_median']:>+9.3f}"
        print(f"  {g:>6g}  {f4} {100 * a['center_coverage']:>8.0f}%   "
              f"{f3} {100 * b['center_coverage']:>8.0f}%")

    # 6c. §A.2's coverage pair. That pair compares two *gate settings* on one population, so the
    # regroup changes both of its numbers without touching the comparison it makes.
    print("\n  §A.2's coverage pair (grid-wide, all γ and lags, module A):")
    print(f"  {'gate':>10}  {'4-corner':>9}  {'3-corner':>9}")
    cov = {}
    for k, label in ((1.0, "1× floor"), (FLOOR_GATE_K, f"{FLOOR_GATE_K:g}× floor")):
        row = {}
        for name, conds in (("four", FOUR), ("three", THREE)):
            rows = sel(main_rows, conditions=conds)
            ok = sum(center_collapse_share(
                r["before_pt"], r["after_pt"],
                min_dlog_R=k * float(np.log1p(R_EFF_FLOOR_CV[r["gamma"]])),
            )["attributable_fraction"] is not None for r in rows)
            row[name] = ok / len(rows)
        cov[label] = row
        print(f"  {label:>10}  {100 * row['four']:>8.0f}%  {100 * row['three']:>8.0f}%")
    report["panel_d_coverage"] = cov

    # 7. Sign audit — is the benign corner the only place capacity rises?
    print("\n" + "=" * 92)
    print("7. SIGN AUDIT — every cell in the grid, width arms and γ=30 probe")
    print("=" * 92)
    gains, resolved_gains = [], []
    for src, rows in (("grid", main_rows), ("width", width_rows), ("gamma30", ext_rows)):
        cells = defaultdict(list)
        for r in rows:
            cells[(r["gamma"], r["condition"], r["N"], r["task"], r["lag"])].append(r)
        for k, v in sorted(cells.items()):
            p = pooled(v)
            if p["dlog_alpha"] > 0:
                rec = {"source": src, "gamma": k[0], "condition": k[1], "N": k[2],
                       "task": k[3], "lag": k[4], "floors": p["floors_signed"], "n": p["n"]}
                gains.append(rec)
                if p["floors_signed"] >= MIN_FLOORS:
                    resolved_gains.append(rec)
    n_cells = sum(len({(r["gamma"], r["condition"], r["N"], r["task"], r["lag"])
                       for r in rows}) for rows in (main_rows, width_rows, ext_rows))
    other = [r for r in resolved_gains if r["condition"] != BENIGN]
    print(f"  {n_cells} cells examined; {len(gains)} have positive mean Δ log α, "
          f"{len(resolved_gains)} of those clear ±{MIN_FLOORS:g} floors")
    print(f"  of the resolvable gains, {len(resolved_gains) - len(other)} are S-HH "
          f"and {len(other)} are not")
    for r in sorted(other, key=lambda r: -r["floors"])[:20]:
        print(f"    {r['source']:>8}  γ={r['gamma']:<5g} {r['condition']:<5} N={r['N']:<4} "
              f"task {r['task']:<3} lag {r['lag']:<3} {r['floors']:>+7.1f} floors  n={r['n']}")
    unresolved = [r for r in gains if r not in resolved_gains and r["condition"] != BENIGN]
    print(f"  sub-floor positive cells outside S-HH: {len(unresolved)} "
          f"(max {max([r['floors'] for r in unresolved], default=0.0):+.2f} floors)")
    report["sign_audit"] = {"n_cells": n_cells, "gains": gains,
                            "resolved_gains": resolved_gains,
                            "resolved_gains_outside_benign": other}

    OUT.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nwrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
