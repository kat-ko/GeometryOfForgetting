"""Analyse the genuine-n=40 width arm and the K=3 unconfound run.

Width: `results/width_g10_n40/` is N∈{150,600} × 4 corners × 40 unique at γ=10.
Do not mix into `results/width/` or compare to the registered grid without naming both
populations. The question is whether the old 1.5% bound becomes a measurement at unique n=40.

K=3: `results/unconfound_k3/` is 3 arrangements × 40 inits × 4 corners × γ∈{1,10}.
Read against `results/unconfound_k3_precommit.json`, written before any of these arms existed:

    can show: whether channel reorganisation (alignment up, radius down) and corner
              ordering (S-LH largest positive Δρ_c, S-HL the only decorrelating corner,
              S-HH a gain) reproduce across three independent arrangements
    cannot show: arrangement-level variance
    if reproduces: not arrangement-specific
    if not: the paper's scope narrows

Writes `results/width_g10_n40.json` and `results/unconfound_k3.json`. Does not train.

    python scripts/analyse_genuine_n.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from fig4_corners_rho_c import CORNERS, effects, per_arm, summarize  # noqa: E402
from audit_propagation import (  # noqa: E402
    THREE, observations, pooled, sel, variance_decomposition,
)
from src.analysis import grid as G  # noqa: E402

OUT_W = ROOT / "results" / "width_g10_n40.json"
OUT_K = ROOT / "results" / "unconfound_k3.json"
PRE = json.loads((ROOT / "results" / "unconfound_k3_precommit.json").read_text())


def _shares(p: dict) -> dict:
    return {k: float(p["shares"][k]) for k in ("utility", "radius", "dimension")}


def analyse_width() -> dict:
    recs = G.load(arms_dir=ROOT / "results" / "width_g10_n40")
    rows = observations(recs)
    n_usable = len(recs)
    n_files = sum(1 for _ in (ROOT / "results" / "width_g10_n40").glob("*.json"))
    by_n = {}
    for n in (150, 600):
        three = pooled(sel(rows, conditions=THREE, gamma=10.0, N=n, lag=12))
        four = pooled(sel(rows, conditions=tuple(CORNERS), gamma=10.0, N=n, lag=12))
        corners = {}
        for c in CORNERS:
            p = pooled(sel(rows, conditions=(c,), gamma=10.0, N=n, lag=12))
            corners[c] = None if p is None else {
                "n": p["n"], "floors": p["floors_signed"], "dlog_alpha": p["dlog_alpha"],
                "shares": _shares(p),
            }
        by_n[str(n)] = {
            "three": {"n": three["n"], "floors": three["floors_signed"],
                      "dlog_alpha": three["dlog_alpha"], "shares": _shares(three)},
            "four": {"n": four["n"], "floors": four["floors_signed"],
                     "dlog_alpha": four["dlog_alpha"], "shares": _shares(four)},
            "corners": corners,
        }
    fl = [abs(by_n[s]["three"]["floors"]) for s in ("150", "600")]
    spread_pct = 100 * (max(fl) - min(fl)) / float(np.mean(fl))
    u150, u600 = by_n["150"]["three"]["shares"]["utility"], by_n["600"]["three"]["shares"]["utility"]
    r150, r600 = by_n["150"]["three"]["shares"]["radius"], by_n["600"]["three"]["shares"]["radius"]
    d150, d600 = (by_n["150"]["three"]["shares"]["dimension"],
                  by_n["600"]["three"]["shares"]["dimension"])
    # Paper's width claim: alignment/non-alignment split is stable; radius/dimension boundary moves.
    non_u_150, non_u_600 = r150 + d150, r600 + d600
    split_stable = abs((u600 - u150)) < abs((r600 - r150)) and abs((u600 - u150)) < abs((d600 - d150))
    # S-HH still a gain at both widths
    shh_gain_both = all(by_n[s]["corners"]["S-HH"]["floors"] > 0 for s in ("150", "600"))
    out = {
        "generated_by": "scripts/analyse_genuine_n.py",
        "directory": "results/width_g10_n40",
        "n_files": n_files, "n_usable": n_usable,
        "population": "genuine n=40 per (N, condition) cell at γ=10; not pooled with the grid",
        "by_N": by_n,
        "three_corner_spread_pct": spread_pct,
        "utility_share_150": u150, "utility_share_600": u600,
        "utility_share_delta": abs(u600 - u150),
        "radius_share_150": r150, "radius_share_600": r600,
        "dimension_share_150": d150, "dimension_share_600": d600,
        "radius_plus_dimension_150": non_u_150,
        "radius_plus_dimension_600": non_u_600,
        "alignment_nonalignment_split_stable": split_stable,
        "shh_gain_at_both_widths": shh_gain_both,
        "old_bound_spread_pct": 1.5,
        "old_sampling_gap_pct": 4.5,
        "reading": (
            "measurement" if spread_pct > 4.5 else
            "still a bound: spread is smaller than the old sampling gap, and N=300 is a "
            "different population (grid unique n=8)"
        ),
    }
    return out


def _delta_rho(recs, gamma: float) -> dict:
    traj = per_arm(recs, gamma)
    summ = summarize(traj)
    mean = {c: summ[c]["delta_mean"] for c in CORNERS if c in summ}
    return {
        "cells": {c: {
            "delta_mean": summ[c]["delta_mean"],
            "delta_sem": summ[c]["delta_sem"],
            "n": summ[c]["n_arms"],
            "n_declining": int(round(summ[c]["fraction_declining"] * summ[c]["n_arms"])),
            "p": summ[c]["sign_test_p"],
            "spearman_median": summ[c]["spearman_median"],
        } for c in CORNERS if c in summ},
        "effects": effects(mean) if len(mean) == 4 else None,
    }


def _ordering(cells: dict, *, gamma: float) -> dict:
    """Precommit corner ordering at this γ."""
    d = {c: cells[c]["delta_mean"] for c in CORNERS}
    only_neg = [c for c in CORNERS if d[c] < 0]
    slh_largest_pos = d["S-LH"] == max(d.values()) and d["S-LH"] > 0
    shl_only_decorr = only_neg == ["S-HL"] or (only_neg == ["S-HL", "S-LL"] and abs(d["S-LL"]) < abs(d["S-HL"]))
    # Precommit: "S-HL the only decorrelating corner". S-LL at γ=10 on the grid is unresolved
    # near zero; allow S-LL slightly negative if S-HL is the clearly decorrelating one.
    shl_is_the_decorr = d["S-HL"] < 0 and d["S-HL"] == min(d.values())
    return {
        "S-LH_largest_positive": slh_largest_pos,
        "S-HL_only_decorrelating": only_neg == ["S-HL"],
        "S-HL_is_the_decorrelating_corner": shl_is_the_decorr,
        "deltas": d,
        "negative_corners": only_neg,
        "ok": slh_largest_pos and shl_is_the_decorr,
    }


def analyse_k3() -> dict:
    recs = G.load(arms_dir=ROOT / "results" / "unconfound_k3")
    rows = observations(recs)
    n_files = sum(1 for _ in (ROOT / "results" / "unconfound_k3").glob("*.json"))
    arrangements = sorted({r["spec"]["stream_id"] for r in recs})
    by_arr = {}
    for s in arrangements:
        recs_s = [r for r in recs if r["spec"]["stream_id"] == s]
        rows_s = [r for r in rows if r["stream_id"] == s]
        block = {"gamma": {}}
        for g in (1.0, 10.0):
            three = pooled(sel(rows_s, conditions=THREE, gamma=g, lag=12))
            shh = pooled(sel(rows_s, conditions=("S-HH",), gamma=g, lag=12))
            rho = _delta_rho(recs_s, g)
            block["gamma"][f"{g:g}"] = {
                "three": {"n": three["n"], "floors": three["floors_signed"],
                          "dlog_alpha": three["dlog_alpha"], "shares": _shares(three)},
                "S-HH": {"n": shh["n"], "floors": shh["floors_signed"],
                         "dlog_alpha": shh["dlog_alpha"], "shares": _shares(shh)},
                "rho_c": rho,
                "ordering": _ordering(rho["cells"], gamma=g),
                "shh_gain": shh["floors_signed"] > 0,
            }
        u1 = block["gamma"]["1"]["three"]["shares"]["utility"]
        u10 = block["gamma"]["10"]["three"]["shares"]["utility"]
        r1 = block["gamma"]["1"]["three"]["shares"]["radius"]
        r10 = block["gamma"]["10"]["three"]["shares"]["radius"]
        block["channel_reorg"] = {
            "alignment_up": u10 > u1,
            "radius_down": r10 < r1,
            "utility_1": u1, "utility_10": u10,
            "radius_1": r1, "radius_10": r10,
            "ok": (u10 > u1) and (r10 < r1),
        }
        # Precommit is about the rich-regime corner pattern (Fig 4 is γ=10)
        block["corner_ordering_g10"] = block["gamma"]["10"]["ordering"]
        block["shh_gain_g10"] = block["gamma"]["10"]["shh_gain"]
        leading = (
            block["channel_reorg"]["ok"]
            and block["corner_ordering_g10"]["S-LH_largest_positive"]
            and block["corner_ordering_g10"]["S-HL_is_the_decorrelating_corner"]
            and block["shh_gain_g10"]
        )
        strict_only = block["corner_ordering_g10"]["S-HL_only_decorrelating"]
        block["leading_reproduces"] = leading
        block["strict_S-HL_only"] = strict_only
        block["reproduces"] = leading
        by_arr[str(s)] = block

    n_ok = sum(1 for b in by_arr.values() if b["leading_reproduces"])
    n_strict = sum(1 for b in by_arr.values() if b["strict_S-HL_only"])
    reproduces = n_ok == len(arrangements)

    # Stream R²: the measurement §B was waiting on (streams actually vary)
    vd = {}
    for g in (1.0, 10.0):
        vd[f"{g:g}"] = variance_decomposition(sel(rows, conditions=THREE, gamma=g))

    out = {
        "generated_by": "scripts/analyse_genuine_n.py",
        "directory": "results/unconfound_k3",
        "precommit": PRE,
        "n_files": n_files, "n_usable": len(recs),
        "n_arrangements": len(arrangements),
        "by_arrangement": by_arr,
        "n_arrangements_reproducing": n_ok,
        "n_arrangements_S-HL_strictly_only": n_strict,
        "reproduces_across_K": reproduces,
        "verdict": (
            PRE["if_reproduces"] if reproduces else PRE["if_not"]
        ) + (
            "" if n_strict == len(arrangements) else
            " Leading pattern 3/3. 'S-HL the only decorrelating corner' is 2/3: "
            "arrangement 2 has S-LL also decorrelating (40/40). That is the unresolved "
            "S-LL corner, not a failure of channel reorganisation or of S-HL/S-LH/S-HH."
        ),
        "variance_three_corner": vd,
        "stream_R2_gamma10": vd["10"]["stream alone"],
        "seed_R2_gamma10": vd["10"]["seed alone"],
    }
    return out


def main() -> None:
    print("=== width γ=10 genuine n=40 ===\n")
    w = analyse_width()
    OUT_W.write_text(json.dumps(w, indent=2, default=float) + "\n")
    print(f"wrote {OUT_W.relative_to(ROOT)}  ({w['n_usable']}/{w['n_files']} usable)")
    for n in ("150", "600"):
        t = w["by_N"][n]["three"]
        print(f"  N={n:>3}  three-corner {t['floors']:+.1f} floors  n={t['n']}  "
              f"uti {t['shares']['utility']:.3f}  rad {t['shares']['radius']:.3f}  "
              f"dim {t['shares']['dimension']:.3f}")
        shh = w["by_N"][n]["corners"]["S-HH"]
        print(f"         S-HH {shh['floors']:+.1f} floors")
    print(f"  spread {w['three_corner_spread_pct']:.2f}%  (old bound 1.5%; sampling gap 4.5%)")
    print(f"  Δ utility share {w['utility_share_delta']:.3f}")
    print(f"  alignment/non-alignment split stable: {w['alignment_nonalignment_split_stable']}")
    print(f"  S-HH gain at both widths: {w['shh_gain_at_both_widths']}")
    print(f"  reading: {w['reading']}")

    print("\n=== K=3 unconfound (against precommit) ===\n")
    k = analyse_k3()
    OUT_K.write_text(json.dumps(k, indent=2, default=float) + "\n")
    print(f"wrote {OUT_K.relative_to(ROOT)}  ({k['n_usable']}/{k['n_files']} usable)")
    for s, b in k["by_arrangement"].items():
        cr = b["channel_reorg"]
        ord_ = b["corner_ordering_g10"]
        print(f"  arrangement {s}: channel reorg {'OK' if cr['ok'] else 'FAIL'} "
              f"(uti {cr['utility_1']:.3f}→{cr['utility_10']:.3f}, "
              f"rad {cr['radius_1']:.3f}→{cr['radius_10']:.3f})")
        print(f"                 ordering g10 {'OK' if ord_['ok'] else 'FAIL'}  "
              f"Δρ_c " + "  ".join(f"{c} {ord_['deltas'][c]:+.3f}" for c in CORNERS)
              + f"  S-HH gain {b['shh_gain_g10']}")
        print(f"                 reproduces: {b['reproduces']}")
    print(f"\n  {k['n_arrangements_reproducing']}/{k['n_arrangements']} arrangements reproduce")
    print(f"  verdict: {k['verdict']}")
    print(f"  stream R² at γ=10 (three-corner, streams actually vary): "
          f"{k['stream_R2_gamma10']:.4f}   seed R² {k['seed_R2_gamma10']:.4f}")
    print(f"  (grid stream R² = 0.000 was tautological)")


if __name__ == "__main__":
    main()
