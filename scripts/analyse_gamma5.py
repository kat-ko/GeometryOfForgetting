"""Where does the `S-HL` center decorrelation begin?

At γ₀=3 it is absent (+0.0034); at γ₀=10 it is −0.0550 on 40 of 40 arms; at γ₀=30 it is stronger
still. The registered sweep steps by 3.3× between 3 and 10, so the onset was bracketed by a factor
of 3.3 and nothing in the record narrowed it (`docs/12-record-report.md` Q10). γ₀=5 sits inside the
bracket.

Uses `fig4_corners_rho_c.per_arm`, `summarize` and `effects` unchanged, so the γ₀=5 numbers are
computed the same way as every other richness level in Figure 4 -- the comparison is the whole point
and a second implementation would undermine it.

Writes `results/gamma5_onset.json`.

    python scripts/analyse_gamma5.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from fig4_corners_rho_c import effects, per_arm, summarize  # noqa: E402

from src.analysis import grid as G  # noqa: E402

CORNERS = ("S-HH", "S-HL", "S-LH", "S-LL")


def parse_args():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--gamma5-dir", default="results/gamma_5",
                   help="directory of γ=5 arms; default is the n=16 duplicate-stream set")
    p.add_argument("--out", default="results/gamma5_onset.json",
                   help="do not overwrite gamma5_onset.json when analysing the unique-n set")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    gamma5_dir = ROOT / args.gamma5_dir
    dest = ROOT / args.out
    # Every richness level with a result set, and where its arms live. The registered grid filters to
    # its own gammas, so the off-design directories have to be named.
    sets = [
        (3.0, None), (10.0, None),
        (5.0, gamma5_dir),
        (30.0, ROOT / "results" / "gamma_ext"),
    ]
    out = {"generated_by": "scripts/analyse_gamma5.py",
           "gamma5_dir": str(gamma5_dir.relative_to(ROOT)),
           "question": "does gamma=5 locate the S-HL decorrelation onset better than one grid step",
           "by_gamma": {}}

    for gamma, arms_dir in sets:
        recs = G.load(arms_dir=arms_dir) if arms_dir else G.load()
        traj = per_arm(recs, gamma)
        if not traj:
            continue
        summ = summarize(traj)
        mean = {c: summ[c]["delta_mean"] for c in CORNERS if c in summ}
        row = {"n_arms_per_corner": {c: summ[c]["n_arms"] for c in CORNERS if c in summ},
               "delta_rho_c": mean,
               "delta_rho_c_sem": {c: summ[c]["delta_sem"] for c in CORNERS if c in summ},
               "S-HL": summ.get("S-HL"),
               "effects": effects(mean) if len(mean) == 4 else None}
        if row["effects"]:
            # Each effect is a half-difference of four independent cell means, so its standard
            # error is half the root-sum-square of theirs. Worth carrying: the interaction is the
            # smallest quantity in Figure 4 and the one a reader will want bounded.
            sems = [summ[c]["delta_sem"] for c in CORNERS]
            half = 0.5 * float(np.sqrt(sum(s ** 2 for s in sems)))
            row["effect_sem"] = half
            # The additive prediction error is lh + hl - ll - hh: four means, unit weights.
            row["additive_error_sem"] = float(np.sqrt(sum(s ** 2 for s in sems)))
            row["additive_error"] = (row["effects"]["measured_SHH"]
                                     - row["effects"]["additive_prediction_SHH"])
        out["by_gamma"][f"{gamma:g}"] = row

    # The onset question, stated as the bracket it narrows.
    shl = {g: r["delta_rho_c"]["S-HL"] for g, r in out["by_gamma"].items()
           if "S-HL" in r["delta_rho_c"]}
    ordered = sorted(((float(g), v) for g, v in shl.items()))
    negatives = [g for g, v in ordered if v < 0]
    positives = [g for g, v in ordered if v >= 0]
    if negatives and positives:
        lo = max(g for g in positives if g < min(negatives))
        hi = min(negatives)
        out["onset"] = {"last_non_negative": lo, "first_negative": hi,
                        "bracket_factor": round(hi / lo, 3),
                        "previous_bracket_factor": round(10.0 / 3.0, 3)}
    out["S-HL_series"] = {f"{g:g}": v for g, v in ordered}

    dest.write_text(json.dumps(out, indent=2, default=float) + "\n")

    print("Δρ_c by corner (generic ensemble, first to last boundary):\n")
    hdr = f"{'γ₀':>6}  " + "  ".join(f"{c:>9}" for c in CORNERS) + "   n/corner"
    print(hdr)
    for g in sorted(out["by_gamma"], key=float):
        r = out["by_gamma"][g]
        cells = "  ".join(f"{r['delta_rho_c'].get(c, float('nan')):+9.4f}" for c in CORNERS)
        ns = sorted(set(r["n_arms_per_corner"].values()))
        print(f"{float(g):>6g}  {cells}   {ns}")

    print("\nS-HL, the decorrelation corner:")
    for g in sorted(out["by_gamma"], key=float):
        s = out["by_gamma"][g].get("S-HL")
        if not s:
            continue
        print(f"  γ₀={float(g):<5g} Δρ_c = {s['delta_mean']:+.4f} ± {s['delta_sem']:.4f}   "
              f"declining on {s['fraction_declining'] * s['n_arms']:.0f}/{s['n_arms']} arms, "
              f"sign-test p={s['sign_test_p']:.3g}, median Spearman {s['spearman_median']:+.3f}")

    print("\n2×2 main effects and interaction:")
    for g in sorted(out["by_gamma"], key=float):
        e = out["by_gamma"][g].get("effects")
        if not e:
            continue
        r = out["by_gamma"][g]
        s, se = r["effect_sem"], r["additive_error_sem"]
        ok = "resolved" if abs(r["additive_error"]) >= 2 * se else "not resolved"
        print(f"  γ₀={float(g):<5g} readout {e['readout']:+.4f}  feature {e['feature']:+.4f}  "
              f"interaction {e['interaction']:+.4f} (±{s:.4f})   "
              f"additive error {r['additive_error']:+.4f} (±{se:.4f}, {ok})")

    if "onset" in out:
        o = out["onset"]
        print(f"\nonset bracket: last non-negative γ₀={o['last_non_negative']:g}, "
              f"first negative γ₀={o['first_negative']:g} "
              f"-- factor {o['bracket_factor']:g}, was {o['previous_bracket_factor']:g}")
    print(f"\nwrote {dest.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
