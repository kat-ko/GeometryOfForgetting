"""Score the γ=5 8-arrangement run against its pre-commit.

`results/gamma5_k8_precommit.json` was written before any of these arms existed. The sampling
unit is arrangement: 8 inits are averaged first, then the eight arrangement means are the
observations. The decision rule is in the pre-commit and is not reopened here.

Also checks that arrangements 0–4 reproduce `results/gamma_5_n40/` on the measured geometry.
Those specs are identical; a mismatch means the measurement path moved and invalidates the
comparison rather than the new arms.

Writes `results/gamma5_k8.json`. Does not train.

    python scripts/analyse_gamma5_k8.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from audit_seed_security import group_means, keyed_rho, sign_test, spread_stats, three_ways  # noqa: E402
from fig4_corners_rho_c import CORNERS  # noqa: E402
from src.analysis import grid as G  # noqa: E402

PRE = json.loads((ROOT / "results" / "gamma5_k8_precommit.json").read_text())
OUT = ROOT / "results" / "gamma5_k8.json"
DIR_K8 = ROOT / "results" / "gamma_5_k8"
DIR_N40 = ROOT / "results" / "gamma_5_n40"


def _geom_signature(rec: dict) -> list[tuple]:
    return sorted(
        (g["module"], g["ensemble"], g["task"], g["boundary"],
         round(float(g["rho_c_signed"]), 12), round(float(g["alpha"]), 12))
        for g in rec["geometry"]
    )


def overlap_check(k8: list[dict], n40: list[dict]) -> dict:
    """Arrangements 0–4 × seeds 0–7 must match the earlier 5×8 set on measured geometry."""
    def key(r):
        s = r["spec"]
        return (s["condition"], s["stream_id"], s["seed"])

    a = {key(r): r for r in k8 if r["spec"]["stream_id"] < 5}
    b = {key(r): r for r in n40}
    missing_k8 = sorted(set(b) - set(a), key=str)
    missing_n40 = sorted(set(a) - set(b), key=str)
    mismatches = []
    n_match = 0
    worst = 0.0
    for k, rec_b in b.items():
        rec_a = a.get(k)
        if rec_a is None:
            continue
        sa, sb = _geom_signature(rec_a), _geom_signature(rec_b)
        if sa == sb:
            n_match += 1
            continue
        # Fall back: max |Δρ_c| difference if the table lengths match.
        da = [x[4] for x in sa]
        db = [x[4] for x in sb]
        if len(da) == len(db):
            d = float(max(abs(x - y) for x, y in zip(da, db)))
            worst = max(worst, d)
        else:
            d = float("inf")
            worst = d
        mismatches.append({"key": k, "max_abs_rho_c_diff": d})
    ok = not missing_k8 and not missing_n40 and not mismatches
    return {
        "n_overlap_expected": len(b),
        "n_identical": n_match,
        "n_mismatched": len(mismatches),
        "missing_in_k8": [str(x) for x in missing_k8[:8]],
        "missing_in_n40": [str(x) for x in missing_n40[:8]],
        "worst_rho_c_diff": worst,
        "ok": ok,
        "verdict": ("identical: arrangements 0–4 reproduce gamma_5_n40"
                    if ok else
                    "MISMATCH — the measurement path moved; do not score the new arms "
                    "against the 5-arrangement set"),
    }


def decide(n_neg: int, ci_excludes_zero: bool) -> dict:
    rule = PRE["decision_rule"]
    if n_neg >= 7 and ci_excludes_zero:
        key = "stands_at_arrangement_level"
    elif n_neg >= 7:
        key = "suggestive_only"
    else:
        key = "arrangement_dependent"
    return {"outcome": key, "licenses": rule[key]}


def main() -> None:
    recs = G.load(arms_dir=DIR_K8)
    n40 = G.load(arms_dir=DIR_N40) if DIR_N40.is_dir() else []
    n_files = sum(1 for _ in DIR_K8.glob("*.json"))
    overlap = overlap_check(recs, n40)

    ent = keyed_rho(recs, 5.0)
    corners = {c: three_ways(ent[c]) for c in CORNERS if ent.get(c)}
    shl = corners["S-HL"]
    au = shl["arrangement_unit"]
    means = au["means"]
    n_neg = au["sign_test"]["n_declining"]
    ci = au["spread"]["t_ci_95"]
    excludes = au["spread"]["ci_excludes_zero"]
    verdict = decide(n_neg, bool(excludes))

    lazy = [0.0051, 0.0101]
    in_lazy = {
        str(a): bool(lazy[0] <= m <= lazy[1] or (m > 0 and m <= lazy[1] + 0.001))
        for a, m in means.items()
    }

    out = {
        "generated_by": "scripts/analyse_gamma5_k8.py",
        "directory": "results/gamma_5_k8",
        "precommit": PRE,
        "n_files": n_files,
        "n_usable": len(recs),
        "n_arrangements": au["n"],
        "n_inits": shl["init_unit"]["n"],
        "overlap_with_gamma_5_n40": overlap,
        "corners": corners,
        "S-HL_arrangement_means": {str(k): v for k, v in means.items()},
        "S-HL_n_arrangements_negative": n_neg,
        "S-HL_t_ci_95": ci,
        "S-HL_ci_excludes_zero": excludes,
        "S-HL_sign_test": au["sign_test"],
        "S-HL_in_lazy_arm_band": in_lazy,
        "verdict": verdict,
        "paper_sentence_this_licenses": verdict["licenses"],
    }
    OUT.write_text(json.dumps(out, indent=2, default=float) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}  ({out['n_usable']}/{out['n_files']} usable)")
    print(f"overlap with gamma_5_n40: {overlap['verdict']}  "
          f"({overlap['n_identical']}/{overlap['n_overlap_expected']} identical, "
          f"worst |Δρ_c| {overlap['worst_rho_c_diff']})")
    print("S-HL arrangement means:")
    for a, m in means.items():
        signs = au["per_arrangement_arm_signs"][str(a)]
        print(f"  arr {a}: {m:+.4f}  arms declining "
              f"{signs['n_declining']}/{signs['n']}")
    print(f"  {n_neg}/8 negative  95% t CI [{ci[0]:+.4f}, {ci[1]:+.4f}]  "
          f"excludes zero: {excludes}")
    print(f"  sign-test p={au['sign_test']['p']:.4g}  "
          f"(ceiling {au['sign_test']['p_floor']:.4g})")
    print(f"verdict: {verdict['outcome']}")
    print(f"  {verdict['licenses']}")


if __name__ == "__main__":
    main()
