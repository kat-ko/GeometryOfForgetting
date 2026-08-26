"""How many independent draws each claim actually rests on.

`audit_unique_n.py` established the *count*: on the registered grid, 40 files per cell are
8 unique seeds and five unused copies. This script asks the next question, which the count
alone does not answer — **what is the sampling unit, and how much does the answer move when
the unit changes?**

Three units exist in this design and they are not interchangeable:

* **initialisation** (`seed` → `paired_init`), shared across conditions so that a corner
  contrast is not competing with initialisation variance;
* **arrangement** (`stream_id` → `stream_rng`), which draws the manifolds and the stream;
* **the arm**, the crossed (arrangement, initialisation) cell that a file corresponds to.

A sign test or SEM over arms assumes the arms are independent. In the RNG-fixed arms they
are crossed, not independent: `gamma_5_n40` and `width_g10_n40` are 5 arrangements × 8
initialisations, so 40 arms carry at most 5 independent arrangements and 8 independent
initialisations. `unconfound_k3` is the other shape — 40 initialisations × 3 arrangements —
so it estimates the initialisation direction well and the arrangement direction barely.

What this script writes down, per claim:

1. the resolution ceiling of a sign test at each n we have (at n = 4 no result can reach
   α = 0.05 however unanimous; at n = 8 one flipped seed takes p from 0.008 to 0.070);
2. the same effect re-tested with the arrangement, and then the initialisation, as the unit;
3. the between-arrangement spread of every quantity the paper quotes a within-arm SEM for,
   which is the variance component the registered grid could not see at all.

Writes `results/seed_security.json`. Reads stored geometry only; trains nothing.

    python scripts/audit_seed_security.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from audit_propagation import THREE, observations, pooled, sel  # noqa: E402
from fig4_corners_rho_c import CORNERS  # noqa: E402
from src.analysis import grid as G  # noqa: E402

OUT = ROOT / "results" / "seed_security.json"
RES = ROOT / "results"


# --- sampling units -----------------------------------------------------------

def keyed_rho(recs: list[dict], gamma: float) -> dict[str, list[dict]]:
    """`corner -> [{arrangement, init, delta, spearman}, ...]`, one entry per arm.

    Same per-arm trajectory as Figure 4, but carrying the two design keys, so the same
    numbers can be re-pooled with either as the sampling unit.
    """
    out: dict[str, list[dict]] = defaultdict(list)
    for r in recs:
        spec = r["spec"]
        if spec["gamma_0"] != gamma:
            continue
        pts = sorted((g["boundary"], g["rho_c_signed"]) for g in r["geometry"]
                     if g["ensemble"] == "generic")
        if len(pts) < 4:
            continue
        rho = stats.spearmanr([b for b, _ in pts], [x for _, x in pts]).statistic
        if not np.isfinite(rho):
            continue
        out[spec["condition"]].append({
            "arrangement": spec.get("stream_id", 0),
            "init": spec["seed"],
            "delta": float(pts[-1][1] - pts[0][1]),
            "spearman": float(rho),
        })
    return out


def sign_test(values: list[float]) -> dict:
    """Two-sided sign test on 'is this negative', with the ceiling it is measured against."""
    v = [x for x in values if np.isfinite(x)]
    n = len(v)
    neg = sum(1 for x in v if x < 0)
    return {
        "n": n,
        "n_declining": neg,
        "p": float(stats.binomtest(neg, n, 0.5).pvalue) if n else None,
        "p_floor": float(stats.binomtest(n, n, 0.5).pvalue) if n else None,
        "p_one_flip": float(stats.binomtest(max(n - 1, 0), n, 0.5).pvalue) if n else None,
        "can_reach_005": bool(n and stats.binomtest(n, n, 0.5).pvalue <= 0.05),
    }


def group_means(entries: list[dict], key: str, field: str = "delta") -> dict[int, float]:
    g: dict[int, list[float]] = defaultdict(list)
    for e in entries:
        g[e[key]].append(e[field])
    return {k: float(np.mean(v)) for k, v in sorted(g.items())}


def t_ci(values: list[float]) -> list[float] | None:
    """Two-sided 95% t interval over the units given. n=3 and n=5 are small on purpose:
    the point is to show how wide the interval is once the unit is the one we want to
    generalise over, not to pretend the interval is narrow."""
    v = np.asarray(values, dtype=float)
    if v.size < 2:
        return None
    half = float(stats.t.ppf(0.975, v.size - 1) * v.std(ddof=1) / np.sqrt(v.size))
    return [float(v.mean() - half), float(v.mean() + half)]


def spread_stats(values: list[float]) -> dict:
    v = np.asarray(values, dtype=float)
    ci = t_ci(list(v))
    return {
        "n_units": int(v.size),
        "mean": float(v.mean()),
        "min": float(v.min()),
        "max": float(v.max()),
        "sd": float(v.std(ddof=1)) if v.size > 1 else None,
        "sem": float(v.std(ddof=1) / np.sqrt(v.size)) if v.size > 1 else None,
        "t_ci_95": ci,
        "ci_excludes_zero": None if ci is None else bool(ci[0] > 0 or ci[1] < 0),
        "sign_stable": bool(np.all(v > 0) or np.all(v < 0)),
    }


def three_ways(entries: list[dict]) -> dict:
    """The same effect with the arm, the arrangement, and the initialisation as the unit."""
    by_arr = group_means(entries, "arrangement")
    by_init = group_means(entries, "init")
    arm_delta = [e["delta"] for e in entries]
    return {
        "mean_delta": float(np.mean(arm_delta)),
        "arm_unit": {
            "sign_test_on_spearman": sign_test([e["spearman"] for e in entries]),
            "sem_assuming_independent_arms": float(
                np.std(arm_delta, ddof=1) / np.sqrt(len(arm_delta))),
        },
        "arrangement_unit": {
            "n": len(by_arr),
            "means": by_arr,
            "spread": spread_stats(list(by_arr.values())),
            "sign_test": sign_test(list(by_arr.values())),
            "per_arrangement_arm_signs": {
                str(a): sign_test([e["spearman"] for e in entries if e["arrangement"] == a])
                for a in sorted({e["arrangement"] for e in entries})
            },
        },
        "init_unit": {
            "n": len(by_init),
            "spread": spread_stats(list(by_init.values())),
            "sign_test": sign_test(list(by_init.values())),
        },
    }


# --- 1. the grid: are the copies literally copies? ----------------------------

def grid_duplication() -> dict:
    recs = [r for r in G.load() if r["spec"].get("a", 0) == 0]
    ent = keyed_rho(recs, 10.0)
    worst = 0.0
    cells = {}
    for c in CORNERS:
        by_init: dict[int, list[float]] = defaultdict(list)
        for e in ent[c]:
            by_init[e["init"]].append(e["delta"])
        spreads = [max(v) - min(v) for v in by_init.values()]
        copies = sorted({len(v) for v in by_init.values()})
        worst = max(worst, max(spreads))
        cells[c] = {"unique_inits": len(by_init), "copies_per_init": copies,
                    "max_spread_within_init": float(max(spreads))}
    return {
        "question": "are the five stream_id files per seed identical, or merely similar?",
        "gamma": 10.0,
        "by_corner": cells,
        "worst_spread_across_copies": float(worst),
        "verdict": ("identical to floating point: the five copies of each seed carry the "
                    "same Δρ_c, so the extra files add no information"
                    if worst < 1e-12 else
                    "copies differ — re-check the duplication claim"),
        "arms_on_disk_a0": len(recs),
        "sign_test_n_on_grid": 8,
    }


# --- 2. sign-test ceilings ----------------------------------------------------

def ceilings() -> dict:
    out = {}
    for n in (4, 5, 8, 16, 40):
        out[str(n)] = {
            "p_if_unanimous": float(stats.binomtest(n, n, 0.5).pvalue),
            "p_if_one_flips": float(stats.binomtest(n - 1, n, 0.5).pvalue),
            "can_reach_005_when_unanimous": bool(
                stats.binomtest(n, n, 0.5).pvalue <= 0.05),
            "still_005_after_one_flip": bool(
                stats.binomtest(n - 1, n, 0.5).pvalue <= 0.05),
        }
    return {
        "question": "what is the best a two-sided sign test can do at each n we have?",
        "by_n": out,
        "reading": ("n=4 (γ=30 probe, duplicate-stream γ=5) cannot reach α=0.05 however "
                    "unanimous. n=8 (registered grid) reaches 0.0078 when unanimous and "
                    "0.070 — above 0.05 — as soon as one seed flips, so every 8/8 result "
                    "on the grid is one seed from not clearing."),
    }


# --- 3. the RNG-fixed arms, re-pooled by unit ---------------------------------

def onset_gamma5() -> dict:
    recs = G.load(arms_dir=RES / "gamma_5_n40")
    ent = keyed_rho(recs, 5.0)
    return {
        "question": "the γ=5 onset at 5 arrangements × 8 initialisations, by unit",
        "directory": "results/gamma_5_n40",
        "corners": {c: three_ways(ent[c]) for c in CORNERS if ent.get(c)},
    }


def width_by_arrangement() -> dict:
    recs = G.load(arms_dir=RES / "width_g10_n40")
    rows = observations(recs)
    per_N = {}
    for n in (150, 600):
        arrs = sorted({r["stream_id"] for r in rows})
        floors = {}
        for a in arrs:
            p = pooled(sel(rows, conditions=THREE, gamma=10.0, N=n, lag=12, stream_id=a))
            if p is not None:
                floors[a] = abs(p["floors_signed"])
        allp = pooled(sel(rows, conditions=THREE, gamma=10.0, N=n, lag=12))
        vals = list(floors.values())
        per_N[str(n)] = {
            "pooled_floors": abs(allp["floors_signed"]),
            "per_arrangement_floors": floors,
            "spread_pct_across_arrangements":
                100 * (max(vals) - min(vals)) / float(np.mean(vals)),
            "spread": spread_stats(vals),
        }
    a, b = (per_N["150"]["pooled_floors"], per_N["600"]["pooled_floors"])
    width_gap = 100 * abs(a - b) / float(np.mean([a, b]))
    worst_arr = max(per_N[k]["spread_pct_across_arrangements"] for k in per_N)

    # The contrast is paired: both widths were run on the same five arrangements, so the
    # arrangement offsets that dominate the marginal spread cancel. This is the fair test.
    arrs = sorted(per_N["150"]["per_arrangement_floors"])
    diffs = [per_N["600"]["per_arrangement_floors"][a_]
             - per_N["150"]["per_arrangement_floors"][a_] for a_ in arrs]
    paired = spread_stats(diffs)
    paired_sign = sign_test([-d for d in diffs])  # 'is N=600 the larger loss'
    return {
        "question": ("is the 0.62% width gap larger than the arrangement-to-arrangement "
                     "spread at fixed width, and does it survive pairing?"),
        "directory": "results/width_g10_n40",
        "by_N": per_N,
        "width_gap_pct": width_gap,
        "worst_within_width_arrangement_spread_pct": worst_arr,
        "arrangement_spread_over_width_gap": float(worst_arr / width_gap),
        "paired_within_arrangement": {
            "differences_floors_600_minus_150": dict(zip(map(str, arrs), diffs)),
            "spread": paired,
            "n_arrangements_with_600_larger_loss": paired_sign["n_declining"],
            "sign_test_p": paired_sign["p"],
            "sign_test_p_floor": paired_sign["p_floor"],
        },
        "reading": ("marginally, arrangement moves the magnitude "
                    f"{worst_arr / width_gap:.0f}× further than width does. Paired within "
                    "arrangement the direction is consistent (N=600 the slightly larger "
                    "loss in every arrangement) but the interval includes zero and five "
                    "arrangements cannot reach α=0.05, so 0.62% is a bound either way"),
    }


def k3_by_arrangement() -> dict:
    """K=3 is the only set with a genuine arrangement axis. What does it say about the SEMs?"""
    k3 = json.loads((RES / "unconfound_k3.json").read_text())
    grid = json.loads((RES / "gamma5_n40_onset.json").read_text())["by_gamma"]["10"]
    u8 = json.loads((RES / "unique_n8.json").read_text())["fig4"]["10"]["cells"]
    recs = G.load(arms_dir=RES / "unconfound_k3")
    ent = keyed_rho(recs, 10.0)

    corners = {}
    for c in CORNERS:
        by_arr = group_means(ent[c], "arrangement")
        within = [k3["by_arrangement"][str(a)]["gamma"]["10"]["rho_c"]["cells"][c]["delta_sem"]
                  for a in sorted(by_arr)]
        sp = spread_stats(list(by_arr.values()))
        corners[c] = {
            "per_arrangement_mean": by_arr,
            "spread": sp,
            "median_within_arrangement_sem": float(np.median(within)),
            "between_over_within_sem": (
                None if sp["sem"] is None
                else float(sp["sem"] / np.median(within))),
            # The grid's 8 unique arms each carry their own arrangement (confounded with
            # init), so SEM8 should already be an arrangement-scale interval. Is it?
            "grid_sem_unique_n8": u8[c]["sem_unique"],
            "grid_sem_over_between_arrangement_sem": (
                None if sp["sem"] is None else float(u8[c]["sem_unique"] / sp["sem"])),
            "grid_value_at_unique_n8": grid["delta_rho_c"][c],
            "grid_value_inside_arrangement_range": bool(
                sp["min"] <= grid["delta_rho_c"][c] <= sp["max"]),
            # 40 inits per arrangement, so the init axis is the well-sampled one here
            "per_arrangement_arm_sign_tests": {
                str(a): sign_test([e["spearman"] for e in ent[c] if e["arrangement"] == a])
                for a in sorted(by_arr)
            },
        }

    eff = {}
    for name in ("readout", "feature", "interaction"):
        vals = [k3["by_arrangement"][a]["gamma"]["10"]["rho_c"]["effects"][name]
                for a in sorted(k3["by_arrangement"])]
        eff[name] = {
            "per_arrangement": {a: v for a, v in zip(sorted(k3["by_arrangement"]), vals)},
            "spread": spread_stats(vals),
            "grid_value_at_unique_n8": grid["effects"][name],
        }

    # The headline gain, per arrangement, at both richnesses.
    shh = {}
    for g in ("1", "10"):
        vals = [k3["by_arrangement"][a]["gamma"][g]["S-HH"]["floors"]
                for a in sorted(k3["by_arrangement"])]
        shh[g] = {
            "per_arrangement_floors": dict(zip(sorted(k3["by_arrangement"]), vals)),
            "spread": spread_stats(vals),
        }

    # Which initialisations behave the same way in every arrangement? The init axis is
    # 40 wide here, so this is the one consistency question K=3 can answer well.
    shl = ent["S-HL"]
    by_init: dict[int, list[float]] = defaultdict(list)
    for e in shl:
        by_init[e["init"]].append(e["spearman"])
    unanimous = sum(1 for v in by_init.values() if all(x < 0 for x in v))
    return {
        "S-HH_gain_by_arrangement": shh,
        "question": ("with arrangement as a real factor, how big is the component the "
                     "registered grid could not see?"),
        "directory": "results/unconfound_k3",
        "n_arrangements": k3["n_arrangements"],
        "n_inits": len(by_init),
        "corners": corners,
        "effects": eff,
        "S-HL_inits_declining_in_all_arrangements": unanimous,
        "S-HL_inits_total": len(by_init),
        "caveat": ("three arrangements estimate a standard deviation to about ±50% "
                   "(relative SE of an SD at n=3 is 1/sqrt(2(n-1)) = 0.50), so the "
                   "between-arrangement numbers below are themselves coarse; they bound "
                   "the component rather than measure it"),
    }


# --- 4. the floors everything is denominated in -------------------------------

def floor_security() -> dict:
    mn = json.loads((RES / "measurement_null.json").read_text())
    n_seeds = mn.get("n_seeds") or mn.get("seeds") or 4
    if isinstance(n_seeds, list):
        n_seeds = len(n_seeds)
    rel_se = 1.0 / np.sqrt(2 * (n_seeds - 1))
    return {
        "question": "how well is the ruler itself known?",
        "n_measurement_seeds": int(n_seeds),
        "relative_se_of_a_cv": float(rel_se),
        "factor_uncertainty_on_a_floor": float(np.exp(rel_se)),
        "ratio_between_plus_and_minus_1se_ends": float(np.exp(2 * rel_se)),
        "reading": ("a CV from 4 seeds carries ~41% relative standard error, so any floor "
                    "is good to about ×/÷1.5 and margins under ~40% of a floor are inside "
                    "the ruler's own uncertainty"),
    }


def main() -> None:
    out = {
        "generated_by": "scripts/audit_seed_security.py",
        "grid_duplication": grid_duplication(),
        "sign_test_ceilings": ceilings(),
        "gamma5_onset": onset_gamma5(),
        "width": width_by_arrangement(),
        "k3_arrangement_component": k3_by_arrangement(),
        "floors": floor_security(),
    }
    OUT.write_text(json.dumps(out, indent=2, default=float) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}\n")

    g = out["grid_duplication"]
    print(f"grid: {g['arms_on_disk_a0']} a=0 arms; worst Δρ_c spread across the five "
          f"copies of a seed = {g['worst_spread_across_copies']:.2e}")
    print("sign-test ceilings:")
    for n, v in out["sign_test_ceilings"]["by_n"].items():
        print(f"  n={n:>2}  unanimous p={v['p_if_unanimous']:.4f}  "
              f"one flip p={v['p_if_one_flips']:.4f}  "
              f"reaches 0.05: {v['can_reach_005_when_unanimous']}")

    print("\nγ=5 onset, S-HL, by unit:")
    s = out["gamma5_onset"]["corners"]["S-HL"]
    print(f"  arms      mean {s['mean_delta']:+.4f}  "
          f"{s['arm_unit']['sign_test_on_spearman']['n_declining']}/"
          f"{s['arm_unit']['sign_test_on_spearman']['n']}  "
          f"p={s['arm_unit']['sign_test_on_spearman']['p']:.2e}  "
          f"SEM {s['arm_unit']['sem_assuming_independent_arms']:.4f}")
    au = s["arrangement_unit"]
    print(f"  arrangements ({au['n']})  "
          + "  ".join(f"{k}:{v:+.4f}" for k, v in au["means"].items()))
    print(f"     per-arrangement arm signs: "
          + "  ".join(f"{k}:{v['n_declining']}/{v['n']}"
                      for k, v in au["per_arrangement_arm_signs"].items()))
    print(f"     SEM over arrangements {au['spread']['sem']:.4f}  "
          f"sign test {au['sign_test']['n_declining']}/{au['sign_test']['n']} "
          f"p={au['sign_test']['p']:.3f} (ceiling {au['sign_test']['p_floor']:.3f})")
    print(f"     95% t CI over arrangements "
          f"[{au['spread']['t_ci_95'][0]:+.4f}, {au['spread']['t_ci_95'][1]:+.4f}]  "
          f"excludes zero: {au['spread']['ci_excludes_zero']}")
    iu = s["init_unit"]
    print(f"  inits ({iu['n']})  SEM {iu['spread']['sem']:.4f}  "
          f"sign test {iu['sign_test']['n_declining']}/{iu['sign_test']['n']} "
          f"p={iu['sign_test']['p']:.4f}  95% t CI "
          f"[{iu['spread']['t_ci_95'][0]:+.4f}, {iu['spread']['t_ci_95'][1]:+.4f}]")

    w = out["width"]
    print(f"\nwidth: gap between widths {w['width_gap_pct']:.2f}%; "
          f"worst spread across arrangements at fixed width "
          f"{w['worst_within_width_arrangement_spread_pct']:.2f}% "
          f"({w['arrangement_spread_over_width_gap']:.0f}× the width gap)")
    pw = w["paired_within_arrangement"]
    print("  paired per arrangement (600 − 150, floors): "
          + "  ".join(f"{k}:{v:+.2f}" for k, v in
                      pw["differences_floors_600_minus_150"].items()))
    print(f"  mean {pw['spread']['mean']:+.2f} floors  95% t CI "
          f"[{pw['spread']['t_ci_95'][0]:+.2f}, {pw['spread']['t_ci_95'][1]:+.2f}]  "
          f"excludes zero: {pw['spread']['ci_excludes_zero']}  "
          f"{pw['n_arrangements_with_600_larger_loss']}/5 same direction "
          f"(p={pw['sign_test_p']:.3f}, ceiling {pw['sign_test_p_floor']:.3f})")

    k = out["k3_arrangement_component"]
    print(f"\nK=3 ({k['n_arrangements']} arrangements × {k['n_inits']} inits), γ=10 Δρ_c:")
    for c, v in k["corners"].items():
        sp = v["spread"]
        print(f"  {c}  " + " ".join(f"{x:+.4f}" for x in v["per_arrangement_mean"].values())
              + f"   between-arr SEM {sp['sem']:.4f} vs within-arr SEM "
                f"{v['median_within_arrangement_sem']:.4f} "
                f"({v['between_over_within_sem']:.1f}×)  "
                f"sign-stable {sp['sign_stable']}  "
                f"grid {v['grid_value_at_unique_n8']:+.4f} in range "
                f"{v['grid_value_inside_arrangement_range']}  "
                f"grid SEM8 {v['grid_sem_unique_n8']:.4f} "
                f"({v['grid_sem_over_between_arrangement_sem']:.2f}× between-arr SEM)")
    for name, v in k["effects"].items():
        print(f"  effect {name:<11} " + " ".join(f"{x:+.4f}" for x in v["per_arrangement"].values())
              + f"   sign-stable {v['spread']['sign_stable']}  "
                f"grid {v['grid_value_at_unique_n8']:+.4f}")
    print(f"  S-HL inits declining in all {k['n_arrangements']} arrangements: "
          f"{k['S-HL_inits_declining_in_all_arrangements']}/{k['S-HL_inits_total']}")
    for g, v in k["S-HH_gain_by_arrangement"].items():
        sp = v["spread"]
        print(f"  S-HH gain at γ={g:<2} "
              + " ".join(f"{x:+.2f}" for x in v["per_arrangement_floors"].values())
              + f" floors   sign-stable {sp['sign_stable']}  "
                f"range [{sp['min']:+.2f}, {sp['max']:+.2f}]")

    f = out["floors"]
    print(f"\nfloors: {f['n_measurement_seeds']} measurement seeds → "
          f"{100 * f['relative_se_of_a_cv']:.0f}% relative SE on a CV "
          f"(×/÷{f['factor_uncertainty_on_a_floor']:.2f} on a floor)")


if __name__ == "__main__":
    main()
