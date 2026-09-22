"""S-HH vs S-LH at unique n=8: input drift at a fixed dichotomy.

Both corners repeat y_0 for all 16 tasks (P=16 Hamming-0). They differ only in
whether the arrangement drifts (s_f=0.9) or jumps (s_f=0.1). Zero training.

    python scripts/analyse_shh_vs_slh.py
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from audit_unique_n import unique_obs, unique_recs  # noqa: E402
from fig4_corners_rho_c import effects, per_arm, summarize as summarize_rho  # noqa: E402
from src.analysis import grid as G  # noqa: E402
from src.analysis.attribution import FACTORS  # noqa: E402
from src.analysis.ledger import (  # noqa: E402
    BOOTSTRAP_N,
    BOOTSTRAP_SEED,
    MIN_FLOORS,
    observations,
)

OUT = ROOT / "results" / "shh_vs_slh.json"
A, B = "S-HH", "S-LH"
GAMMAS = G.REGISTERED_GAMMAS
LAG12, TASK0 = 12, 0
LAGS = (4, 8, 12, 15)
POSITIONS = (0, 4, 8)
MATCHED_LAG = 4


def _fmt(x) -> float | None:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return None
    return float(x)


def summarize(vals) -> dict:
    v = np.asarray(list(vals), dtype=float)
    v = v[np.isfinite(v)]
    n = int(v.size)
    if n == 0:
        return {"n": 0, "mean": None, "sd": None, "sem": None, "min": None, "max": None}
    sd = float(v.std(ddof=1)) if n > 1 else 0.0
    return {
        "n": n,
        "mean": _fmt(v.mean()),
        "sd": _fmt(sd),
        "sem": _fmt(sd / np.sqrt(n)) if n else None,
        "min": _fmt(v.min()),
        "max": _fmt(v.max()),
        "n_positive": int(np.sum(v > 0)),
        "n_negative": int(np.sum(v < 0)),
    }


def floors(dlog: float) -> float | None:
    if dlog is None or not np.isfinite(dlog):
        return None
    return float(np.sign(dlog) * G.floors(dlog, "alpha"))


def bootstrap_mean_ci(vals, n: int = BOOTSTRAP_N) -> list[float | None]:
    v = np.asarray(vals, float)
    v = v[np.isfinite(v)]
    if v.size < 2:
        return [None, None]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(n)]
    return [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]


def contrast(hh, lh) -> dict:
    x = np.asarray(hh, float)
    y = np.asarray(lh, float)
    x, y = x[np.isfinite(x)], y[np.isfinite(y)]
    d = float(x.mean() - y.mean()) if x.size and y.size else None
    out = {
        "n_hh": int(x.size),
        "n_lh": int(y.size),
        "diff_hh_minus_lh": _fmt(d),
        "diff_floors": floors(d) if d is not None else None,
        "welch_p": None,
        "mannwhitney_p": None,
        "wilcoxon_paired_p": None,
        "n_paired": 0,
        "ci_diff": [None, None],
    }
    if x.size >= 2 and y.size >= 2:
        out["welch_p"] = _fmt(stats.ttest_ind(x, y, equal_var=False).pvalue)
        out["mannwhitney_p"] = _fmt(stats.mannwhitneyu(x, y, alternative="two-sided").pvalue)
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        draws = []
        for _ in range(BOOTSTRAP_N):
            draws.append(x[rng.integers(0, len(x), len(x))].mean()
                         - y[rng.integers(0, len(y), len(y))].mean())
        out["ci_diff"] = [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]
    return out


def contrast_paired(hh_by_seed: dict, lh_by_seed: dict) -> dict:
    keys = sorted(set(hh_by_seed) & set(lh_by_seed))
    if len(keys) < 3:
        return {"n_paired": len(keys), "wilcoxon_paired_p": None, "mean_paired_diff": None}
    d = np.array([hh_by_seed[k] - lh_by_seed[k] for k in keys], float)
    try:
        p = float(stats.wilcoxon(d).pvalue)
    except ValueError:
        p = None
    return {
        "n_paired": len(keys),
        "mean_paired_diff": _fmt(d.mean()),
        "wilcoxon_paired_p": _fmt(p),
        "n_hh_greater": int(np.sum(d > 0)),
    }


def cell_pack(vals, *, as_floors: bool = False) -> dict:
    s = summarize(vals)
    s["ci"] = bootstrap_mean_ci(vals)
    if as_floors and s["mean"] is not None:
        s["mean_floors"] = floors(s["mean"])
        s["ci_floors"] = [floors(c) if c is not None else None for c in s["ci"]]
        s["resolved_gain"] = bool(
            s["mean_floors"] is not None
            and s["mean_floors"] >= MIN_FLOORS
            and s["ci"][0] is not None
            and s["ci"][0] > 0
        )
        s["resolved_loss"] = bool(
            s["mean_floors"] is not None
            and s["mean_floors"] <= -MIN_FLOORS
            and s["ci"][1] is not None
            and s["ci"][1] < 0
        )
    return s


def pick_obs(obs, gamma, cond, task, lag) -> list:
    return [o for o in obs
            if o["gamma"] == gamma and o["condition"] == cond
            and o["task"] == task and o["lag"] == lag]


def arm_generic(r: dict, module: str = "A") -> list[dict]:
    pts = [g for g in r["geometry"]
           if g["ensemble"] == "generic" and g["module"] == module]
    return sorted(pts, key=lambda g: g["boundary"])


def last_probe_margin(r: dict, module: str = "A") -> float | None:
    checks = r.get("manipulation_checks") or []
    if not checks:
        return None
    last = max(checks, key=lambda c: c["boundary"])
    d = last.get("probe_decodability", {}).get(module, {})
    m = d.get("margin")
    return float(m) if m is not None else None


def first_probe_margin(r: dict, module: str = "A") -> float | None:
    checks = r.get("manipulation_checks") or []
    if not checks:
        return None
    first = min(checks, key=lambda c: c["boundary"])
    d = first.get("probe_decodability", {}).get(module, {})
    m = d.get("margin")
    return float(m) if m is not None else None


def mean_steps(r: dict) -> float | None:
    tasks = r.get("tasks") or []
    if not tasks:
        return None
    return float(np.mean([t["steps_taken"] for t in tasks]))


def task0_cfr(r: dict) -> float | None:
    pt = (r.get("forgetting") or {}).get("per_task") or {}
    v = pt.get("0") or pt.get(0)
    if not v:
        return None
    return float(v["CFr"])


def main() -> None:
    recs = unique_recs(G.load())
    recs_ab = [r for r in recs if r["spec"]["condition"] in (A, B)]
    obs = unique_obs(observations(recs, module="A"))
    by_gc = defaultdict(list)
    for r in recs_ab:
        by_gc[(r["spec"]["gamma_0"], r["spec"]["condition"])].append(r)

    retained_lag12 = {}
    terms_lag12 = {}
    for g in GAMMAS:
        packs = {}
        term_packs = {f: {} for f in FACTORS}
        by_seed = {}
        for cond in (A, B):
            rows = pick_obs(obs, g, cond, TASK0, LAG12)
            dlogs = [o["att"].dlog_alpha for o in rows]
            packs[cond] = cell_pack(dlogs, as_floors=True)
            packs[cond]["n_resolved_gain_arms"] = int(sum(
                floors(v) is not None and floors(v) >= MIN_FLOORS and v > 0
                for v in dlogs
            ))
            by_seed[cond] = {o["seed"]: o["att"].dlog_alpha for o in rows}
            for f in FACTORS:
                term_packs[f][cond] = cell_pack([o["att"].terms[f] for o in rows])
        c = contrast(list(by_seed[A].values()), list(by_seed[B].values()))
        c.update(contrast_paired(by_seed[A], by_seed[B]))
        retained_lag12[f"{g:g}"] = {
            A: packs[A], B: packs[B], "contrast": c,
            "reading": (
                "HH gain, LH loss" if packs[A].get("resolved_gain") and packs[B].get("resolved_loss")
                else "HH gain, LH unresolved" if packs[A].get("resolved_gain")
                else "neither resolved as gain"
            ),
        }
        terms_lag12[f"{g:g}"] = {}
        for f in FACTORS:
            hh = [o["att"].terms[f] for o in pick_obs(obs, g, A, TASK0, LAG12)]
            lh = [o["att"].terms[f] for o in pick_obs(obs, g, B, TASK0, LAG12)]
            terms_lag12[f"{g:g}"][f] = {
                A: term_packs[f][A], B: term_packs[f][B], "contrast": contrast(hh, lh),
            }

    lag_series = {}
    for g in (1.0, 10.0):
        lag_series[f"{g:g}"] = {}
        for lag in LAGS:
            packs = {cond: cell_pack(
                [o["att"].dlog_alpha for o in pick_obs(obs, g, cond, TASK0, lag)],
                as_floors=True,
            ) for cond in (A, B)}
            packs["contrast"] = contrast(
                [o["att"].dlog_alpha for o in pick_obs(obs, g, A, TASK0, lag)],
                [o["att"].dlog_alpha for o in pick_obs(obs, g, B, TASK0, lag)],
            )
            lag_series[f"{g:g}"][str(lag)] = packs

    position_series = {}
    for g in (1.0, 10.0):
        position_series[f"{g:g}"] = {}
        for task in POSITIONS:
            packs = {cond: cell_pack(
                [o["att"].dlog_alpha for o in pick_obs(obs, g, cond, task, MATCHED_LAG)],
                as_floors=True,
            ) for cond in (A, B)}
            packs["contrast"] = contrast(
                [o["att"].dlog_alpha for o in pick_obs(obs, g, A, task, MATCHED_LAG)],
                [o["att"].dlog_alpha for o in pick_obs(obs, g, B, task, MATCHED_LAG)],
            )
            position_series[f"{g:g}"][str(task)] = packs

    rho = {}
    for g in GAMMAS:
        traj = per_arm(recs, g)
        s = summarize_rho({c: traj[c] for c in (A, B) if c in traj})
        hh = [p[-1][1] - p[0][1] for p in traj.get(A, [])]
        lh = [p[-1][1] - p[0][1] for p in traj.get(B, [])]
        rho[f"{g:g}"] = {
            A: s.get(A),
            B: s.get(B),
            "contrast": contrast(hh, lh),
        }

    four_rho = {}
    for g in (3.0, 10.0):
        traj = per_arm(recs, g)
        s = summarize_rho(traj)
        mean = {c: s[c]["delta_mean"] for c in s}
        four_rho[f"{g:g}"] = {
            "cells": {c: {
                "delta_mean": s[c]["delta_mean"],
                "delta_sem": s[c]["delta_sem"],
                "n": s[c]["n_arms"],
                "spearman_median": s[c]["spearman_median"],
                "sign_test_p": s[c]["sign_test_p"],
                "label": {
                    "S-HH": "same task, drift",
                    "S-LH": "same task, jump",
                    "S-HL": "different tasks, drift",
                    "S-LL": "different tasks, jump",
                }[c],
            } for c in s},
            "effects_as_published": effects(mean),
            "same_task_mean": _fmt(0.5 * (mean["S-HH"] + mean["S-LH"])),
            "different_task_mean": _fmt(0.5 * (mean["S-HL"] + mean["S-LL"])),
            "drift_at_fixed_same_task": _fmt(mean["S-HH"] - mean["S-LH"]),
            "jump_minus_drift_at_fixed_same_task": _fmt(mean["S-LH"] - mean["S-HH"]),
        }

    generic_alpha = {}
    behavioral = {}
    probe = {}
    steps = {}
    for g in GAMMAS:
        generic_alpha[f"{g:g}"] = {}
        behavioral[f"{g:g}"] = {}
        probe[f"{g:g}"] = {}
        steps[f"{g:g}"] = {}
        for cond in (A, B):
            arms = by_gc[(g, cond)]
            dlog_gen, glue_delta, signed_delta = [], [], []
            cfr, cf, acc_final, t0_cfr = [], [], [], []
            p_first, p_last, st = [], [], []
            for r in arms:
                gen = arm_generic(r)
                if len(gen) >= 2 and gen[0]["alpha"] > 0 and gen[-1]["alpha"] > 0:
                    dlog_gen.append(float(np.log(gen[-1]["alpha"]) - np.log(gen[0]["alpha"])))
                    glue_delta.append(float(gen[-1]["rho_c_glue"] - gen[0]["rho_c_glue"]))
                    signed_delta.append(float(gen[-1]["rho_c_signed"] - gen[0]["rho_c_signed"]))
                fr = r.get("forgetting") or {}
                if "CFr" in fr:
                    cfr.append(float(fr["CFr"]))
                if "CF" in fr:
                    cf.append(float(fr["CF"]))
                if "final_mean_accuracy" in fr:
                    acc_final.append(float(fr["final_mean_accuracy"]))
                t0 = task0_cfr(r)
                if t0 is not None:
                    t0_cfr.append(t0)
                a0, a1 = first_probe_margin(r), last_probe_margin(r)
                if a0 is not None:
                    p_first.append(a0)
                if a1 is not None:
                    p_last.append(a1)
                ms = mean_steps(r)
                if ms is not None:
                    st.append(ms)
            generic_alpha[f"{g:g}"][cond] = {
                "dlog_alpha_generic": cell_pack(dlog_gen, as_floors=True),
                "delta_rho_c_glue": cell_pack(glue_delta),
                "delta_rho_c_signed": cell_pack(signed_delta),
            }
            behavioral[f"{g:g}"][cond] = {
                "CFr": cell_pack(cfr),
                "CF": cell_pack(cf),
                "final_mean_accuracy": cell_pack(acc_final),
                "task0_CFr": cell_pack(t0_cfr),
            }
            probe[f"{g:g}"][cond] = {
                "margin_first": cell_pack(p_first),
                "margin_last": cell_pack(p_last),
            }
            steps[f"{g:g}"][cond] = cell_pack(st)
        generic_alpha[f"{g:g}"]["contrast_dlog"] = contrast(
            [float(np.log(arm_generic(r)[-1]["alpha"]) - np.log(arm_generic(r)[0]["alpha"]))
             for r in by_gc[(g, A)] if len(arm_generic(r)) >= 2],
            [float(np.log(arm_generic(r)[-1]["alpha"]) - np.log(arm_generic(r)[0]["alpha"]))
             for r in by_gc[(g, B)] if len(arm_generic(r)) >= 2],
        )
        behavioral[f"{g:g}"]["contrast_CFr"] = contrast(
            [(r.get("forgetting") or {}).get("CFr") for r in by_gc[(g, A)]],
            [(r.get("forgetting") or {}).get("CFr") for r in by_gc[(g, B)]],
        )
        behavioral[f"{g:g}"]["contrast_task0_CFr"] = contrast(
            [task0_cfr(r) for r in by_gc[(g, A)]],
            [task0_cfr(r) for r in by_gc[(g, B)]],
        )

    headline = []
    for g in GAMMAS:
        row = retained_lag12[f"{g:g}"]
        headline.append({
            "gamma": g,
            "S-HH_floors": row[A].get("mean_floors"),
            "S-LH_floors": row[B].get("mean_floors"),
            "S-HH_resolved_gain": row[A].get("resolved_gain"),
            "S-LH_resolved_loss": row[B].get("resolved_loss"),
            "diff_floors": row["contrast"]["diff_floors"],
            "welch_p": row["contrast"]["welch_p"],
            "wilcoxon_paired_p": row["contrast"]["wilcoxon_paired_p"],
        })

    out = {
        "generated_by": "scripts/analyse_shh_vs_slh.py",
        "question": "At unique n=8, what does repeating the same dichotomy do when the arrangement drifts versus jumps?",
        "unit": "unique (arrangement, init) draw; n=8 per (γ, corner); module A",
        "no_new_training": True,
        "reframe": {
            "S-HH": "same task × 16, slowly drifting arrangement (AR(1) s_f=0.9)",
            "S-LH": "same task × 16, fresh arrangement each boundary (s_f=0.1)",
            "isolated": "input change at a fixed dichotomy. Not a similarity-level contrast.",
        },
        "n_unique_per_cell": 8,
        "gammas": list(GAMMAS),
        "headline_task0_lag12": headline,
        "retained_task0_lag12": retained_lag12,
        "terms_task0_lag12": terms_lag12,
        "lag_series_task0": lag_series,
        "position_series_matched_lag4": position_series,
        "rho_c_signed_generic": rho,
        "finding4_2x2_reread": four_rho,
        "generic_alpha": generic_alpha,
        "behavioral": behavioral,
        "probe_margin": probe,
        "mean_steps_per_task": steps,
        "gates": {"min_floors": MIN_FLOORS, "bootstrap_n": BOOTSTRAP_N,
                  "bootstrap_seed": BOOTSTRAP_SEED},
    }
    OUT.write_text(json.dumps(out, indent=2, default=str))
    print(f"wrote {OUT}")
    print("task 0 lag 12 unique n=8:")
    for row in headline:
        print(
            f"  γ={row['gamma']:g}  HH {row['S-HH_floors']:+.1f} fl  "
            f"LH {row['S-LH_floors']:+.1f} fl  "
            f"Δ {row['diff_floors']:+.1f}  p_welch={row['welch_p']:.3g}"
        )


if __name__ == "__main__":
    main()
