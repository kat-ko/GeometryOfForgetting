"""Realized T×T stream structure on the registered unique-n=8 grid.

Zero training. Reads `results/phase1/*.json` stream matrices and lag-12
retained-capacity comparisons. Writes `results/realized_stream_structure.json`.

    python scripts/analyse_realized_stream_structure.py
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
from src.analysis import grid as G  # noqa: E402
from src.analysis.ledger import observations  # noqa: E402
from src.manifolds.dichotomies import (  # noqa: E402
    hamming_for_readout_similarity,
    readout_similarity,
    sample_balanced,
)

OUT = ROOT / "results" / "realized_stream_structure.json"
CORNERS = ("S-HH", "S-HL", "S-LH", "S-LL")
NOMINAL = {"S-HH": (0.9, 0.9), "S-HL": (0.9, 0.1), "S-LH": (0.1, 0.9), "S-LL": (0.1, 0.1)}
P, T, LAG = 16, 16, 12


def _fmt(x) -> float | None:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return None
    return float(x)


def summarize(vals: np.ndarray) -> dict:
    v = np.asarray(vals, dtype=float)
    return {
        "n": int(v.size),
        "mean": _fmt(v.mean()) if v.size else None,
        "sd": _fmt(v.std(ddof=1)) if v.size > 1 else None,
        "min": _fmt(v.min()) if v.size else None,
        "max": _fmt(v.max()) if v.size else None,
    }


def linreg(x, y) -> dict:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if x.size < 3 or np.std(x) == 0:
        return {
            "n": int(x.size), "slope": None, "intercept": None, "r": None,
            "r2": None, "p": None, "predictor_constant": bool(np.std(x) == 0),
        }
    r = stats.linregress(x, y)
    return {
        "n": int(x.size),
        "slope": _fmt(r.slope),
        "intercept": _fmt(r.intercept),
        "r": _fmt(r.rvalue),
        "r2": _fmt(r.rvalue ** 2),
        "p": _fmt(r.pvalue),
        "predictor_constant": False,
    }


def spearman(x, y) -> dict:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if x.size < 3 or np.std(x) == 0 or np.std(y) == 0:
        return {"n": int(x.size), "rho": None, "p": None,
                "predictor_constant": bool(np.std(x) == 0)}
    rho, p = stats.spearmanr(x, y)
    return {"n": int(x.size), "rho": _fmt(rho), "p": _fmt(p), "predictor_constant": False}


def r2_groups(y, labels) -> dict:
    y = np.asarray(y, float)
    labels = np.asarray(labels)
    if y.size < 3:
        return {"n": int(y.size), "r2": None}
    grand = y.mean()
    sst = np.sum((y - grand) ** 2)
    ssr = 0.0
    for lab in np.unique(labels):
        g = y[labels == lab]
        ssr += np.sum((g - g.mean()) ** 2)
    return {"n": int(y.size), "r2": _fmt(1.0 - ssr / sst) if sst > 0 else None}


def random_pair_baseline(n: int = 20_000, seed: int = 20260909) -> dict:
    rng = np.random.default_rng(seed)
    vals = np.empty(n)
    for i in range(n):
        a, b = sample_balanced(P, rng), sample_balanced(P, rng)
        vals[i] = readout_similarity(a, b)
    return {
        "n_pairs": n,
        "mean": _fmt(vals.mean()),
        "sd": _fmt(vals.std(ddof=1)),
        "p_eq_1": _fmt(np.mean(vals >= 1.0 - 1e-12)),
        "p_eq_0": _fmt(np.mean(vals <= 1e-12)),
    }


def unique_streams(recs: list[dict]) -> list[dict]:
    keep = {}
    for r in recs:
        s = r["spec"]
        key = (s["condition"], s["seed"])
        if key in keep:
            a = np.asarray(keep[key]["stream"]["S_f"])
            b = np.asarray(r["stream"]["S_f"])
            c = np.asarray(keep[key]["stream"]["S_r"])
            d = np.asarray(r["stream"]["S_r"])
            if not (np.allclose(a, b) and np.allclose(c, d)):
                raise ValueError(f"S_f/S_r mismatch across γ at {key}")
            continue
        keep[key] = r
    return list(keep.values())


def stream_row(r: dict) -> dict:
    cond = r["spec"]["condition"]
    Sf = np.asarray(r["stream"]["S_f"], float)
    Sr = np.asarray(r["stream"]["S_r"], float)
    sf_lag = [float(Sf[0, t]) for t in range(1, T)]
    sr_lag = [float(Sr[0, t]) for t in range(1, T)]
    t12 = list(range(1, LAG + 1))
    return {
        "condition": cond,
        "seed": r["spec"]["seed"],
        "sf_0_t": sf_lag,
        "sr_0_t": sr_lag,
        "sf_0_12": float(Sf[0, LAG]),
        "sr_0_12": float(Sr[0, LAG]),
        "min_sr_1_12": float(min(Sr[0, t] for t in t12)),
        "max_sf_1_12": float(max(Sf[0, t] for t in t12)),
        "net_disp_12": float(1.0 - Sf[0, LAG]),
        "path_disp_2_12": float(sum(1.0 - Sf[0, t] for t in range(2, LAG + 1))),
        "sr_all_eq_1": bool(np.allclose(Sr, 1.0)),
        "sf_subdiag": float(Sf[0, 1]),
        "sr_subdiag": float(Sr[0, 1]),
        "n_unique_sr_rows": int(len({tuple(np.round(row, 12)) for row in Sr})),
    }


def main() -> None:
    recs = unique_recs(G.load())
    streams = unique_streams(recs)
    h_high = hamming_for_readout_similarity(P, 0.9)
    h_low = hamming_for_readout_similarity(P, 0.1)
    s_from_h = lambda h: abs(1.0 - 2.0 * h / P)

    by_corner = defaultdict(list)
    for r in streams:
        by_corner[r["spec"]["condition"]].append(stream_row(r))

    lag_tables = {}
    lag12 = {}
    decay = {}
    approach = {}
    for cond in CORNERS:
        rows = by_corner[cond]
        sf_mat = np.array([r["sf_0_t"] for r in rows])
        sr_mat = np.array([r["sr_0_t"] for r in rows])
        nom_sf, nom_sr = NOMINAL[cond]
        lag_tables[cond] = {
            "n_streams": len(rows),
            "nominal_s_f": nom_sf,
            "nominal_s_r": nom_sr,
            "by_lag": [
                {
                    "t": t + 1,
                    "s_f": summarize(sf_mat[:, t]),
                    "s_r": summarize(sr_mat[:, t]),
                    "ar1_s_f_t": _fmt(nom_sf ** (t + 1)),
                }
                for t in range(T - 1)
            ],
        }
        lag12[cond] = {
            "s_f_0_12": summarize(np.array([r["sf_0_12"] for r in rows])),
            "s_r_0_12": summarize(np.array([r["sr_0_12"] for r in rows])),
            "ar1": _fmt(nom_sf ** LAG),
        }
        decay[cond] = {
            "ar1_mae_t2_to_15": _fmt(np.mean([
                abs(sf_mat[:, t].mean() - nom_sf ** (t + 1))
                for t in range(1, T - 1)
            ])),
            "mean_s_f_0_t_vs_ar1": [
                {
                    "t": t + 1,
                    "mean_realized": _fmt(sf_mat[:, t].mean()),
                    "ar1": _fmt(nom_sf ** (t + 1)),
                    "subdiagonal_is_nominal": t == 0,
                }
                for t in range(T - 1)
            ],
        }
        approach[cond] = {
            "min_s_r_1_12": summarize(np.array([r["min_sr_1_12"] for r in rows])),
            "max_s_f_1_12": summarize(np.array([r["max_sf_1_12"] for r in rows])),
            "n_sr_identically_1": int(sum(r["sr_all_eq_1"] for r in rows)),
            "n_unique_dichotomies_mean": _fmt(np.mean([r["n_unique_sr_rows"] for r in rows])),
        }

    obs = unique_obs(observations(recs, module="A"))
    cell = [o for o in obs if o["task"] == 0 and o["lag"] == LAG]
    by_g_c = defaultdict(list)
    for o in cell:
        by_g_c[(o["gamma"], o["condition"])].append(o)

    stream_ix = {(r["condition"], r["seed"]): r for c in CORNERS for r in by_corner[c]}

    def attach(gamma: float, cond: str) -> list[dict]:
        out = []
        for o in by_g_c[(gamma, cond)]:
            s = stream_ix[(cond, o["seed"])]
            out.append({
                "seed": o["seed"],
                "dlog_alpha": float(o["att"].dlog_alpha),
                "floors": float(np.sign(o["att"].dlog_alpha) * G.floors(o["att"].dlog_alpha, "alpha")),
                **{k: s[k] for k in (
                    "sf_0_12", "sr_0_12", "min_sr_1_12", "max_sf_1_12",
                    "net_disp_12", "path_disp_2_12",
                )},
            })
        return out

    finding3 = {}
    for g in (1.0, 10.0):
        rows = attach(g, "S-HH")
        y = [r["dlog_alpha"] for r in rows]
        finding3[f"{g:g}"] = {
            "n": len(rows),
            "mean_dlog_alpha": _fmt(np.mean(y)),
            "mean_floors": _fmt(np.mean([r["floors"] for r in rows])),
            "vs_min_sr": spearman([r["min_sr_1_12"] for r in rows], y),
            "vs_max_sf": spearman([r["max_sf_1_12"] for r in rows], y),
            "vs_sf_0_12": spearman([r["sf_0_12"] for r in rows], y),
            "vs_net_disp": spearman([r["net_disp_12"] for r in rows], y),
            "ols_min_sr": linreg([r["min_sr_1_12"] for r in rows], y),
            "ols_max_sf": linreg([r["max_sf_1_12"] for r in rows], y),
            "ols_sf_0_12": linreg([r["sf_0_12"] for r in rows], y),
            "ols_net_disp": linreg([r["net_disp_12"] for r in rows], y),
        }

    which_wins = {}
    for g in (1.0, 10.0):
        rows = []
        labels = []
        for cond in CORNERS:
            for r in attach(g, cond):
                rows.append(r)
                labels.append(cond)
        y = [r["dlog_alpha"] for r in rows]
        which_wins[f"{g:g}"] = {
            "n": len(rows),
            "r2_corner_label": r2_groups(y, labels),
            "ols_sr_0_12": linreg([r["sr_0_12"] for r in rows], y),
            "ols_sf_0_12": linreg([r["sf_0_12"] for r in rows], y),
            "ols_min_sr": linreg([r["min_sr_1_12"] for r in rows], y),
            "ols_max_sf": linreg([r["max_sf_1_12"] for r in rows], y),
            "ols_net_disp": linreg([r["net_disp_12"] for r in rows], y),
            "ols_path_disp": linreg([r["path_disp_2_12"] for r in rows], y),
            "spearman_sr_0_12": spearman([r["sr_0_12"] for r in rows], y),
            "spearman_sf_0_12": spearman([r["sf_0_12"] for r in rows], y),
            "spearman_min_sr": spearman([r["min_sr_1_12"] for r in rows], y),
            "spearman_net_disp": spearman([r["net_disp_12"] for r in rows], y),
        }

    sep = {
        "s_f_0_12_HH_minus_LL": _fmt(
            lag12["S-HH"]["s_f_0_12"]["mean"] - lag12["S-LL"]["s_f_0_12"]["mean"]
        ),
        "s_f_0_12_high_minus_low": _fmt(
            0.5 * (lag12["S-HH"]["s_f_0_12"]["mean"] + lag12["S-HL"]["s_f_0_12"]["mean"])
            - 0.5 * (lag12["S-LH"]["s_f_0_12"]["mean"] + lag12["S-LL"]["s_f_0_12"]["mean"])
        ),
        "s_r_0_12_high_minus_low": _fmt(
            0.5 * (lag12["S-HH"]["s_r_0_12"]["mean"] + lag12["S-LH"]["s_r_0_12"]["mean"])
            - 0.5 * (lag12["S-HL"]["s_r_0_12"]["mean"] + lag12["S-LL"]["s_r_0_12"]["mean"])
        ),
        "nominal_s_f_gap": 0.8,
        "nominal_s_r_gap": 0.8,
    }

    out = {
        "generated_by": "scripts/analyse_realized_stream_structure.py",
        "training": False,
        "population": "registered phase1 unique n=8 (one file per γ, condition, seed); a=0, N=300",
        "unit_streams": "(condition, seed); S_f/S_r identical across γ",
        "unit_outcomes": "unique (arrangement, init)=seed; task 0 lag 12 module A",
        "verify": {
            "arrangements_change_in_2x2": True,
            "why": "make_stream redraws centers at s_f via redraw_centers_correlated; s_f∈{0.1,0.9}<1 so A_t ≠ A_0. S-fixed-r is s_f=1 and was not run.",
            "notebook_section_4": "false for the 2×2; true only of unrun S-fixed-r",
            "consecutive_S_f_subdiagonal": "written as the generating s_f, not the realized cosine",
            "lag_ge_2_S_f": "realized mean pairwise cosine of centers",
            "hamming_P16_sr_0.9": {"h": h_high, "realized_consecutive_s_r": s_from_h(h_high)},
            "hamming_P16_sr_0.1": {"h": h_low, "realized_consecutive_s_r": s_from_h(h_low)},
            "S_HH_and_S_LH_dichotomy": "h=0 so y_t = y_0 for every t (up to the copy, not a walk)",
        },
        "n_unique_streams": {c: len(by_corner[c]) for c in CORNERS},
        "n_unique_arms": len(recs),
        "random_balanced_pair_baseline": random_pair_baseline(),
        "lag_tables": lag_tables,
        "lag12": lag12,
        "effective_separation": sep,
        "decay": decay,
        "unintended_recurrence": approach,
        "finding3_S_HH_gain_vs_approach": finding3,
        "which_predictor": which_wins,
        "change_entangled_with_similarity": {
            "fact": "High s_f drifts (AR(1) at 0.9); low s_f jumps. Consecutive change is 1-s_f by construction and is constant within a corner. Line 2 change co-varies with feature similarity; it is not a held-constant of the 2×2.",
            "consecutive_change_high_sf": 0.1,
            "consecutive_change_low_sf": 0.9,
        },
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps({
        "verify": out["verify"],
        "lag12": lag12,
        "effective_separation": sep,
        "unintended_recurrence": {c: approach[c] for c in CORNERS},
        "finding3": finding3,
        "which_predictor": which_wins,
    }, indent=2))


if __name__ == "__main__":
    main()
