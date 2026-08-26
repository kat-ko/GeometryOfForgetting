"""Three short follow-ups that finish the validation sheet.

1. **Is `α_sim`'s drift with ambient `N` an artifact?** `results/gate_2a.json` shows
   `α_sim` rising 0.420 → 0.460 as `N` goes 300 → 1200 at fixed generated geometry,
   while `α_core` stays flat. Here `N` is raised by **zero-padding the same point
   clouds**, so the geometry is not merely statistically equivalent, it is
   *identical*. Capacity must be invariant. If the estimate moves, the drift is
   confirmed as an estimator artifact.

2. **What does `M` actually cost?** The `M`-sweep gave the accuracy side of the
   `M` tradeoff (`results/scale_compression.json`); this measures the cost side at
   project parameters, so the choice of `M = 150` can be justified rather than
   inherited, and a high-`M` robustness arm can be sized.

3. **MDE at 5 vs 8 seeds** from the measured noise floors, so the seed count is
   chosen on evidence (`05` Day-14 report).

Writes `results/estimator_followups.json`.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from _par import pin_threads, pmap

pin_threads()

import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

from src.glue import core  # noqa: E402
from src.glue.adapters import simcap  # noqa: E402
from src.manifolds import generator  # noqa: E402

PAD_P, PAD_M, PAD_D, PAD_R = 16, 60, 4, 1.0
PAD_N_BASE = 300
PAD_N = (300, 600, 1200)
PAD_SEEDS = (0, 1, 2, 3, 4)

COST_P, COST_N, COST_NT = 16, 300, 10
COST_M = (150, 400, 800)

# Full Phase 1 grid, from results/cost_model.json.
GRID_EVALS = 48_000
BASE_S_PER_EVAL = 52.4          # P=16, M=150, N=300, n_t=200, one BLAS thread
WORKERS = 248

NOISE_FLOORS_CV_PCT = {          # results/cost_model.json, n_t=200, 12 replicates
    "alpha": 1.87, "D_eff": 1.26, "R_eff": 0.50,
    "Psi_eff": 1.39, "rho_c_glue": 0.97,
}


def _pad_job(spec):
    kind, N, seed = spec
    rng = np.random.default_rng(seed)
    arr = generator.make_arrangement(PAD_P, PAD_N_BASE, PAD_D, PAD_R, PAD_M, rng)
    mans = core.from_arrangement(arr.points)
    if N > PAD_N_BASE:  # identical geometry, embedded in a larger ambient space
        pad = np.zeros((N - PAD_N_BASE, PAD_M))
        mans = [np.vstack([X, pad]) for X in mans]
    if kind == "sim":
        val = simcap.simcap(mans, np.random.default_rng(seed + 2000), n_rep=10).alpha_sim
    else:
        val = core.glue_measures(mans, np.random.default_rng(seed + 1000), n_t=200).alpha
    return {"kind": kind, "N": N, "seed": seed, "alpha": val}


def _cost_job(M):
    rng = np.random.default_rng(0)
    arr = generator.make_arrangement(COST_P, COST_N, PAD_D, PAD_R, M, rng)
    mans = core.from_arrangement(arr.points)
    t0 = time.perf_counter()
    core.glue_measures(mans, np.random.default_rng(1), n_t=COST_NT)
    return {"M": M, "s_per_sample": (time.perf_counter() - t0) / COST_NT}


def mde(sd_pct: float, n_per_group: int, power: float = 0.8, alpha: float = 0.05) -> float:
    """Two-sided two-sample MDE as a percentage, at the given per-group n."""
    df = 2 * (n_per_group - 1)
    t_crit = stats.t.ppf(1 - alpha / 2, df)
    t_pow = stats.t.ppf(power, df)
    return float((t_crit + t_pow) * sd_pct * np.sqrt(2.0 / n_per_group))


def main() -> None:
    out: dict = {}

    # ---- 1. zero-padding invariance ---------------------------------------
    print("[alpha_sim under zero-padding: identical geometry, larger ambient N]")
    raw = pmap(_pad_job, [(k, N, s) for k in ("sim", "core")
                          for N in PAD_N for s in PAD_SEEDS])
    out["zero_padding"] = {
        "design": "one point cloud per seed, padded with zero rows to raise N; "
                  "capacity is invariant under isometric embedding, so any change "
                  "is estimator artifact",
        "P": PAD_P, "M": PAD_M, "D": PAD_D, "R": PAD_R,
        "N_base": PAD_N_BASE, "seeds": list(PAD_SEEDS), "results": {},
    }
    for kind in ("sim", "core"):
        row = {}
        for N in PAD_N:
            v = np.array([r["alpha"] for r in raw if (r["kind"], r["N"]) == (kind, N)])
            row[str(N)] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1))}
        base = row[str(PAD_N_BASE)]["mean"]
        row["drift_pct_300_to_1200"] = 100 * (row["1200"]["mean"] - base) / base
        out["zero_padding"]["results"][kind] = row
        cells = "  ".join(f"N={N}: {row[str(N)]['mean']:.4f}+-{row[str(N)]['sd']:.4f}"
                          for N in PAD_N)
        print(f"  alpha_{kind:<5} {cells}   drift {row['drift_pct_300_to_1200']:+.1f}%")

    sim_drift = out["zero_padding"]["results"]["sim"]["drift_pct_300_to_1200"]
    core_drift = out["zero_padding"]["results"]["core"]["drift_pct_300_to_1200"]
    out["zero_padding"]["verdict"] = (
        "alpha_sim drift CONFIRMED as estimator artifact"
        if abs(sim_drift) > 3 * max(abs(core_drift), 1.0)
        else "no differential drift under padding; the gate_2a trend is not an "
             "ambient-dimension artifact of alpha_sim"
    )
    print(f"  -> {out['zero_padding']['verdict']}")

    # ---- 2. cost of M ------------------------------------------------------
    print(f"\n[cost vs M at P={COST_P}, N={COST_N}]")
    cost = pmap(_cost_job, list(COST_M))
    base = next(c for c in cost if c["M"] == 150)["s_per_sample"]
    logM = np.log([c["M"] for c in cost])
    logT = np.log([c["s_per_sample"] for c in cost])
    exponent = float(np.polyfit(logM, logT, 1)[0])
    rows = []
    for c in cost:
        factor = c["s_per_sample"] / base
        full_h = GRID_EVALS * BASE_S_PER_EVAL * factor / 3600 / WORKERS
        rows.append({"M": c["M"], "s_per_sample": c["s_per_sample"],
                     "cost_factor_vs_M150": factor,
                     "full_grid_wall_hours": full_h,
                     "robustness_arm_wall_hours": full_h * 0.045})
        print(f"  M={c['M']:<4} {c['s_per_sample']*1000:7.1f} ms/sample  "
              f"{factor:5.2f}x  full grid {full_h:6.2f} h  "
              f"4.5% arm {full_h*0.045:5.2f} h")
    out["cost_vs_M"] = {
        "measured": rows,
        "scaling_exponent": exponent,
        "note": f"cost ~ M^{exponent:.2f} at P={COST_P}, N={COST_N}; "
                "robustness arm = 4.5% of the grid (1 condition x 2 gamma x 3 "
                "streams x 3 seeds = 54 runs)",
    }
    print(f"  scaling: cost ~ M^{exponent:.2f}")

    # ---- 3. MDE at 5 vs 8 seeds -------------------------------------------
    print("\n[MDE (%, two-sided, power 0.8) from measured Monte-Carlo noise floors]")
    out["mde"] = {
        "caveat": "computed from estimator Monte-Carlo CV at fixed manifolds "
                  "(results/cost_model.json). The Phase-1 across-seed variance also "
                  "contains network-init and stream variability, so these are LOWER "
                  "BOUNDS on the true MDE. Recompute from Phase 0 pilot spread.",
        "per_measure": {},
    }
    print("  measure       CV%    MDE n=5   MDE n=8   gain")
    for m, cv in NOISE_FLOORS_CV_PCT.items():
        m5, m8 = mde(cv, 5), mde(cv, 8)
        out["mde"]["per_measure"][m] = {
            "cv_pct": cv, "mde_pct_n5": m5, "mde_pct_n8": m8,
            "relative_gain": 1 - m8 / m5,
        }
        print(f"  {m:<12} {cv:5.2f}  {m5:7.2f}%  {m8:7.2f}%   {100*(1-m8/m5):4.1f}%")
    extra_h = GRID_EVALS * BASE_S_PER_EVAL / 3600 / WORKERS * (8 / 5 - 1)
    out["mde"]["seed_cost"] = {
        "wall_hours_5_seeds": GRID_EVALS * BASE_S_PER_EVAL / 3600 / WORKERS,
        "extra_wall_hours_for_8_seeds": extra_h,
    }
    print(f"  seeds 5 -> 8 costs +{extra_h:.2f} h wall")

    path = Path(__file__).resolve().parents[1] / "results" / "estimator_followups.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
