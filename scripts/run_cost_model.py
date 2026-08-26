"""Cost model for the GLUE core — Phase 1 feasibility (`01` Phase 1, `05` §8).

Three measurements, then arithmetic:

1. **per-sample time** vs `(P, M, N)`, and the linearity of total time in `n_t`;
2. **Monte-Carlo noise** vs `n_t` — the precision side of the same knob, so the
   cut order can be argued rather than asserted;
3. **α_sim** cost, for the `§2a` gate rather than Phase 1.

Writes `results/cost_model.json`.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from _par import pin_threads, pmap

pin_threads()

import numpy as np  # noqa: E402

from src.glue import core  # noqa: E402
from src.glue.adapters import simcap  # noqa: E402
from src.manifolds import generator  # noqa: E402

BASE = {"P": 16, "M": 150, "N": 300, "D": 4, "R": 1.0}
N_T_TIMING = 12
NOISE_N_T = (50, 100, 200, 400, 800)
NOISE_REPS = 12

# Phase 1 grid as specified in `01`.
GRID = {
    "T_tasks": 16,
    "tier2_every": 4,
    "evals_per_run": 40,
    "gamma": 6,
    "conditions": 4,
    "streams": 10,
    "seeds": 5,
}


def _manifolds(P, M, N, D, R, seed):
    rng = np.random.default_rng(seed)
    arr = generator.make_arrangement(P, N, D, R, M, rng)
    return core.from_arrangement(arr.points)


def _time_one(job):
    P, M, N, n_t = job
    mans = _manifolds(P, M, N, BASE["D"], BASE["R"], seed=0)
    t0 = time.perf_counter()
    core.glue_measures(mans, np.random.default_rng(1), n_t=n_t)
    dt = time.perf_counter() - t0
    return {"P": P, "M": M, "N": N, "n_t": n_t, "seconds": dt, "s_per_sample": dt / n_t}


def _noise_one(job):
    n_t, rep = job
    mans = _manifolds(BASE["P"], BASE["M"], BASE["N"], BASE["D"], BASE["R"], seed=0)
    r = core.glue_measures(mans, np.random.default_rng(10_000 + rep), n_t=n_t)
    return {"n_t": n_t, **{k: getattr(r, k) for k in
            ("alpha", "D_eff", "R_eff", "Psi_eff", "rho_c_glue", "rho_c_signed")}}


def _simcap_one(job):
    P, M, N = job
    mans = _manifolds(P, M, N, BASE["D"], BASE["R"], seed=0)
    t0 = time.perf_counter()
    simcap.simcap(mans, np.random.default_rng(1), n_rep=10)
    return {"P": P, "M": M, "N": N, "seconds": time.perf_counter() - t0}


def main() -> None:
    out: dict = {"machine": {}, "base_point": BASE}

    import os

    out["machine"] = {
        "logical_cores": len(os.sched_getaffinity(0)),
        "note": "workers pinned to 1 BLAS thread each",
    }

    # ---- 1. timing surface -------------------------------------------------
    print("[timing]", flush=True)
    jobs = []
    for P in (8, 16, 32):
        for N in (300, 600, 1200):
            jobs.append((P, BASE["M"], N, N_T_TIMING))
    for M in (50, 100, 150):
        jobs.append((BASE["P"], M, BASE["N"], N_T_TIMING))
    for n_t in (12, 25, 50):
        jobs.append((BASE["P"], BASE["M"], BASE["N"], n_t))
    timing = pmap(_time_one, jobs)
    out["timing"] = timing
    for row in timing:
        print(
            f"  P={row['P']:<3} M={row['M']:<4} N={row['N']:<5} n_t={row['n_t']:<4}"
            f" {row['s_per_sample']*1000:8.1f} ms/sample",
            flush=True,
        )

    base_row = next(
        r for r in timing
        if (r["P"], r["M"], r["N"], r["n_t"]) == (BASE["P"], BASE["M"], BASE["N"], 50)
    )
    s_per_sample = base_row["s_per_sample"]

    # ---- 2. noise vs n_t ---------------------------------------------------
    print("[noise vs n_t]", flush=True)
    noise_raw = pmap(_noise_one, [(n, r) for n in NOISE_N_T for r in range(NOISE_REPS)])
    noise = []
    for n_t in NOISE_N_T:
        rows = [r for r in noise_raw if r["n_t"] == n_t]
        entry = {"n_t": n_t, "reps": len(rows)}
        for m in ("alpha", "D_eff", "R_eff", "Psi_eff", "rho_c_glue", "rho_c_signed"):
            v = np.array([r[m] for r in rows], dtype=float)
            entry[m] = {
                "mean": float(v.mean()),
                "sd": float(v.std(ddof=1)),
                "cv_pct": float(100 * v.std(ddof=1) / abs(v.mean())) if v.mean() else None,
            }
        entry["seconds_per_eval"] = s_per_sample * n_t
        noise.append(entry)
        print(
            f"  n_t={n_t:<5} {entry['seconds_per_eval']:6.1f}s/eval  CV%: "
            f"alpha={entry['alpha']['cv_pct']:.2f} D_eff={entry['D_eff']['cv_pct']:.2f} "
            f"R_eff={entry['R_eff']['cv_pct']:.2f} Psi={entry['Psi_eff']['cv_pct']:.2f} "
            f"rho_c={entry['rho_c_glue']['cv_pct']:.2f}",
            flush=True,
        )
    out["noise_vs_n_t"] = noise

    # ---- 3. alpha_sim cost -------------------------------------------------
    print("[alpha_sim]", flush=True)
    out["simcap_timing"] = pmap(
        _simcap_one, [(P, 60, N) for P in (8, 16, 32) for N in (300, 600, 1200)]
    )
    for row in out["simcap_timing"]:
        print(f"  P={row['P']:<3} N={row['N']:<5} {row['seconds']:6.1f}s", flush=True)

    # ---- 4. grid arithmetic ------------------------------------------------
    runs = GRID["gamma"] * GRID["conditions"] * GRID["streams"] * GRID["seeds"]
    evals = runs * GRID["evals_per_run"]
    workers = out["machine"]["logical_cores"] - 8
    plan = []
    for n_t in NOISE_N_T:
        sec = s_per_sample * n_t
        serial_h = evals * sec / 3600
        plan.append(
            {
                "n_t": n_t,
                "seconds_per_eval": round(sec, 2),
                "serial_hours": round(serial_h, 1),
                "wall_hours_at_workers": round(serial_h / workers, 2),
                "alpha_cv_pct": next(e["alpha"]["cv_pct"] for e in noise if e["n_t"] == n_t),
            }
        )
    out["phase1_grid"] = {
        **GRID,
        "runs": runs,
        "evaluations": evals,
        "workers_assumed": workers,
        "plan_vs_n_t": plan,
    }
    print("\n[Phase 1]", flush=True)
    for p in plan:
        print(
            f"  n_t={p['n_t']:<5} {p['seconds_per_eval']:5.1f}s/eval  "
            f"serial={p['serial_hours']:7.1f}h  wall@{workers}={p['wall_hours_at_workers']:6.2f}h  "
            f"alpha CV={p['alpha_cv_pct']:.2f}%",
            flush=True,
        )

    path = Path(__file__).resolve().parents[1] / "results" / "cost_model.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
