"""`02` §2a — capacity validity over the thermodynamic-limit knobs `P` and `N`.

Three capacity numbers on **identical** point clouds at κ = 0:

- `α_sim`  — simulation ground truth (`src/glue/adapters/simcap.py`);
- `α_core` — our GLUE core;
- `α_mf`   — replicaMFT, **generic only** and included at β = 0 only, reduced with
             the harmonic mean (`AGENTS.md` §4).

Mean-field error should fall as `O(1/N)`. This is what decides the `P = 16`
question (`01` Phase 0): the pre-specified fix order is raise `N`, then accept and
report capacity as approximate, then raise `P` as a last resort.

Writes `results/gate_2a.json`.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from _par import pin_threads, pmap

pin_threads()

import numpy as np  # noqa: E402

from src.glue import core  # noqa: E402
from src.glue.adapters import simcap  # noqa: E402
from src.manifolds import generator  # noqa: E402

P_VALUES = (8, 16, 32)
N_VALUES = (300, 600, 1200)
M, D, R = 60, 4, 1.0
N_T = 200
N_REP = 10
SEEDS = (0, 1, 2)


def _manifolds(P, N, seed):
    rng = np.random.default_rng(seed)
    return core.from_arrangement(generator.make_arrangement(P, N, D, R, M, rng).points)


def _job(spec):
    kind, P, N, seed = spec
    mans = _manifolds(P, N, seed)
    t0 = time.perf_counter()
    if kind == "core":
        r = core.glue_measures(mans, np.random.default_rng(seed + 1000), n_t=N_T)
        val = {"alpha": r.alpha, "D_eff": r.D_eff, "R_eff": r.R_eff, "Psi_eff": r.Psi_eff}
    elif kind == "sim":
        r = simcap.simcap(mans, np.random.default_rng(seed + 2000), n_rep=N_REP)
        val = {"alpha": r.alpha_sim, "N_c": r.N_c}
    else:
        vendor = Path(__file__).resolve().parents[1] / "third_party" / "replicaMFT"
        sys.path.insert(0, str(vendor))
        from mftma.manifold_analysis_correlation import manifold_analysis_corr

        alpha_vec, R_M, D_M, res_coeff0, _ = manifold_analysis_corr(mans, 0, N_T)
        val = {"alpha": core.harmonic_mean(alpha_vec), "R_M": float(np.mean(R_M)),
               "D_M": float(np.mean(D_M)), "center_cos_abs": float(res_coeff0)}
    return {"kind": kind, "P": P, "N": N, "seed": seed,
            "seconds": time.perf_counter() - t0, **val}


def main() -> None:
    t0 = time.perf_counter()
    specs = [
        (kind, P, N, s)
        for kind in ("core", "sim", "mf")
        for P in P_VALUES
        for N in N_VALUES
        for s in SEEDS
    ]
    print(f"[{len(specs)} estimations]", flush=True)
    raw = pmap(_job, specs)

    out = {
        "settings": {"M": M, "D": D, "R": R, "n_t": N_T, "n_rep": N_REP,
                     "seeds": list(SEEDS), "kappa": 0, "beta": 0,
                     "center_policy": "all"},
        "note": "alpha_mf is label-invariant; present at beta=0 only (00 §6.1)",
        "cells": [],
    }

    def agg(kind, P, N, field="alpha"):
        v = [r[field] for r in raw if (r["kind"], r["P"], r["N"]) == (kind, P, N)]
        return float(np.mean(v)), float(np.std(v, ddof=1))

    print("\n  P    N     alpha_sim       alpha_core      alpha_mf        "
          "core-vs-sim   mf-vs-sim")
    for P in P_VALUES:
        for N in N_VALUES:
            s_m, s_sd = agg("sim", P, N)
            c_m, c_sd = agg("core", P, N)
            m_m, m_sd = agg("mf", P, N)
            cell = {
                "P": P, "N": N,
                "alpha_sim": {"mean": s_m, "sd": s_sd},
                "alpha_core": {"mean": c_m, "sd": c_sd},
                "alpha_mf": {"mean": m_m, "sd": m_sd},
                "core_rel_error": abs(c_m - s_m) / s_m,
                "mf_rel_error": abs(m_m - s_m) / s_m,
            }
            out["cells"].append(cell)
            print(
                f"  {P:<4} {N:<5} {s_m:.4f}+-{s_sd:.4f}  {c_m:.4f}+-{c_sd:.4f}  "
                f"{m_m:.4f}+-{m_sd:.4f}  {cell['core_rel_error']*100:8.1f}%  "
                f"{cell['mf_rel_error']*100:8.1f}%",
                flush=True,
            )

    # O(1/N) trend per P
    out["scaling"] = {}
    print("\n  1/N trend (rel. error vs N, per P)")
    for P in P_VALUES:
        cells = [c for c in out["cells"] if c["P"] == P]
        inv_N = np.array([1.0 / c["N"] for c in cells])
        for key in ("core_rel_error", "mf_rel_error"):
            e = np.array([c[key] for c in cells])
            slope, intercept = np.polyfit(inv_N, e, 1)
            out["scaling"].setdefault(f"P={P}", {})[key] = {
                "slope_vs_inv_N": float(slope),
                "intercept": float(intercept),
                "errors_by_N": {str(c["N"]): c[key] for c in cells},
            }
        cr = out["scaling"][f"P={P}"]["core_rel_error"]
        mr = out["scaling"][f"P={P}"]["mf_rel_error"]
        print(
            f"  P={P:<3} core: " + " ".join(
                f"N={n}:{v*100:.1f}%" for n, v in cr["errors_by_N"].items()
            ) + f"  -> intercept {cr['intercept']*100:.1f}%"
        )
        print(
            f"        mf:   " + " ".join(
                f"N={n}:{v*100:.1f}%" for n, v in mr["errors_by_N"].items()
            ) + f"  -> intercept {mr['intercept']*100:.1f}%"
        )

    out["elapsed_s"] = round(time.perf_counter() - t0, 1)
    path = Path(__file__).resolve().parents[1] / "results" / "gate_2a.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nwrote {path}  ({out['elapsed_s']}s)")


if __name__ == "__main__":
    main()
