"""§B.5 ground-truth recovery sweeps for `src/glue/core.py` (`02` §1).

Acceptance criterion for the GLUE core: sweeping each ground-truth knob **with the
others held fixed** must move the corresponding effective measure monotonically
and in the sign the theory requires. Chou et al. App. B.5 settings — P = 2,
M = 200, N = 1000, and `D = 4, R = 1, ρ_C = 0` off the swept axis.

The three sweeps are independent, not a diagonal: `D` and `R` are each swept at
zero correlation, then `D = 4, R = 1` are held while `ρ_C` sweeps.

Two extra axes, both decidable in-house:

- ``center_policy ∈ {"all", "active"}`` — how `s⁰_μ = E[s^μ(y,t)]` treats samples
  where manifold `μ` was inactive. The active fraction is ~0.5, so this is half
  the data. The α = 2 point-manifold test does **not** discriminate: `α` depends
  only on `S`, while `R_eff` and both ρ_c conventions normalize by `‖s⁰‖`.
- ``n_t ∈ {200, 1000}`` — whether the under-recovery of large `D` and `R` is
  sampling error (closes with `n_t`) or estimator bias (persists). This matters
  because the stream varies `D`: a bias correlated with a manipulated variable is
  a confound, not a nuisance.

Writes `results/glue_core_recovery.json`.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from _par import pin_threads, pmap

pin_threads()

import numpy as np  # noqa: E402

from src.glue import core  # noqa: E402
from src.manifolds import generator  # noqa: E402

P, M, N = 2, 200, 1000
D_FIXED, R_FIXED = 4, 1.0
SEEDS = (0, 1, 2)
N_T_VALUES = (200, 1000)
POLICIES = ("all", "active")

SWEEPS = {
    "dimension": ("D", (2, 4, 6, 8, 10)),
    "radius": ("R", (0.8, 1.0, 1.4, 1.7, 2.0)),
    "center_correlation": ("rho_C", (0.0, 0.2, 0.4, 0.6, 0.8)),
}
MEASURES = (
    "alpha", "D_eff", "R_eff", "Psi_eff",
    "rho_c_glue", "rho_c_signed", "active_fraction", "identity_residual",
)


def _job(spec: tuple) -> dict:
    sweep, key, value, seed, n_t, policy = spec
    kw = {"D": D_FIXED, "R": R_FIXED, "rho_C": 0.0}
    kw[key] = value
    rng = np.random.default_rng(seed)
    arr = generator.make_arrangement(
        P, N, int(kw["D"]), float(kw["R"]), M, rng, rho_C=float(kw["rho_C"])
    )
    res = core.glue_measures(
        core.from_arrangement(arr.points),
        np.random.default_rng(seed + 1000),
        n_t=n_t,
        center_policy=policy,
    )
    return {"sweep": sweep, "key": key, "value": value, "seed": seed,
            "n_t": n_t, "policy": policy, **{m: getattr(res, m) for m in MEASURES}}


def _aggregate(rows: list[dict]) -> dict:
    out = {}
    for m in MEASURES:
        v = np.array([r[m] for r in rows], dtype=float)
        out[m] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if v.size > 1 else 0.0}
    return out


def _recovery_error(rows: list[dict], key: str, measure: str) -> dict:
    """Mean |estimate − ground truth| / ground truth over the swept axis."""
    errs = [
        abs(r[measure]["mean"] - r["ground_truth"]) / r["ground_truth"]
        for r in rows
        if r["ground_truth"] > 0
    ]
    return {"mean_abs_rel_error": float(np.mean(errs)),
            "max_abs_rel_error": float(np.max(errs))}


def main() -> None:
    t0 = time.perf_counter()
    specs = [
        (sweep, key, value, seed, n_t, policy)
        for sweep, (key, values) in SWEEPS.items()
        for value in values
        for seed in SEEDS
        for n_t in N_T_VALUES
        for policy in POLICIES
    ]
    print(f"[{len(specs)} estimations]", flush=True)
    raw = pmap(_job, specs)

    out = {
        "estimator": "glue_core",
        "settings": {
            "P": P, "M": M, "N": N, "seeds": list(SEEDS), "kappa": 0,
            "D_fixed": D_FIXED, "R_fixed": R_FIXED, "rho_C_fixed": 0.0,
            "n_t_values": list(N_T_VALUES), "center_policies": list(POLICIES),
            "inactive_policy": "zero",
        },
        "source": "Chou et al. ICML 2025 App. B.5 recovery protocol",
        "design_note": "three independent sweeps; off-axis knobs held at D=4, R=1, rho_C=0",
        "sweeps": {},
        "recovery_error": {},
    }

    # measure whose ground truth each sweep recovers
    TARGET = {"dimension": "D_eff", "radius": "R_eff", "center_correlation": "rho_c_glue"}

    for sweep, (key, values) in SWEEPS.items():
        out["sweeps"][sweep] = {}
        for n_t in N_T_VALUES:
            for policy in POLICIES:
                rows = []
                for value in values:
                    sel = [r for r in raw if (r["sweep"], r["value"], r["n_t"], r["policy"])
                           == (sweep, value, n_t, policy)]
                    rows.append({"ground_truth": float(value), **_aggregate(sel)})
                tag = f"n_t={n_t},policy={policy}"
                out["sweeps"][sweep][tag] = rows
                out["recovery_error"].setdefault(sweep, {})[tag] = _recovery_error(
                    rows, key, TARGET[sweep]
                )

    out["elapsed_s"] = round(time.perf_counter() - t0, 1)

    # ---- report ------------------------------------------------------------
    for sweep, (key, values) in SWEEPS.items():
        target = TARGET[sweep]
        print(f"\n[{sweep}]  ground truth {key} -> {target}   "
              f"(off-axis fixed at D={D_FIXED}, R={R_FIXED}, rho_C=0)")
        header = "  " + "".join(f"{key}={v:<7}" for v in values)
        print(header + "   mean|rel err|")
        for n_t in N_T_VALUES:
            for policy in POLICIES:
                tag = f"n_t={n_t},policy={policy}"
                rows = out["sweeps"][sweep][tag]
                cells = "".join(f"{r[target]['mean']:<9.3f}" for r in rows)
                err = out["recovery_error"][sweep][tag]["mean_abs_rel_error"]
                print(f"  {cells}   {err*100:5.1f}%   [{tag}]")

    path = Path(__file__).resolve().parents[1] / "results" / "glue_core_recovery.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nwrote {path}  ({out['elapsed_s']}s)")


if __name__ == "__main__":
    main()
