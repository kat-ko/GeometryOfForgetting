"""Phase 0 — manipulation validation (`01` §4, table).

Four of the five checks. The time-reparameterization test (`00` §12) needs
geometry trajectories through training and lands with `src/analysis/`.

| Check | Pass criterion | Failure action |
|---|---|---|
| Capacity-at-init tracks `a` | monotone, range ≥ 2× noise floor | **Gate 2** — fall back to `wealth_knob: input_dim` |
| Capacity-at-init flat in `γ` | within noise floor | parameterization bug — stop |
| `‖ΔW‖/‖W‖` separates across `γ` | ≥ 1 order of magnitude | parameterization bug — stop |
| `pairwise` vs `full_P` | agree within noise floor | report both |

Also checks that every task converges across the γ grid, since under-training
would enter the Phase 1 forgetting numbers as forgetting.

Writes `results/phase0.json`.
"""

from __future__ import annotations

import json
from pathlib import Path

from _par import n_workers, pin_threads, pmap

pin_threads()

import numpy as np  # noqa: E402

from src.glue import core  # noqa: E402
from src.manifolds import dichotomies, generator  # noqa: E402
from src.models import MODULES, ScalingConfig, TwoModuleNet, paired_init  # noqa: E402
from src.models.alignment import aligned_init, center_subspace  # noqa: E402
from src.train import TrainConfig, run_stream  # noqa: E402

# Project configuration (`01` §1).
P, D_AMB, M, N = 16, 150, 150, 300
D_INT, R = 4, 1.0
N_T = 200
GAMMAS = (0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
A_GRID = tuple(np.linspace(0.0, 1.0, 7))
SEEDS = (0, 1, 2)
T_TASKS = 4
TRAIN = TrainConfig(steps_per_task=20_000, record_every=200,
                    stopping="matched_loss", target_loss=0.05)

# Measured Monte-Carlo noise floors (CV %, results/cost_model.json).
NOISE_FLOOR_PCT = {"alpha": 1.87, "D_eff": 1.26, "R_eff": 0.50,
                   "Psi_eff": 1.39, "rho_c_glue": 0.97}


def _arrangement(seed):
    return generator.make_arrangement(
        P, D_AMB, D_INT, R, M, paired_init(seed)["data"], rho_C=0.0, rho_A=0.0
    )


def _model(seed, gamma, a=None, arr=None):
    streams = paired_init(seed)
    cfg = {m: ScalingConfig(N=N, d=D_AMB, gamma_0=gamma, lr0=0.2) for m in MODULES}
    W_init = None
    if a is not None:
        # One shape matrix per seed, reused across conditions (`01` §1 paired init).
        base = streams["shape"].standard_normal((N, D_AMB))
        U_C, _ = center_subspace(arr.centers, D=D_INT)
        W_init = {m: aligned_init(base, U_C, a) for m in MODULES}
    return TwoModuleNet.init(cfg, streams["shape"], W_init=W_init)


def _init_geometry(model, arr, seed, module="A"):
    reps = model.manifold_representation(arr.points, module=module)
    return core.glue_measures(reps, np.random.default_rng(seed + 9000), n_t=N_T)


# --- check 1: Gate 2, capacity-at-init vs alignment -------------------------

def _gate2_job(spec):
    a, seed = spec
    arr = _arrangement(seed)
    r = _init_geometry(_model(seed, 1.0, a=a, arr=arr), arr, seed)
    return {"a": float(a), "seed": seed, "alpha": r.alpha,
            "D_eff": r.D_eff, "R_eff": r.R_eff, "rho_c_glue": r.rho_c_glue}


def check_gate2():
    raw = pmap(_gate2_job, [(a, s) for a in A_GRID for s in SEEDS])
    by_a = {}
    for a in A_GRID:
        v = np.array([r["alpha"] for r in raw if r["a"] == float(a)])
        by_a[round(float(a), 4)] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1))}
    means = np.array([by_a[round(float(a), 4)]["mean"] for a in A_GRID])
    rng_pct = 100 * (means.max() - means.min()) / means.mean()
    d = np.diff(means)
    monotone = bool(np.all(d >= -1e-12) or np.all(d <= 1e-12))
    spearman = float(np.corrcoef(np.argsort(np.argsort(means)),
                                 np.arange(len(means)))[0, 1])
    passed = monotone and rng_pct >= 2 * NOISE_FLOOR_PCT["alpha"]
    print(f"  alpha(a): {np.array2string(means, precision=4)}")
    print(f"  range {rng_pct:.2f}% (need >= {2*NOISE_FLOOR_PCT['alpha']:.2f}%), "
          f"monotone={monotone}, rank-corr={spearman:+.2f} -> "
          f"{'PASS' if passed else 'FAIL'}")
    return {"per_a": by_a, "range_pct": rng_pct, "monotone": monotone,
            "rank_correlation": spearman,
            "threshold_pct": 2 * NOISE_FLOOR_PCT["alpha"],
            "passed": passed, "raw": raw,
            "failure_action": "wealth_knob: input_dim (00 §5.3); do NOT debug the "
                              "geodesic (02 §4)"}


# --- check 2: capacity-at-init flat in gamma --------------------------------

def check_gamma_flat():
    """Structural, not statistical: `h_m = ReLU(β₀ W_m x)` contains no γ.

    γ enters only the output scale and the learning rate, so the *representation*
    at init is bitwise identical across γ and capacity-at-init cannot vary. We
    assert the bitwise identity and confirm the estimator agrees exactly.
    """
    seed = SEEDS[0]
    arr = _arrangement(seed)
    ref_rep = _model(seed, GAMMAS[0]).manifold_representation(arr.points, module="A")
    bitwise = all(
        np.array_equal(x, y)
        for g in GAMMAS[1:]
        for x, y in zip(ref_rep,
                        _model(seed, g).manifold_representation(arr.points, module="A"))
    )
    alphas = {g: _init_geometry(_model(seed, g), arr, seed).alpha for g in (GAMMAS[0], GAMMAS[-1])}
    exact = alphas[GAMMAS[0]] == alphas[GAMMAS[-1]]
    print(f"  representations bitwise identical across gamma: {bitwise}")
    print(f"  alpha({GAMMAS[0]}) == alpha({GAMMAS[-1]}): {exact} "
          f"({alphas[GAMMAS[0]]:.6f}) -> {'PASS' if bitwise and exact else 'FAIL'}")
    return {"bitwise_identical_representations": bitwise,
            "alpha_endpoints": {str(k): v for k, v in alphas.items()},
            "exactly_equal": bool(exact), "passed": bool(bitwise and exact),
            "note": "guaranteed by construction (I6), not an empirical result: the "
                    "hidden layer has no gamma in it and the readout is zero-init"}


# --- check 3: weight change separates across gamma --------------------------

def _richness_job(spec):
    gamma, seed = spec
    arr = _arrangement(seed)
    streams = paired_init(seed)
    model = _model(seed, gamma)
    ys = np.stack([dichotomies.sample_balanced(P, streams["stream"])
                   for _ in range(T_TASKS)])
    tasks, boundaries = run_stream(model, [arr.points] * T_TASKS, ys, TRAIN,
                                   streams["data"])
    return {"gamma": gamma, "seed": seed,
            "dW": boundaries[-1].weight_change["A"],
            "all_converged": all(t.converged for t in tasks),
            "task_accuracy": [t.train_accuracy for t in tasks],
            "final_loss": [t.final_loss for t in tasks],
            "steps": [t.steps_taken for t in tasks]}


def check_richness():
    raw = pmap(_richness_job, [(g, s) for g in GAMMAS for s in SEEDS])
    per_gamma = {}
    for g in GAMMAS:
        v = np.array([r["dW"] for r in raw if r["gamma"] == g])
        conv = all(r["all_converged"] for r in raw if r["gamma"] == g)
        steps = np.array([sum(r["steps"]) for r in raw if r["gamma"] == g])
        per_gamma[str(g)] = {"dW_mean": float(v.mean()), "dW_sd": float(v.std(ddof=1)),
                             "all_tasks_converged": conv,
                             "total_steps_mean": float(steps.mean())}
        print(f"  gamma={g:<6} dW/W = {v.mean():.5f} +- {v.std(ddof=1):.5f}   "
              f"steps={steps.mean():8.0f}   converged={conv}")
    lo = per_gamma[str(GAMMAS[0])]["dW_mean"]
    hi = per_gamma[str(GAMMAS[-1])]["dW_mean"]
    decades = float(np.log10(hi / lo)) if lo > 0 else float("inf")
    conv_all = all(v["all_tasks_converged"] for v in per_gamma.values())
    passed = decades >= 1.0
    print(f"  spread {decades:.2f} decades (need >= 1) -> {'PASS' if passed else 'FAIL'}")
    print(f"  every task converged at every gamma: {conv_all}"
          f"{'' if conv_all else '   <-- under-training would enter Phase 1 as forgetting'}")
    return {"per_gamma": per_gamma, "decades": decades, "passed": passed,
            "all_converged_everywhere": conv_all, "raw": raw,
            "failure_action": "parameterization bug -- STOP"}


# --- check 4: pairwise vs full_P --------------------------------------------

def _mode_job(spec):
    mode, seed = spec
    arr = _arrangement(seed)
    reps = _model(seed, 1.0).manifold_representation(arr.points, module="A")
    rng = np.random.default_rng(seed + 4242)
    if mode == "full_P":
        r = core.glue_measures(reps, rng, n_t=N_T)
        out = {k: getattr(r, k) for k in
               ("alpha", "D_eff", "R_eff", "Psi_eff", "rho_c_glue", "rho_c_signed")}
    else:
        out = core.pairwise_measures(reps, rng, n_t=N_T)
    return {"mode": mode, "seed": seed, **{k: out[k] for k in
            ("alpha", "D_eff", "R_eff", "Psi_eff", "rho_c_glue", "rho_c_signed")}}


def check_estimation_mode():
    raw = pmap(_mode_job, [(m, s) for m in ("full_P", "pairwise") for s in SEEDS])
    keys = ("alpha", "D_eff", "R_eff", "Psi_eff", "rho_c_glue", "rho_c_signed")
    table, verdicts = {}, {}
    print("  measure       full_P    pairwise    rel.diff   floor   verdict")
    for k in keys:
        f = np.array([r[k] for r in raw if r["mode"] == "full_P"])
        p = np.array([r[k] for r in raw if r["mode"] == "pairwise"])
        rel = 100 * abs(p.mean() - f.mean()) / abs(f.mean())
        floor = NOISE_FLOOR_PCT.get(k, NOISE_FLOOR_PCT["rho_c_glue"])
        agree = rel <= floor
        table[k] = {"full_P": float(f.mean()), "full_P_sd": float(f.std(ddof=1)),
                    "pairwise": float(p.mean()), "pairwise_sd": float(p.std(ddof=1)),
                    "rel_diff_pct": rel, "noise_floor_pct": floor,
                    "agrees_within_floor": bool(agree)}
        verdicts[k] = agree
        print(f"  {k:<12} {f.mean():8.4f}  {p.mean():8.4f}   {rel:7.2f}%  "
              f"{floor:5.2f}%   {'agree' if agree else 'DIVERGE'}")
    return {"per_measure": table, "all_agree": all(verdicts.values()), "raw": raw,
            "action": "report both; pairwise = comparable-to-published (lab "
                      "standard), full_P = primary (01 §4)"}


def main() -> None:
    print(f"Phase 0 at P={P}, N={N}, M={M}, d={D_AMB}, D={D_INT}, R={R}, "
          f"n_t={N_T}, {n_workers()} workers\n")
    out = {"config": {"P": P, "N": N, "M": M, "d": D_AMB, "D": D_INT, "R": R,
                      "n_t": N_T, "gammas": list(GAMMAS), "a_grid": [float(a) for a in A_GRID],
                      "seeds": list(SEEDS), "T": T_TASKS,
                      "steps_per_task": TRAIN.steps_per_task,
                      "target_accuracy": TRAIN.target_accuracy}}

    print("[1/4] GATE 2 -- capacity-at-init vs alignment a")
    out["gate2_alignment"] = check_gate2()
    print("\n[2/4] capacity-at-init flat in gamma")
    out["gamma_flat"] = check_gamma_flat()
    print("\n[3/4] weight change separates across gamma")
    out["richness_separation"] = check_richness()
    print("\n[4/4] estimation mode: pairwise vs full_P")
    out["estimation_mode"] = check_estimation_mode()

    gates = {"gate2_alignment": out["gate2_alignment"]["passed"],
             "gamma_flat": out["gamma_flat"]["passed"],
             "richness_separation": out["richness_separation"]["passed"]}
    out["summary"] = gates
    print("\n" + "=" * 62)
    for k, v in gates.items():
        print(f"  {k:<24} {'PASS' if v else 'FAIL'}")
    print(f"  {'pairwise vs full_P':<24} "
          f"{'agree' if out['estimation_mode']['all_agree'] else 'DIVERGE - report both'}")

    path = Path(__file__).resolve().parents[1] / "results" / "phase0.json"
    path.write_text(json.dumps(out, indent=2, default=float) + "\n")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
