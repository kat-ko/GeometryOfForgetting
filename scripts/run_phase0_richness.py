"""Phase 0 check 3, redone under matched-loss stopping.

The first pass stopped on train accuracy and was invalid: from `u = 0`, one step
gives the kernel readout `u ∝ Σ_b y_b h(x_b)`, whose *sign* is independent of the
learning rate, so accuracy hits ~0.99 at step 1 for every γ and training halts
before any feature learning — the thing under study. `01` §1 specifies
`stopping: "matched_loss"`, which is what this uses.

Two outputs:

1. `‖ΔW‖/‖W‖` across γ at **matched loss**, so the richness separation is not
   confounded with how far each arm trained.
2. **Steps to target loss vs γ** — the size of that confound, and a Phase 1
   budgeting input. μP's `η ∝ γ₀²` is designed to make the *function-space* rate
   γ-independent, so the residual spread here is feature learning speeding the
   fit, not the learning rate.

Shortened relative to `run_phase0.py`: `T = 2` tasks, since `ΔW` separation does
not need a long stream. Merges into `results/phase0.json`.
"""

from __future__ import annotations

import json
from pathlib import Path

from _par import n_workers, pin_threads, pmap

pin_threads()

import numpy as np  # noqa: E402

from src.manifolds import dichotomies, generator  # noqa: E402
from src.models import MODULES, ScalingConfig, TwoModuleNet, paired_init  # noqa: E402
from src.train import TrainConfig, run_stream  # noqa: E402

P, D_AMB, M, N = 16, 150, 150, 300
D_INT, R = 4, 1.0
GAMMAS = (0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
SEEDS = (0, 1, 2)
T_TASKS = 2
MAX_STEPS = 5000
LR0 = 5.0        # lr0=0.2 could not reach target_loss below gamma~3; see LOG
TARGET_LOSS = 0.05


def _job(spec):
    gamma, seed = spec
    streams = paired_init(seed)
    arr = generator.make_arrangement(P, D_AMB, D_INT, R, M, streams["data"],
                                     rho_C=0.0, rho_A=0.0)
    cfg = {m: ScalingConfig(N=N, d=D_AMB, gamma_0=gamma, lr0=LR0) for m in MODULES}
    model = TwoModuleNet.init(cfg, streams["shape"])
    ys = np.stack([dichotomies.sample_balanced(P, streams["stream"])
                   for _ in range(T_TASKS)])
    tcfg = TrainConfig(steps_per_task=MAX_STEPS, record_every=200,
                       stopping="matched_loss", target_loss=TARGET_LOSS)
    tasks, boundaries = run_stream(model, [arr.points] * T_TASKS, ys, tcfg,
                                   streams["data"])
    return {"gamma": gamma, "seed": seed,
            "dW": boundaries[-1].weight_change["A"],
            "all_converged": all(t.converged for t in tasks),
            "final_loss": [t.final_loss for t in tasks],
            "steps": [t.steps_taken for t in tasks],
            "total_steps": sum(t.steps_taken for t in tasks)}


def main() -> None:
    print(f"matched-loss richness at P={P}, N={N}, M={M}, T={T_TASKS}, "
          f"target_loss={TARGET_LOSS}, lr0={LR0}, cap={MAX_STEPS}, {n_workers()} workers")
    raw = pmap(_job, [(g, s) for g in GAMMAS for s in SEEDS])

    per_gamma = {}
    print("\n  gamma    dW/W                  steps    loss     converged")
    for g in GAMMAS:
        sel = [r for r in raw if r["gamma"] == g]
        v = np.array([r["dW"] for r in sel])
        st = np.array([r["total_steps"] for r in sel])
        ls = np.array([max(r["final_loss"]) for r in sel])
        conv = all(r["all_converged"] for r in sel)
        per_gamma[str(g)] = {
            "dW_mean": float(v.mean()), "dW_sd": float(v.std(ddof=1)),
            "total_steps_mean": float(st.mean()),
            "worst_final_loss": float(ls.max()),
            "all_tasks_converged": bool(conv),
        }
        print(f"  {g:<7} {v.mean():.5f} +- {v.std(ddof=1):.5f}   {st.mean():7.0f}  "
              f"{ls.max():.4f}   {conv}")

    lo = per_gamma[str(GAMMAS[0])]["dW_mean"]
    hi = per_gamma[str(GAMMAS[-1])]["dW_mean"]
    decades = float(np.log10(hi / lo)) if lo > 0 else float("inf")
    conv_all = all(v["all_tasks_converged"] for v in per_gamma.values())
    steps = np.array([per_gamma[str(g)]["total_steps_mean"] for g in GAMMAS])
    step_ratio = float(steps.max() / steps.min())
    passed = decades >= 1.0

    print(f"\n  dW spread {decades:.2f} decades (need >= 1) -> "
          f"{'PASS' if passed else 'FAIL'}")
    print(f"  steps-to-target spread {step_ratio:.1f}x across gamma")
    print(f"  matched loss reached at every gamma: {conv_all}")

    out = {
        "per_gamma": per_gamma, "decades": decades, "passed": passed,
        "all_converged_everywhere": conv_all,
        "steps_spread_ratio": step_ratio,
        "stopping": "matched_loss", "target_loss": TARGET_LOSS,
        "max_steps": MAX_STEPS, "lr0": LR0, "T": T_TASKS, "raw": raw,
        "failure_action": "parameterization bug -- STOP",
        "note": "stopping on accuracy is invalid here: one step from u=0 gives the "
                "kernel readout, whose sign is lr-independent, so accuracy "
                "saturates at step 1 identically for every gamma",
    }

    path = Path(__file__).resolve().parents[1] / "results" / "phase0.json"
    d = json.loads(path.read_text())
    d["richness_separation"] = out
    d["summary"]["richness_separation"] = passed
    d["config"].update({"stopping": "matched_loss", "target_loss": TARGET_LOSS,
                        "richness_T": T_TASKS, "richness_max_steps": MAX_STEPS,
                        "lr0": LR0})
    path.write_text(json.dumps(d, indent=2, default=float) + "\n")
    print(f"\nupdated {path}")


if __name__ == "__main__":
    main()
