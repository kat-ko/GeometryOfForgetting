"""Phase 0's fifth check — the time-reparameterization test (`00` §12).

Determines whether H3 is testable at all. Atanasov et al.'s claim is about
**large** `γ`, so the spec-relevant pairs are those with both `γ ≳ 1`; `γ = 0.03`
is included as the lazy contrast, where the trajectories are expected to differ
for uninteresting reasons and which therefore acts as a positive control.

`01` Phase 0 measured a ~40× spread in steps-to-target across `γ`, so the
parameter-free "measured" warp has real content here rather than being a guess.

Writes `results/timewarp.json`.
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

from _par import n_workers, pin_threads, pmap

pin_threads()

import numpy as np  # noqa: E402

from src.analysis import timewarp, trajectories  # noqa: E402
from src.manifolds import dichotomies, generator  # noqa: E402
from src.models import MODULES, ScalingConfig, TwoModuleNet, paired_init  # noqa: E402

P, D_AMB, M, N = 16, 150, 150, 300
D_INT, R = 4, 1.0
LR0, TARGET_LOSS, N_T = 5.0, 0.05, 200
GAMMAS = (0.03, 1.0, 3.0, 10.0)
LARGE = (1.0, 3.0, 10.0)          # the pairs `00` §12 is actually about
SEEDS = (0, 1, 2)
MAX_STEPS = 4000
N_CHECKPOINTS = 9


def _job(spec):
    gamma, seed = spec
    streams = paired_init(seed)
    arr = generator.make_arrangement(P, D_AMB, D_INT, R, M, streams["data"],
                                     rho_C=0.0, rho_A=0.0)
    cfg = {m: ScalingConfig(N=N, d=D_AMB, gamma_0=gamma, lr0=LR0) for m in MODULES}
    model = TwoModuleNet.init(cfg, streams["shape"])
    y = dichotomies.sample_balanced(P, streams["stream"])
    traj = trajectories.record_geometry_trajectory(
        model, arr.points, y, streams["data"],
        checkpoints=trajectories.log_checkpoints(MAX_STEPS, N_CHECKPOINTS),
        n_t=N_T, measure_rng_seed=seed,
    )
    traj.gamma = gamma
    return traj


def main() -> None:
    path = Path(__file__).resolve().parents[1] / "results" / "timewarp.json"
    reuse = "--reanalyze" in sys.argv and path.exists()
    if reuse:
        # Trajectories are the expensive part and are fully recorded, so the
        # analysis can be revised without retraining.
        print(f"reusing trajectories from {path}")
        trajs = [trajectories.GeometryTrajectory.from_dict(t)
                 for t in json.loads(path.read_text())["trajectories"]]
    else:
        print(f"trajectories at P={P}, N={N}, M={M}, gammas={GAMMAS}, "
              f"{N_CHECKPOINTS} checkpoints to {MAX_STEPS} steps, "
              f"{n_workers()} workers")
        trajs = pmap(_job, [(g, s) for g in GAMMAS for s in SEEDS])
    by = {(t.gamma, t.seed): t for t in trajs}

    print("\n  geometry excursion over the window, in noise floors "
          "(< 3 makes the warp test vacuous):")
    for g in GAMMAS:
        exc = [by[(g, s)].excursion_in_floors(timewarp.NOISE_FLOOR_CV) for s in SEEDS]
        print(f"    gamma={g:<6} {np.mean(exc):8.1f} +- {np.std(exc, ddof=1):.1f}"
              f"{'   <-- barely moves' if np.mean(exc) < 3 else ''}")

    print("\n  gamma  seed   loss at first/last checkpoint   D_eff first->last")
    for g in GAMMAS:
        for s in SEEDS:
            t = by[(g, s)]
            print(f"  {g:<6} {s}      {t.loss[0]:.4f} -> {t.loss[-1]:.4f}"
                  f"            {t.channels['D_eff'][0]:.3f} -> "
                  f"{t.channels['D_eff'][-1]:.3f}")

    print("\n  residuals in units of the noise floor (<= 1.0 means coincide)")
    print("  pair            measured(scale)   best_rate(scale)   best_monotone   verdict")
    tests = []
    for ga, gb in itertools.combinations(GAMMAS, 2):
        per_seed = [timewarp.time_reparameterization_test(
            by[(ga, s)], by[(gb, s)], target_loss=TARGET_LOSS) for s in SEEDS]
        agg = {"gamma_a": ga, "gamma_b": gb,
               "both_large": ga in LARGE and gb in LARGE,
               "per_seed": per_seed}
        for i, name in enumerate(("measured", "best_rate", "best_monotone_dtw")):
            vals = np.array([p["warps"][i]["residual_in_floors"] for p in per_seed])
            scales = [p["warps"][i]["scale"] for p in per_seed]
            agg[name] = {"mean": float(vals.mean()), "sd": float(vals.std(ddof=1)),
                         "scale_mean": (float(np.mean([s for s in scales if s]))
                                        if scales[0] else None)}
        agg["vacuous"] = all(p["vacuous"] for p in per_seed)
        agg["h3_testable"] = all(p["h3_testable"] for p in per_seed)
        agg["verdict"] = ("VACUOUS - a trajectory barely moves" if agg["vacuous"]
                          else "DIFFER - no monotone warp aligns them"
                          if agg["h3_testable"] else "COINCIDE")
        tests.append(agg)
        sc_m = agg["measured"]["scale_mean"]
        sc_r = agg["best_rate"]["scale_mean"]
        print(f"  g={ga:<5}vs g={gb:<5} {agg['measured']['mean']:6.2f}"
              f"({sc_m:6.1f})   {agg['best_rate']['mean']:6.2f}({sc_r:6.1f})   "
              f"{agg['best_monotone_dtw']['mean']:8.2f}      {agg['verdict']}")

    large = [t for t in tests if t["both_large"] and not t["vacuous"]]
    differ = [t for t in large if t["h3_testable"]]
    # Existence, not universality: H3 needs *some* large-gamma pair whose
    # trajectories differ in shape. A pair where both runs barely move cannot
    # distinguish shape from rate and is excluded above, not counted as agreement.
    h3_alive = len(differ) > 0
    print("\n" + "=" * 70)
    print(f"  Informative pairs with both gamma >= 1 (what 00 §12 is about): "
          f"{len(large)} of {len([t for t in tests if t['both_large']])}")
    print(f"  ... of which trajectories DIFFER: {len(differ)} "
          f"({', '.join(f'{t[chr(103)+chr(97)+chr(109)+chr(109)+chr(97)+chr(95)+chr(97)]} vs {t[chr(103)+chr(97)+chr(109)+chr(109)+chr(97)+chr(95)+chr(98)]}' for t in differ)})")
    print(f"  H3 TESTABLE: {h3_alive}")
    if h3_alive:
        print("  -> trajectories at different large gamma are not one trajectory at "
              "two speeds.\n     The division-of-labour hypothesis survives; "
              "heterogeneity contrasts are unrestricted.")
    else:
        print("  -> restrict all heterogeneity contrasts to gamma << 1 vs gamma ~ 1 "
              "and record\n     that H3 is not testable in the large-gamma range "
              "(00 §12 consequence).")

    out = {"config": {"P": P, "N": N, "M": M, "d": D_AMB, "D": D_INT, "R": R,
                      "lr0": LR0, "target_loss": TARGET_LOSS, "n_t": N_T,
                      "gammas": list(GAMMAS), "seeds": list(SEEDS),
                      "max_steps": MAX_STEPS, "checkpoints": N_CHECKPOINTS},
           "trajectories": [t.to_dict() for t in trajs],
           "tests": tests,
           "h3_testable_in_large_gamma": h3_alive,
           "informative_large_pairs": len(large),
           "differing_large_pairs": [[t["gamma_a"], t["gamma_b"]] for t in differ],
           "consequence": ("heterogeneity contrasts unrestricted, but must use "
                           "gamma pairs whose trajectories differ" if h3_alive else
                           "restrict contrasts to gamma << 1 vs gamma ~ 1")}
    path.write_text(json.dumps(out, indent=2, default=float) + "\n")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
