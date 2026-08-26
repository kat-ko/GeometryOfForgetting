"""Is the `pairwise` / `full_P` offset constant across geometries?

Phase 0 found that `pairwise` inflates `D_eff` and `Ψ_eff` by the same factor
(1.2609 and 1.2676), which cancels in `α` to 0.5%. That was **one** geometry, and
one point cannot distinguish a constant offset from a varying one.

It matters because Figure 2 attributes forgetting to exactly those two channels in
log space. If the factor is constant it cancels in `Δlog D_eff` and the
attribution is mode-independent; if it drifts with geometry it contaminates the
headline result in a way no downstream analysis can detect.

Design: 6 synthetic points spanning the Phase 1 range on the three axes that move
the geometry (`D`, `R`, `ρ_C`), plus 3 **network representations** — at init and
after training at the lazy and rich ends — since Phase 1 measures representations,
not generated manifolds.

The decisive statistic is not the ratio's spread but whether `Δlog D_eff` between
conditions agrees across modes, because that is the quantity the attribution uses.

Writes `results/mode_constancy.json`.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

from _par import n_workers, pin_threads, pmap

pin_threads()

import numpy as np  # noqa: E402

from src.glue import core  # noqa: E402
from src.manifolds import dichotomies, generator  # noqa: E402
from src.models import MODULES, ScalingConfig, TwoModuleNet, paired_init  # noqa: E402
from src.train import TrainConfig, train_task  # noqa: E402

P, D_AMB, M, N = 16, 150, 150, 300
# Synthetic manifolds are placed in the *measurement* ambient dimension N, not the
# input dimension d. At d = 150 the arrangement is nearly degenerate for large D:
# P(D+1) = 144 of 150 dimensions at D = 8, so the manifolds are no longer in
# general position and D_eff collapses for reasons that have nothing to do with
# estimation mode. (B.5 uses N = 1000 for the same reason.)
AMBIENT = N
N_T = 200
SEEDS = (0, 1, 2)
LR0, TARGET_LOSS = 5.0, 0.05

# (label, D, R, rho_C) — one axis moved at a time from the Phase 1 base point.
SYNTHETIC = [
    ("base",     4, 1.0, 0.0),
    ("D_low",    2, 1.0, 0.0),
    ("D_high",   8, 1.0, 0.0),
    ("R_low",    4, 0.5, 0.0),
    ("R_high",   4, 1.5, 0.0),
    ("rho_high", 4, 1.0, 0.4),
]
# Representations: at init (gamma-independent) and trained at both ends of gamma.
REPRESENTATION = [("rep_init", None), ("rep_lazy", 0.03), ("rep_rich", 10.0)]
KEYS = ("alpha", "D_eff", "R_eff", "Psi_eff", "rho_c_glue", "rho_c_signed")


def _manifolds(label, seed):
    streams = paired_init(seed)
    spec = next((s for s in SYNTHETIC if s[0] == label), None)
    if spec is not None:
        _, D, R, rho_C = spec
        arr = generator.make_arrangement(P, AMBIENT, D, R, M, streams["data"],
                                         rho_C=rho_C, rho_A=0.0)
        return core.from_arrangement(arr.points)

    gamma = dict(REPRESENTATION)[label]
    arr = generator.make_arrangement(P, D_AMB, 4, 1.0, M, streams["data"],
                                     rho_C=0.0, rho_A=0.0)
    cfg = {m: ScalingConfig(N=N, d=D_AMB, gamma_0=gamma or 1.0, lr0=LR0)
           for m in MODULES}
    model = TwoModuleNet.init(cfg, streams["shape"])
    if gamma is not None:
        y = dichotomies.sample_balanced(P, streams["stream"])
        train_task(model, arr.points, y,
                   TrainConfig(steps_per_task=5000, record_every=500,
                               stopping="matched_loss", target_loss=TARGET_LOSS),
                   streams["data"])
    return model.manifold_representation(arr.points, module="A")


def _job(spec):
    label, seed, mode = spec
    mans = _manifolds(label, seed)
    rng = np.random.default_rng(hash((label, seed)) % (2**32))
    if mode == "full_P":
        r = core.glue_measures(mans, rng, n_t=N_T)
        vals = {k: getattr(r, k) for k in KEYS}
    else:
        vals = core.pairwise_measures(mans, rng, n_t=N_T)
    return {"label": label, "seed": seed, "mode": mode,
            **{k: float(vals[k]) for k in KEYS}}


def main() -> None:
    labels = [s[0] for s in SYNTHETIC] + [r[0] for r in REPRESENTATION]
    print(f"mode constancy over {len(labels)} geometries x {len(SEEDS)} seeds, "
          f"{n_workers()} workers")
    raw = pmap(_job, [(lab, s, m) for lab in labels for s in SEEDS
                      for m in ("full_P", "pairwise")])

    def get(label, mode, key):
        return np.array([r[key] for r in raw
                         if (r["label"], r["mode"]) == (label, mode)])

    per_geometry, ratios = {}, {"D_eff": [], "Psi_eff": [], "alpha": [], "R_eff": []}
    print("\n  geometry     D_eff full_P  pairwise   ratio    Psi ratio   alpha ratio")
    for lab in labels:
        row = {}
        for k in KEYS:
            f, p = get(lab, "full_P", k), get(lab, "pairwise", k)
            row[k] = {"full_P": float(f.mean()), "full_P_sd": float(f.std(ddof=1)),
                      "pairwise": float(p.mean()), "pairwise_sd": float(p.std(ddof=1)),
                      "ratio": float(p.mean() / f.mean())}
        for k in ratios:
            ratios[k].append(row[k]["ratio"])
        per_geometry[lab] = row
        print(f"  {lab:<12} {row['D_eff']['full_P']:9.3f}  {row['D_eff']['pairwise']:8.3f}  "
              f"{row['D_eff']['ratio']:7.4f}  {row['Psi_eff']['ratio']:9.4f}  "
              f"{row['alpha']['ratio']:10.4f}")

    # --- constancy of the ratio -------------------------------------------
    print("\n  ratio spread across geometries (CV%):")
    spread = {}
    for k, v in ratios.items():
        arr = np.array(v)
        spread[k] = {"mean": float(arr.mean()), "sd": float(arr.std(ddof=1)),
                     "cv_pct": float(100 * arr.std(ddof=1) / arr.mean()),
                     "min": float(arr.min()), "max": float(arr.max())}
        print(f"    {k:<10} mean {arr.mean():.4f}  CV {spread[k]['cv_pct']:5.2f}%  "
              f"range [{arr.min():.4f}, {arr.max():.4f}]")

    # --- the decisive test: does Delta-log agree across modes? -------------
    print("\n  Delta log D_eff between geometry pairs (the attribution's quantity):")
    deltas = []
    for a, b in itertools.combinations(labels, 2):
        df = np.log(per_geometry[b]["D_eff"]["full_P"] / per_geometry[a]["D_eff"]["full_P"])
        dp = np.log(per_geometry[b]["D_eff"]["pairwise"] / per_geometry[a]["D_eff"]["pairwise"])
        deltas.append({"from": a, "to": b, "full_P": float(df), "pairwise": float(dp),
                       "abs_error": float(abs(dp - df))})
    err = np.array([d["abs_error"] for d in deltas])
    big = sorted(deltas, key=lambda d: -d["abs_error"])[:4]
    for d in big:
        print(f"    {d['from']:>10} -> {d['to']:<10} full_P {d['full_P']:+.4f}  "
              f"pairwise {d['pairwise']:+.4f}  err {d['abs_error']:.4f}")
    print(f"    max |error| in Delta log D_eff = {err.max():.4f} "
          f"(mean {err.mean():.4f})")

    # A 1.26% error in Delta log D_eff is the D_eff noise floor (results/cost_model).
    floor = np.log1p(0.0126)
    verdict_constant = bool(err.max() <= 2 * floor)
    print(f"\n  noise-floor equivalent for Delta log D_eff: {floor:.4f} "
          f"(2x = {2*floor:.4f})")
    print(f"  VERDICT: offset is {'CONSTANT' if verdict_constant else 'NOT constant'} "
          f"across the Phase 1 geometry range")
    if verdict_constant:
        print("  -> cancels in Delta log; attribution is mode-independent. Record "
              "the cancellation argument for the appendix.")
    else:
        print("  -> full_P is MANDATORY for all Tier-2 measurement; pairwise is "
              "reported only for comparability with published numbers.")

    out = {"config": {"P": P, "N": N, "M": M, "d": D_AMB, "n_t": N_T,
                      "seeds": list(SEEDS), "lr0": LR0,
                      "ambient_synthetic": AMBIENT,
                      "synthetic": [list(s) for s in SYNTHETIC],
                      "representations": [list(r) for r in REPRESENTATION]},
           "per_geometry": per_geometry, "ratio_spread": spread,
           "delta_log_D_eff": deltas,
           "max_abs_delta_log_error": float(err.max()),
           "noise_floor_delta_log": float(floor),
           "offset_is_constant": verdict_constant,
           "consequence": ("cancels in Delta log; attribution mode-independent"
                           if verdict_constant else
                           "full_P mandatory for Tier-2; pairwise for comparability only"),
           "raw": raw}
    path = Path(__file__).resolve().parents[1] / "results" / "mode_constancy.json"
    path.write_text(json.dumps(out, indent=2, default=float) + "\n")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
