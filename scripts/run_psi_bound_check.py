"""Is retained `Ψ_eff > 1` a property of the fixed-`y` ensemble, or a defect?

The grid puts 21.6% of retained measurements above 1 (max 1.78) while the generic
ensemble never exceeds 0.655, with the identity closing to 4e-16 throughout. The
`Ψ_eff ∈ [0,1]` range was exported from the GLUE papers, where capacity is an
expectation over *random* dichotomies. The hypothesis under test is that the bound
depends on that label average and does not survive fixing `y`: `a` uses
`(S_y S_yᵀ)†` while `c` uses `(S_{y,0}S_{y,0}ᵀ + S_{y,1}S_{y,1}ᵀ)†`, and the two
differ by center-axis cross-terms that plausibly vanish under `E_y` but survive at
fixed `y`.

Two checks, following `results/LOG.md`:

- **α cross-check** (`--real`): on a real γ=10 arm at its own boundary, where every
  lag-0 measurement exceeds 1, does `α_sim` at fixed `y` agree with the GLUE `α`?
  Agreement means the capacity is right and only `Ψ_eff`'s interpretive range needs
  rescoping — the identity guarantees the decomposition either way.
- **Synthetic pin** (`--synthetic`): on ground truth deliberately organized for one
  dichotomy, is fixed-`y` `Ψ_eff > 1` while generic `Ψ_eff ≤ 1` on the *same*
  arrangement? That converts the cross-term hypothesis into a demonstrated mechanism.

Run with neither flag to do both.
"""

from __future__ import annotations

import argparse
import glob
import json
import time

import numpy as np

from src import provenance
from src.glue.adapters.simcap import simcap
from src.glue.core import Ensemble, glue_measures
from src.models import paired_init
from src.pipeline import Phase1Spec, build_model, build_stream
from src.train.loop import train_task

_SOURCE = provenance.register(__file__)


def aligned_arrangement(
    P: int, N: int, M: int, y: np.ndarray, sep: float, radius: float,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """`P` manifolds whose centers are separated *along* the `y` direction.

    Centers sit at `±sep·u` by the sign of `y_μ`, so the `y` dichotomy is read off a
    single direction while a random dichotomy has to cut a cloud that is, in that
    direction, two overlapping blobs. Axes are isotropic and shared in scale, so the
    only thing distinguishing the ensembles is center placement relative to `y`.
    """
    u = rng.standard_normal(N)
    u /= np.linalg.norm(u)
    mans = []
    for mu in range(P):
        axes = radius * rng.standard_normal((N, M)) / np.sqrt(N)
        mans.append((sep * y[mu] * u)[:, None] + axes)
    return mans


def measure_both(mans, y, seed: int, n_t: int) -> dict:
    """GLUE measures for the generic and the fixed-`y` ensemble on one arrangement."""
    out = {}
    for label, ens in (("generic", Ensemble()),
                       ("retained", Ensemble(kind="retained", y_ref=np.asarray(y, float)))):
        r = glue_measures(mans, np.random.default_rng(seed), n_t=n_t, ensemble=ens)
        out[label] = dict(alpha=r.alpha, D_eff=r.D_eff, R_eff=r.R_eff,
                          Psi_eff=r.Psi_eff, identity_residual=r.identity_residual)
    return out


def synthetic_pin(args) -> dict:
    """Does an arrangement built for one dichotomy put fixed-`y` Ψ_eff above 1?"""
    rng = np.random.default_rng(args.seed)
    P, N, M = args.P, args.N, args.M
    y = np.sign(rng.standard_normal(P))
    y[y == 0] = 1.0

    rows = []
    print(f"synthetic pin: P={P} N={N} M={M}, centers along the y direction")
    print(f"{'sep/radius':>11} | {'generic Ψ':>10} {'retained Ψ':>11} | "
          f"{'generic α':>10} {'retained α':>11} | {'max|resid|':>10}")
    for sep in args.sep:
        mans = aligned_arrangement(P, N, M, y, sep, args.radius, rng)
        m = measure_both(mans, y, args.seed, args.n_t)
        resid = max(abs(m['generic']['identity_residual']),
                    abs(m['retained']['identity_residual']))
        print(f"{sep / args.radius:>11.2f} | {m['generic']['Psi_eff']:>10.4f} "
              f"{m['retained']['Psi_eff']:>11.4f} | {m['generic']['alpha']:>10.4f} "
              f"{m['retained']['alpha']:>11.4f} | {resid:>10.1e}")
        rows.append(dict(sep=sep, ratio=sep / args.radius, **m))

    gen_max = max(r["generic"]["Psi_eff"] for r in rows)
    ret_max = max(r["retained"]["Psi_eff"] for r in rows)
    print(f"\n  generic Ψ_eff max {gen_max:.4f} (bound holds: {gen_max <= 1.0})")
    print(f"  retained Ψ_eff max {ret_max:.4f} (exceeds 1: {ret_max > 1.0})")
    if ret_max > 1.0 and gen_max <= 1.0:
        print("  => MECHANISM DEMONSTRATED on ground truth: the bound is a property of\n"
              "     the label average, not of the estimator.")
    return dict(check="synthetic_pin", P=P, N=N, M=M, radius=args.radius, rows=rows,
                generic_max=gen_max, retained_max=ret_max)


def real_arm(args) -> dict:
    """Train one real γ=10 arm to its first boundary and cross-check α at fixed `y`."""
    cands = []
    for f in glob.glob("results/phase1/*.json"):
        rec = json.load(open(f))
        if rec["spec"]["gamma_0"] == args.gamma and rec["spec"]["a"] == 0.0:
            over = [g for g in rec["geometry"]
                    if g["ensemble"] == "retained" and g["task"] == g["boundary"]
                    and g["Psi_eff"] > 1]
            if over:
                cands.append((rec, over))
    if not cands:
        print(f"no γ={args.gamma}, a=0 arm on disk with a lag-0 retained Ψ_eff > 1")
        return dict(check="real_arm", found=False)

    rec, over = sorted(cands, key=lambda c: c[0]["key"])[0]
    spec = Phase1Spec(**rec["spec"])
    stored = sorted(over, key=lambda g: -g["Psi_eff"])[0]
    print(f"real arm {spec.key}: stored lag-0 retained Ψ_eff = {stored['Psi_eff']:.4f} "
          f"(module {stored['module']}, boundary {stored['boundary']})")

    # Reproduce the arm's own state at boundary 0: same paired init, same stream, and
    # training on task 0 only. Anything past boundary 0 is unnecessary — lag 0 at this
    # γ is where every measurement is above 1.
    streams = paired_init(spec.seed)
    stream = build_stream(spec, streams["stream"])
    model = build_model(spec, stream)
    pts, y = stream.arrangements[0].points, stream.dichotomies[0]
    train_task(model, pts, y, spec.train_config, streams["data"], task_index=0)

    module = stored["module"]
    mans = model.manifold_representation(pts, module)
    ens = Ensemble(kind="retained", y_ref=np.asarray(y, float))
    # `run_arm`'s measurement seed for this (module, task), so the reproduction is exact
    # rather than a fresh Monte Carlo draw — otherwise a `n_t` sampling difference looks
    # like an estimator disagreement.
    seed = (int(np.random.default_rng([spec.seed, 20260811]).integers(2**32))
            + 1013 * spec.module_list.index(module) + 1 + int(stored["boundary"]))
    g = glue_measures(mans, np.random.default_rng(seed), n_t=spec.n_t, ensemble=ens)
    print(f"  reproduced: Ψ_eff {g.Psi_eff:.4f}  α_glue {g.alpha:.4f}  "
          f"D_eff {g.D_eff:.4f}  R_eff {g.R_eff:.4f}  resid {g.identity_residual:.1e}")

    t0 = time.time()
    s = simcap(mans, np.random.default_rng(seed), ensemble=ens, n_rep=args.n_rep)
    print(f"  α_sim at fixed y: {s.alpha_sim:.4f}  (N_c={s.N_c:.1f}, "
          f"{time.time() - t0:.0f}s)")
    rel = abs(g.alpha - s.alpha_sim) / s.alpha_sim
    print(f"  |α_glue − α_sim| / α_sim = {rel:.3f}")
    if rel < args.tol:
        print(f"  => α AGREES within {args.tol:.0%}: the capacity is right where "
              f"Ψ_eff > 1.\n     Only Ψ_eff's interpretive range needs rescoping.")
    else:
        print(f"  => α DISAGREES by more than {args.tol:.0%}: this is a defect in the\n"
              f"     tilted path, not a bound question. STOP.")
    return dict(check="real_arm", found=True, key=spec.key, module=module,
                stored_Psi_eff=stored["Psi_eff"], Psi_eff=g.Psi_eff,
                alpha_glue=g.alpha, alpha_sim=s.alpha_sim, rel_err=rel,
                identity_residual=g.identity_residual, agrees=bool(rel < args.tol))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--real", action="store_true")
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--gamma", type=float, default=10.0)
    p.add_argument("--P", type=int, default=8)
    p.add_argument("--N", type=int, default=120)
    p.add_argument("--M", type=int, default=20)
    p.add_argument("--radius", type=float, default=1.0)
    p.add_argument("--sep", type=float, nargs="+", default=[0.0, 0.5, 1.0, 2.0, 4.0])
    p.add_argument("--n_t", type=int, default=200)
    p.add_argument("--n_rep", type=int, default=10)
    p.add_argument("--tol", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=20260812)
    p.add_argument("--out", default="results/psi_bound_check.json")
    args = p.parse_args()
    both = not (args.real or args.synthetic)

    out = {"code": provenance.code_stamp(), "args": vars(args), "checks": []}
    if both or args.synthetic:
        out["checks"].append(synthetic_pin(args))
        print()
    if both or args.real:
        out["checks"].append(real_arm(args))

    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
