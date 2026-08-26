"""W4 — the per-γ measurement null: how much does the estimator move on a fixed representation?

Every "noise floor" claim in this project divides by `timewarp.NOISE_FLOOR_CV`, which was
measured at one setting. If the estimator's Monte-Carlo dispersion depends on γ, then a change
quoted as "50 floors at γ=10" and one quoted as "5 floors at γ=0.3" are being measured with
different rulers, and the γ-dependence of the ruler is silently inside the γ-dependence of the
result. This measures the ruler at every registered γ.

**The design isolates measurement from everything else.** One arm per γ, trained once; then the
*same* trained network, the *same* manifolds, re-measured under several measurement seeds. The
only thing that varies is the anchor-point draw inside the GLUE estimator, so the dispersion is
attributable to measurement and nothing else — not initialization, not stream, not training.

It also supplies the per-γ null the module A-vs-B comparison needs. That comparison currently
has a null only at γ=1, because every `a>0` arm sits there, so an A-vs-B difference at another γ
cannot presently be called resolved or not.

    python scripts/run_measurement_null.py [--seeds 4]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from _par import pin_threads  # noqa: E402

pin_threads()

import numpy as np  # noqa: E402

from _par import pmap  # noqa: E402
from src import pipeline as pl  # noqa: E402
from src import provenance  # noqa: E402
from src.analysis.grid import REGISTERED_GAMMAS  # noqa: E402

CHANNELS = ("alpha", "D_eff", "R_eff", "Psi_eff", "rho_c_signed")
OUT = ROOT / "results" / "measurement_null.json"


def _job(args: tuple[float, int]) -> dict:
    """Train one arm at γ, then re-measure its final representation under `n_seeds` seeds."""
    gamma, n_seeds = args
    provenance.assert_current()
    spec = pl.Phase1Spec(gamma_0=gamma, a=0.0, condition="S-HL", stream_id=0, seed=0)

    t0 = time.time()
    streams = pl.paired_init(spec.seed)
    stream = pl.build_stream(spec, streams["stream"])
    model = pl.build_model(spec, stream)
    converged = []
    for t in range(spec.T):
        pts, y = stream.arrangements[t].points, stream.dichotomies[t]
        rec = pl.train_task(model, pts, y, spec.train_config, streams["data"], task_index=t)
        converged.append(bool(rec.converged))
    t_train = time.time() - t0

    # The trained network is now fixed. Everything below re-measures it.
    last = spec.T - 1
    out: dict[str, list[dict]] = {}
    t0 = time.time()
    for s in range(n_seeds):
        # Seeds far apart in the integer line, and unrelated to the pipeline's own
        # `meas_base`, so this is not accidentally re-running one of its draws.
        rng_seed = 700_000_003 + 7919 * s
        for module in spec.module_list:
            for task, ens in ((0, "retained"), (None, "generic")):
                pts = stream.arrangements[0 if task == 0 else last].points
                y = stream.dichotomies[0] if task == 0 else None
                g = pl._measure(model, pts, y, spec, module, last, task, rng_seed)
                out.setdefault(f"{module}|{ens}", []).append(asdict(g))
    return {"gamma": gamma, "converged": all(converged), "n_seeds": n_seeds,
            "train_seconds": t_train, "measure_seconds": time.time() - t0,
            "cells": out}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--cap", type=int, default=None)
    args = ap.parse_args()

    jobs = [(g, args.seeds) for g in REGISTERED_GAMMAS]
    print(f"measurement null: {len(jobs)} arms (one per registered γ) × {args.seeds} "
          f"measurement seeds", flush=True)
    t0 = time.time()
    res = pmap(_job, jobs, cap=args.cap)
    print(f"  {time.time() - t0:.0f} s wall\n", flush=True)

    summary = {}
    for r in res:
        per = {}
        for cell, recs in r["cells"].items():
            per[cell] = {}
            for ch in CHANNELS:
                v = np.array([x[ch] for x in recs], dtype=float)
                mean = float(np.mean(v))
                sd = float(np.std(v, ddof=1))
                per[cell][ch] = {"mean": mean, "sd": sd,
                                 "cv": abs(sd / mean) if mean else float("nan"),
                                 "sd_log": float(np.std(np.log(np.abs(v)), ddof=1))
                                 if np.all(v > 0) else None}
        summary[f"{r['gamma']:g}"] = {"converged": r["converged"],
                                      "train_seconds": r["train_seconds"],
                                      "measure_seconds": r["measure_seconds"], "cells": per}

    OUT.write_text(json.dumps({"generated_by": "scripts/run_measurement_null.py",
                               "n_seeds": args.seeds, "condition": "S-HL, a=0, seed 0",
                               "measured_at": "final boundary, both modules",
                               "code": provenance.code_stamp(),
                               "per_gamma": summary}, indent=2))
    print(f"wrote {OUT.relative_to(ROOT)}\n")

    print("coefficient of variation over measurement seeds, module A generic:")
    print(f"  {'γ':>6} " + " ".join(f"{c:>12}" for c in CHANNELS) + "  converged")
    for g in REGISTERED_GAMMAS:
        c = summary[f"{g:g}"]["cells"]["A|generic"]
        print(f"  {g:>6g} " + " ".join(f"{c[ch]['cv']:>12.5f}" for ch in CHANNELS)
              + f"  {summary[f'{g:g}']['converged']}")
    print("\nsame, module A retained (task 0):")
    print(f"  {'γ':>6} " + " ".join(f"{c:>12}" for c in CHANNELS))
    for g in REGISTERED_GAMMAS:
        c = summary[f"{g:g}"]["cells"]["A|retained"]
        print(f"  {g:>6g} " + " ".join(f"{c[ch]['cv']:>12.5f}" for ch in CHANNELS))


if __name__ == "__main__":
    main()
