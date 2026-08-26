"""Is the under-recovery of large D and R sampling error or estimator bias?

`results/glue_core_recovery.json` shows `D_eff` and `R_eff` over-report small
ground-truth values and under-report large ones. Two candidate explanations, and
they have different consequences:

- **sampling** — closes as `n_t` grows; fix by raising `n_t` in high-D conditions;
- **bias** — persists; goes in the paper as a stated caveat.

It matters because the stream manipulates `D`: a bias correlated with a
manipulated variable is a confound, not a nuisance.

`n_t ∈ {200, 1000}` is covered by the recovery script. This one adds two axes that
script cannot:

- **P** — the B.5 protocol runs at `P = 2`, where `S` has only two rows, but the
  project runs at `P = 16`. Whether the compression under the authors' protocol is
  the compression at *our* P is a separate question.
- **M** — the candidate mechanism. Anchors are extreme points of a *finite* sample
  of each manifold, and a finite sample under-represents the extent of a
  high-dimensional or large-radius manifold. If that is the cause, compression
  must shrink as `M` grows at fixed `D`; if it does not, the bias is intrinsic.

Writes `results/scale_compression.json`.
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

N, M, N_T = 1000, 100, 200
SEEDS = (0, 1, 2)
P_VALUES = (2, 8, 16)
D_SWEEP = (2, 4, 6, 8, 10)
R_SWEEP = (0.8, 1.0, 1.4, 1.7, 2.0)


M_SWEEP = (50, 100, 200, 400, 800)
M_SWEEP_D = (2, 6, 10)


def _job(spec):
    axis, value, P, seed = spec
    D = int(value) if axis == "D" else 4
    R = float(value) if axis == "R" else 1.0
    rng = np.random.default_rng(seed)
    arr = generator.make_arrangement(P, N, D, R, M, rng)
    res = core.glue_measures(
        core.from_arrangement(arr.points), np.random.default_rng(seed + 1000), n_t=N_T
    )
    return {"axis": axis, "value": float(value), "P": P, "seed": seed,
            "D_eff": res.D_eff, "R_eff": res.R_eff, "alpha": res.alpha,
            "Psi_eff": res.Psi_eff}


def _job_M(spec):
    D, m, seed = spec
    rng = np.random.default_rng(seed)
    arr = generator.make_arrangement(2, N, D, 1.0, m, rng)
    res = core.glue_measures(
        core.from_arrangement(arr.points), np.random.default_rng(seed + 1000), n_t=N_T
    )
    return {"D": D, "M": m, "seed": seed, "D_eff": res.D_eff, "R_eff": res.R_eff}


def main() -> None:
    t0 = time.perf_counter()
    specs = (
        [("D", v, P, s) for v in D_SWEEP for P in P_VALUES for s in SEEDS]
        + [("R", v, P, s) for v in R_SWEEP for P in P_VALUES for s in SEEDS]
    )
    print(f"[{len(specs)} estimations]", flush=True)
    raw = pmap(_job, specs)

    out = {
        "settings": {"N": N, "M": M, "n_t": N_T, "seeds": list(SEEDS),
                     "P_values": list(P_VALUES), "center_policy": "all"},
        "question": "does the D/R compression depend on P, or is it fixed by the estimator?",
        "axes": {},
    }

    for axis, values, target in (("D", D_SWEEP, "D_eff"), ("R", R_SWEEP, "R_eff")):
        out["axes"][axis] = {}
        print(f"\n[{axis} -> {target}]   ratio estimate/truth   (N={N}, M={M}, n_t={N_T})")
        print("  P    " + "".join(f"{axis}={v:<8}" for v in values) + " mean|rel err|")
        for P in P_VALUES:
            rows, ratios = [], []
            for v in values:
                sel = [r for r in raw if (r["axis"], r["value"], r["P"]) == (axis, float(v), P)]
                mean = float(np.mean([r[target] for r in sel]))
                rows.append({"ground_truth": float(v), "mean": mean,
                             "sd": float(np.std([r[target] for r in sel], ddof=1)),
                             "ratio": mean / float(v)})
                ratios.append(abs(mean - float(v)) / float(v))
            out["axes"][axis][f"P={P}"] = {
                "points": rows,
                "mean_abs_rel_error": float(np.mean(ratios)),
                "compression_slope": float(
                    np.polyfit([r["ground_truth"] for r in rows],
                               [r["ratio"] for r in rows], 1)[0]
                ),
            }
            cells = "".join(f"{r['ratio']:<10.3f}" for r in rows)
            print(f"  {P:<4} {cells} {np.mean(ratios)*100:5.1f}%")

    # ---- mechanism: does more manifold sampling reduce the compression? -----
    print(f"\n[M sweep at P=2]   D_eff/D   (N={N}, n_t={N_T})")
    m_raw = pmap(_job_M, [(D, m, s) for D in M_SWEEP_D for m in M_SWEEP for s in SEEDS])
    out["M_sweep"] = {}
    print("  D    " + "".join(f"M={m:<8}" for m in M_SWEEP))
    for D in M_SWEEP_D:
        pts = []
        for m in M_SWEEP:
            sel = [r for r in m_raw if (r["D"], r["M"]) == (D, m)]
            mean = float(np.mean([r["D_eff"] for r in sel]))
            pts.append({"M": m, "D_eff": mean, "ratio": mean / D,
                        "sd": float(np.std([r["D_eff"] for r in sel], ddof=1))})
        out["M_sweep"][f"D={D}"] = pts
        print("  " + f"{D:<4} " + "".join(f"{p['ratio']:<10.3f}" for p in pts))

    out["verdict"] = {
        "n_t": "results/glue_core_recovery.json: n_t 200 -> 1000 does not close the "
               "gap, so it is not Monte-Carlo sampling of (y, t)",
        "P": "compression is nearly P-independent, so it is not an artifact of the "
             "B.5 protocol's P = 2",
        "M": "see M_sweep: whether finite sampling of each manifold is the mechanism",
    }
    out["elapsed_s"] = round(time.perf_counter() - t0, 1)

    path = Path(__file__).resolve().parents[1] / "results" / "scale_compression.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nwrote {path}  ({out['elapsed_s']}s)")


if __name__ == "__main__":
    main()
