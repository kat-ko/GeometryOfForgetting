"""Parallel throughput vs worker count — correcting the Phase 1 cost model.

`results/cost_model.json` measured 52.4 s/eval **in isolation** and projected wall time
by dividing total core-seconds by 248 workers. That assumes perfect scaling. It does
not hold: measured under a live 254-worker grid, one evaluation took **775.8 s**, a
14.8× slowdown, which turns the projected 2.8–9 h grid into ~49 h.

Two causes, both structural rather than incidental:

- **128 physical cores, not 256.** `nproc` reports 256 on this 2×64-core EPYC 7763
  because of SMT, so 254 workers is already 2× oversubscribed.
- **The anchor QP is not small.** It has `P·M = 2400` variables, so its Gram matrix is
  2400² × 8 B ≈ 46 MB — larger than the 32 MB L3. Every worker streams that from DRAM
  on every one of `n_t` samples, so throughput is memory-bandwidth-bound and adding
  workers past a point buys nothing.

This finds the throughput plateau empirically. `n_t` is reduced to keep the measurement
short; that is legitimate because cost is linear in `n_t` while the memory footprint
that drives contention is set by `P·M`, which is unchanged.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from _par import pin_threads  # noqa: E402

pin_threads()

import multiprocessing as mp  # noqa: E402

import numpy as np  # noqa: E402

N_T = 200  # must match the grid; measuring at n_t=20 mischaracterised the peak
WORKER_COUNTS = (1, 8, 16, 32, 64, 96, 128, 192, 254)
EVALS_PER_WORKER = 2


def _one(seed: int) -> float:
    from src import pipeline as pl
    from src.glue.core import glue_measures
    from src.models import paired_init

    spec = pl.Phase1Spec()
    streams = paired_init(seed % 8)
    stream = pl.build_stream(spec, streams["stream"])
    model = pl.build_model(spec, stream)
    reps = model.manifold_representation(stream.arrangements[0].points, "A")
    t = time.time()
    glue_measures(reps, np.random.default_rng(seed), n_t=N_T)
    return time.time() - t


def main() -> None:
    counts = [int(x) for x in sys.argv[1:]] or list(WORKER_COUNTS)
    rows = []
    ctx = mp.get_context("fork")

    for w in counts:
        jobs = list(range(w * EVALS_PER_WORKER))
        t0 = time.time()
        with ctx.Pool(processes=w) as pool:
            lat = pool.map(_one, jobs)
        wall = time.time() - t0
        per_eval = float(np.mean(lat))
        thr = len(jobs) / wall                       # evals per second, aggregate
        rows.append({
            "workers": w, "evals": len(jobs), "wall_s": wall,
            "mean_eval_latency_s": per_eval,
            "throughput_evals_per_s": thr,
            "effective_cores": thr * per_eval,       # Little's law
        })
        print(f"  {w:4d} workers: latency {per_eval:7.2f}s  throughput "
              f"{thr:6.3f} eval/s  effective cores {thr * per_eval:6.1f}", flush=True)

    base = rows[0]["mean_eval_latency_s"]
    best = max(rows, key=lambda r: r["throughput_evals_per_s"])
    for r in rows:
        r["slowdown_vs_isolated"] = r["mean_eval_latency_s"] / base
        r["parallel_efficiency"] = r["effective_cores"] / r["workers"]

    # Full-grid projection at the design point: cost is linear in n_t.
    scale = 200 / N_T
    evals = 1280 * 38
    out = {
        "n_t_measured": N_T, "n_t_design": 200,
        "isolated_latency_s": base, "isolated_latency_s_at_design_n_t": base * scale,
        "rows": rows,
        "best_workers": best["workers"],
        "best_throughput_evals_per_s": best["throughput_evals_per_s"],
        "grid_projection": {
            "evals": evals,
            "wall_hours_at_best": evals / (best["throughput_evals_per_s"] / scale) / 3600,
            "wall_hours_naive_254": evals * base * scale / 254 / 3600,
        },
    }
    (ROOT / "results" / "scaling.json").write_text(json.dumps(out, indent=2))

    print(f"\nisolated latency at n_t={N_T}: {base:.2f}s "
          f"(-> {base * scale:.1f}s at n_t=200)")
    print(f"throughput peaks at {best['workers']} workers "
          f"({best['throughput_evals_per_s']:.3f} eval/s); "
          f"efficiency there {best['effective_cores'] / best['workers']:.2f}")
    print(f"\nfull grid ({evals:,} evals at n_t=200):")
    print(f"  at best worker count : {out['grid_projection']['wall_hours_at_best']:.1f} h")
    print(f"  cost model's estimate: {out['grid_projection']['wall_hours_naive_254']:.1f} h")


if __name__ == "__main__":
    main()
