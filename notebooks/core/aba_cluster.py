"""Cluster driver for the A–B–A grid (sbatch only; never on the login node).

    python notebooks/core/aba_cluster.py calibrate            # bottleneck n on the FULL init representation
    python notebooks/core/aba_cluster.py timing               # three FULL arms, sequential, wall + peak RSS
    python notebooks/core/aba_cluster.py grid --workers 32    # FULL grid into results/aba/FULL_<hash>.jsonl

Timing arms go to results/aba/timing_<hash>.jsonl, never into the grid checkpoint.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import aba_lib as L  # noqa: E402

CANDIDATES = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64)
TARGET_ACC = 0.75


def peak_rss_mb() -> float:
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024   # Linux: KiB
    except ImportError:
        return float("nan")


def calibrate(cfg: L.ABAConfig) -> None:
    t0 = time.perf_counter()
    table = L.calibrate_bottleneck(cfg, range(cfg.n_seeds), CANDIDATES)
    best = min(table, key=lambda n: abs(table[n]["acc_mean"] - TARGET_ACC))
    print(f"bottleneck calibration ({'QUICK' if cfg.quick else 'FULL'} init representation, "
          f"M={cfg.M}, seeds 0..{cfg.n_seeds - 1}, "
          f"budget {cfg.refit_steps} steps), {time.perf_counter() - t0:.0f}s")
    print(f"{'n':>4} {'acc mean':>9} {'min':>7} {'max':>7} {'loss':>8}")
    for n, v in table.items():
        mark = "  <- closest to 0.75" if n == best else ""
        print(f"{n:>4} {v['acc_mean']:>9.4f} {v['acc_min']:>7.4f} {v['acc_max']:>7.4f} {v['loss_mean']:>8.4f}{mark}")
    out = L.REPO / "results" / "aba" / f"calibration_{L.lib_hash()}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"target": TARGET_ACC, "chosen_n": best, "table": table,
                               "stamp": L.git_stamp(), "config": cfg.key_fields()}, indent=1))
    print(f"chosen n = {best}; written to {out}")


def timing(cfg: L.ABAConfig) -> None:
    specs = [L.ArmSpec(0, g, max(cfg.hammings), min(cfg.s_fs)) for g in cfg.gammas]
    out = L.REPO / "results" / "aba" / f"timing_{L.lib_hash()}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    for spec in specs:
        t0 = time.perf_counter()
        res = L.run_arm(cfg, spec)
        wall = time.perf_counter() - t0
        tm = res["timing"]
        print(f"{spec.key}: wall {wall:.0f}s (train {tm['train_s']:.0f}, GLUE {tm['glue_s']:.0f}, "
              f"NTK {tm['ntk_s']:.0f}, probes {tm['refit_s']:.0f}); steps {res['behaviour']['steps']}; "
              f"usable {res['usable']}; peak RSS so far {peak_rss_mb():.0f} MB", flush=True)
        with out.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(res) + "\n")


def grid(cfg: L.ABAConfig, workers: int) -> None:
    t0 = time.perf_counter()
    rows = L.run_grid(cfg, L.all_specs(cfg), workers=workers, progress=lambda m: print(m, flush=True))
    print(f"{len(rows)} arms in {L.cache_path(cfg)}; wall {time.perf_counter() - t0:.0f}s on {workers} workers")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("calibrate", "timing", "grid"))
    ap.add_argument("--quick", action="store_true", help="QUICK config instead of FULL")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    a = ap.parse_args()
    cfg = L.ABAConfig.make(a.quick)
    print(f"mode {a.mode}  {'QUICK' if a.quick else 'FULL'}  lib {L.lib_hash()}  stamp {L.git_stamp()}  "
          f"bottleneck_n {cfg.bottleneck_n}", flush=True)
    {"calibrate": lambda: calibrate(cfg), "timing": lambda: timing(cfg),
     "grid": lambda: grid(cfg, a.workers)}[a.mode]()
