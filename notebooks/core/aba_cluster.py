"""Cluster driver for the A–B–A grid (sbatch only; never on the login node).

    python notebooks/core/aba_cluster.py calibrate            # bottleneck n on the FULL init representation
    python notebooks/core/aba_cluster.py timing               # three FULL arms, sequential, wall + peak RSS
    python notebooks/core/aba_cluster.py grid --workers 32    # FULL grid into results/aba/FULL_<hash>.jsonl
    python notebooks/core/aba_cluster.py grid --seed-offset 8 # held-out replication, seeds 8-15 (docs/28 amendment)
    python notebooks/core/aba_cluster.py reproduce            # one FULL seed-0 arm vs job 4349, bit for bit

Timing arms go to results/aba/timing_<hash>.jsonl, never into the grid checkpoint.
"""
from __future__ import annotations

import argparse
import dataclasses
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
                               "stamp": L.git_stamp(), "config": cfg.key_fields()}, indent=1), encoding="utf-8")
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


# Job 4349 (commit a7f0398, lib 2528909f2feaf3f3): the reference for the seed_offset change.
REF_JSONL = L.REPO / "results" / "aba" / "FULL_2528909f2feaf3f3.jsonl"
REF_SPEC = L.ArmSpec(0, 10.0, 8, 0.1)
IGNORED = ("lib_hash", "stamp", "timing")


def _diff(a, b, path=""):
    """Paths where a and b differ; floats compared exactly (bit for bit after JSON)."""
    if isinstance(a, dict) and isinstance(b, dict):
        out = [f"{path}.{k} only on one side" for k in sorted(set(a) ^ set(b))]
        for k in sorted(set(a) & set(b)):
            out += _diff(a[k], b[k], f"{path}.{k}")
        return out
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{path} length {len(a)} vs {len(b)}"]
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in _diff(x, y, f"{path}[{i}]")]
    return [] if a == b or (a != a and b != b) else [f"{path}: {a!r} vs {b!r}"]


def reproduce(cfg: L.ABAConfig) -> None:
    ref = next((r for r in map(json.loads, REF_JSONL.read_text(encoding="utf-8").splitlines())
                if r["key"] == REF_SPEC.key), None)
    if ref is None:
        sys.exit(f"{REF_SPEC.key} not found in {REF_JSONL}")
    t0 = time.perf_counter()
    # Through JSON, exactly as run_grid stores a row.
    new = json.loads(json.dumps(L.run_arm(cfg, REF_SPEC)))
    a = {k: v for k, v in ref.items() if k not in IGNORED}
    b = {k: v for k, v in new.items() if k not in IGNORED}
    diffs = _diff(a, b)
    print(f"{REF_SPEC.key}: wall {time.perf_counter() - t0:.0f}s; reference lib {ref['lib_hash']} "
          f"stamp {ref['stamp']}; this lib {new['lib_hash']} stamp {new['stamp']}")
    print(f"compared {len(a)} top-level fields (ignoring {', '.join(IGNORED)})")
    if diffs:
        print(f"NOT bit-identical: {len(diffs)} differences")
        for d in diffs[:50]:
            print("  ", d)
        sys.exit(1)
    print("bit-identical")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("calibrate", "timing", "grid", "reproduce"))
    ap.add_argument("--quick", action="store_true", help="QUICK config instead of FULL")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    ap.add_argument("--seed-offset", type=int, default=0, help="first seed of the run (8: replication)")
    a = ap.parse_args()
    cfg = dataclasses.replace(L.ABAConfig.make(a.quick), seed_offset=a.seed_offset)
    print(f"mode {a.mode}  {'QUICK' if a.quick else 'FULL'}  lib {L.lib_hash()}  stamp {L.git_stamp()}  "
          f"bottleneck_n {cfg.bottleneck_n}  seeds {cfg.seeds.start}..{cfg.seeds.stop - 1}", flush=True)
    {"calibrate": lambda: calibrate(cfg), "timing": lambda: timing(cfg),
     "grid": lambda: grid(cfg, a.workers), "reproduce": lambda: reproduce(cfg)}[a.mode]()
