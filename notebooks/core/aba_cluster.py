"""Cluster driver for the A–B–A grid (sbatch only; never on the login node).

    python notebooks/core/aba_cluster.py calibrate            # bottleneck n on the FULL init representation
    python notebooks/core/aba_cluster.py timing               # three FULL arms, sequential, wall + peak RSS
    python notebooks/core/aba_cluster.py grid --workers 32    # FULL grid into results/aba/FULL_<hash>.jsonl
    python notebooks/core/aba_cluster.py grid --seed-offset 8 # held-out replication, seeds 8-15 (docs/28 amendment)
    python notebooks/core/aba_cluster.py reproduce            # one FULL seed-0 arm vs job 4349, bit for bit
    python notebooks/core/aba_cluster.py arm --spec 8,0.1,8,1 --stopping fixed --budget 2300,2400,2000
                                                              # one arm, sequential, timed (timing file only)
    python notebooks/core/aba_cluster.py grid --seed-offset 8 --stopping fixed --budget 2300,2400,2000
                                                              # docs/31 stopping-rule ablation
    python notebooks/core/aba_cluster.py floors --seed-offset 8 [--stopping fixed --budget ...]
                                                              # trained floors per γ0 on the run's first seed, K = 8
    python notebooks/core/aba_cluster.py arm --spec 8,0.1,4,1 --stopping fixed --budget 2300,-,2000
                                                              # docs/32: '-' = criterion for that phase
    python notebooks/core/aba_cluster.py cells --workers 32   # docs/32 2x2: 4 cells x 32 arms
    python notebooks/core/aba_cluster.py match [--spec ... --stopping ... --budget ...]
                                                              # docs/32 integrity check vs jobs 4396 / 4458
    python notebooks/core/aba_cluster.py arm --map --spec 16,0.1,16,1
                                                              # docs/34: one map arm, timed
    python notebooks/core/aba_cluster.py map1 --workers 32    # docs/34 Phase 1 (771 arms, seeds 16-23)
    python notebooks/core/aba_cluster.py reproduce --map      # docs/36: one job-4651 map arm, bit for bit
    python notebooks/core/aba_cluster.py arm --map --spec 19,0.1,16,1 --stopping fixed --budget -,3200,-
                                                              # docs/36: the timed fixed-B arm
    python notebooks/core/aba_cluster.py map1b --workers 64   # docs/36 Phase 1b (264 arms, B fixed per γ0)
    python notebooks/core/aba_cluster.py check1b              # docs/36 integrity checks vs job 4651

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


def arm(cfg: L.ABAConfig, spec: L.ArmSpec) -> None:
    out = L.REPO / "results" / "aba" / f"timing_{L.lib_hash()}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    res = L.run_arm(cfg, spec)
    wall = time.perf_counter() - t0
    tm = res["timing"]
    print(f"{spec.key} ({cfg.stopping}): wall {wall:.0f}s (train {tm['train_s']:.0f}, GLUE {tm['glue_s']:.0f}, "
          f"NTK {tm['ntk_s']:.0f}, probes {tm['refit_s']:.0f}); steps {res['behaviour']['steps']}; "
          f"first_hit {res['behaviour'].get('first_hit')}; "
          f"status {res['status']}; final loss {res['behaviour'].get('final_loss')}; "
          f"usable {res['usable']}; peak RSS {peak_rss_mb():.0f} MB", flush=True)
    with out.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(res) + "\n")


K_FLOOR = 8   # docs/28 §3: K = 8 estimator seeds


def floors(cfg: L.ABAConfig) -> None:
    t0 = time.perf_counter()
    fl = {f"{g:g}": L.trained_floor(cfg, g, seed=cfg.seeds[0], K=K_FLOOR) for g in cfg.gammas}
    tag = "" if cfg.stopping == "criterion" else f"_{cfg.stopping}"
    out = L.REPO / "results" / "aba" / f"floors_{L.lib_hash()}{tag}_seed{cfg.seeds[0]}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"floors": fl, "stamp": L.git_stamp(), "lib_hash": L.lib_hash(),
                               "config": cfg.key_fields()}, indent=1), encoding="utf-8")
    for g, f in fl.items():
        print(f"γ0 {g}: floor {f['floor']:.4f} (K {f['K']}, mean log α_A after A {f['mean_log_alpha']:+.3f})")
    print(f"written to {out}; {time.perf_counter() - t0:.0f}s")


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


# docs/32: the A–B–A stopping 2x2. Cells are (A rule, B rule); A2 follows A, fresh B follows B.
CELLS_2X2 = {
    "Cr,Cr": ("criterion", ()),
    "Fx,Cr": ("fixed", (2300, None, 2000)),
    "Cr,Fx": ("fixed", (None, 2400, None)),
    "Fx,Fx": ("fixed", (2300, 2400, 2000)),
}
SPECS_2X2 = [L.ArmSpec(s, g, 4, f) for s in range(8, 16) for g in (0.1, 10.0) for f in (1.0, 0.9)]
# The existing cells' reference rows: job 4396 (criterion replication) and job 4458 (docs/31 fixed).
REF_2X2 = {"Cr,Cr": L.REPO / "results" / "aba" / "FULL_4e110835b8fc2863.jsonl",
           "Fx,Fx": L.REPO / "results" / "aba" / "FULL_0ee97f82b6bee621_fixed.jsonl"}


def cell_cfg(base: L.ABAConfig, cell: str) -> L.ABAConfig:
    stopping, budget = CELLS_2X2[cell]
    return dataclasses.replace(base, seed_offset=8, stopping=stopping, fixed_steps=budget)


def cell_of(cfg: L.ABAConfig) -> str | None:
    want = (cfg.stopping, tuple(cfg.fixed_steps))
    return next((c for c, v in CELLS_2X2.items() if (v[0], tuple(v[1])) == want), None)


def cells(base: L.ABAConfig, workers: int) -> None:
    t0 = time.perf_counter()
    for cell in CELLS_2X2:
        cfg = cell_cfg(base, cell)
        t1 = time.perf_counter()
        rows = L.run_grid(cfg, SPECS_2X2, workers=workers, progress=lambda m: print(m, flush=True))
        print(f"cell ({cell}): {len(rows)} rows in {L.cache_path(cfg).name}; "
              f"wall {time.perf_counter() - t1:.0f}s", flush=True)
    print(f"docs/32 cells done; wall {time.perf_counter() - t0:.0f}s on {workers} workers")


def _ref_rows(cell: str) -> dict:
    want = json.dumps(cell_cfg(L.ABAConfig.make(False), cell).key_fields(), sort_keys=True)
    out = {}
    for line in REF_2X2[cell].read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if json.dumps(r["config"], sort_keys=True) == want:
            out[r["key"]] = r
    return out


def match(base: L.ABAConfig, spec: L.ArmSpec | None) -> None:
    """docs/32 §2 integrity check. With a spec: run that arm now in the cell given by
    --stopping/--budget and compare it with its reference row. Without: compare every (Cr,Cr)
    and (Fx,Fx) row of this lib's checkpoints with the reference rows."""
    if spec is not None:
        cell = cell_of(base)
        if cell not in REF_2X2:
            sys.exit(f"match --spec needs the (Cr,Cr) or (Fx,Fx) cell, got {base.stopping} {base.fixed_steps}")
        t0 = time.perf_counter()
        pairs = [(cell, spec.key, json.loads(json.dumps(L.run_arm(cell_cfg(base, cell), spec))))]
        print(f"({cell}) {spec.key}: ran in {time.perf_counter() - t0:.0f}s", flush=True)
    else:
        keys = {s.key for s in SPECS_2X2}
        pairs = [(cell, r["key"], r) for cell in REF_2X2
                 for r in L.load_results(cell_cfg(base, cell)) if r["key"] in keys]
        if len(pairs) != 2 * len(SPECS_2X2):
            sys.exit(f"expected {2 * len(SPECS_2X2)} rows in the (Cr,Cr) and (Fx,Fx) checkpoints, "
                     f"found {len(pairs)}")
    bad = 0
    for cell, key, new in pairs:
        ref = _ref_rows(cell).get(key)
        if ref is None:
            sys.exit(f"({cell}) {key} not in {REF_2X2[cell].name}")
        diffs = _diff({k: v for k, v in ref.items() if k not in IGNORED},
                      {k: v for k, v in new.items() if k not in IGNORED})
        if diffs:
            bad += 1
            print(f"({cell}) {key}: NOT bit-identical to {REF_2X2[cell].name} ({len(diffs)} differences)")
            for d in diffs[:20]:
                print("  ", d)
    print(f"match: {len(pairs)} rows compared (ignoring {', '.join(IGNORED)}); {bad} differ")
    if bad:
        sys.exit(1)
    print("bit-identical")


def _only_new(ref: dict, d: str) -> bool:
    """True if the 'only on one side' path in diff line `d` is absent from the reference."""
    parts = d.split(" ")[0].lstrip(".").split(".")
    v = ref
    for p in parts:
        if not isinstance(v, dict) or p not in v:
            return True
        v = v[p]
    return False


# docs/34 similarity map, Phase 1 (amended 2026-11-10): 768 design arms + 3 fresh-noise identity arms.
MAP_TIMING_SPEC = L.ArmSpec(16, 0.1, 16, 1.0)     # new H level, lazy learner


def map1(base: L.ABAConfig, workers: int) -> None:
    cfg = L.map_config(base)
    specs = L.phase1_specs(cfg)
    t0 = time.perf_counter()
    rows = L.run_grid(cfg, specs, workers=workers, progress=lambda m: print(m, flush=True))
    print(f"docs/34 Phase 1: {len(rows)}/{len(specs)} arms in {L.cache_path(cfg).name}; "
          f"usable {sum(r['usable'] for r in rows)}; wall {time.perf_counter() - t0:.0f}s on {workers} workers")


def reproduce(cfg: L.ABAConfig, ref_jsonl: Path = REF_JSONL, ref_spec: L.ArmSpec = REF_SPEC) -> None:
    ref = next((r for r in map(json.loads, ref_jsonl.read_text(encoding="utf-8").splitlines())
                if r["key"] == ref_spec.key
                and json.dumps(r["config"], sort_keys=True) == json.dumps(cfg.key_fields(), sort_keys=True)),
               None)
    if ref is None:
        sys.exit(f"{ref_spec.key} with this config not found in {ref_jsonl}")
    t0 = time.perf_counter()
    # Through JSON, exactly as run_grid stores a row.
    new = json.loads(json.dumps(L.run_arm(cfg, ref_spec)))
    a = {k: v for k, v in ref.items() if k not in IGNORED}
    b = {k: v for k, v in new.items() if k not in IGNORED}
    # docs/34: fields added since the reference (relations.taskvec_*) are listed, not compared;
    # every field the reference has must match bit for bit.
    added = [d for d in _diff(a, b) if d.endswith("only on one side") and _only_new(a, d)]
    diffs = [d for d in _diff(a, b) if d not in added]
    print(f"{ref_spec.key}: wall {time.perf_counter() - t0:.0f}s; reference lib {ref['lib_hash']} "
          f"stamp {ref['stamp']}; this lib {new['lib_hash']} stamp {new['stamp']}")
    print(f"compared {len(a)} top-level fields (ignoring {', '.join(IGNORED)}); "
          f"fields added since the reference: {[d.split(' ')[0] for d in added] or 'none'}")
    if diffs:
        print(f"NOT bit-identical: {len(diffs)} differences")
        for d in diffs[:50]:
            print("  ", d)
        sys.exit(1)
    print("bit-identical")


# docs/36: Phase 1b, fixed budget on phase B per γ0 (extreme cases, run 1), against job 4651.
MAP_REF_JSONL = L.REPO / "results" / "aba" / "FULL_e3011fa49aca1994.jsonl"   # job 4651
MAP_REF_SPEC = L.ArmSpec(16, 0.1, 16, 1.0)
P1B_TIMING_SPEC = L.ArmSpec(19, 0.1, 16, 1.0)      # job 4651's largest B count (3132)


def map1b(base: L.ABAConfig, workers: int) -> None:
    """The 264 docs/36 arms in ONE pool (three configs, one per γ0; longest γ0 first), appended to
    the shared fixed checkpoint and separated there by the config dict. Resumes per config."""
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from multiprocessing import get_context
    cfgs = [L.phase1b_cfg(base, g) for g in (0.1, 1.0, 10.0)]
    path = L.cache_path(cfgs[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    todo = []
    for cfg in cfgs:
        done = {r["key"] for r in L.load_results(cfg)}
        todo += [(cfg, s) for s in L.phase1b_specs(cfg) if s.key not in done]
    stamp = L.git_stamp()
    t0 = time.perf_counter()
    n = sum(len(L.phase1b_specs(c)) for c in cfgs)
    print(f"docs/36 Phase 1b: {n} arms, {len(todo)} to run, budgets {L.PHASE1B_BUDGET}, into {path.name}",
          flush=True)
    if todo:
        with ProcessPoolExecutor(max_workers=min(workers, len(todo)), mp_context=get_context("spawn")) as ex:
            futs = {ex.submit(L._arm_job, (cfg, s, stamp)): s for cfg, s in todo}
            for i, fut in enumerate(as_completed(futs), 1):
                res = fut.result()
                with path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(res) + "\n")
                print(f"[{i}/{len(todo)}] {res['key']}  wall {res['timing']['wall_s']:.0f}s  "
                      f"first_hit B {res['behaviour']['first_hit']['B']}  usable={res['usable']}", flush=True)
    rows = sum(len(L.load_results(c)) for c in cfgs)
    print(f"docs/36 Phase 1b: {rows}/{n} arms in {path.name}; wall {time.perf_counter() - t0:.0f}s "
          f"on {workers} workers")


def check1b(base: L.ABAConfig) -> None:
    """docs/36 §2 integrity checks 1-3 against job 4651. Exit 1 on any failure."""
    import aba_fixedb_analysis as FB
    checks = FB.integrity(FB.load_fixed(base), FB.load_criterion(MAP_REF_JSONL, base))
    for name, ok, det in checks:
        print(f"{'PASS' if ok else 'FAIL'}  {name}  {det}", flush=True)
    if not all(ok for _, ok, _ in checks):
        sys.exit(1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("calibrate", "timing", "grid", "reproduce", "arm", "floors",
                                       "cells", "match", "map1", "map1b", "check1b"))
    ap.add_argument("--map", action="store_true", help="arm mode: use the docs/34 map config (seeds 16-23)")
    ap.add_argument("--quick", action="store_true", help="QUICK config instead of FULL")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    ap.add_argument("--seed-offset", type=int, default=0, help="first seed of the run (8: replication)")
    ap.add_argument("--stopping", choices=("criterion", "fixed"), default="criterion")
    ap.add_argument("--budget", default="", help="fixed budgets A1,B,A2 (with --stopping fixed); '-' = criterion")
    ap.add_argument("--spec", default="", help="arm mode: seed,gamma,hamming,s_f")
    a = ap.parse_args()
    budget = tuple(None if x == "-" else int(x) for x in a.budget.split(",")) if a.budget else ()
    if (a.stopping == "fixed") != (len(budget) == 3):
        sys.exit("--stopping fixed needs --budget A1,B,A2 (and only then)")
    cfg = dataclasses.replace(L.ABAConfig.make(a.quick), seed_offset=a.seed_offset,
                              stopping=a.stopping, fixed_steps=budget)
    print(f"mode {a.mode}  {'QUICK' if a.quick else 'FULL'}  lib {L.lib_hash()}  stamp {L.git_stamp()}  "
          f"bottleneck_n {cfg.bottleneck_n}  seeds {cfg.seeds.start}..{cfg.seeds.stop - 1}  "
          f"stopping {cfg.stopping} {cfg.fixed_steps or ''}", flush=True)
    spec = (L.ArmSpec(int(a.spec.split(",")[0]), float(a.spec.split(",")[1]),
                      int(a.spec.split(",")[2]), float(a.spec.split(",")[3])) if a.spec else None)
    {"calibrate": lambda: calibrate(cfg), "timing": lambda: timing(cfg),
     "grid": lambda: grid(cfg, a.workers),
     "reproduce": lambda: (reproduce(L.map_config(cfg), MAP_REF_JSONL, MAP_REF_SPEC) if a.map
                           else reproduce(cfg)),
     "map1b": lambda: map1b(cfg, a.workers), "check1b": lambda: check1b(cfg),
     "floors": lambda: floors(cfg), "arm": lambda: arm(L.map_config(cfg) if a.map else cfg, spec),
     "cells": lambda: cells(cfg, a.workers), "match": lambda: match(cfg, spec),
     "map1": lambda: map1(cfg, a.workers)}[a.mode]()
