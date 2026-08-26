"""Full Hiratani 2×2 at N=32 and N=100, 8 seeds, CF + GLUE.

Pre-commit: `results/small_n_grid_precommit.json`. Imports 20 existing arms
(S-HH seeds 0–3; S-HL seed 0) and trains the rest.

    python scripts/run_small_n_grid.py

128 arms: N∈{32,100} × γ∈{0.03,10} × {S-HH,S-HL,S-LH,S-LL} × stream 0 × seeds 0..7.
Writes `results/small_n_grid/`, never `results/phase1/`.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from _par import pin_threads  # noqa: E402

pin_threads()

from _par import pmap  # noqa: E402
from src import pipeline as pl  # noqa: E402
from src import provenance  # noqa: E402

from run_width_smoke import _path, _run_arm_with_clouds  # noqa: E402

PRE_DEFAULT = ROOT / "results" / "small_n_grid_precommit.json"
N_TRAIN_DEFAULT = (32, 100)
GAMMAS = (0.03, 10.0)
CONDITIONS = ("S-HH", "S-HL", "S-LH", "S-LL")
SEEDS = tuple(range(8))
STREAM = 0
OVERLAP_DIRS = (
    ROOT / "results" / "shh_width",
    ROOT / "results" / "width_smoke",
)


def _require_precommit(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(f"missing {path}")
    return json.loads(path.read_text())


def _spec(N: int, gamma: float, condition: str, seed: int) -> pl.Phase1Spec:
    return pl.Phase1Spec(
        gamma_0=gamma, a=0.0, condition=condition,
        stream_id=STREAM, seed=seed, N=N,
    )


def arms(widths: tuple[int, ...]) -> list[pl.Phase1Spec]:
    return [
        _spec(n, g, c, s)
        for n in widths for g in GAMMAS for c in CONDITIONS for s in SEEDS
    ]


def import_overlap(out: Path, widths: tuple[int, ...]) -> list[str]:
    """Copy already-run matching specs. Do not retrain them."""
    copied = []
    wanted = {_path(s, out).name: s for s in arms(widths)}
    for src_dir in OVERLAP_DIRS:
        if not src_dir.is_dir():
            continue
        for src in src_dir.glob("N*.json"):
            dest = out / src.name
            if src.name not in wanted:
                continue
            rec = json.loads(src.read_text())
            spec = wanted[src.name]
            if rec.get("spec") != json.loads(json.dumps(asdict(spec))):
                continue
            if not dest.exists():
                shutil.copy2(src, dest)
                copied.append(src.name)
            clouds = src.with_suffix(".clouds.npz")
            if spec.N == 32 and spec.seed == 0 and clouds.exists():
                cdest = dest.with_suffix(".clouds.npz")
                if not cdest.exists():
                    shutil.copy2(clouds, cdest)
    return copied


def _job(payload: tuple[dict, str]) -> dict:
    provenance.assert_current()
    spec = pl.Phase1Spec(**payload[0])
    out = Path(payload[1])
    path = _path(spec, out)
    if path.exists():
        stored = json.loads(path.read_text())
        if stored.get("spec") == json.loads(json.dumps(asdict(spec))):
            return {
                "key": spec.key, "N": spec.N, "seed": spec.seed,
                "condition": spec.condition, "gamma_0": spec.gamma_0,
                "skipped": True, "usable": stored.get("usable"),
                "CF": stored.get("forgetting", {}).get("CF"),
            }
    rec, clouds = _run_arm_with_clouds(spec)
    path.write_text(json.dumps(rec))
    if spec.N in (16, 32) and spec.seed == 0:
        import numpy as np
        npz = path.with_suffix(".clouds.npz")
        save = {k: v for k, v in clouds.items() if k not in ("N", "gamma_0")}
        np.savez_compressed(
            npz, **{k: np.asarray(v) for k, v in save.items()},
            N=clouds["N"], gamma_0=clouds["gamma_0"],
        )
    return {
        "key": spec.key, "N": spec.N, "gamma_0": spec.gamma_0,
        "condition": spec.condition, "seed": spec.seed,
        "skipped": False, "usable": rec["usable"],
        "wall_seconds": rec["wall_seconds"],
        "CF": rec["forgetting"]["CF"],
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=str, default="results/small_n_grid")
    p.add_argument("--cap", type=int, default=32)
    p.add_argument("--widths", type=int, nargs="+", default=None)
    p.add_argument("--precommit", type=str, default=None)
    args = p.parse_args()
    widths = tuple(args.widths) if args.widths else N_TRAIN_DEFAULT
    pre_path = Path(args.precommit) if args.precommit else PRE_DEFAULT
    if not pre_path.is_absolute():
        pre_path = ROOT / pre_path
    pre = _require_precommit(pre_path)
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    if out.resolve() == (ROOT / "results" / "phase1").resolve():
        raise SystemExit("refusing to write into results/phase1/")
    out.mkdir(parents=True, exist_ok=True)
    copied = import_overlap(out, widths) if set(widths) & {32, 100} else []
    if copied:
        print(f"imported {len(copied)} overlap arms", flush=True)

    specs = arms(widths)
    print(
        f"small-n grid: {len(specs)} arms N={widths} γ={GAMMAS} "
        f"cond={CONDITIONS} seeds={SEEDS} precommit={pre['written']} "
        f"-> {out.relative_to(ROOT)}",
        flush=True,
    )
    t0 = time.time()
    res = pmap(_job, [(asdict(s), str(out)) for s in specs], cap=args.cap)
    done = [r for r in res if not r.get("skipped")]
    print(f"\n  {len(done)} run, {len(res) - len(done)} skipped, "
          f"{time.time() - t0:.0f}s wall", flush=True)
    for r in sorted(res, key=lambda x: (x["N"], x.get("gamma_0"), x.get("condition"), x.get("seed"))):
        extra = "skipped" if r.get("skipped") else f"{r.get('wall_seconds', 0):.0f}s"
        cf = r.get("CF")
        cf_s = f"CF={cf:.3f}" if isinstance(cf, float) else ""
        print(
            f"  N={r['N']} γ={r.get('gamma_0')} {r.get('condition')} "
            f"seed={r.get('seed')} usable={r.get('usable')} {cf_s} {extra}",
            flush=True,
        )


if __name__ == "__main__":
    main()
