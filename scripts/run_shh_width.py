"""S-HH at N=32 and N=100: does behavioural forgetting stay ~0?

Pre-commit: `results/shh_width_precommit.json` — written before any arm here.
Primary statistic is CF / past-task accuracy, not GLUE shares.

    python scripts/run_shh_width.py

16 arms: N∈{32,100} × γ∈{0.03,10} × S-HH × stream 0 × seeds 0..3.
Writes `results/shh_width/`, never `results/phase1/`.
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
sys.path.insert(0, str(ROOT / "scripts"))

from _par import pin_threads  # noqa: E402

pin_threads()

from _par import pmap  # noqa: E402
from src import pipeline as pl  # noqa: E402
from src import provenance  # noqa: E402

from run_width_smoke import _path, _run_arm_with_clouds  # noqa: E402

PRE = ROOT / "results" / "shh_width_precommit.json"
N_TRAIN = (32, 100)
GAMMAS = (0.03, 10.0)
SEEDS = (0, 1, 2, 3)
STREAM = 0
CONDITION = "S-HH"


def _require_precommit() -> dict:
    if not PRE.is_file():
        raise SystemExit(f"missing {PRE.relative_to(ROOT)}")
    return json.loads(PRE.read_text())


def _spec(N: int, gamma: float, seed: int) -> pl.Phase1Spec:
    return pl.Phase1Spec(
        gamma_0=gamma, a=0.0, condition=CONDITION,
        stream_id=STREAM, seed=seed, N=N,
    )


def arms() -> list[pl.Phase1Spec]:
    return [_spec(n, g, s) for n in N_TRAIN for g in GAMMAS for s in SEEDS]


def _job(payload: tuple[dict, str]) -> dict:
    provenance.assert_current()
    spec = pl.Phase1Spec(**payload[0])
    out = Path(payload[1])
    path = _path(spec, out)
    if path.exists():
        stored = json.loads(path.read_text())
        if stored.get("spec") == json.loads(json.dumps(asdict(spec))):
            return {"key": spec.key, "N": spec.N, "seed": spec.seed, "skipped": True,
                    "usable": stored.get("usable"), "gamma_0": spec.gamma_0}
    rec, clouds = _run_arm_with_clouds(spec)
    path.write_text(json.dumps(rec))
    if spec.N == 32:
        import numpy as np
        npz = path.with_suffix(".clouds.npz")
        save = {k: v for k, v in clouds.items() if k not in ("N", "gamma_0")}
        np.savez_compressed(
            npz, **{k: np.asarray(v) for k, v in save.items()},
            N=clouds["N"], gamma_0=clouds["gamma_0"],
        )
    return {
        "key": spec.key, "N": spec.N, "gamma_0": spec.gamma_0, "seed": spec.seed,
        "skipped": False, "usable": rec["usable"],
        "wall_seconds": rec["wall_seconds"],
        "n_converged": sum(1 for t in rec["tasks"] if t["converged"]),
        "n_tasks": len(rec["tasks"]),
        "CF": rec["forgetting"]["CF"],
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=str, default="results/shh_width")
    p.add_argument("--cap", type=int, default=16)
    args = p.parse_args()
    pre = _require_precommit()
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    if out.resolve() == (ROOT / "results" / "phase1").resolve():
        raise SystemExit("refusing to write into results/phase1/")
    out.mkdir(parents=True, exist_ok=True)
    if any(out.glob("N*.json")):
        print("note: some N*.json already in out; identical specs will skip", flush=True)

    specs = arms()
    print(
        f"S-HH width: {len(specs)} arms N={N_TRAIN} γ={GAMMAS} seeds={SEEDS} "
        f"precommit={pre['written']} -> {out.relative_to(ROOT)}",
        flush=True,
    )
    t0 = time.time()
    res = pmap(_job, [(asdict(s), str(out)) for s in specs], cap=args.cap)
    done = [r for r in res if not r.get("skipped")]
    print(f"\n  {len(done)} run, {len(res) - len(done)} skipped, "
          f"{time.time() - t0:.0f}s wall", flush=True)
    for r in res:
        extra = "skipped" if r.get("skipped") else f"{r.get('wall_seconds', 0):.0f}s"
        cf = r.get("CF")
        cf_s = f"CF={cf:.3f}" if isinstance(cf, float) else ""
        print(
            f"  N={r['N']} γ={r.get('gamma_0')} seed={r.get('seed')} "
            f"usable={r.get('usable')} {cf_s} {extra}",
            flush=True,
        )


if __name__ == "__main__":
    main()
