"""K=3 unconfounding: arrangement crossed with initialisation.

The registered grid stored `stream_id` in the filename and never consumed it, so 5×8 files
are 8 unique (init, arrangement, dichotomy) triples, seed-confounded, each written five
times. This run is the only measurement that separates the two:

    stream_id ∈ {0, 1, 2}     — 3 independent arrangements (`pipeline.stream_rng`)
    seed      ∈ {0, …, 39}    — 40 independent inits (`paired_init`)
    γ         ∈ {1, 10}
    4 corners

960 arms. Writes to `results/unconfound_k3/`, **never** `results/phase1/`: γ=1 and γ=10 are
registered values, and dropping them into the grid directory would mix RNG recipes with the
duplicate-stream files and silently change every figure.

    python scripts/run_unconfound.py
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from _par import pin_threads  # noqa: E402

pin_threads()

from _par import pmap  # noqa: E402
from src import pipeline as pl  # noqa: E402
from src import provenance  # noqa: E402

GAMMAS = (10.0, 1.0)  # cheaper first
CONDITIONS = ("S-HH", "S-HL", "S-LH", "S-LL")
N_ARRANGEMENTS = 3
N_INITS = 40
OUT = ROOT / "results" / "unconfound_k3"


def arms(*, n_arr: int = N_ARRANGEMENTS, n_init: int = N_INITS) -> list[pl.Phase1Spec]:
    return [
        pl.Phase1Spec(gamma_0=g, a=0.0, condition=c, stream_id=s, seed=sd)
        for g, c, s, sd in itertools.product(GAMMAS, CONDITIONS, range(n_arr), range(n_init))
    ]


def _path(spec: pl.Phase1Spec, out: Path) -> Path:
    return out / f"{spec.key.replace(',', '__').replace('=', '-')}.json"


def _job(payload: tuple[pl.Phase1Spec, str]) -> dict:
    spec, out_s = payload
    provenance.assert_current()
    path = _path(spec, Path(out_s))
    if path.exists():
        try:
            stored = json.loads(path.read_text())["spec"]
        except (json.JSONDecodeError, KeyError):
            stored = None
        if stored == json.loads(json.dumps(asdict(spec))):
            return {"key": spec.key, "skipped": True}
    t0 = time.time()
    rec = pl.run_arm(spec)
    path.write_text(json.dumps(rec))
    return {"key": spec.key, "skipped": False, "usable": rec["usable"],
            "wall_seconds": time.time() - t0}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arrangements", type=int, default=N_ARRANGEMENTS)
    p.add_argument("--inits", type=int, default=N_INITS)
    p.add_argument("--out", type=str, default="results/unconfound_k3")
    p.add_argument("--cap", type=int, default=None)
    args = p.parse_args()

    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    if out.resolve() == (ROOT / "results" / "phase1").resolve():
        raise SystemExit("refusing to write unconfound arms into the registered grid directory")
    out.mkdir(parents=True, exist_ok=True)

    specs = arms(n_arr=args.arrangements, n_init=args.inits)
    print(f"K={args.arrangements} unconfound: {len(specs)} arms "
          f"(γ={GAMMAS}, {CONDITIONS}, stream_id=0..{args.arrangements - 1}, "
          f"seed=0..{args.inits - 1}) -> "
          f"{out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}", flush=True)
    t0 = time.time()
    res = pmap(_job, [(s, str(out)) for s in specs], cap=args.cap)
    done = [r for r in res if not r.get("skipped")]
    print(f"\n  {len(done)} run, {len(res) - len(done)} skipped, "
          f"{time.time() - t0:.0f} s wall")
    bad = [r for r in done if not r.get("usable")]
    if bad:
        print(f"  WARNING {len(bad)} arms not usable: {[r['key'] for r in bad][:5]}")


if __name__ == "__main__":
    main()
