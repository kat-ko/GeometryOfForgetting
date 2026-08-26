"""Width robustness: does the channel reorganization survive changing `N`?

Everything in Phase 1 is reported in noise floors and capacity ratios, and both depend on
`N` — capacity is `P/N_crit`, and the load is `P/N`. So "one architecture" is the most
obvious vulnerability of the γ result, and it is cheap to close now that the grid is done:
re-run the γ extremes and the reference point at half and double width, and ask whether the
attribution's channel mix reorganizes the same way.

`N` is the hidden width; `d = 150` is the input dimension and is held fixed. Load moves
`P/N` = 16/150 = 0.107, 16/300 = 0.053, 16/600 = 0.027, which spans a 4× range around the
design point.

**Writes to `results/width/`, not `results/phase1/`.** `Phase1Spec.key` omits `N`, so a
width arm's filename collides with a completed grid arm's; the file name here carries `N`
explicitly and the loader checks it.

Cost is measured, not projected — the Phase 1 cost model was wrong by 9× (see
`results/LOG.md`), so `--time-one` runs a single arm per width and reports before any grid
is launched.
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

WIDTHS = (150, 600)                 # 300 is the completed grid
GAMMAS = (0.03, 1.0, 10.0)          # extremes plus the reference point
CONDITIONS = ("S-HH", "S-HL", "S-LH", "S-LL")

OUT = ROOT / "results" / "width"


def arms(*, widths=WIDTHS, gammas=GAMMAS, conditions=CONDITIONS,
         streams: int = 2, seeds: int = 2) -> list[pl.Phase1Spec]:
    """All four conditions kept, seeds/streams cut — Figure 2 pools over conditions, so
    dropping to one condition would change the comparison rather than replicate it."""
    out = [
        pl.Phase1Spec(gamma_0=g, a=0.0, condition=c, stream_id=s, seed=sd, N=n)
        for n, g, c, s, sd in itertools.product(
            widths, gammas, conditions, range(streams), range(seeds))
    ]
    # Cheapest width first: if the estimate is wrong, it is wrong early and visibly.
    return sorted(out, key=lambda s: (s.N, s.gamma_0))


def _path(spec: pl.Phase1Spec, out: Path) -> Path:
    return out / f"N{spec.N}__{spec.key.replace(',', '__').replace('=', '-')}.json"


def _job(payload: tuple[pl.Phase1Spec, str]) -> dict:
    spec, out_s = payload
    provenance.assert_current()
    out = Path(out_s)
    path = _path(spec, out)
    if path.exists():
        try:
            stored = json.loads(path.read_text())["spec"]
        except (json.JSONDecodeError, KeyError):
            stored = None
        if stored == json.loads(json.dumps(asdict(spec))):
            return {"key": spec.key, "N": spec.N, "skipped": True}
    t0 = time.time()
    rec = pl.run_arm(spec)
    path.write_text(json.dumps(rec))
    return {"key": spec.key, "N": spec.N, "skipped": False, "usable": rec["usable"],
            "wall_seconds": time.time() - t0}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--time-one", action="store_true",
                   help="one arm per width, report cost, launch nothing")
    p.add_argument("--streams", type=int, default=2)
    p.add_argument("--seeds", type=int, default=2)
    p.add_argument("--gammas", type=float, nargs="+", default=None,
                   help="subset of γ; default is the three-point (0.03, 1, 10) grid")
    p.add_argument("--out", type=str, default="results/width",
                   help="directory under the project root, or an absolute path. New n=40 "
                        "cells at a fixed γ must not mix with the 2×2 files whose stream_id "
                        "did not key the arrangement.")
    p.add_argument("--cap", type=int, default=None, help="max workers (see `_par.pmap`)")
    args = p.parse_args()
    gammas = tuple(args.gammas) if args.gammas is not None else GAMMAS
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)

    if args.time_one:
        for n in WIDTHS:
            spec = pl.Phase1Spec(gamma_0=max(gammas), a=0.0, condition="S-HL",
                                 stream_id=0, seed=0, N=n)
            t0 = time.time()
            _job((spec, str(out)))
            dt = time.time() - t0
            print(f"  N={n:<4d} one arm at γ={spec.gamma_0:g}: {dt:7.1f} s "
                  f"({dt / 60:.1f} min single-threaded-ish)", flush=True)
        full = len(arms(streams=args.streams, seeds=args.seeds, gammas=gammas))
        print(f"\n  this width grid would be {full} arms; scale the per-arm times above by\n"
              f"  {full} / n_workers to project, and remember the γ=10 arm timed here is\n"
              f"  the *cheapest* in steps-to-target — lazy arms take more SGD steps.")
        return

    specs = arms(streams=args.streams, seeds=args.seeds, gammas=gammas)
    print(f"width grid: {len(specs)} arms over N={WIDTHS}, γ={gammas} -> "
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
