"""γ = 30: does the S-HL decorrelation extend, or is it an edge-of-sweep effect?

§5.4's weakest point is that `S-HL` center decorrelation exists at exactly one γ. At γ=3 it
is absent (+0.0034, p=1.0); at γ=10 it is −0.0550 on 40 of 40 arms. One further point on the
log-spaced ladder turns a single value into a trend, or fails to and says so.

**All four corners, not just S-HL.** The finding that replaced H6 is the *additive*
decomposition into a readout effect and a feature effect, and both main effects are
half-differences over all four cells — two corners cannot produce either. Extending the
decorrelation without extending the decomposition would test the weaker claim.

**Writes to `results/gamma_ext/`, deliberately.** `Phase1Spec.key` carries γ, so a γ=30 arm
would not overwrite a grid arm — it would be silently *adopted* by `grid.load()` and change
Figure 2, Figure 4 and the scope audit without anything failing. γ=30 is outside the
registered sweep and stays in its own directory until something explicitly asks for it.

**γ is now a parameter, and each value gets its own directory.** The registered sweep steps by
3.3× from γ=3 to γ=10, so the `S-HL` decorrelation onset is bracketed by a factor of 3.3 and no
re-analysis can narrow it (`docs/12-record-report.md` Q10). γ=5 sits inside that bracket. It is
off-design in exactly the way γ=30 is, so it gets the same treatment: its own directory, never
adopted by `grid.load()` unless something asks for it by name.

    python scripts/run_gamma_ext.py --time-one     # cost first, per the 9× cost-model error
    python scripts/run_gamma_ext.py                # γ=30, results/gamma_ext/
    python scripts/run_gamma_ext.py --gamma 5      # γ=5,  results/gamma_5/ (n=16 duplicate-stream)
    python scripts/run_gamma_ext.py --gamma 5 --streams 5 --seeds 8 --out results/gamma_5_n40
        # 40 unique arms/corner. Own directory so it does not mix with the n=16 set.
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
from src.analysis import grid as gridmod  # noqa: E402

GAMMA = 30.0
CONDITIONS = ("S-HH", "S-HL", "S-LH", "S-LL")

# γ=30 keeps the original directory name so no existing path or manifest entry moves; anything
# else is `results/gamma_<value>/`. Never the registered grid directory, whatever the value.
DIRS = {30.0: "gamma_ext"}


def out_dir(gamma: float) -> Path:
    name = DIRS.get(gamma) or f"gamma_{gamma:g}".replace(".", "p")
    return ROOT / "results" / name


def arms(gamma: float = GAMMA, *, streams: int = 4, seeds: int = 4) -> list[pl.Phase1Spec]:
    """16 arms per corner. At γ=10 the SEM on Δρ_c was 0.0033 with 40 arms, so 16 gives
    roughly 0.005 against effects of 0.055–0.110 — ample, and 2.5× cheaper."""
    return [
        pl.Phase1Spec(gamma_0=gamma, a=0.0, condition=c, stream_id=s, seed=sd)
        for c, s, sd in itertools.product(CONDITIONS, range(streams), range(seeds))
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
    p.add_argument("--time-one", action="store_true")
    p.add_argument("--gamma", type=float, default=GAMMA)
    p.add_argument("--streams", type=int, default=4)
    p.add_argument("--seeds", type=int, default=4)
    p.add_argument("--out", type=str, default=None,
                   help="directory under the project root. Required when writing a unique-n "
                        "rerun that must not mix with duplicate-stream files at the default path.")
    p.add_argument("--cap", type=int, default=None)
    args = p.parse_args()

    gamma = args.gamma
    if gamma in gridmod.REGISTERED_GAMMAS:
        raise SystemExit(
            f"γ={gamma:g} is in the registered sweep. This script writes off-design arms to their "
            f"own directory; a registered value belongs in results/phase1/ via run_phase1.py.")
    out = Path(args.out) if args.out else out_dir(gamma)
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)

    if args.time_one:
        for cond in ("S-HL", "S-LH"):
            spec = pl.Phase1Spec(gamma_0=gamma, a=0.0, condition=cond, stream_id=0, seed=0)
            t0 = time.time()
            r = _job((spec, str(out)))
            dt = time.time() - t0
            print(f"  {cond}  one arm at γ={gamma:g}: {dt:7.1f} s  "
                  f"usable={r.get('usable')}", flush=True)
        n = len(arms(gamma, streams=args.streams, seeds=args.seeds))
        print(f"\n  full extension is {n} arms. Rich arms are the *cheapest* in the grid "
              f"(fewest\n  SGD steps to matched loss), so this is an upper-bound-free "
              f"estimate: n/workers × per-arm.")
        return

    specs = arms(gamma, streams=args.streams, seeds=args.seeds)
    print(f"γ={gamma:g} extension: {len(specs)} arms over {CONDITIONS} -> "
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
