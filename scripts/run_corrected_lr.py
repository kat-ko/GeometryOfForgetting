"""Corrected-LR reference at L=2, unique n=8, old arrangements (`docs/16` §0.1).

192 arms: 6 γ × 4 conditions × 8 legacy seeds. `arrangement_source=legacy_seed`.
Writes under `results/corrected_lr/`, never `results/phase1/`.

    python scripts/run_corrected_lr.py --one-arm
    python scripts/run_corrected_lr.py
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from _par import pin_threads  # noqa: E402

pin_threads()

from _par import n_workers, pmap  # noqa: E402
from src import pipeline as pl  # noqa: E402
from src import provenance  # noqa: E402
from src.reservations import old_init_seeds, reserved_stream_ids  # noqa: E402

GAMMAS = (0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
CONDITIONS = ("S-HH", "S-HL", "S-LH", "S-LL")
SEEDS = old_init_seeds()
OUT = ROOT / "results" / "corrected_lr"
PRECOMMIT = ROOT / "results" / "corrected_lr_precommit.json"


@dataclass(frozen=True)
class CorrectedLRSpec(pl.Phase1Spec):
    lr_scaling: str = "corrected"
    arrangement_source: str = "legacy_seed"


def arms() -> list[CorrectedLRSpec]:
    out = [
        CorrectedLRSpec(gamma_0=g, a=0.0, condition=c, seed=s, stream_id=0)
        for g, c, s in itertools.product(GAMMAS, CONDITIONS, SEEDS)
    ]
    reserved = reserved_stream_ids()
    for spec in out:
        if spec.stream_id in reserved:
            raise RuntimeError("reserved stream id in a design-decision run")
        if spec.arrangement_source != "legacy_seed":
            raise RuntimeError("§0.1 must use the old arrangement population")
    return out


def _path(spec: CorrectedLRSpec) -> Path:
    return OUT / (spec.key.replace(",", "__").replace("=", "-") + ".json")


def _job(spec: CorrectedLRSpec) -> dict:
    provenance.assert_current()
    path = _path(spec)
    if path.exists():
        stored = json.loads(path.read_text()).get("spec")
        if stored == json.loads(json.dumps(asdict(spec))):
            return {"key": spec.key, "path": str(path), "skipped": True}
    rec = pl.run_arm(spec)
    path.write_text(json.dumps(rec))
    n_miss = sum(1 for t in rec["tasks"] if not t["converged"])
    return {
        "key": rec["key"], "path": str(path), "skipped": False,
        "usable": rec["usable"], "wall_seconds": rec["wall_seconds"],
        "n_tasks_missed": n_miss,
    }


def one_arm() -> CorrectedLRSpec:
    return CorrectedLRSpec(gamma_0=10.0, a=0.0, condition="S-HL", seed=0, stream_id=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--one-arm", action="store_true",
                    help="γ=10 S-HL seed=0, the measured arm before launch")
    ap.add_argument("--cap", type=int, default=None)
    args = ap.parse_args()

    pre = json.loads(PRECOMMIT.read_text())
    if not pre.get("written_before_running"):
        sys.exit("pre-commit missing")
    OUT.mkdir(parents=True, exist_ok=True)

    specs = [one_arm()] if args.one_arm else arms()
    print(f"corrected-LR: {len(specs)} arm(s), unique n={len(SEEDS)}, "
          f"legacy_seed, lr0={specs[0].lr0}, target={specs[0].target_loss}, "
          f"dir={OUT}", flush=True)
    t0 = time.time()
    results = pmap(_job, specs, cap=args.cap)
    print(json.dumps({
        "n": len(results),
        "n_workers": n_workers(args.cap),
        "wall_seconds": time.time() - t0,
        "usable": sum(1 for r in results if r.get("usable")),
        "missed": sum(1 for r in results if r.get("usable") is False),
        "skipped": sum(1 for r in results if r.get("skipped")),
        "one_arm": bool(args.one_arm),
        "first": results[0] if results else None,
    }, indent=2))


if __name__ == "__main__":
    main()
