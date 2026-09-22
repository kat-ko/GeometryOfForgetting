"""Isotropic-Gaussian D.1.1 control (`docs/16` A10 / A15).

64 arms: 2 γ × 4 conditions × 8 reserved stream_ids. No rank sweep.
Writes under `results/isotropic/`, never `results/phase1/` or `results/rank/`.

    python scripts/run_isotropic.py --one-arm
    python scripts/run_isotropic.py
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
from src.reservations import reserved_stream_ids  # noqa: E402

OUT = ROOT / "results" / "isotropic"
PRECOMMIT = ROOT / "results" / "isotropic_precommit.json"
SEED = 0


@dataclass(frozen=True)
class IsotropicSpec(pl.Phase1Spec):
    arrangement_source: str = "stream_rng"
    lr_scaling: str = "quadratic"
    manifold_kind: str = "isotropic_gaussian"

    @property
    def key(self) -> str:
        return super().key + ",kind=iso"


def _pre() -> dict:
    pre = json.loads(PRECOMMIT.read_text())
    if not pre.get("written_before_running"):
        sys.exit("pre-commit missing")
    if pre.get("written_before_running_applies_to") != "64-arm isotropic-Gaussian control":
        sys.exit("this runner governs the isotropic control; check isotropic_precommit.json")
    return pre


def arms() -> list[IsotropicSpec]:
    pre = _pre()
    sl = pre["design"]["first_slice"]
    gammas = tuple(float(g) for g in sl["gammas"])
    conditions = tuple(sl["conditions"])
    sids = tuple(sorted(reserved_stream_ids()))
    n = len(gammas) * len(conditions) * len(sids)
    if n != int(sl["n_arms"]):
        sys.exit(f"grid product {n} != pre-commit n_arms {sl['n_arms']}")
    reserved = reserved_stream_ids()
    out = [
        IsotropicSpec(
            gamma_0=g, a=0.0, condition=c, seed=SEED, stream_id=sid,
            arrangement_source="stream_rng", lr_scaling="quadratic",
            manifold_kind="isotropic_gaussian",
        )
        for g, c, sid in itertools.product(gammas, conditions, sids)
    ]
    for spec in out:
        if spec.stream_id not in reserved:
            raise RuntimeError("non-reserved stream id in the isotropic control")
        if spec.manifold_kind != "isotropic_gaussian":
            raise RuntimeError("isotropic control must use isotropic_gaussian")
        if spec.lr_scaling != "quadratic":
            raise RuntimeError("primary law is quadratic")
    return out


def one_arm() -> IsotropicSpec:
    sid = min(reserved_stream_ids())
    return IsotropicSpec(
        gamma_0=10.0, a=0.0, condition="S-HL", seed=SEED, stream_id=sid,
        arrangement_source="stream_rng", manifold_kind="isotropic_gaussian",
    )


def _path(spec: IsotropicSpec) -> Path:
    return OUT / (spec.key.replace(",", "__").replace("=", "-") + ".json")


def _norm(spec: IsotropicSpec) -> dict:
    return json.loads(json.dumps(asdict(spec)))


def _job(spec: IsotropicSpec) -> dict:
    provenance.assert_current()
    path = _path(spec)
    if path.exists():
        stored = json.loads(path.read_text()).get("spec")
        if stored == _norm(spec):
            return {"key": spec.key, "path": str(path), "skipped": True}
        sys.exit(f"refusing to overwrite {path.name}: stored spec does not match")
    rec = pl.run_arm(spec)
    path.write_text(json.dumps(rec))
    n_miss = sum(1 for t in rec["tasks"] if not t["converged"])
    return {
        "key": rec["key"], "path": str(path), "skipped": False,
        "usable": rec["usable"], "wall_seconds": rec["wall_seconds"],
        "n_tasks_missed": n_miss, "stream_id": spec.stream_id,
        "manifold_kind": spec.manifold_kind,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--one-arm", action="store_true")
    ap.add_argument("--cap", type=int, default=None)
    args = ap.parse_args()
    _pre()
    OUT.mkdir(parents=True, exist_ok=True)
    specs = [one_arm()] if args.one_arm else arms()
    workers = n_workers(args.cap)
    print(
        f"isotropic: {len(specs)} arm(s), unique n={len(reserved_stream_ids())}, "
        f"stream_rng, quadratic, lr0={specs[0].lr0}, workers={workers}, dir={OUT}",
        flush=True,
    )
    t0 = time.time()
    results = pmap(_job, specs, cap=args.cap)
    print(json.dumps({
        "n": len(results),
        "n_workers": workers,
        "wall_seconds": time.time() - t0,
        "usable": sum(1 for r in results if r.get("usable")),
        "missed": sum(1 for r in results if r.get("usable") is False),
        "skipped": sum(1 for r in results if r.get("skipped")),
        "one_arm": bool(args.one_arm),
        "first": results[0] if results else None,
    }, indent=2))


if __name__ == "__main__":
    main()
