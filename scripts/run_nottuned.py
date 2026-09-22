"""Not-tuned reserved-set duplicate of the headline (`docs/16` A3).

192 arms: 6 γ × 4 conditions × 8 reserved stream_ids. Quadratic, spherical,
unconstrained readout (Q = I). Writes under `results/nottuned/`, never
`results/phase1/` or `results/rank/`.

    python scripts/run_nottuned.py --one-arm
    python scripts/run_nottuned.py
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

OUT = ROOT / "results" / "nottuned"
PRECOMMIT = ROOT / "results" / "nottuned_precommit.json"
SEED = 0


@dataclass(frozen=True)
class NottunedSpec(pl.Phase1Spec):
    arrangement_source: str = "stream_rng"
    lr_scaling: str = "quadratic"
    manifold_kind: str = "spherical"

    @property
    def key(self) -> str:
        return super().key + ",pop=reserved"


def _pre() -> dict:
    pre = json.loads(PRECOMMIT.read_text())
    if not pre.get("written_before_running"):
        sys.exit("pre-commit missing")
    if pre.get("written_before_running_applies_to") != "192-arm not-tuned reserved duplicate":
        sys.exit("this runner governs the not-tuned duplicate; check nottuned_precommit.json")
    return pre


def arms() -> list[NottunedSpec]:
    pre = _pre()
    sl = pre["design"]["grid"]
    gammas = tuple(float(g) for g in sl["gammas"])
    conditions = tuple(sl["conditions"])
    sids = tuple(sorted(reserved_stream_ids()))
    n = len(gammas) * len(conditions) * len(sids)
    if n != int(sl["n_arms"]):
        sys.exit(f"grid product {n} != pre-commit n_arms {sl['n_arms']}")
    reserved = reserved_stream_ids()
    out = [
        NottunedSpec(
            gamma_0=g, a=0.0, condition=c, seed=SEED, stream_id=sid,
            arrangement_source="stream_rng", lr_scaling="quadratic",
            manifold_kind="spherical",
        )
        for g, c, sid in itertools.product(gammas, conditions, sids)
    ]
    for spec in out:
        if spec.stream_id not in reserved:
            raise RuntimeError("non-reserved stream id in the not-tuned duplicate")
        if spec.manifold_kind != "spherical":
            raise RuntimeError("not-tuned duplicate is the registered spherical generator")
        if spec.lr_scaling != "quadratic":
            raise RuntimeError("primary law is quadratic")
        if spec.arrangement_source != "stream_rng":
            raise RuntimeError("not-tuned uses the reserved population, not legacy_seed")
    return out


def one_arm() -> NottunedSpec:
    sid = min(reserved_stream_ids())
    return NottunedSpec(
        gamma_0=10.0, a=0.0, condition="S-HL", seed=SEED, stream_id=sid,
        arrangement_source="stream_rng", manifold_kind="spherical",
    )


def _path(spec: NottunedSpec) -> Path:
    return OUT / (spec.key.replace(",", "__").replace("=", "-") + ".json")


def _norm(spec: NottunedSpec) -> dict:
    return json.loads(json.dumps(asdict(spec)))


def _job(spec: NottunedSpec) -> dict:
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
        f"nottuned: {len(specs)} arm(s), unique n={len(reserved_stream_ids())}, "
        f"stream_rng, quadratic, spherical, lr0={specs[0].lr0}, "
        f"workers={workers}, dir={OUT}",
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
