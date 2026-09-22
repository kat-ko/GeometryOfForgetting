"""Rank first-slice arms on reserved arrangements (`docs/16` §1 / A5 / A14).

288 arms: 3 γ × 3 r × 4 conditions × 8 reserved stream_ids. Nested Q, quadratic,
seed=0. Writes under `results/rank/`, never `results/phase1/`.

The un-nested one-arm that preceded the pre-commit commit is not in this
directory and is not skipped-into the slice.

    python scripts/run_rank.py --one-arm
    python scripts/run_rank.py
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

OUT = ROOT / "results" / "rank"
PRECOMMIT = ROOT / "results" / "rank_precommit.json"
SEED = 0


@dataclass(frozen=True)
class RankSpec(pl.Phase1Spec):
    readout_rank: int = 300
    arrangement_source: str = "stream_rng"
    lr_scaling: str = "quadratic"
    q_nested: bool = True

    @property
    def key(self) -> str:
        return super().key + f",rank={self.readout_rank}"


def _pre() -> dict:
    pre = json.loads(PRECOMMIT.read_text())
    if not pre.get("written_before_running"):
        sys.exit("pre-commit missing")
    if pre.get("written_before_running_applies_to") != "288-arm first slice":
        sys.exit("this runner governs the 288-arm slice; check rank_precommit.json")
    return pre


def _grid(pre: dict) -> tuple[tuple[float, ...], tuple[int, ...], tuple[str, ...], tuple[int, ...]]:
    sl = pre["design"]["first_slice"]
    gammas = tuple(float(g) for g in sl["gammas"])
    ranks = tuple(int(r) for r in sl["ranks"])
    conditions = tuple(sl["conditions"])
    sids = tuple(sorted(reserved_stream_ids()))
    if ranks != (4, 16, 300):
        sys.exit(f"rank grid {ranks} is not the A5 slice")
    n = len(gammas) * len(ranks) * len(conditions) * len(sids)
    if n != int(sl["n_arms"]):
        sys.exit(f"grid product {n} != pre-commit n_arms {sl['n_arms']}")
    return gammas, ranks, conditions, sids


def arms() -> list[RankSpec]:
    pre = _pre()
    gammas, ranks, conditions, sids = _grid(pre)
    reserved = reserved_stream_ids()
    out = [
        RankSpec(
            gamma_0=g, a=0.0, condition=c, seed=SEED, stream_id=sid,
            readout_rank=r, arrangement_source="stream_rng", lr_scaling="quadratic",
            q_nested=True,
        )
        for g, r, c, sid in itertools.product(gammas, ranks, conditions, sids)
    ]
    for spec in out:
        if spec.stream_id not in reserved:
            raise RuntimeError("non-reserved stream id in the rank slice")
        if spec.arrangement_source != "stream_rng":
            raise RuntimeError("rank uses the reserved population, not legacy_seed")
        if spec.lr_scaling != "quadratic":
            raise RuntimeError("rank primary law is quadratic")
        if not spec.q_nested:
            raise RuntimeError("slice Q must be nested")
    return out


def one_arm() -> RankSpec:
    """Same cell as the un-nested one-arm, now under nested Q. Timing, then the slice."""
    sid = min(reserved_stream_ids())
    spec = RankSpec(
        gamma_0=10.0, a=0.0, condition="S-HL", seed=SEED, stream_id=sid,
        readout_rank=4, arrangement_source="stream_rng",
    )
    if spec.stream_id not in reserved_stream_ids():
        sys.exit("one-arm must use a reserved arrangement")
    return spec


def _path(spec: RankSpec) -> Path:
    return OUT / (spec.key.replace(",", "__").replace("=", "-") + ".json")


def _norm(spec: RankSpec) -> dict:
    return json.loads(json.dumps(asdict(spec)))


def _job(spec: RankSpec) -> dict:
    provenance.assert_current()
    path = _path(spec)
    if path.exists():
        stored = json.loads(path.read_text()).get("spec")
        if stored == _norm(spec):
            return {"key": spec.key, "path": str(path), "skipped": True}
        sys.exit(
            f"refusing to overwrite {path.name}: stored spec does not match this runner "
            "(un-nested one-arm must not sit in the slice directory)"
        )
    rec = pl.run_arm(spec)
    path.write_text(json.dumps(rec))
    n_miss = sum(1 for t in rec["tasks"] if not t["converged"])
    return {
        "key": rec["key"], "path": str(path), "skipped": False,
        "usable": rec["usable"], "wall_seconds": rec["wall_seconds"],
        "n_tasks_missed": n_miss,
        "readout_rank": spec.readout_rank,
        "stream_id": spec.stream_id,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--one-arm", action="store_true",
                    help="nested-Q re-time of γ=10 r=4 S-HL first reserved arrangement")
    ap.add_argument("--cap", type=int, default=None)
    args = ap.parse_args()

    pre = _pre()
    OUT.mkdir(parents=True, exist_ok=True)
    specs = [one_arm()] if args.one_arm else arms()
    workers = n_workers(args.cap)
    print(
        f"rank: {len(specs)} arm(s), unique n={len(reserved_stream_ids())}, "
        f"stream_rng, nested Q, quadratic, lr0={specs[0].lr0}, "
        f"target={specs[0].target_loss}, workers={workers}, dir={OUT}",
        flush=True,
    )
    if not args.one_arm:
        measured = pre["cost"]["one_arm_wall_seconds"]
        print(
            f"cost: un-nested one-arm was {measured}s serial (lower bound). "
            f"project {len(specs)}/{workers} × that ≈ "
            f"{len(specs) / workers * measured / 3600:.2f} h if measurement-dominated; "
            f"budget 60-100 h serial. nested-Q one-arm is the current-code time.",
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
        "q_nested": True,
        "first": results[0] if results else None,
    }, indent=2))


if __name__ == "__main__":
    main()
