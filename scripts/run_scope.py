"""Scope granularity first slice (`docs/16` §3 / A7).

192 arms: 2 γ × 3 K × 4 conditions × 8 reserved stream_ids.
Worst-of-K stopping. Writes under `results/scope/`, never `results/phase1/`.

    python scripts/run_scope.py --one-arm
    python scripts/run_scope.py
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
from src.train.loop import TrainConfig  # noqa: E402
from src.reservations import reserved_stream_ids  # noqa: E402

OUT = ROOT / "results" / "scope"
PRECOMMIT = ROOT / "results" / "scope_precommit.json"
SEED = 0


@dataclass(frozen=True)
class ScopeSpec(pl.Phase1Spec):
    arrangement_source: str = "stream_rng"
    lr_scaling: str = "quadratic"
    scope_K: int = 1
    n_outputs: int = 1

    @property
    def key(self) -> str:
        return super().key + f",K={self.scope_K}"

    @property
    def train_config(self) -> TrainConfig:
        return TrainConfig(
            steps_per_task=self.steps_per_task, stopping="worst_of_k",
            target_loss=self.target_loss, record_every=self.record_every,
        )


def _pre() -> dict:
    pre = json.loads(PRECOMMIT.read_text())
    if not pre.get("written_before_running"):
        sys.exit("pre-commit missing")
    if pre.get("written_before_running_applies_to") != "192-arm scope granularity first slice":
        sys.exit("this runner governs the scope first slice; check scope_precommit.json")
    return pre


def arms() -> list[ScopeSpec]:
    pre = _pre()
    sl = pre["design"]["first_slice"]
    gammas = tuple(float(g) for g in sl["gammas"])
    ks = tuple(int(k) for k in sl["K"])
    conditions = tuple(sl["conditions"])
    sids = tuple(sorted(reserved_stream_ids()))
    n = len(gammas) * len(ks) * len(conditions) * len(sids)
    if n != int(sl["n_arms"]):
        sys.exit(f"grid product {n} != pre-commit n_arms {sl['n_arms']}")
    reserved = reserved_stream_ids()
    out = [
        ScopeSpec(
            gamma_0=g, a=0.0, condition=c, seed=SEED, stream_id=sid,
            arrangement_source="stream_rng", lr_scaling="quadratic",
            scope_K=k, n_outputs=k,
        )
        for g, k, c, sid in itertools.product(gammas, ks, conditions, sids)
    ]
    for spec in out:
        if spec.stream_id not in reserved:
            raise RuntimeError("non-reserved stream id in the scope slice")
        if spec.n_outputs != spec.scope_K:
            raise RuntimeError("n_outputs must equal scope_K")
        if spec.lr_scaling != "quadratic":
            raise RuntimeError("primary law is quadratic")
        if spec.train_config.stopping != "worst_of_k":
            raise RuntimeError("scope uses worst-of-K stopping")
    return out


def one_arm() -> ScopeSpec:
    sid = min(reserved_stream_ids())
    return ScopeSpec(
        gamma_0=10.0, a=0.0, condition="S-HL", seed=SEED, stream_id=sid,
        arrangement_source="stream_rng", scope_K=4, n_outputs=4,
    )


def _path(spec: ScopeSpec) -> Path:
    return OUT / (spec.key.replace(",", "__").replace("=", "-") + ".json")


def _norm(spec: ScopeSpec) -> dict:
    return json.loads(json.dumps(asdict(spec)))


def _job(spec: ScopeSpec) -> dict:
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
        "scope_K": spec.scope_K,
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
        f"scope: {len(specs)} arm(s), unique n={len(reserved_stream_ids())}, "
        f"stream_rng, quadratic, worst-of-K, lr0={specs[0].lr0}, "
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
