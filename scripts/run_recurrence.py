"""Recurrence first slice (`docs/16` §4 / A8).

64 arms: 2 γ × 2 modes × 2 conditions (S-HL, S-LL) × 8 reserved stream_ids.
Re-present y_0 at k=12, versus a novel dichotomy matched on s_r to y_{11}.
S-HH and S-LH are refused: consecutive s_r=0.9 rounds to Hamming 0 at P=16.
Writes under `results/recurrence/`, never `results/phase1/`.

    python scripts/run_recurrence.py --one-arm
    python scripts/run_recurrence.py
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

OUT = ROOT / "results" / "recurrence"
PRECOMMIT = ROOT / "results" / "recurrence_precommit.json"
SEED = 0
K = 12
MODES = ("represent", "control")


@dataclass(frozen=True)
class RecurrenceSpec(pl.Phase1Spec):
    arrangement_source: str = "stream_rng"
    lr_scaling: str = "quadratic"
    recurrence_mode: str = "represent"
    recurrence_k: int = K

    @property
    def key(self) -> str:
        return super().key + f",recur={self.recurrence_mode},k={self.recurrence_k}"


def _pre() -> dict:
    pre = json.loads(PRECOMMIT.read_text())
    if not pre.get("written_before_running"):
        sys.exit("pre-commit missing")
    if pre.get("written_before_running_applies_to") != "64-arm recurrence first slice":
        sys.exit("this runner governs the recurrence first slice; check recurrence_precommit.json")
    if int(pre["design"]["k"]) != K:
        sys.exit("runner k is not the pre-commit k")
    return pre


def arms() -> list[RecurrenceSpec]:
    pre = _pre()
    sl = pre["design"]["first_slice"]
    gammas = tuple(float(g) for g in sl["gammas"])
    modes = tuple(sl["modes"])
    conditions = tuple(sl["conditions"])
    sids = tuple(sorted(reserved_stream_ids()))
    n = len(gammas) * len(modes) * len(conditions) * len(sids)
    if n != int(sl["n_arms"]):
        sys.exit(f"grid product {n} != pre-commit n_arms {sl['n_arms']}")
    if modes != MODES:
        sys.exit(f"modes {modes} != {MODES}")
    reserved = reserved_stream_ids()
    out = [
        RecurrenceSpec(
            gamma_0=g, a=0.0, condition=c, seed=SEED, stream_id=sid,
            arrangement_source="stream_rng", lr_scaling="quadratic",
            recurrence_mode=mode, recurrence_k=K,
        )
        for g, mode, c, sid in itertools.product(gammas, modes, conditions, sids)
    ]
    for spec in out:
        if spec.stream_id not in reserved:
            raise RuntimeError("non-reserved stream id in the recurrence slice")
        if spec.lr_scaling != "quadratic":
            raise RuntimeError("primary law is quadratic")
        if spec.arrangement_source != "stream_rng":
            raise RuntimeError("recurrence uses the reserved population")
        if spec.recurrence_k != K:
            raise RuntimeError("k is locked at 12")
        if spec.train_config.stopping != "matched_loss":
            raise RuntimeError("recurrence is sequential matched_loss")
    return out


def one_arm() -> RecurrenceSpec:
    sid = min(reserved_stream_ids())
    return RecurrenceSpec(
        gamma_0=10.0, a=0.0, condition="S-HL", seed=SEED, stream_id=sid,
        arrangement_source="stream_rng", recurrence_mode="represent",
        recurrence_k=K,
    )


def _path(spec: RecurrenceSpec) -> Path:
    return OUT / (spec.key.replace(",", "__").replace("=", "-") + ".json")


def _norm(spec: RecurrenceSpec) -> dict:
    return json.loads(json.dumps(asdict(spec)))


def _job(spec: RecurrenceSpec) -> dict:
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
    steps = [t["steps_taken"] for t in rec["tasks"]]
    return {
        "key": rec["key"], "path": str(path), "skipped": False,
        "usable": rec["usable"], "wall_seconds": rec["wall_seconds"],
        "n_tasks_missed": n_miss, "stream_id": spec.stream_id,
        "recurrence_mode": spec.recurrence_mode,
        "steps_at_k": steps[spec.recurrence_k] if spec.recurrence_k < len(steps) else None,
        "steps_at_0": steps[0] if steps else None,
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
        f"recurrence: {len(specs)} arm(s), unique n={len(reserved_stream_ids())}, "
        f"k={K}, stream_rng, quadratic, matched_loss, lr0={specs[0].lr0}, "
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
