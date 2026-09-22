"""Hamming dose × input-change first slice (`docs/20` §5).

192 arms: 2 γ × 3 input levels × 4 Hamming levels × 8 reserved stream_ids.
Frozen / drift / jump at s_f ∈ {1.0, 0.9, 0.1}; s_r ∈ {1.0, 0.75, 0.5, 0.25}
(exact Hamming 0, 2, 4, 6 at P=16). T=16, tracked_stride=2.
Writes under `results/hamming/`, never `results/phase1/`.

    python scripts/run_hamming.py --one-arm
    python scripts/run_hamming.py
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

OUT = ROOT / "results" / "hamming"
PRECOMMIT = ROOT / "results" / "hamming_precommit.json"
SEED = 0
TRACKED_STRIDE = 2
INPUT_LEVELS = ("frozen", "drift", "jump")
INPUT_S_F = {"frozen": 1.0, "drift": 0.9, "jump": 0.1}
READOUT = (1.0, 0.75, 0.5, 0.25)


@dataclass(frozen=True)
class HammingSpec(pl.Phase1Spec):
    arrangement_source: str = "stream_rng"
    lr_scaling: str = "quadratic"
    input_level: str = "jump"
    feature_similarity: float = 0.1
    readout_similarity: float = 0.5
    tracked_stride: int = TRACKED_STRIDE

    @property
    def key(self) -> str:
        return (f"gamma={self.gamma_0:g},a={self.a:g},"
                f"input={self.input_level},s_r={self.readout_similarity:g},"
                f"stream={self.stream_id},seed={self.seed}")


def _pre() -> dict:
    pre = json.loads(PRECOMMIT.read_text())
    if not pre.get("written_before_running"):
        sys.exit("pre-commit missing")
    if pre.get("written_before_running_applies_to") != "192-arm Hamming first slice":
        sys.exit("this runner governs the Hamming first slice; check hamming_precommit.json")
    held = pre["design"]["held_fixed"]
    if int(held["tracked_stride"]) != TRACKED_STRIDE:
        sys.exit("runner stride is not the pre-commit stride")
    if int(held["P"]) != 16:
        sys.exit("runner P is not the pre-commit P")
    return pre


def _spec(gamma: float, level: str, s_r: float, sid: int) -> HammingSpec:
    if level not in INPUT_S_F:
        raise ValueError(f"unknown input level {level!r}")
    return HammingSpec(
        gamma_0=gamma, a=0.0, condition=level, seed=SEED, stream_id=sid,
        arrangement_source="stream_rng", lr_scaling="quadratic",
        input_level=level, feature_similarity=INPUT_S_F[level],
        readout_similarity=s_r, tracked_stride=TRACKED_STRIDE,
    )


def arms() -> list[HammingSpec]:
    pre = _pre()
    sl = pre["design"]["first_slice"]
    gammas = tuple(float(g) for g in sl["gammas"])
    levels = tuple(sl["input_levels"])
    s_rs = tuple(float(s) for s in sl["s_r"])
    sids = tuple(sorted(reserved_stream_ids()))
    n = len(gammas) * len(levels) * len(s_rs) * len(sids)
    if n != int(sl["n_arms"]):
        sys.exit(f"grid product {n} != pre-commit n_arms {sl['n_arms']}")
    if levels != INPUT_LEVELS:
        sys.exit(f"input levels {levels} != {INPUT_LEVELS}")
    if s_rs != READOUT:
        sys.exit(f"s_r {s_rs} != {READOUT}")
    reserved = reserved_stream_ids()
    out = [
        _spec(g, level, s_r, sid)
        for g, level, s_r, sid in itertools.product(gammas, levels, s_rs, sids)
    ]
    for spec in out:
        if spec.stream_id not in reserved:
            raise RuntimeError("non-reserved stream id in the Hamming slice")
        if spec.lr_scaling != "quadratic":
            raise RuntimeError("primary law is quadratic")
        if spec.arrangement_source != "stream_rng":
            raise RuntimeError("Hamming uses the reserved population")
        if spec.tracked_stride != TRACKED_STRIDE:
            raise RuntimeError("tracked_stride is locked at 2")
        if spec.feature_similarity != INPUT_S_F[spec.input_level]:
            raise RuntimeError("s_f does not match the named input level")
        if spec.condition != spec.input_level:
            raise RuntimeError("condition must be the input-level name")
        if spec.train_config.stopping != "matched_loss":
            raise RuntimeError("Hamming is sequential matched_loss")
        if spec.train_config.optimizer != "sgd":
            raise RuntimeError("Hamming is SGD; Adam is docs/22, never this runner")
        if spec.P != 16:
            raise RuntimeError("P is locked at 16")
    return out


def one_arm() -> HammingSpec:
    return _spec(10.0, "frozen", 0.5, min(reserved_stream_ids()))


def _path(spec: HammingSpec) -> Path:
    return OUT / (spec.key.replace(",", "__").replace("=", "-") + ".json")


def _norm(spec: HammingSpec) -> dict:
    return json.loads(json.dumps(asdict(spec)))


def _job(spec: HammingSpec) -> dict:
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
        "n_evals": rec["n_evals"], "n_tasks_missed": n_miss,
        "stream_id": spec.stream_id, "input_level": spec.input_level,
        "readout_similarity": spec.readout_similarity,
        "tracked_stride": spec.tracked_stride,
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
        f"hamming: {len(specs)} arm(s), unique n={len(reserved_stream_ids())}, "
        f"stride={TRACKED_STRIDE}, stream_rng, quadratic, matched_loss, "
        f"lr0={specs[0].lr0}, workers={workers}, dir={OUT}",
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
