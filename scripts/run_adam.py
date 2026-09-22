"""Adam Hamming first slice (`docs/22`). Same streams, same MSE pin, Adam update.

192 arms: 2 γ × 3 input × 4 Hamming × 8 ids. Refuses unless the richness
gate reads `clears_decade`. After `partial_manipulation`, the arm that
can still run is Hamming-only at γ=10 (96). Writes under `results/adam/`,
never `results/phase1/`. I3 exception licensed only here.

    python scripts/run_adam.py --one-arm
    python scripts/run_adam.py --richness-gate
    python scripts/run_adam.py --one-arm-hamming
    python scripts/run_adam.py --hamming-only
    python scripts/run_adam.py
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
from run_hamming import INPUT_LEVELS, INPUT_S_F, READOUT, TRACKED_STRIDE, HammingSpec  # noqa: E402
from src import pipeline as pl  # noqa: E402
from src import provenance  # noqa: E402
from src.models import paired_init  # noqa: E402
from src.reservations import reserved_stream_ids  # noqa: E402
from src.train.loop import train_task  # noqa: E402

OUT = ROOT / "results" / "adam"
PRECOMMIT = ROOT / "results" / "adam_precommit.json"
HAMMING_PRECOMMIT = ROOT / "results" / "adam_hamming_precommit.json"
ONE_ARM_REPORT = ROOT / "results" / "adam_one_arm.json"
HAMMING_ONE_ARM_REPORT = ROOT / "results" / "adam_hamming_one_arm.json"
GATE_REPORT = ROOT / "results" / "adam_richness_gate.json"
SEED = 0
GATE_S_R = 0.5
GATE_INPUT = "frozen"
HAMMING_ONE_ARM_INPUT = "drift"
HAMMING_ONE_ARM_S_R = 0.5
HAMMING_ONLY_GAMMA = 10.0


@dataclass(frozen=True)
class AdamSpec(HammingSpec):
    optimizer: str = "adam"

    @property
    def key(self) -> str:
        return (f"gamma={self.gamma_0:g},a={self.a:g},"
                f"input={self.input_level},s_r={self.readout_similarity:g},"
                f"stream={self.stream_id},seed={self.seed},opt=adam")


def _pre() -> dict:
    pre = json.loads(PRECOMMIT.read_text())
    if not pre.get("written_before_running"):
        sys.exit("pre-commit missing")
    if pre.get("written_before_running_applies_to") != "192-arm Adam Hamming first slice":
        sys.exit("this runner governs the Adam Hamming first slice; check adam_precommit.json")
    if pre.get("not_in_phase1") is not True:
        sys.exit("Adam is off-design; not_in_phase1 must be true")
    if pre["design"]["loss"] != "mse":
        sys.exit("Adam arm holds MSE")
    if pre["design"]["optimizer"]["name"] != "adam":
        sys.exit("Adam arm is Adam")
    held = pre["design"]["held_fixed"]
    if int(held["tracked_stride"]) != TRACKED_STRIDE:
        sys.exit("runner stride is not the pre-commit stride")
    if int(held["P"]) != 16:
        sys.exit("runner P is not the pre-commit P")
    if float(held["target_loss"]) != 0.05:
        sys.exit("runner target_loss is not the registered pin")
    return pre


def _lr0(pre: dict | None = None) -> float:
    held = (pre or _pre())["design"]["held_fixed"]
    return float(held["lr0"])


def _spec(gamma: float, level: str, s_r: float, sid: int, *, lr0: float | None = None) -> AdamSpec:
    if level not in INPUT_S_F:
        raise ValueError(f"unknown input level {level!r}")
    return AdamSpec(
        gamma_0=gamma, a=0.0, condition=level, seed=SEED, stream_id=sid,
        arrangement_source="stream_rng", lr_scaling="quadratic",
        input_level=level, feature_similarity=INPUT_S_F[level],
        readout_similarity=s_r, tracked_stride=TRACKED_STRIDE,
        optimizer="adam", lr0=_lr0() if lr0 is None else lr0,
    )


def arms() -> list[AdamSpec]:
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
            raise RuntimeError("non-reserved stream id in the Adam slice")
        if spec.lr_scaling != "quadratic":
            raise RuntimeError("primary law is quadratic")
        if spec.optimizer != "adam":
            raise RuntimeError("this runner is Adam")
        if spec.train_config.optimizer != "adam":
            raise RuntimeError("train_config did not carry Adam")
        if spec.train_config.loss != "mse":
            raise RuntimeError("Adam arm holds MSE")
        if spec.arrangement_source != "stream_rng":
            raise RuntimeError("Adam uses the reserved population")
        if spec.tracked_stride != TRACKED_STRIDE:
            raise RuntimeError("tracked_stride is locked at 2")
        if spec.feature_similarity != INPUT_S_F[spec.input_level]:
            raise RuntimeError("s_f does not match the named input level")
        if spec.condition != spec.input_level:
            raise RuntimeError("condition must be the input-level name")
        if spec.train_config.stopping != "matched_loss":
            raise RuntimeError("Adam is sequential matched_loss")
        if spec.P != 16:
            raise RuntimeError("P is locked at 16")
        if spec.lr0 != _lr0(pre):
            raise RuntimeError("lr0 is the pre-commit pin")
    return out


def one_arm() -> AdamSpec:
    return _spec(10.0, GATE_INPUT, GATE_S_R, min(reserved_stream_ids()))


def one_arm_hamming() -> AdamSpec:
    """γ=10, drift × s_r=0.5, stream 10000. Not the richness-gate cell."""
    return _spec(HAMMING_ONLY_GAMMA, HAMMING_ONE_ARM_INPUT, HAMMING_ONE_ARM_S_R,
                 min(reserved_stream_ids()))


def _pre_hamming() -> dict:
    pre = json.loads(HAMMING_PRECOMMIT.read_text())
    if not pre.get("written_before_running"):
        sys.exit("hamming-only pre-commit missing")
    if pre.get("written_before_running_applies_to") != "96-arm Adam Hamming-only at γ=10":
        sys.exit("this runner's --hamming-only governs the 96; check adam_hamming_precommit.json")
    if pre.get("not_in_phase1") is not True:
        sys.exit("Adam is off-design; not_in_phase1 must be true")
    if pre.get("not_the_192") is not True:
        sys.exit("hamming-only is not the 192")
    if pre["design"]["loss"] != "mse":
        sys.exit("Adam arm holds MSE")
    if pre["design"]["optimizer"]["name"] != "adam":
        sys.exit("Adam arm is Adam")
    held = pre["design"]["held_fixed"]
    if int(held["tracked_stride"]) != TRACKED_STRIDE:
        sys.exit("runner stride is not the pre-commit stride")
    if int(held["P"]) != 16:
        sys.exit("runner P is not the pre-commit P")
    if float(held["target_loss"]) != 0.05:
        sys.exit("runner target_loss is not the registered pin")
    if float(held["lr0"]) != _lr0():
        sys.exit("hamming-only lr0 is not the signed pin")
    cannot = pre.get("cannot_speak_to") or []
    for item in ("γ", "finding 1", "channel reorganisation"):
        if item not in cannot:
            sys.exit(f"pre-commit must state this arm cannot speak to {item}")
    return pre


def hamming_only_arms() -> list[AdamSpec]:
    pre = _pre_hamming()
    sl = pre["design"]["first_slice"]
    gammas = tuple(float(g) for g in sl["gammas"])
    levels = tuple(sl["input_levels"])
    s_rs = tuple(float(s) for s in sl["s_r"])
    sids = tuple(sorted(reserved_stream_ids()))
    n = len(gammas) * len(levels) * len(s_rs) * len(sids)
    if n != int(sl["n_arms"]):
        sys.exit(f"grid product {n} != pre-commit n_arms {sl['n_arms']}")
    if gammas != (HAMMING_ONLY_GAMMA,):
        sys.exit(f"hamming-only is γ={HAMMING_ONLY_GAMMA:g} only, got {gammas}")
    if levels != INPUT_LEVELS:
        sys.exit(f"input levels {levels} != {INPUT_LEVELS}")
    if s_rs != READOUT:
        sys.exit(f"s_r {s_rs} != {READOUT}")
    if n != 96:
        sys.exit(f"hamming-only is 96 arms, got {n}")
    reserved = reserved_stream_ids()
    out = [
        _spec(g, level, s_r, sid)
        for g, level, s_r, sid in itertools.product(gammas, levels, s_rs, sids)
    ]
    for spec in out:
        if spec.stream_id not in reserved:
            raise RuntimeError("non-reserved stream id in the Adam slice")
        if spec.gamma_0 != HAMMING_ONLY_GAMMA:
            raise RuntimeError("hamming-only is γ=10")
        if spec.optimizer != "adam":
            raise RuntimeError("this runner is Adam")
        if spec.train_config.optimizer != "adam":
            raise RuntimeError("train_config did not carry Adam")
        if spec.lr0 != _lr0():
            raise RuntimeError("lr0 is the pre-commit pin")
    return out


def gate_arms() -> list[AdamSpec]:
    """Unique n=8, frozen × s_r=0.5, both γ. Training-only richness gate."""
    sids = tuple(sorted(reserved_stream_ids()))
    out = [
        _spec(g, GATE_INPUT, GATE_S_R, sid)
        for g, sid in itertools.product((10.0, 1.0), sids)
    ]
    if len(out) != 16:
        raise RuntimeError(f"gate is 16 arms, got {len(out)}")
    return out


def gate_reading(ratio: float) -> str:
    """Named before the 16-arm numbers. One decade is still the bar.

    `clears_decade`: launch the 192.
    `partial_manipulation`: 3× ≤ ratio < 10× — stop for a decision.
    `richness_manipulation_collapses`: ratio < 3× — fail clearly.
    """
    if ratio >= 10.0:
        return "clears_decade"
    if ratio >= 3.0:
        return "partial_manipulation"
    return "richness_manipulation_collapses"


def _gate_job(spec: AdamSpec) -> dict:
    provenance.assert_current()
    stream = pl.build_stream(spec)
    model = pl.build_model(spec, stream)
    rng_train = paired_init(spec.seed)["data"]
    rec = train_task(
        model, stream.arrangements[0].points, stream.dichotomies[0],
        spec.train_config, rng_train, task_index=0,
    )
    return {
        "key": spec.key,
        "gamma_0": spec.gamma_0,
        "stream_id": spec.stream_id,
        "converged": rec.converged,
        "steps": rec.steps_taken,
        "final_loss": rec.final_loss,
        "weight_change": rec.weight_change,
        "dW_A": rec.weight_change["A"],
        "dW_B": rec.weight_change["B"],
    }


def _write_gate_report(jobs: list[dict], wall: float) -> dict:
    g10 = [j for j in jobs if j["gamma_0"] == 10.0]
    g1 = [j for j in jobs if j["gamma_0"] == 1.0]
    g10.sort(key=lambda j: j["stream_id"])
    g1.sort(key=lambda j: j["stream_id"])
    mean10 = sum(j["dW_A"] for j in g10) / len(g10)
    mean1 = sum(j["dW_A"] for j in g1) / len(g1)
    ratio = mean10 / mean1
    reading = gate_reading(ratio)
    mean_steps_10 = sum(j["steps"] for j in g10) / len(g10)
    mean_steps_1 = sum(j["steps"] for j in g1) / len(g1)
    report = {
        "generated_by": "scripts/run_adam.py --richness-gate",
        "precommit": "results/adam_precommit.json",
        "question": (
            "At unique n=8, frozen × s_r=0.5, after task 0, does mean "
            "‖ΔW‖/‖W‖ at γ=10 over γ=1 clear one decade under Adam?"
        ),
        "population": {
            "input_level": GATE_INPUT,
            "s_r": GATE_S_R,
            "unique_n": 8,
            "after": "task 0",
            "quantity": "weight_change A = ‖ΔW‖_F / ‖W(0)‖_F",
            "training_only": True,
            "not": "geometry. Not the 192-arm slice.",
        },
        "bar": "mean(γ=10) / mean(γ=1) ≥ 10",
        "bands": {
            "clears_decade": "ratio ≥ 10: launch the 192",
            "partial_manipulation": "3 ≤ ratio < 10: stop for a decision",
            "richness_manipulation_collapses": "ratio < 3: fail clearly",
        },
        "n_arms": len(jobs),
        "n_missed": sum(1 for j in jobs if not j["converged"]),
        "mean_dW_A_gamma10": mean10,
        "mean_dW_A_gamma1": mean1,
        "ratio": ratio,
        "mean_steps_gamma10": mean_steps_10,
        "mean_steps_gamma1": mean_steps_1,
        "step_ratio": mean_steps_1 / mean_steps_10,
        "reading": reading,
        "wall_seconds": wall,
        "arms": jobs,
        "identity": {
            "stream_10000_gamma10_dW_A": next(
                j["dW_A"] for j in g10 if j["stream_id"] == min(reserved_stream_ids())
            ),
            "one_arm_task0_dW_A": json.loads(ONE_ARM_REPORT.read_text())["task0_weight_change"]["A"],
            "must_match": "the one-arm task-0 weight_change A, same spec",
        },
        "will_not_do": [
            "Launch the 192 unless reading is clears_decade.",
            "Interpret γ-dependent claims if the decade is not cleared.",
            "Re-pin lr0 from this gate.",
            "Write into results/phase1/.",
        ],
    }
    GATE_REPORT.write_text(json.dumps(report, indent=2) + "\n")
    md = [
        "# Adam richness gate — frozen × s_r=0.5, unique n=8, task 0",
        "",
        f"Training-only. Mean ‖ΔW‖/‖W‖ (module A): γ=10 **{mean10:.4f}**, "
        f"γ=1 **{mean1:.4f}**, ratio **{ratio:.3f}** against a one-decade bar.",
        "",
        f"Reading: **`{reading}`**.",
        "",
        f"Mean steps: γ=10 {mean_steps_10:.1f}, γ=1 {mean_steps_1:.1f} "
        f"(step ratio γ=1/γ=10 = {mean_steps_1 / mean_steps_10:.2f}).",
        "",
        f"Missed target: {report['n_missed']} / {len(jobs)}. Wall {wall:.1f} s.",
        "",
        "The 192-arm slice launches only on `clears_decade`. "
        "`partial_manipulation` is a finding, not a failed check; "
        "do not launch 192. The arm that can still run is Hamming-only "
        "at γ=10 (`results/adam_hamming_precommit.json`).",
        "",
        "Source: `results/adam_richness_gate.json`.",
        "",
    ]
    (ROOT / "results" / "adam_richness_gate.md").write_text("\n".join(md))
    return report


def _path(spec: AdamSpec) -> Path:
    return OUT / (spec.key.replace(",", "__").replace("=", "-") + ".json")


def _norm(spec: AdamSpec) -> dict:
    return json.loads(json.dumps(asdict(spec)))


def _job(spec: AdamSpec) -> dict:
    provenance.assert_current()
    path = _path(spec)
    if path.exists():
        stored = json.loads(path.read_text()).get("spec")
        if stored == _norm(spec):
            return {"key": spec.key, "path": str(path), "skipped": True}
        sys.exit(f"refusing to overwrite {path.name}: stored spec does not match")
    rec = pl.run_arm(spec, verbose=True)
    path.write_text(json.dumps(rec))
    n_miss = sum(1 for t in rec["tasks"] if not t["converged"])
    task0 = rec["tasks"][0]
    return {
        "key": rec["key"], "path": str(path), "skipped": False,
        "usable": rec["usable"], "wall_seconds": rec["wall_seconds"],
        "n_evals": rec["n_evals"], "n_tasks_missed": n_miss,
        "stream_id": spec.stream_id, "input_level": spec.input_level,
        "readout_similarity": spec.readout_similarity,
        "tracked_stride": spec.tracked_stride,
        "optimizer": spec.optimizer, "lr0": spec.lr0,
        "steps_taken": [t["steps_taken"] for t in rec["tasks"]],
        "all_converged": bool(rec["usable"]),
        "task0_weight_change": task0["weight_change"],
        "task0_steps": task0["steps_taken"],
        "task0_final_loss": task0["final_loss"],
    }


def _write_one_arm_report(
    job: dict,
    wall: float,
    *,
    path: Path,
    generated_by: str,
    precommit: str,
    input_level: str,
    readout_similarity: float,
    question: str,
) -> None:
    steps = job.get("steps_taken") or []
    reached = bool(job.get("all_converged"))
    report = {
        "generated_by": generated_by,
        "precommit": precommit,
        "question": question,
        "spec": {
            "gamma_0": 10.0,
            "input_level": input_level,
            "feature_similarity": INPUT_S_F[input_level],
            "readout_similarity": readout_similarity,
            "tracked_stride": TRACKED_STRIDE,
            "n_evals": job.get("n_evals"),
            "a": 0.0,
            "N": 300,
            "lr0": _lr0(),
            "target_loss": 0.05,
            "stopping": "matched_loss",
            "lr_scaling": "quadratic",
            "optimizer": "adam",
            "loss": "mse",
            "arrangement_source": "stream_rng",
            "stream_id": 10000,
            "seed": 0,
            "P": 16,
            "T": 16,
            "population": "reserved arrangements",
        },
        "usable": job.get("usable"),
        "n_tasks": len(steps),
        "n_tasks_missed": job.get("n_tasks_missed"),
        "all_converged": reached,
        "steps_taken": steps,
        "steps_at_0": job.get("task0_steps"),
        "mean_steps": (sum(steps) / len(steps)) if steps else None,
        "max_steps": max(steps) if steps else None,
        "task0_weight_change": job.get("task0_weight_change"),
        "task0_final_loss": job.get("task0_final_loss"),
        "wall_seconds": job.get("wall_seconds"),
        "runner_wall_seconds": wall,
        "n_evals": job.get("n_evals"),
        "path": job.get("path"),
        "governs": False,
        "reached_target": reached,
        "miss_rule": (
            "Pinned lr0 from results/adam_precommit.json. Inherited 5 exploded "
            "and was re-pinned once. If this arm still misses: lr0_miss, stop. "
            "Do not re-pin a second time."
        ),
        "reading": (
            "One-arm smoke check. n=1 licenses nothing about the Hamming "
            "dose, the input level, or the richness gate (gate is unique n=8)."
        ),
        "will_not_do": [
            "Re-pin lr0 a second time.",
            "Interpret richness_manipulation_collapses from n=1.",
            "Read hamming_reproduces / hamming_fails / partial from n=1.",
            "Write into results/phase1/.",
            "Launch the 192.",
        ],
        "cannot_speak_to": ["γ", "finding 1", "channel reorganisation"],
    }
    path.write_text(json.dumps(report, indent=2) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--one-arm", action="store_true")
    ap.add_argument("--one-arm-hamming", action="store_true",
                    help="γ=10 drift × s_r=0.5 × stream 10000; Hamming-only smoke check")
    ap.add_argument("--hamming-only", action="store_true",
                    help="96 arms at γ=10; refuses the 192")
    ap.add_argument("--richness-gate", action="store_true",
                    help="16 training-only arms: frozen × s_r=0.5 × both γ × 8 ids")
    ap.add_argument("--cap", type=int, default=None)
    args = ap.parse_args()
    flags = [args.one_arm, args.one_arm_hamming, args.hamming_only, args.richness_gate]
    if sum(bool(x) for x in flags) > 1:
        sys.exit("one flag at a time")
    _pre()
    if args.richness_gate:
        if not ONE_ARM_REPORT.exists() or not json.loads(ONE_ARM_REPORT.read_text()).get("reached_target"):
            sys.exit("one-arm first")
        specs = gate_arms()
        print(
            f"adam richness gate: {len(specs)} training-only arm(s), "
            f"frozen × s_r={GATE_S_R}, γ ∈ {{1,10}}, unique n=8, "
            f"lr0={specs[0].lr0}, workers={n_workers(args.cap)}",
            flush=True,
        )
        t0 = time.time()
        jobs = pmap(_gate_job, specs, cap=args.cap)
        wall = time.time() - t0
        report = _write_gate_report(jobs, wall)
        print(json.dumps({
            "n": len(jobs),
            "wall_seconds": wall,
            "ratio": report["ratio"],
            "reading": report["reading"],
            "mean_dW_A_gamma10": report["mean_dW_A_gamma10"],
            "mean_dW_A_gamma1": report["mean_dW_A_gamma1"],
            "n_missed": report["n_missed"],
        }, indent=2))
        return
    if args.hamming_only:
        if not HAMMING_ONE_ARM_REPORT.exists():
            sys.exit("hamming one-arm first: run scripts/run_adam.py --one-arm-hamming")
        one = json.loads(HAMMING_ONE_ARM_REPORT.read_text())
        if one.get("reached_target") is not True:
            sys.exit("hamming one-arm missed target; lr0_miss, stop")
        if one.get("spec", {}).get("input_level") == GATE_INPUT and one.get("spec", {}).get("readout_similarity") == GATE_S_R:
            sys.exit("hamming one-arm must not be the richness-gate cell frozen × s_r=0.5")
        specs = hamming_only_arms()
    elif args.one_arm_hamming:
        specs = [one_arm_hamming()]
    elif args.one_arm:
        specs = [one_arm()]
    else:
        if not ONE_ARM_REPORT.exists():
            sys.exit("one-arm first: run scripts/run_adam.py --one-arm")
        one = json.loads(ONE_ARM_REPORT.read_text())
        if one.get("reached_target") is not True:
            sys.exit("one-arm missed target; re-pin by docs/22 before the slice")
        if not GATE_REPORT.exists():
            sys.exit("richness gate first: run scripts/run_adam.py --richness-gate")
        gate = json.loads(GATE_REPORT.read_text())
        if gate.get("reading") != "clears_decade":
            sys.exit(
                f"richness gate reading is {gate.get('reading')!r}; "
                "do not launch the 192"
            )
        specs = arms()
    OUT.mkdir(parents=True, exist_ok=True)
    workers = n_workers(args.cap)
    print(
        f"adam: {len(specs)} arm(s), unique n={len(reserved_stream_ids())}, "
        f"stride={TRACKED_STRIDE}, stream_rng, quadratic, matched_loss, "
        f"Adam, MSE, lr0={specs[0].lr0}, workers={workers}, dir={OUT}",
        flush=True,
    )
    t0 = time.time()
    results = pmap(_job, specs, cap=args.cap)
    wall = time.time() - t0
    if args.one_arm:
        _write_one_arm_report(
            results[0], wall,
            path=ONE_ARM_REPORT,
            generated_by="scripts/run_adam.py --one-arm",
            precommit="results/adam_precommit.json",
            input_level=GATE_INPUT,
            readout_similarity=GATE_S_R,
            question=(
                "Does γ=10 frozen × s_r=0.5 on the first reserved arrangement "
                "reach target_loss=0.05 under Adam at the pinned lr0?"
            ),
        )
    if args.one_arm_hamming:
        _write_one_arm_report(
            results[0], wall,
            path=HAMMING_ONE_ARM_REPORT,
            generated_by="scripts/run_adam.py --one-arm-hamming",
            precommit="results/adam_hamming_precommit.json",
            input_level=HAMMING_ONE_ARM_INPUT,
            readout_similarity=HAMMING_ONE_ARM_S_R,
            question=(
                "Does γ=10 drift × s_r=0.5 on the first reserved arrangement "
                "reach target_loss=0.05 under Adam at the pinned lr0?"
            ),
        )
    print(json.dumps({
        "n": len(results),
        "n_workers": workers,
        "wall_seconds": wall,
        "usable": sum(1 for r in results if r.get("usable")),
        "missed": sum(1 for r in results if r.get("usable") is False),
        "skipped": sum(1 for r in results if r.get("skipped")),
        "one_arm": bool(args.one_arm),
        "one_arm_hamming": bool(args.one_arm_hamming),
        "hamming_only": bool(args.hamming_only),
        "first": results[0] if results else None,
    }, indent=2))


if __name__ == "__main__":
    main()
