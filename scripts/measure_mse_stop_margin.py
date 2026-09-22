"""Recover the MSE-stop readout margin on reserved Hamming streams.

Hamming records do not store weights or f. Training is deterministic from
(seed, spec), so a training-only replay is the checkpoint pass. Identity:
steps_taken must match the stored Hamming JSON.

    python scripts/measure_mse_stop_margin.py --one-arm
    python scripts/measure_mse_stop_margin.py --four-streams

`--one-arm` is γ ∈ {1, 10} on the Hamming one-arm cell (frozen, s_r=0.5,
first reserved id). `--four-streams` extends that to the first four reserved
ids so the MSE spread is a measured tolerance, not two points. No geometry.
Never results/phase1/.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from _par import pin_threads  # noqa: E402

pin_threads()

from _par import n_workers, pmap  # noqa: E402
from run_hamming import HammingSpec, one_arm  # noqa: E402
from src import pipeline as pl  # noqa: E402
from src.models import paired_init  # noqa: E402
from src.reservations import reserved_stream_ids  # noqa: E402
from src.train.loop import flatten_task, train_task  # noqa: E402

OUT_ONE = ROOT / "results" / "mse_stop_margin_one_arm.json"
OUT_FOUR = ROOT / "results" / "mse_stop_margin_four_streams.json"
HAMMING = ROOT / "results" / "hamming"
N_FOUR = 4


def readout_margins(f: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Current-task readout margin. y in {±1}; f is the scalar network output."""
    yf = np.asarray(y, dtype=np.float64).ravel() * np.asarray(f, dtype=np.float64).ravel()
    err2 = (np.asarray(f, dtype=np.float64).ravel() - np.asarray(y, dtype=np.float64).ravel()) ** 2
    return {
        "mean": float(np.mean(yf)),
        "p05": float(np.percentile(yf, 5)),
        "p25": float(np.percentile(yf, 25)),
        "min": float(np.min(yf)),
        "post_step_mse": float(0.5 * np.mean(err2)),
        "sign_accuracy": float(np.mean(np.sign(f).ravel() == np.sign(y).ravel())),
    }


def _spec_at(gamma: float, stream_id: int) -> HammingSpec:
    base = one_arm()
    return HammingSpec(
        gamma_0=gamma, a=0.0, condition=base.condition, seed=base.seed,
        stream_id=stream_id, arrangement_source=base.arrangement_source,
        lr_scaling=base.lr_scaling, input_level=base.input_level,
        feature_similarity=base.feature_similarity,
        readout_similarity=base.readout_similarity,
        tracked_stride=base.tracked_stride,
    )


def _spec_at_gamma(gamma: float) -> HammingSpec:
    return _spec_at(gamma, one_arm().stream_id)


def four_stream_ids() -> tuple[int, ...]:
    ids = tuple(sorted(reserved_stream_ids())[:N_FOUR])
    if one_arm().stream_id not in ids:
        raise RuntimeError("one-arm stream is not among the first four reserved ids")
    return ids


def _stored_path(spec: HammingSpec) -> Path:
    return HAMMING / (spec.key.replace(",", "__").replace("=", "-") + ".json")


def replay(spec: HammingSpec) -> dict:
    streams = paired_init(spec.seed)
    rng_train = streams["data"]
    stream = pl.build_stream(spec)
    model = pl.build_model(spec, stream)
    tasks = []
    for t in range(spec.T):
        rec = train_task(
            model, stream.arrangements[t].points, stream.dichotomies[t],
            spec.train_config, rng_train, task_index=t,
        )
        X, y = flatten_task(stream.arrangements[t].points, stream.dichotomies[t])
        f = model.forward(X)
        if f.ndim == 2:
            f = f[:, 0]
        m = readout_margins(f, y)
        tasks.append({
            "task": t,
            "steps_taken": rec.steps_taken,
            "final_loss": rec.final_loss,
            "converged": rec.converged,
            "train_accuracy": rec.train_accuracy,
            **{f"margin_{k}": v for k, v in m.items()},
        })
    stored = _stored_path(spec)
    identity = {"stored_path": str(stored.relative_to(ROOT)), "present": stored.exists()}
    if stored.exists():
        prev = json.loads(stored.read_text())["tasks"]
        replay_steps = [t["steps_taken"] for t in tasks]
        stored_steps = [t["steps_taken"] for t in prev]
        identity["steps_match"] = replay_steps == stored_steps
        identity["stored_steps"] = stored_steps
        identity["replay_steps"] = replay_steps
        identity["stored_final_loss"] = [t["final_loss"] for t in prev]
    means = [t["margin_mean"] for t in tasks]
    return {
        "spec": json.loads(json.dumps(asdict(spec))),
        "key": spec.key,
        "identity": identity,
        "mean_margin_over_tasks": float(np.mean(means)),
        "p05_of_task_means": float(np.percentile(means, 5)),
        "task0_margin_mean": tasks[0]["margin_mean"],
        "task0_margin_p05": tasks[0]["margin_p05"],
        "task0_margin_p25": tasks[0]["margin_p25"],
        "mean_p05_over_tasks": float(np.mean([t["margin_p05"] for t in tasks])),
        "mean_p25_over_tasks": float(np.mean([t["margin_p25"] for t in tasks])),
        "tasks": tasks,
    }


def _summarise(rows: list[dict]) -> dict:
    means = [c["mean_margin_over_tasks"] for c in rows]
    return {
        "n_cells": len(rows),
        "grand_mean": float(np.mean(means)),
        "spread": float(np.max(means) - np.min(means)),
        "min": float(np.min(means)),
        "max": float(np.max(means)),
        "all_steps_match": all(c["identity"].get("steps_match") is True for c in rows),
    }


def _job(spec: HammingSpec) -> dict:
    print(f"replay {spec.key}", flush=True)
    rec = replay(spec)
    print(
        f"  mean m={rec['mean_margin_over_tasks']:.4f} "
        f"task0={rec['task0_margin_mean']:.4f} "
        f"steps_match={rec['identity'].get('steps_match')}",
        flush=True,
    )
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--one-arm", action="store_true")
    g.add_argument("--four-streams", action="store_true")
    ap.add_argument("--cap", type=int, default=None)
    args = ap.parse_args()
    if args.four_streams:
        specs = [_spec_at(g, sid) for g in (10.0, 1.0) for sid in four_stream_ids()]
        out = OUT_FOUR
        generated = "scripts/measure_mse_stop_margin.py --four-streams"
        question = (
            "Where does MSE 0.05 sit in mean readout margin on four reserved "
            "streams × γ ∈ {1, 10}, frozen × s_r=0.5? The range of those "
            "eight cell means is the BCE-pilot tolerance."
        )
        pop = (f"reserved Hamming, frozen × s_r=0.5, stream_ids={list(four_stream_ids())}, "
               "seed=0, γ ∈ {1, 10}")
        will_not = [
            "Freeze m* if any identity check fails.",
            "Choose a tolerance other than the measured range of cell means.",
            "Write results/ce_precommit.json from this file.",
            "Inform lr0 from this replay.",
            "Run geometry.",
        ]
    else:
        specs = [_spec_at_gamma(10.0), _spec_at_gamma(1.0)]
        out = OUT_ONE
        generated = "scripts/measure_mse_stop_margin.py --one-arm"
        question = (
            "Where does MSE target_loss=0.05 sit in current-task readout "
            "margin m=mean(y f) on the Hamming one-arm cell at γ=10 and γ=1?"
        )
        pop = "reserved Hamming, frozen × s_r=0.5, stream_id of one_arm, seed=0"
        will_not = [
            "Freeze m* from n=2.",
            "Convert 0.05 MSE to a margin by assuming homogeneous f.",
            "Write results/ce_precommit.json from this file.",
            "Inform lr0 from this replay.",
            "Run geometry.",
        ]
    t0 = time.time()
    workers = n_workers(args.cap if args.cap is not None else min(len(specs), 8))
    print(f"mse-stop-margin: {len(specs)} cell(s), workers={workers}", flush=True)
    rows = pmap(_job, specs, cap=args.cap if args.cap is not None else min(len(specs), 8))
    summary = _summarise(rows)
    payload = {
        "generated_by": generated,
        "question": question,
        "definition": {
            "margin": "mean_b [y_b f(x_b)] on the current-task batch at the "
                      "state train_task returns",
            "y": "±1",
            "p05_p25": "percentiles of the same pointwise y f; reported, not the pin",
            "naive_homogeneous": "if ½(f−y)²=0.05 and f=y m with m<1, "
                                 "m=1−sqrt(0.1)≈0.6838 — an assumption, not "
                                 "this measurement. Measured mean is ~0.85; "
                                 "the analogy would have stopped BCE ~20% "
                                 "shallower in margin.",
            "tolerance": "range of mean_margin_over_tasks across these cells",
        },
        "population": pop,
        "n_cells": len(rows),
        "wall_seconds": time.time() - t0,
        "summary": summary,
        "cells": rows,
        "will_not_do": will_not,
    }
    out.write_text(json.dumps(payload, indent=2))
    print(json.dumps({
        "path": str(out.relative_to(ROOT)),
        "wall_seconds": payload["wall_seconds"],
        "summary": summary,
        "cells": [
            {
                "gamma": c["spec"]["gamma_0"],
                "stream_id": c["spec"]["stream_id"],
                "mean_margin_over_tasks": c["mean_margin_over_tasks"],
                "mean_p05_over_tasks": c["mean_p05_over_tasks"],
                "mean_p25_over_tasks": c["mean_p25_over_tasks"],
                "task0_margin_mean": c["task0_margin_mean"],
                "steps_match": c["identity"].get("steps_match"),
            }
            for c in rows
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
