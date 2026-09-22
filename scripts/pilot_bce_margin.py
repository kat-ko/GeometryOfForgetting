"""BCE margin-matching pilot on the Hamming one-arm cells (`docs/21`).

Tolerance is the measured range of MSE-stop mean margins on four reserved
streams × two γ (`results/mse_stop_margin_four_streams.json`). One BCE
target; if both γ cannot land in that band, stop — do not pin per-γ.

    python scripts/pilot_bce_margin.py --one-arm
    python scripts/pilot_bce_margin.py --bisect

`--one-arm` is γ=10, frozen, s_r=0.5, first reserved id, at a probe BCE
target (default 0.35). Times the cell. Governs nothing.
`--bisect` requires the four-stream MSE file and the one-arm cost file.
Never results/phase1/. No ce_precommit.json.
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

from _par import pmap  # noqa: E402

from measure_mse_stop_margin import (  # noqa: E402
    _spec_at_gamma, _stored_path, readout_margins,
)
from src import pipeline as pl  # noqa: E402
from src.models import paired_init  # noqa: E402
from src.train.loop import TrainConfig, flatten_task, train_task  # noqa: E402

FOUR = ROOT / "results" / "mse_stop_margin_four_streams.json"
OUT_ONE = ROOT / "results" / "bce_margin_one_arm.json"
OUT_BISECT = ROOT / "results" / "bce_margin_bisect.json"
PROBE_LOSS = 0.35
N_BISECT = 8


def _train_cfg(target: float) -> TrainConfig:
    base = _spec_at_gamma(10.0)
    return TrainConfig(
        steps_per_task=base.steps_per_task,
        stopping="matched_loss",
        target_loss=target,
        record_every=base.record_every,
        loss="bce",
    )


def run_cell(gamma: float, target_loss: float) -> dict:
    spec = _spec_at_gamma(gamma)
    streams = paired_init(spec.seed)
    rng_train = streams["data"]
    stream = pl.build_stream(spec)
    model = pl.build_model(spec, stream)
    cfg = _train_cfg(target_loss)
    tasks = []
    for t in range(spec.T):
        rec = train_task(
            model, stream.arrangements[t].points, stream.dichotomies[t],
            cfg, rng_train, task_index=t,
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
    means = [t["margin_mean"] for t in tasks]
    return {
        "spec": json.loads(json.dumps(asdict(spec))),
        "target_loss": target_loss,
        "loss": "bce",
        "all_converged": all(t["converged"] for t in tasks),
        "mean_margin_over_tasks": float(np.mean(means)),
        "mean_p05_over_tasks": float(np.mean([t["margin_p05"] for t in tasks])),
        "mean_p25_over_tasks": float(np.mean([t["margin_p25"] for t in tasks])),
        "task0_margin_mean": tasks[0]["margin_mean"],
        "task0_margin_p05": tasks[0]["margin_p05"],
        "task0_margin_p25": tasks[0]["margin_p25"],
        "tasks": tasks,
        "mse_identity_path": str(_stored_path(spec).relative_to(ROOT)),
    }


def _band() -> dict:
    if not FOUR.exists():
        sys.exit(f"missing {FOUR.name}; run measure_mse_stop_margin.py --four-streams")
    four = json.loads(FOUR.read_text())
    s = four["summary"]
    if not s.get("all_steps_match"):
        sys.exit("four-stream identity failed; not a tolerance")
    return {
        "lo": float(s["min"]),
        "hi": float(s["max"]),
        "grand_mean": float(s["grand_mean"]),
        "spread": float(s["spread"]),
        "n_cells": int(s["n_cells"]),
        "source": str(FOUR.relative_to(ROOT)),
    }


def _bisect_job(item: tuple[float, float]) -> dict:
    gamma, target_loss = item
    return run_cell(gamma, target_loss)


def _in_band(m: float, band: dict) -> bool:
    return band["lo"] <= m <= band["hi"]


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--one-arm", action="store_true")
    g.add_argument("--bisect", action="store_true")
    ap.add_argument("--target-loss", type=float, default=PROBE_LOSS)
    args = ap.parse_args()

    if args.one_arm:
        print(f"bce one-arm γ=10 target={args.target_loss}", flush=True)
        t0 = time.time()
        rec = run_cell(10.0, args.target_loss)
        payload = {
            "generated_by": "scripts/pilot_bce_margin.py --one-arm",
            "question": "Does BCE at lr0=5 reach a probe target on the Hamming "
                        "one-arm cell, and where does mean margin sit?",
            "probe_target_loss": args.target_loss,
            "wall_seconds": time.time() - t0,
            "cell": rec,
            "will_not_do": [
                "Treat this probe loss as the pin.",
                "Write results/ce_precommit.json from this file.",
                "Inform lr0 from n=1.",
                "Run geometry.",
            ],
        }
        OUT_ONE.write_text(json.dumps(payload, indent=2))
        print(json.dumps({
            "path": str(OUT_ONE.relative_to(ROOT)),
            "wall_seconds": payload["wall_seconds"],
            "converged": rec["all_converged"],
            "mean_margin": rec["mean_margin_over_tasks"],
            "mean_p05": rec["mean_p05_over_tasks"],
            "mean_p25": rec["mean_p25_over_tasks"],
            "steps_task0": rec["tasks"][0]["steps_taken"],
        }, indent=2))
        return

    band = _band()
    if not OUT_ONE.exists():
        sys.exit("run --one-arm first (AGENTS §5)")
    one = json.loads(OUT_ONE.read_text())
    probe_L = float(one["probe_target_loss"])
    probe_m = float(one["cell"]["mean_margin_over_tasks"])
    print(f"bce bisect band=[{band['lo']:.4f}, {band['hi']:.4f}] "
          f"spread={band['spread']:.4f} from {band['source']}", flush=True)
    print(f"  one-arm L={probe_L} m={probe_m:.4f} p05={one['cell']['mean_p05_over_tasks']:.4f}",
          flush=True)

    # Lower BCE target → longer training → higher mean margin (checked at the
    # one-arm: L=0.35 gave m=1.31, above the band). Bracket from that.
    if probe_m > band["hi"]:
        lo_L, hi_L = probe_L, float(np.log(2.0) - 0.01)
    elif probe_m < band["lo"]:
        lo_L, hi_L = 0.05, probe_L
    else:
        lo_L, hi_L = 0.08, 0.55
    history = []
    t0 = time.time()
    pinned = None
    for k in range(N_BISECT):
        L = 0.5 * (lo_L + hi_L)
        print(f"  round {k+1}/{N_BISECT} L={L:.4f}", flush=True)
        g10, g1 = pmap(_bisect_job, ((10.0, L), (1.0, L)), cap=2)
        row = {
            "round": k + 1,
            "target_loss": L,
            "gamma_10": {
                "mean": g10["mean_margin_over_tasks"],
                "p05": g10["mean_p05_over_tasks"],
                "p25": g10["mean_p25_over_tasks"],
                "converged": g10["all_converged"],
            },
            "gamma_1": {
                "mean": g1["mean_margin_over_tasks"],
                "p05": g1["mean_p05_over_tasks"],
                "p25": g1["mean_p25_over_tasks"],
                "converged": g1["all_converged"],
            },
        }
        history.append(row)
        if not (g10["all_converged"] and g1["all_converged"]):
            print("  miss: lr0=5 did not reach target; stop. Re-pin is the signed rule, not this search.",
                  flush=True)
            break
        hit10 = _in_band(g10["mean_margin_over_tasks"], band)
        hit1 = _in_band(g1["mean_margin_over_tasks"], band)
        row["both_in_band"] = hit10 and hit1
        print(
            f"  m10={g10['mean_margin_over_tasks']:.4f} "
            f"m1={g1['mean_margin_over_tasks']:.4f} "
            f"in_band={hit10}/{hit1}",
            flush=True,
        )
        if hit10 and hit1:
            pinned = L
            break
        # Steer on the midpoint of the two γ means vs the MSE grand mean.
        mid = 0.5 * (g10["mean_margin_over_tasks"] + g1["mean_margin_over_tasks"])
        if mid > band["grand_mean"]:
            lo_L = L  # too deep; raise the BCE target
        else:
            hi_L = L
    reading = (
        "shared_operating_point" if pinned is not None
        else "no_shared_operating_point"
    )
    payload = {
        "generated_by": "scripts/pilot_bce_margin.py --bisect",
        "question": "Does one BCE target place both γ in the MSE-stop margin band?",
        "band": band,
        "one_arm_probe": str(OUT_ONE.relative_to(ROOT)),
        "one_arm_wall_seconds": one["wall_seconds"],
        "wall_seconds": time.time() - t0,
        "reading": reading,
        "pinned_target_loss": pinned,
        "history": history,
        "will_not_do": [
            "Pin per-γ if the shared target fails.",
            "Write results/ce_precommit.json unless reading is shared_operating_point.",
            "Re-pin lr0 inside this search.",
            "Run geometry.",
            "Treat a distribution mismatch at matched mean as a re-pin.",
        ],
        "if_no_shared_point": (
            "The CE arm cannot test stream vs learner. The failure is a "
            "measured limitation: the two objectives have no shared operating "
            "point. Report and stop. Do not run the slice."
        ),
    }
    OUT_BISECT.write_text(json.dumps(payload, indent=2))
    print(json.dumps({
        "path": str(OUT_BISECT.relative_to(ROOT)),
        "reading": reading,
        "pinned_target_loss": pinned,
        "wall_seconds": payload["wall_seconds"],
        "last": history[-1] if history else None,
    }, indent=2))


if __name__ == "__main__":
    main()
