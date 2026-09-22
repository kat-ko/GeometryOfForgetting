"""Does matched-loss stopping survive a multi-output readout?

`docs/16` §Amendments A7. Decides granularity (multi-output, worst-of-K
stopping) versus breadth (|S|) before the scope pre-commit. One task, not a
stream. Writes `results/scope_loss_scale.json`.

    python scripts/check_scope_loss_scale.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from _par import pin_threads

pin_threads()

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.manifolds.dichotomies import sample_balanced  # noqa: E402
from src.manifolds.generator import make_arrangement  # noqa: E402
from src.models import MODULES, ScalingConfig, TwoModuleNet, paired_init  # noqa: E402

P, D_AMB, M, N = 16, 150, 150, 300
D_INT, R = 4, 1.0
LR0, GAMMA = 5.0, 1.0
TARGET = 0.05
MAX_STEPS = 20_000
KS = (1, 2, 4)
SEEDS = (0, 1, 2)
OUT = ROOT / "results" / "scope_loss_scale.json"


def _data(seed: int, K: int):
    rng = paired_init(seed)["data"]
    arr = make_arrangement(P, D_AMB, D_INT, R, M, rng, rho_C=0.0, rho_A=0.0)
    X = arr.points.reshape(P * M, D_AMB)
    ys = np.stack([sample_balanced(P, rng) for _ in range(K)], axis=0)  # (K, P)
    if K == 1:
        y = np.repeat(ys[0], M)
    else:
        y = np.repeat(ys.T, M, axis=0)  # (P*M, K)
    return X, y.astype(np.float64)


def _per_output_mse(model, X, y, K: int) -> list[float]:
    pred = model.forward(X)
    err2 = (pred - y) ** 2
    if K == 1:
        return [float(0.5 * np.mean(err2))]
    return [float(0.5 * np.mean(err2[:, k])) for k in range(K)]


def _train(seed: int, K: int, stop: str) -> dict:
    X, y = _data(seed, K)
    cfg = {m: ScalingConfig(N=N, d=D_AMB, gamma_0=GAMMA, lr0=LR0) for m in MODULES}
    model = TwoModuleNet.init(cfg, paired_init(seed)["shape"], n_outputs=K)
    init_loss = float(0.5 * np.mean((model.forward(X) - y) ** 2))
    loss = init_loss
    step = 0
    for step in range(1, MAX_STEPS + 1):
        loss = model.sgd_step(X, y)
        per = _per_output_mse(model, X, y, K)
        hit = (loss <= TARGET) if stop == "mean" else (max(per) <= TARGET)
        if hit:
            break
    per_output = _per_output_mse(model, X, y, K)
    return {
        "seed": seed, "K": K, "stop": stop,
        "init_loss": init_loss, "final_loss": float(loss),
        "steps": step,
        "hit_mean_target": bool(float(np.mean(per_output)) <= TARGET),
        "hit_worst_target": bool(max(per_output) <= TARGET),
        "per_output_mse": per_output,
        "per_output_spread": float(np.max(per_output) - np.min(per_output)),
        "max_per_output_mse": float(np.max(per_output)),
    }


def _summarise(runs, stop: str) -> dict:
    by_k = {}
    for k in KS:
        rs = [r for r in runs if r["K"] == k and r["stop"] == stop]
        by_k[str(k)] = {
            "init_loss_mean": float(np.mean([r["init_loss"] for r in rs])),
            "steps_mean": float(np.mean([r["steps"] for r in rs])),
            "hit_mean_rate": float(np.mean([r["hit_mean_target"] for r in rs])),
            "hit_worst_rate": float(np.mean([r["hit_worst_target"] for r in rs])),
            "spread_mean": float(np.mean([r["per_output_spread"] for r in rs])),
            "max_per_output_mean": float(np.mean([r["max_per_output_mse"] for r in rs])),
        }
    return by_k


def main() -> None:
    runs = [_train(s, k, stop) for stop in ("mean", "worst") for k in KS for s in SEEDS]
    by_mean = _summarise(runs, "mean")
    by_worst = _summarise(runs, "worst")
    init_ok = all(abs(by_mean[str(k)]["init_loss_mean"] - 0.5) < 1e-12 for k in KS)
    mean_leaves_an_output_above = any(
        by_mean[str(k)]["max_per_output_mean"] > TARGET and by_mean[str(k)]["hit_mean_rate"] > 0
        for k in KS if k > 1)
    worst_holds = all(by_worst[str(k)]["hit_worst_rate"] == 1.0 for k in KS)
    out = {
        "generated_by": "scripts/check_scope_loss_scale.py",
        "question": "does matched-loss stopping at target_loss=0.05 survive multi-output?",
        "loss": "0.5 * mean over batch and outputs of (f-y)^2; y=±1",
        "target_loss": TARGET, "lr0": LR0, "gamma_0": GAMMA, "N": N,
        "K": list(KS), "seeds": list(SEEDS), "max_steps": MAX_STEPS,
        "by_K_mean_stop": by_mean,
        "by_K_worst_stop": by_worst,
        "init_loss_independent_of_K": init_ok,
        "mean_hit_with_an_output_above_target": mean_leaves_an_output_above,
        "worst_of_K_holds_every_output_to_target": worst_holds,
        "confound_controlled_cleanly": bool(init_ok and worst_holds),
        "decision": (
            "granularity (multi-output) is licensed with worst-of-K stopping: "
            "halt when max_k ½ mean_b (f_k − y_k)² ≤ 0.05. Mean-reduced stopping "
            "leaves a graded residual (the informative quantity). |S| is class-count "
            "as scope, which Line 2 records as unspecified; do not fall back to it."
            if init_ok and worst_holds else
            "worst-of-K did not hold every output to target; do not license granularity yet."
        ),
        "will_not_do": [
            "Fall back to breadth |S|. That repeats the Line 2 criticism of class-count-as-scope."
        ],
        "runs": runs,
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps({
        "by_K_mean_stop": by_mean,
        "by_K_worst_stop": by_worst,
        "confound_controlled_cleanly": out["confound_controlled_cleanly"],
        "decision": out["decision"],
    }, indent=2))


if __name__ == "__main__":
    main()
