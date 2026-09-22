"""Unique n=8 replicate of the A24/A25 loss-vs-width contrast.

Pre-commit: `results/mup_loss_width_seeds_precommit.json` (written first).

    python scripts/run_mup_loss_width_seeds.py

Does not lift A6. Does not sign Verification 2.
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

from src.models import MODULES, ScalingConfig, TwoModuleNet, paired_init  # noqa: E402

PRE = ROOT / "results" / "mup_loss_width_seeds_precommit.json"
OUT = ROOT / "results" / "mup_loss_width_seeds.json"


def _cfg(N: int, parameterization: str, lr0: float, d: int, gamma_0: float = 1.0):
    return {
        m: ScalingConfig(
            N=N, d=d, gamma_0=gamma_0, parameterization=parameterization, lr0=lr0,
        )
        for m in MODULES
    }


def _model(N, parameterization, lr0, d, seed, n_hidden_layers, gamma_0: float = 1.0):
    return TwoModuleNet.init(
        _cfg(N, parameterization, lr0, d, gamma_0),
        paired_init(seed)["shape"],
        n_hidden_layers=n_hidden_layers,
    )


def _batch(d: int, B: int, data_seed: int):
    rng = np.random.default_rng(data_seed)
    X = rng.standard_normal((B, d))
    y = np.sign(rng.standard_normal(B))
    y[y == 0] = 1.0
    return X, y


def _finite(x) -> bool:
    return bool(np.isfinite(x))


def run_cell(N, parameterization, lr0, d, seed, X, y, n_hidden, module, K, gamma_0=1.0):
    mdl = _model(N, parameterization, lr0, d, seed, n_hidden, gamma_0)
    kw = {"diagnostic": True} if n_hidden == 2 else {}
    z_idx = 1 if n_hidden == 2 else 0
    z0 = mdl.preactivations(X, module)[z_idx].copy()
    loss = float("nan")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for _ in range(K):
            loss = mdl.sgd_step(X, y, **kw)
            if not _finite(loss):
                loss = float("inf")
                break
    rec = {
        "loss": float(loss) if _finite(loss) else float("inf"),
        "weight_change_W": mdl.weight_change(module) if _finite(loss) else float("nan"),
    }
    if n_hidden == 2:
        rec["weight_change_W2"] = (
            mdl.weight_change_W2(module) if _finite(loss) else float("nan")
        )
    rec["rms_delta_z"] = (
        float(np.sqrt(np.mean(np.square(mdl.preactivations(X, module)[z_idx] - z0))))
        if _finite(loss) else float("nan")
    )
    return rec


def contrast(cells: dict, widths: list[int]) -> dict:
    n64, n300 = str(widths[0]), str(widths[1])
    loss = {
        p: {n64: cells[p][n64]["loss"], n300: cells[p][n300]["loss"]}
        for p in ("mup", "ntp")
    }
    ratios = {}
    for p in ("mup", "ntp"):
        a, b = loss[p][n64], loss[p][n300]
        ratios[p] = (b / a) if _finite(a) and a != 0 else float("nan")
    mup_ok = _finite(loss["mup"][n64]) and _finite(loss["mup"][n300]) and loss["mup"][n300] <= loss["mup"][n64]
    ntp_ok = _finite(loss["ntp"][n64]) and _finite(loss["ntp"][n300]) and loss["ntp"][n300] > loss["ntp"][n64]
    return {
        "loss": loss,
        "ratio_N300_over_N64": ratios,
        "mup_does_not_increase": mup_ok,
        "ntp_increases": ntp_ok,
        "present": bool(mup_ok and ntp_ok),
        "cells": cells,
    }


def seed0_matches(pre: dict, l2: dict, l3: dict) -> bool:
    ref = pre["seed0_must_match_a24"]
    pairs = [
        (l2["loss"]["mup"]["64"], ref["L2_mup_64"]),
        (l2["loss"]["mup"]["300"], ref["L2_mup_300"]),
        (l2["loss"]["ntp"]["300"], ref["L2_ntp_300"]),
        (l3["loss"]["mup"]["64"], ref["L3_mup_64"]),
        (l3["loss"]["mup"]["300"], ref["L3_mup_300"]),
        (l3["loss"]["ntp"]["300"], ref["L3_ntp_300"]),
    ]
    return all(a == b for a, b in pairs)


def apply_reading(n_l2: int, n_l3: int) -> str:
    if n_l2 < 8:
        return "instrument_blind"
    if n_l3 < 8:
        return "l3_problem"
    return "all_supported"


def summarize_ratios(rows: list[dict], depth: str) -> dict:
    mup = [r[depth]["ratio_N300_over_N64"]["mup"] for r in rows]
    ntp = [r[depth]["ratio_N300_over_N64"]["ntp"] for r in rows]
    def _stats(xs):
        arr = np.array(xs, dtype=np.float64)
        return {
            "mean": float(arr.mean()),
            "min": float(arr.min()),
            "max": float(arr.max()),
        }
    return {"mup": _stats(mup), "ntp": _stats(ntp)}


def main() -> None:
    pre = json.loads(PRE.read_text())
    assert pre["written_before_running"] is True
    held = pre["held_fixed"]
    seeds = list(held["seeds"])
    widths = list(held["widths"])
    d, lr0, K, module = held["d"], held["named_lr0"], held["K"], held["module"]
    rows = []
    for seed in seeds:
        data_seed = seed + 1
        X, y = _batch(d, held["batch"], data_seed)
        per_depth = {}
        for depth, n_hidden in (("L2", 1), ("L3", 2)):
            cells = {}
            for parameterization in ("mup", "ntp"):
                cells[parameterization] = {}
                for N in widths:
                    cells[parameterization][str(N)] = run_cell(
                        N, parameterization, lr0, d, seed, X, y, n_hidden, module, K,
                    )
            per_depth[depth] = contrast(cells, widths)
        rows.append({
            "seed": seed,
            "data_seed": data_seed,
            "L2": per_depth["L2"],
            "L3": per_depth["L3"],
        })

    seed0 = next(r for r in rows if r["seed"] == 0)
    if not seed0_matches(pre, seed0["L2"], seed0["L3"]):
        out = {
            "generated_by": "scripts/run_mup_loss_width_seeds.py",
            "precommit": str(PRE.relative_to(ROOT)),
            "governs_stream_arms": False,
            "signed_finding": False,
            "applied_reading": "seed0_mismatch",
            "reading": pre["readings_committed_before_seeing_numbers"]["seed0_mismatch"],
            "per_seed": rows,
        }
        OUT.write_text(json.dumps(out, indent=2))
        print(json.dumps({"applied_reading": "seed0_mismatch"}, indent=2))
        return

    n_l2 = sum(1 for r in rows if r["L2"]["present"])
    n_l3 = sum(1 for r in rows if r["L3"]["present"])
    key = apply_reading(n_l2, n_l3)
    out = {
        "generated_by": "scripts/run_mup_loss_width_seeds.py",
        "precommit": str(PRE.relative_to(ROOT)),
        "governs_stream_arms": False,
        "signed_finding": False,
        "n_L2_present": n_l2,
        "n_L3_present": n_l3,
        "unique_n": 8,
        "seed0_matches_a24": True,
        "ratio_stats_not_a_reading": {
            "L2": summarize_ratios(rows, "L2"),
            "L3": summarize_ratios(rows, "L3"),
        },
        "per_seed": rows,
        "applied_reading": key,
        "reading": pre["readings_committed_before_seeing_numbers"][key],
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps({
        "seed0_matches_a24": True,
        "n_L2_present": n_l2,
        "n_L3_present": n_l3,
        "per_seed_present": [
            {"seed": r["seed"], "L2": r["L2"]["present"], "L3": r["L3"]["present"],
             "L2_mup_ratio": r["L2"]["ratio_N300_over_N64"]["mup"],
             "L2_ntp_ratio": r["L2"]["ratio_N300_over_N64"]["ntp"],
             "L3_mup_ratio": r["L3"]["ratio_N300_over_N64"]["mup"],
             "L3_ntp_ratio": r["L3"]["ratio_N300_over_N64"]["ntp"]}
            for r in rows
        ],
        "ratio_stats_not_a_reading": out["ratio_stats_not_a_reading"],
        "applied_reading": key,
        "reading": out["reading"],
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
