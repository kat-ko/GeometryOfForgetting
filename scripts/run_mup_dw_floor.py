"""L=2 ‖ΔW‖ floor, then A24 loss-vs-width contrast at that K.

Pre-commit: `results/mup_dw_floor_precommit.json` (written first).

    python scripts/run_mup_dw_floor.py

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

PRE = ROOT / "results" / "mup_dw_floor_precommit.json"
FROZEN = ROOT / "results" / "mup_dw_floor_frozen.json"
OUT = ROOT / "results" / "mup_dw_floor.json"


def _cfg(N: int, parameterization: str, lr0: float, d: int):
    return {
        m: ScalingConfig(
            N=N, d=d, gamma_0=1.0, parameterization=parameterization, lr0=lr0,
        )
        for m in MODULES
    }


def _model(N, parameterization, lr0, d, seed, n_hidden_layers):
    return TwoModuleNet.init(
        _cfg(N, parameterization, lr0, d),
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


def _rms(a: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(a))))


def snapshot_trajectory(
    N, parameterization, lr0, d, seed, X, y, n_hidden, module, Ks, *, stop_floor=None,
):
    mdl = _model(N, parameterization, lr0, d, seed, n_hidden)
    kw = {"diagnostic": True} if n_hidden == 2 else {}
    z_idx = 1 if n_hidden == 2 else 0
    z0 = mdl.preactivations(X, module)[z_idx].copy()
    want = set(Ks)
    out = []
    max_k = max(Ks)
    loss = float("nan")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for step in range(1, max_k + 1):
            loss = mdl.sgd_step(X, y, **kw)
            if step not in want:
                continue
            rec = {
                "K": step,
                "loss": float(loss) if _finite(loss) else float("inf"),
                "weight_change_W": mdl.weight_change(module) if _finite(loss) else float("nan"),
                "rms_delta_z": _rms(mdl.preactivations(X, module)[z_idx] - z0) if _finite(loss) else float("nan"),
            }
            if n_hidden == 2:
                rec["weight_change_W2"] = (
                    mdl.weight_change_W2(module) if _finite(loss) else float("nan")
                )
            out.append(rec)
            if not _finite(loss):
                break
            if stop_floor is not None and _finite(rec["weight_change_W"]) and rec["weight_change_W"] >= stop_floor:
                break
    return out


def run_k_steps(N, parameterization, lr0, d, seed, X, y, n_hidden, module, K):
    mdl = _model(N, parameterization, lr0, d, seed, n_hidden)
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
        "rms_delta_z": (
            _rms(mdl.preactivations(X, module)[z_idx] - z0) if _finite(loss) else float("nan")
        ),
    }
    if n_hidden == 2:
        rec["weight_change_W2"] = (
            mdl.weight_change_W2(module) if _finite(loss) else float("nan")
        )
    return rec


def contrast_from_cells(cells: dict, widths: list[int]) -> dict:
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


def apply_reading(l2_present: bool, l3_present: bool) -> str:
    if not l2_present:
        return "instrument_blind"
    if l3_present:
        return "table_supported_at_floor"
    return "l3_problem"


def main() -> None:
    pre = json.loads(PRE.read_text())
    assert pre["written_before_running"] is True
    assert pre["governs_stream_arms"] is False
    held = pre["held_fixed"]
    search = pre["floor_search"]
    widths = list(held["widths"])
    d, seed, lr0 = held["d"], held["seed"], held["named_lr0"]
    module = held["module"]
    Ks = list(search["K_sequence"])
    floor = float(search["floor"])
    X, y = _batch(d, held["batch"], held["data_seed"])

    freeze_traj = snapshot_trajectory(
        64, "mup", lr0, d, seed, X, y, 1, module, Ks, stop_floor=floor,
    )
    frozen_K = None
    freeze_row = None
    for row in freeze_traj:
        wc = row["weight_change_W"]
        if _finite(wc) and wc >= floor:
            frozen_K = int(row["K"])
            freeze_row = row
            break

    if frozen_K is None:
        trajectories = {"mup_N64": freeze_traj}
        out = {
            "generated_by": "scripts/run_mup_dw_floor.py",
            "precommit": str(PRE.relative_to(ROOT)),
            "governs_stream_arms": False,
            "signed_finding": False,
            "l2_trajectories": trajectories,
            "frozen_K": None,
            "l3_ran": False,
            "applied_reading": "search_failed",
            "reading": pre["readings_committed_before_seeing_numbers"]["search_failed"],
        }
        OUT.write_text(json.dumps(out, indent=2))
        print(json.dumps({
            "applied_reading": "search_failed",
            "max_weight_change_mup_N64": max(
                (r["weight_change_W"] for r in freeze_traj if _finite(r["weight_change_W"])),
                default=None,
            ),
            "reading": out["reading"],
        }, indent=2))
        return

    Ks_upto = [k for k in Ks if k <= frozen_K]
    trajectories = {"mup_N64": [r for r in freeze_traj if r["K"] <= frozen_K]}
    for parameterization in ("mup", "ntp"):
        for N in widths:
            key = f"{parameterization}_N{N}"
            if key == "mup_N64":
                continue
            trajectories[key] = snapshot_trajectory(
                N, parameterization, lr0, d, seed, X, y, 1, module, Ks_upto,
            )

    frozen_body = {
        "generated_by": "scripts/run_mup_dw_floor.py, after L=2 floor search, before L=3 contrast",
        "written_before_running": True,
        "written_before_running_applies_to": "L=2 and L=3 A24 contrast at frozen K",
        "governs_stream_arms": False,
        "frozen_K": frozen_K,
        "floor": floor,
        "l2_mup_N64_at_freeze": freeze_row,
        "named_lr0": lr0,
    }
    FROZEN.write_text(json.dumps(frozen_body, indent=2))

    cells = {"L2": {}, "L3": {}}
    for depth, n_hidden in (("L2", 1), ("L3", 2)):
        cells[depth] = {}
        for parameterization in ("mup", "ntp"):
            cells[depth][parameterization] = {}
            for N in widths:
                cells[depth][parameterization][str(N)] = run_k_steps(
                    N, parameterization, lr0, d, seed, X, y, n_hidden, module, frozen_K,
                )

    l2 = contrast_from_cells(cells["L2"], widths)
    l3 = contrast_from_cells(cells["L3"], widths)
    key = apply_reading(l2["present"], l3["present"])
    l3_floor_met = bool(
        _finite(cells["L3"]["mup"]["64"]["weight_change_W"])
        and cells["L3"]["mup"]["64"]["weight_change_W"] >= floor
    )
    out = {
        "generated_by": "scripts/run_mup_dw_floor.py",
        "precommit": str(PRE.relative_to(ROOT)),
        "frozen_file": str(FROZEN.relative_to(ROOT)),
        "governs_stream_arms": False,
        "signed_finding": False,
        "l2_trajectories": trajectories,
        "frozen_K": frozen_K,
        "l2_mup_N64_at_freeze": freeze_row,
        "l3_mup_N64_meets_floor_at_frozen_K_not_a_reading": l3_floor_met,
        "L2": l2,
        "L3": l3,
        "l3_ran": True,
        "applied_reading": key,
        "reading": pre["readings_committed_before_seeing_numbers"][key],
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps({
        "frozen_K": frozen_K,
        "l2_mup_N64_weight_change": freeze_row["weight_change_W"],
        "L2_present": l2["present"],
        "L2_mup_ratio": l2["ratio_N300_over_N64"]["mup"],
        "L2_ntp_ratio": l2["ratio_N300_over_N64"]["ntp"],
        "L2_weight_change_W": {
            p: {n: cells["L2"][p][n]["weight_change_W"] for n in ("64", "300")}
            for p in ("mup", "ntp")
        },
        "L3_present": l3["present"],
        "L3_mup_ratio": l3["ratio_N300_over_N64"]["mup"],
        "L3_ntp_ratio": l3["ratio_N300_over_N64"]["ntp"],
        "L3_weight_change_W": {
            p: {n: cells["L3"][p][n]["weight_change_W"] for n in ("64", "300")}
            for p in ("mup", "ntp")
        },
        "L3_weight_change_W2": {
            p: {n: cells["L3"][p][n]["weight_change_W2"] for n in ("64", "300")}
            for p in ("mup", "ntp")
        },
        "l3_mup_N64_meets_floor": l3_floor_met,
        "applied_reading": key,
        "reading": out["reading"],
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
