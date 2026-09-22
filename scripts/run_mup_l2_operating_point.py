"""L=2 operating-point search for Gate 2 check B, then one L=3 apply.

Pre-commit: `results/mup_l2_operating_point_precommit.json` (written first).
If a point freezes, writes `results/mup_l3_gate2_frozen_precommit.json`
before any L=3 number, then applies check B at L=3 once.

    python scripts/run_mup_l2_operating_point.py

Does not lift A6. Does not sign Verification 2. Does not retune the original
L=3 Gate 2 contract.
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

PRE = ROOT / "results" / "mup_l2_operating_point_precommit.json"
SEARCH_OUT = ROOT / "results" / "mup_l2_operating_point.json"
FROZEN_PRE = ROOT / "results" / "mup_l3_gate2_frozen_precommit.json"
L3_OUT = ROOT / "results" / "mup_l3_gate2_frozen.json"


def _cfg(N: int, gamma: float, parameterization: str, lr0: float, d: int):
    return {
        m: ScalingConfig(
            N=N, d=d, gamma_0=gamma, parameterization=parameterization, lr0=lr0,
        )
        for m in MODULES
    }


def _model(N, gamma, parameterization, lr0, d, seed, n_hidden_layers):
    return TwoModuleNet.init(
        _cfg(N, gamma, parameterization, lr0, d),
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


def loss_after(N, parameterization, lr0, K, d, seed, X, y, n_hidden_layers) -> float:
    mdl = _model(N, 1.0, parameterization, lr0, d, seed, n_hidden_layers)
    kw = {"diagnostic": True} if n_hidden_layers == 2 else {}
    loss = float("nan")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for _ in range(K):
            loss = mdl.sgd_step(X, y, **kw)
            if not _finite(loss):
                return float("inf")
    return float(loss) if _finite(loss) else float("inf")


def argmin_prefix(losses: dict[str, float], grid: list[float]) -> dict:
    finite = [(lr0, losses[str(lr0)]) for lr0 in grid if _finite(losses[str(lr0)])]
    if not finite:
        return {"lr0": None, "loss": float("inf"), "index": None}
    lr0, loss = min(finite, key=lambda t: t[1])
    return {"lr0": lr0, "loss": loss, "index": grid.index(lr0)}


def index_dist(a: dict, b: dict) -> int | None:
    if a["index"] is None or b["index"] is None:
        return None
    return abs(a["index"] - b["index"])


def eval_B(table: dict, grid: list[float], widths: list[int]) -> dict:
    argmins = {}
    for parameterization in ("mup", "ntp"):
        argmins[parameterization] = {
            str(N): argmin_prefix(table[parameterization][str(N)], grid)
            for N in widths
        }
    mup_d = index_dist(argmins["mup"][str(widths[0])], argmins["mup"][str(widths[1])])
    ntp_d = index_dist(argmins["ntp"][str(widths[0])], argmins["ntp"][str(widths[1])])
    mup_ok = mup_d is not None and mup_d <= 1
    ntp_slides = ntp_d is not None and ntp_d >= 1
    return {
        "grid": grid,
        "argmins": argmins,
        "mup_index_distance": mup_d,
        "ntp_index_distance": ntp_d,
        "mup_stable": mup_ok,
        "ntp_slides": ntp_slides,
        "pass": bool(mup_ok and ntp_slides),
    }


def fill_table(K, lr0s, widths, d, seed, X, y, n_hidden_layers, cache: dict) -> dict:
    table = {p: {str(N): {} for N in widths} for p in ("mup", "ntp")}
    for parameterization in ("mup", "ntp"):
        for N in widths:
            for lr0 in lr0s:
                key = (K, parameterization, N, lr0, n_hidden_layers)
                if key not in cache:
                    cache[key] = loss_after(
                        N, parameterization, lr0, K, d, seed, X, y, n_hidden_layers,
                    )
                table[parameterization][str(N)][str(lr0)] = cache[key]
    return table


def check_A(K, lr0, widths, d, seed, X, y, n_hidden_layers, module):
    rms = {}
    kw = {"diagnostic": True} if n_hidden_layers == 2 else {}
    z_idx = 1 if n_hidden_layers == 2 else 0
    for N in widths:
        mdl = _model(N, 1.0, "mup", lr0, d, seed, n_hidden_layers)
        z0 = mdl.preactivations(X, module)[z_idx]
        for _ in range(K):
            mdl.sgd_step(X, y, **kw)
        zK = mdl.preactivations(X, module)[z_idx]
        rms[str(N)] = _rms(zK - z0)
    lo, hi = rms[str(widths[0])], rms[str(widths[1])]
    ratio = hi / lo if lo > 0 else float("nan")
    passed = _finite(lo) and _finite(hi) and lo > 0 and hi > 0 and (1.0 / 3.0) <= ratio <= 3.0
    return {"rms_delta_z": rms, "ratio_N300_over_N64": ratio, "pass": passed}


def check_C(K, lr0, widths, d, seed, X, y, n_hidden_layers, module):
    rows = {}
    passed = True
    kw = {"diagnostic": True} if n_hidden_layers == 2 else {}
    for N in widths:
        ch = {}
        for gamma in (1.0, 10.0):
            mdl = _model(N, gamma, "mup", lr0, d, seed, n_hidden_layers)
            for _ in range(K):
                mdl.sgd_step(X, y, **kw)
            rec = {"W": mdl.weight_change(module)}
            if n_hidden_layers == 2:
                rec["W2"] = mdl.weight_change_W2(module)
            ch[str(gamma)] = rec
        order_W = _finite(ch["1.0"]["W"]) and _finite(ch["10.0"]["W"]) and ch["10.0"]["W"] > ch["1.0"]["W"]
        order_W2 = True
        if n_hidden_layers == 2:
            order_W2 = (
                _finite(ch["1.0"]["W2"]) and _finite(ch["10.0"]["W2"])
                and ch["10.0"]["W2"] > ch["1.0"]["W2"]
            )
        rows[str(N)] = {"by_gamma": ch, "W_separates": order_W, "W2_separates": order_W2}
        passed = passed and order_W and order_W2
    return {"per_width": rows, "pass": passed}


def write_frozen_precommit(pre: dict, freeze: dict) -> None:
    body = {
        "generated_by": "scripts/run_mup_l2_operating_point.py, after L=2 search, before L=3",
        "written_before_running": True,
        "written_before_running_applies_to": "one-shot L=3 check B at the L=2-frozen operating point",
        "status": "unsigned diagnostic contract — not Verification 2",
        "governs_stream_arms": False,
        "signed_finding": False,
        "parent_precommit": str(PRE.relative_to(ROOT)),
        "frozen_from_l2": {
            "K": freeze["K"],
            "n_tail": freeze["n_tail"],
            "lr0_grid": freeze["grid"],
            "l2_check_B": {
                "mup_index_distance": freeze["eval"]["mup_index_distance"],
                "ntp_index_distance": freeze["eval"]["ntp_index_distance"],
                "argmins": freeze["eval"]["argmins"],
            },
        },
        "identification": "check B at this (K, lr0_grid), L=3, diagnostic SGD, no stream",
        "pass_B": pre["l3_apply"]["pass_B"],
        "overall_pass": pre["l3_apply"]["overall_pass"],
        "shared": {
            "d": pre["held_fixed"]["d"],
            "batch": pre["held_fixed"]["batch"],
            "seed": pre["held_fixed"]["seed"],
            "data_seed": pre["held_fixed"]["data_seed"],
            "module": pre["held_fixed"]["module"],
            "widths": pre["held_fixed"]["widths"],
            "n_hidden_layers": 2,
            "K": freeze["K"],
            "lr0_grid": freeze["grid"],
        },
        "readings_committed_before_seeing_l3_numbers": {
            "l3_b_passes": pre["readings_committed_before_seeing_numbers"]["l3_b_passes"],
            "l3_b_fails": pre["readings_committed_before_seeing_numbers"]["l3_b_fails"],
        },
    }
    FROZEN_PRE.write_text(json.dumps(body, indent=2))


def main() -> None:
    pre = json.loads(PRE.read_text())
    assert pre["written_before_running"] is True
    assert pre["governs_stream_arms"] is False
    held = pre["held_fixed"]
    search = pre["search"]
    G0 = list(held["G0"])
    H = list(search["high_tail_H"])
    widths = list(held["widths"])
    d, seed = held["d"], held["seed"]
    X, y = _batch(d, held["batch"], held["data_seed"])
    cache: dict = {}
    evaluated = []
    freeze = None

    for K in search["K_sequence"]:
        for n_tail in range(0, len(H) + 1):
            if K == search["skip"]["K"] and n_tail == search["skip"]["n_tail"]:
                continue
            grid = G0 + H[:n_tail]
            table = fill_table(K, grid, widths, d, seed, X, y, 1, cache)
            ev = eval_B(table, grid, widths)
            row = {"K": K, "n_tail": n_tail, "grid": grid, "eval": ev}
            evaluated.append(row)
            if ev["pass"]:
                freeze = row
                break
        if freeze is not None:
            break

    if freeze is None:
        out = {
            "generated_by": "scripts/run_mup_l2_operating_point.py",
            "precommit": str(PRE.relative_to(ROOT)),
            "governs_stream_arms": False,
            "signed_finding": False,
            "search_evaluated": [
                {
                    "K": r["K"], "n_tail": r["n_tail"], "pass": r["eval"]["pass"],
                    "mup_index_distance": r["eval"]["mup_index_distance"],
                    "ntp_index_distance": r["eval"]["ntp_index_distance"],
                    "argmins": r["eval"]["argmins"],
                }
                for r in evaluated
            ],
            "frozen": None,
            "l3_ran": False,
            "applied_reading": "search_failed",
            "reading": pre["readings_committed_before_seeing_numbers"]["search_failed"],
        }
        SEARCH_OUT.write_text(json.dumps(out, indent=2))
        print(json.dumps({
            "applied_reading": out["applied_reading"],
            "n_evaluated": len(evaluated),
            "reading": out["reading"],
        }, indent=2))
        return

    write_frozen_precommit(pre, freeze)
    frozen_pre = json.loads(FROZEN_PRE.read_text())
    assert frozen_pre["written_before_running"] is True
    assert frozen_pre["frozen_from_l2"]["K"] == freeze["K"]

    l3_table = fill_table(
        freeze["K"], freeze["grid"], widths, d, seed, X, y, 2, cache,
    )
    l3_b = eval_B(l3_table, freeze["grid"], widths)
    l3_a = check_A(freeze["K"], 0.01, widths, d, seed, X, y, 2, held["module"])
    l3_c = check_C(freeze["K"], 0.001, widths, d, seed, X, y, 2, held["module"])
    key = "l3_b_passes" if l3_b["pass"] else "l3_b_fails"

    search_out = {
        "generated_by": "scripts/run_mup_l2_operating_point.py",
        "precommit": str(PRE.relative_to(ROOT)),
        "governs_stream_arms": False,
        "signed_finding": False,
        "search_evaluated": [
            {
                "K": r["K"], "n_tail": r["n_tail"], "pass": r["eval"]["pass"],
                "mup_index_distance": r["eval"]["mup_index_distance"],
                "ntp_index_distance": r["eval"]["ntp_index_distance"],
                "argmins": r["eval"]["argmins"],
            }
            for r in evaluated
        ],
        "frozen": {
            "K": freeze["K"],
            "n_tail": freeze["n_tail"],
            "lr0_grid": freeze["grid"],
            "l2_eval": freeze["eval"],
        },
        "l3_ran": True,
        "applied_reading": key,
        "reading": pre["readings_committed_before_seeing_numbers"][key],
    }
    SEARCH_OUT.write_text(json.dumps(search_out, indent=2))

    l3_out = {
        "generated_by": "scripts/run_mup_l2_operating_point.py",
        "precommit": str(FROZEN_PRE.relative_to(ROOT)),
        "governs_stream_arms": False,
        "signed_finding": False,
        "check_B": l3_b,
        "check_A_recorded_not_a_licence": l3_a,
        "check_C_recorded_not_a_licence": l3_c,
        "overall_pass": l3_b["pass"],
        "applied_reading": key,
        "reading": frozen_pre["readings_committed_before_seeing_l3_numbers"][key],
    }
    L3_OUT.write_text(json.dumps(l3_out, indent=2))
    print(json.dumps({
        "frozen_K": freeze["K"],
        "frozen_n_tail": freeze["n_tail"],
        "frozen_grid": freeze["grid"],
        "l2_B_pass": freeze["eval"]["pass"],
        "l2_mup_d": freeze["eval"]["mup_index_distance"],
        "l2_ntp_d": freeze["eval"]["ntp_index_distance"],
        "l3_B_pass": l3_b["pass"],
        "l3_mup_d": l3_b["mup_index_distance"],
        "l3_ntp_d": l3_b["ntp_index_distance"],
        "applied_reading": key,
        "reading": search_out["reading"],
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
