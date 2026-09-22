"""Loss-vs-width contrast at named lr0=10 on the A23 frozen grid.

Pre-commit: `results/mup_loss_width_precommit.json` (written first).
L=2 and L=3 in one shot. No search. Identification is not argmin.

    python scripts/run_mup_loss_width.py

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

PRE = ROOT / "results" / "mup_loss_width_precommit.json"
OUT = ROOT / "results" / "mup_loss_width.json"


def _cfg(N: int, gamma: float, parameterization: str, lr0: float, d: int):
    return {
        m: ScalingConfig(
            N=N, d=d, gamma_0=gamma, parameterization=parameterization, lr0=lr0,
        )
        for m in MODULES
    }


def _model(N, parameterization, lr0, d, seed, n_hidden_layers):
    return TwoModuleNet.init(
        _cfg(N, 1.0, parameterization, lr0, d),
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


def loss_after(N, parameterization, lr0, K, d, seed, X, y, n_hidden_layers) -> float:
    mdl = _model(N, parameterization, lr0, d, seed, n_hidden_layers)
    kw = {"diagnostic": True} if n_hidden_layers == 2 else {}
    loss = float("nan")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for _ in range(K):
            loss = mdl.sgd_step(X, y, **kw)
            if not _finite(loss):
                return float("inf")
    return float(loss) if _finite(loss) else float("inf")


def fill_table(K, grid, widths, d, seed, X, y, n_hidden_layers) -> dict:
    table = {p: {str(N): {} for N in widths} for p in ("mup", "ntp")}
    for parameterization in ("mup", "ntp"):
        for N in widths:
            for lr0 in grid:
                table[parameterization][str(N)][str(lr0)] = loss_after(
                    N, parameterization, lr0, K, d, seed, X, y, n_hidden_layers,
                )
    return table


def contrast_at(table: dict, lr0: float, widths: list[int]) -> dict:
    n64, n300 = str(widths[0]), str(widths[1])
    key = str(lr0)
    out = {"named_lr0": lr0, "loss": {}, "ratio_N300_over_N64": {}, "mup_does_not_increase": None,
           "ntp_increases": None, "present": False}
    for p in ("mup", "ntp"):
        a, b = table[p][n64][key], table[p][n300][key]
        out["loss"][p] = {n64: a, n300: b}
        out["ratio_N300_over_N64"][p] = (b / a) if _finite(a) and a != 0 else float("nan")
    mup_a, mup_b = out["loss"]["mup"][n64], out["loss"]["mup"][n300]
    ntp_a, ntp_b = out["loss"]["ntp"][n64], out["loss"]["ntp"][n300]
    mup_ok = _finite(mup_a) and _finite(mup_b) and mup_b <= mup_a
    ntp_ok = _finite(ntp_a) and _finite(ntp_b) and ntp_b > ntp_a
    out["mup_does_not_increase"] = mup_ok
    out["ntp_increases"] = ntp_ok
    out["present"] = bool(mup_ok and ntp_ok)
    out["n64_mup_equals_ntp"] = bool(
        _finite(mup_a) and _finite(ntp_a) and mup_a == ntp_a
    )
    return out


def apply_reading(pre: dict, l2: dict, l3: dict) -> str:
    if not l2["present"]:
        return "instrument_blind"
    if l3["present"]:
        return "table_supported"
    return "l3_problem"


def main() -> None:
    pre = json.loads(PRE.read_text())
    assert pre["written_before_running"] is True
    assert pre["governs_stream_arms"] is False
    assert pre["no_search"] is True
    held = pre["held_fixed"]
    widths = list(held["widths"])
    grid = list(held["lr0_grid"])
    K = int(held["K"])
    named = float(held["named_lr0"])
    assert named in grid
    X, y = _batch(held["d"], held["batch"], held["data_seed"])

    tables = {
        "L2": fill_table(K, grid, widths, held["d"], held["seed"], X, y, 1),
        "L3": fill_table(K, grid, widths, held["d"], held["seed"], X, y, 2),
    }
    l2 = contrast_at(tables["L2"], named, widths)
    l3 = contrast_at(tables["L3"], named, widths)
    key = apply_reading(pre, l2, l3)
    out = {
        "generated_by": "scripts/run_mup_loss_width.py",
        "precommit": str(PRE.relative_to(ROOT)),
        "governs_stream_arms": False,
        "signed_finding": False,
        "loss_tables_full_grid_not_identification": tables,
        "L2": l2,
        "L3": l3,
        "applied_reading": key,
        "reading": pre["readings_committed_before_seeing_numbers"][key],
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps({
        "L2_present": l2["present"],
        "L2_mup_ratio": l2["ratio_N300_over_N64"]["mup"],
        "L2_ntp_ratio": l2["ratio_N300_over_N64"]["ntp"],
        "L2_losses": l2["loss"],
        "L3_present": l3["present"],
        "L3_mup_ratio": l3["ratio_N300_over_N64"]["mup"],
        "L3_ntp_ratio": l3["ratio_N300_over_N64"]["ntp"],
        "L3_losses": l3["loss"],
        "applied_reading": key,
        "reading": out["reading"],
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
