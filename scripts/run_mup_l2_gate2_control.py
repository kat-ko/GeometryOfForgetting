"""L=2 clone of Gate 2 — control for the L=3 mixed fail.

Pass/fail from `results/mup_l2_gate2_control_precommit.json`, written before
these numbers. Same (d, batch, seeds, K, lr0 grid, widths) as the L=3
diagnostic. Only depth changes.

    python scripts/run_mup_l2_gate2_control.py

Does not lift A6. Does not retune the L=3 contract. Does not sign Verification 2.
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
from src.models.parameterization import N_BASE  # noqa: E402

PRE = ROOT / "results" / "mup_l2_gate2_control_precommit.json"
OUT = ROOT / "results" / "mup_l2_gate2_control.json"


def _cfg(N: int, gamma: float, parameterization: str, lr0: float, d: int):
    return {
        m: ScalingConfig(
            N=N, d=d, gamma_0=gamma, parameterization=parameterization, lr0=lr0,
        )
        for m in MODULES
    }


def _model(N: int, gamma: float, parameterization: str, lr0: float, d: int, seed: int):
    return TwoModuleNet.init(
        _cfg(N, gamma, parameterization, lr0, d),
        paired_init(seed)["shape"],
        n_hidden_layers=1,
    )


def _batch(d: int, B: int, data_seed: int):
    rng = np.random.default_rng(data_seed)
    X = rng.standard_normal((B, d))
    y = np.sign(rng.standard_normal(B))
    y[y == 0] = 1.0
    return X, y


def _rms(a: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(a))))


def _finite(x) -> bool:
    return bool(np.isfinite(x))


def gate1(d: int, B: int, seed: int, data_seed: int) -> dict:
    X, y = _batch(d, B, data_seed)
    ntp = _model(N_BASE, 1.0, "ntp", 0.01, d, seed)
    mup = _model(N_BASE, 1.0, "mup", 0.01, d, seed)
    assert ntp.n_hidden_layers == 1 and mup.n_hidden_layers == 1
    fwd_eq = bool(np.array_equal(ntp.forward(X), mup.forward(X)))
    ntp.sgd_step(X, y)
    mup.sgd_step(X, y)
    tensors = {}
    for name, a, b in (("W", ntp.W, mup.W), ("u", ntp.u, mup.u)):
        tensors[name] = {m: bool(np.array_equal(a[m], b[m])) for m in MODULES}
    passed = fwd_eq and all(all(v.values()) for v in tensors.values())
    return {
        "setting": "N=64, gamma_0=1, L=2, first step, float64",
        "forward_equal": fwd_eq,
        "tensors_equal_after_step": tensors,
        "lr_equal": bool(mup.cfg["A"].lr == ntp.cfg["A"].lr),
        "output_scale_equal": bool(mup.cfg["A"].output_scale == ntp.cfg["A"].output_scale),
        "pass": passed,
    }


def check_A(h: dict, X: np.ndarray, y: np.ndarray) -> dict:
    spec = h["gates"]["2"]["check_A"]
    shared = h["gates"]["2"]["shared"]
    module = shared["module"]
    rms = {}
    for N in spec["widths"]:
        mdl = _model(N, spec["gamma_0"], spec["parameterization"], spec["lr0"],
                     shared["d"], shared["seed"])
        z0 = mdl.preactivations(X, module)[0]
        for _ in range(spec["K"]):
            mdl.sgd_step(X, y)
        zK = mdl.preactivations(X, module)[0]
        rms[str(N)] = _rms(zK - z0)
    lo, hi = rms[str(spec["widths"][0])], rms[str(spec["widths"][1])]
    ratio = hi / lo if lo > 0 else float("nan")
    passed = _finite(lo) and _finite(hi) and lo > 0 and hi > 0 and (1.0 / 3.0) <= ratio <= 3.0
    return {
        "rms_delta_z1": rms,
        "ratio_N300_over_N64": ratio,
        "pass_interval": [1.0 / 3.0, 3.0],
        "pass": passed,
    }


def _loss_after(N, parameterization, lr0, K, d, seed, X, y) -> float:
    mdl = _model(N, 1.0, parameterization, lr0, d, seed)
    loss = float("nan")
    for _ in range(K):
        loss = mdl.sgd_step(X, y)
        if not _finite(loss):
            return float("inf")
    return float(loss)


def _argmin_grid(losses: dict[str, float], grid: list[float]) -> dict:
    finite = [(lr0, losses[str(lr0)]) for lr0 in grid if _finite(losses[str(lr0)])]
    if not finite:
        return {"lr0": None, "loss": float("inf"), "index": None}
    lr0, loss = min(finite, key=lambda t: t[1])
    return {"lr0": lr0, "loss": loss, "index": grid.index(lr0)}


def _index_dist(argmins: dict, widths: list[int]) -> int | None:
    a, b = argmins[str(widths[0])]["index"], argmins[str(widths[1])]["index"]
    if a is None or b is None:
        return None
    return abs(a - b)


def check_B(h: dict, X: np.ndarray, y: np.ndarray) -> dict:
    spec = h["gates"]["2"]["check_B"]
    shared = h["gates"]["2"]["shared"]
    grid = spec["lr0_grid"]
    table = {}
    argmins = {}
    for parameterization in ("mup", "ntp"):
        argmins[parameterization] = {}
        table[parameterization] = {}
        for N in spec["widths"]:
            losses = {
                str(lr0): _loss_after(
                    N, parameterization, lr0, spec["K"], shared["d"], shared["seed"], X, y,
                )
                for lr0 in grid
            }
            table[parameterization][str(N)] = losses
            argmins[parameterization][str(N)] = _argmin_grid(losses, grid)
    mup_d = _index_dist(argmins["mup"], spec["widths"])
    ntp_d = _index_dist(argmins["ntp"], spec["widths"])
    mup_ok = mup_d is not None and mup_d <= 1
    ntp_slides = ntp_d is not None and ntp_d >= 1
    both_at_edge = all(
        argmins[p][str(N)]["lr0"] == grid[-1]
        for p in ("mup", "ntp") for N in spec["widths"]
    )
    return {
        "loss_after_K": table,
        "argmins": argmins,
        "mup_index_distance": mup_d,
        "ntp_index_distance": ntp_d,
        "mup_stable": mup_ok,
        "ntp_slides": ntp_slides,
        "both_argmins_at_grid_edge": both_at_edge,
        "pass": bool(mup_ok and ntp_slides),
    }


def check_C(h: dict, X: np.ndarray, y: np.ndarray) -> dict:
    spec = h["gates"]["2"]["check_C"]
    shared = h["gates"]["2"]["shared"]
    module = shared["module"]
    rows = {}
    passed = True
    for N in spec["widths"]:
        ch = {}
        for gamma in spec["gammas"]:
            mdl = _model(N, gamma, spec["parameterization"], spec["lr0"],
                         shared["d"], shared["seed"])
            for _ in range(spec["K"]):
                mdl.sgd_step(X, y)
            ch[str(gamma)] = {"W": mdl.weight_change(module)}
        g1, g10 = ch["1.0"], ch["10.0"]
        order_W = _finite(g1["W"]) and _finite(g10["W"]) and g10["W"] > g1["W"]
        rows[str(N)] = {"by_gamma": ch, "W_separates": order_W}
        passed = passed and order_W
    return {"per_width": rows, "pass": passed}


def apply_reading(pre: dict, g1: dict, b: dict) -> str:
    readings = pre["readings_committed_before_seeing_numbers"]
    if not g1["pass"]:
        return "gate1_fail"
    if b["pass"]:
        return "protocol_is_strong_enough"
    return "protocol_cannot_certify"


def main() -> None:
    pre = json.loads(PRE.read_text())
    assert pre["written_before_running"] is True
    assert pre["governs_stream_arms"] is False
    assert pre["gates"]["2"]["shared"]["n_hidden_layers"] == 1
    g2 = pre["gates"]["2"]["shared"]
    X, y = _batch(g2["d"], g2["batch"], g2["data_seed"])

    g1 = gate1(g2["d"], g2["batch"], g2["seed"], g2["data_seed"])
    if not g1["pass"]:
        key = "gate1_fail"
        out = {
            "generated_by": "scripts/run_mup_l2_gate2_control.py",
            "precommit": str(PRE.relative_to(ROOT)),
            "governs_stream_arms": False,
            "signed_finding": False,
            "gate1": g1,
            "gate2": None,
            "gate2_ran": False,
            "applied_reading": key,
            "reading": pre["readings_committed_before_seeing_numbers"][key],
        }
        OUT.write_text(json.dumps(out, indent=2))
        print(json.dumps(out, indent=2))
        return

    a = check_A(pre, X, y)
    b = check_B(pre, X, y)
    c = check_C(pre, X, y)
    key = apply_reading(pre, g1, b)
    out = {
        "generated_by": "scripts/run_mup_l2_gate2_control.py",
        "precommit": str(PRE.relative_to(ROOT)),
        "governs_stream_arms": False,
        "signed_finding": False,
        "gate1": g1,
        "gate2": {
            "check_A": a,
            "check_B": b,
            "check_C": c,
            "primary": "check_B",
            "primary_pass": b["pass"],
        },
        "gate2_ran": True,
        "applied_reading": key,
        "reading": pre["readings_committed_before_seeing_numbers"][key],
        "empirical_floors_not_a_reading": {
            "check_A_rms_delta_z1": a["rms_delta_z1"],
            "check_C_weight_change": {
                str(N): c["per_width"][str(N)]["by_gamma"] for N in (64, 300)
            },
        },
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps({
        "gate1_pass": g1["pass"],
        "check_A_pass": a["pass"],
        "check_A_ratio": a["ratio_N300_over_N64"],
        "check_A_rms": a["rms_delta_z1"],
        "check_B_pass": b["pass"],
        "check_B_mup_distance": b["mup_index_distance"],
        "check_B_ntp_distance": b["ntp_index_distance"],
        "check_B_both_at_edge": b["both_argmins_at_grid_edge"],
        "check_C_pass": c["pass"],
        "applied_reading": key,
        "reading": out["reading"],
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
