"""Gate 2 empirical μP checks at L=3 — toy batch, no stream.

Pass/fail criteria are loaded from `results/mup_l3_hypothesis.json`, written
before any of these numbers existed. This script does not retune them.

    python scripts/run_mup_l3_diagnostics.py

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
from src.models.parameterization import N_BASE  # noqa: E402

HYP = ROOT / "results" / "mup_l3_hypothesis.json"
OUT = ROOT / "results" / "mup_l3_diagnostics.json"


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
        n_hidden_layers=2,
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
    fwd_eq = bool(np.array_equal(ntp.forward(X), mup.forward(X)))
    ntp.sgd_step(X, y, diagnostic=True)
    mup.sgd_step(X, y, diagnostic=True)
    tensors = {}
    for name, a, b in (
        ("W", ntp.W, mup.W),
        ("W2", ntp.W2, mup.W2),
        ("u", ntp.u, mup.u),
    ):
        tensors[name] = {
            m: bool(np.array_equal(a[m], b[m])) for m in MODULES
        }
    passed = fwd_eq and all(all(v.values()) for v in tensors.values())
    return {
        "setting": "N=64, gamma_0=1, L=3, first diagnostic step, float64",
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
        z0 = mdl.preactivations(X, module)[1]
        for _ in range(spec["K"]):
            mdl.sgd_step(X, y, diagnostic=True)
        zK = mdl.preactivations(X, module)[1]
        rms[str(N)] = _rms(zK - z0)
    lo, hi = rms[str(spec["widths"][0])], rms[str(spec["widths"][1])]
    ratio = hi / lo if lo > 0 else float("nan")
    passed = _finite(lo) and _finite(hi) and lo > 0 and hi > 0 and (1.0 / 3.0) <= ratio <= 3.0
    return {
        "rms_delta_z2": rms,
        "ratio_N300_over_N64": ratio,
        "pass_interval": [1.0 / 3.0, 3.0],
        "pass": passed,
    }


def _loss_after(N, parameterization, lr0, K, d, seed, X, y) -> float:
    mdl = _model(N, 1.0, parameterization, lr0, d, seed)
    loss = float("nan")
    for _ in range(K):
        loss = mdl.sgd_step(X, y, diagnostic=True)
        if not _finite(loss):
            return float("inf")
    return float(loss)


def _argmin_grid(losses: dict[str, float], grid: list[float]) -> dict:
    finite = [(lr0, losses[str(lr0)]) for lr0 in grid if _finite(losses[str(lr0)])]
    if not finite:
        return {"lr0": None, "loss": float("inf"), "index": None}
    lr0, loss = min(finite, key=lambda t: t[1])
    return {"lr0": lr0, "loss": loss, "index": grid.index(lr0)}


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
    return {
        "loss_after_K": table,
        "argmins": argmins,
        "mup_index_distance": mup_d,
        "ntp_index_distance": ntp_d,
        "mup_stable": mup_ok,
        "ntp_slides": ntp_slides,
        "pass": bool(mup_ok and ntp_slides),
    }


def _index_dist(argmins: dict, widths: list[int]) -> int | None:
    a, b = argmins[str(widths[0])]["index"], argmins[str(widths[1])]["index"]
    if a is None or b is None:
        return None
    return abs(a - b)


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
                mdl.sgd_step(X, y, diagnostic=True)
            ch[str(gamma)] = {
                "W": mdl.weight_change(module),
                "W2": mdl.weight_change_W2(module),
            }
        g1, g10 = ch["1.0"], ch["10.0"]
        order_W = _finite(g1["W"]) and _finite(g10["W"]) and g10["W"] > g1["W"]
        order_W2 = _finite(g1["W2"]) and _finite(g10["W2"]) and g10["W2"] > g1["W2"]
        rows[str(N)] = {
            "by_gamma": ch,
            "W_separates": order_W,
            "W2_separates": order_W2,
        }
        passed = passed and order_W and order_W2
    return {"per_width": rows, "pass": passed}


def main() -> None:
    h = json.loads(HYP.read_text())
    assert h["written_before_running"] is True
    assert h["governs_stream_arms"] is False
    g2 = h["gates"]["2"]["shared"]
    X, y = _batch(g2["d"], g2["batch"], g2["data_seed"])

    g1 = gate1(g2["d"], g2["batch"], g2["seed"], g2["data_seed"])
    if not g1["pass"]:
        out = {
            "generated_by": "scripts/run_mup_l3_diagnostics.py",
            "hypothesis": str(HYP.relative_to(ROOT)),
            "governs_stream_arms": False,
            "signed_finding": False,
            "gate1": g1,
            "gate2": None,
            "gate2_ran": False,
            "overall_pass": False,
            "reading": "Gate 1 failed. Stop. Do not proceed to Gate 2. Do not sign Verification 2.",
        }
        OUT.write_text(json.dumps(out, indent=2))
        print(json.dumps(out, indent=2))
        return

    a = check_A(h, X, y)
    b = check_B(h, X, y)
    c = check_C(h, X, y)
    g2_pass = bool(a["pass"] and b["pass"] and c["pass"])
    mixed = (a["pass"] or b["pass"] or c["pass"]) and not g2_pass
    overall = bool(g1["pass"] and g2_pass)
    if not g2_pass:
        reading = (
            "Gate 2 failed"
            + (" (mixed: not a partial licence)" if mixed else "")
            + ". Stop. Do not sign Verification 2. Do not lift stream training."
        )
    else:
        reading = (
            "Gate 1 and Gate 2 passed the pre-committed criteria. This is not "
            "Verification 2. Stream training stays blocked until Kati signs."
        )
    out = {
        "generated_by": "scripts/run_mup_l3_diagnostics.py",
        "hypothesis": str(HYP.relative_to(ROOT)),
        "governs_stream_arms": False,
        "signed_finding": False,
        "gate1": g1,
        "gate2": {
            "check_A": a,
            "check_B": b,
            "check_C": c,
            "pass": g2_pass,
            "mixed": mixed,
        },
        "gate2_ran": True,
        "overall_pass": overall,
        "reading": reading,
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps({
        "gate1_pass": g1["pass"],
        "check_A_pass": a["pass"],
        "check_A_ratio": a["ratio_N300_over_N64"],
        "check_B_pass": b["pass"],
        "check_B_mup_distance": b["mup_index_distance"],
        "check_B_ntp_distance": b["ntp_index_distance"],
        "check_C_pass": c["pass"],
        "overall_pass": overall,
        "reading": reading,
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
