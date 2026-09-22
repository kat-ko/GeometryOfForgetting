"""Init-α at L = 2 versus L = 3 — forward pass only, no training, no stream.

`docs/16` §Amendments A6. Reports both parameter-count-matched and width-matched
L=3, so depth and width are not collinear. Writes `results/init_alpha_depth.json`.

    python scripts/run_init_alpha_depth.py
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

from src.glue import core  # noqa: E402
from src.manifolds import generator  # noqa: E402
from src.models import MODULES, ScalingConfig, TwoModuleNet, paired_init  # noqa: E402
from src.models.parameterization import module_param_count, width_matching_param_count  # noqa: E402

P, D_AMB, M, N_L2 = 16, 150, 150, 300
D_INT, R = 4, 1.0
N_T = 200
SEEDS = tuple(range(8))
GAMMA = 1.0
N_L3_PARAM = width_matching_param_count(N_L2, D_AMB, n_hidden_layers=2)
N_L3_WIDTH = N_L2
OUT = ROOT / "results" / "init_alpha_depth.json"


def _arrangement(seed: int):
    return generator.make_arrangement(
        P, D_AMB, D_INT, R, M, paired_init(seed)["data"], rho_C=0.0, rho_A=0.0)


def _model(seed: int, N: int, n_hidden_layers: int):
    cfg = {m: ScalingConfig(N=N, d=D_AMB, gamma_0=GAMMA, lr0=5.0) for m in MODULES}
    return TwoModuleNet.init(cfg, paired_init(seed)["shape"], n_hidden_layers=n_hidden_layers)


def _alpha(model, arr, seed, module="A") -> dict:
    reps = model.manifold_representation(arr.points, module=module)
    r = core.glue_measures(reps, np.random.default_rng(seed + 9000), n_t=N_T)
    return {"alpha": r.alpha, "D_eff": r.D_eff, "R_eff": r.R_eff, "Psi_eff": r.Psi_eff}


def _pair(a: np.ndarray, b: np.ndarray) -> dict:
    d = b - a
    n_b_lower = int(np.sum(b < a))
    return {
        "mean_a": float(a.mean()), "mean_b": float(b.mean()),
        "delta_mean": float(d.mean()),
        "sem_a": float(a.std(ddof=1) / np.sqrt(len(a))),
        "sem_b": float(b.std(ddof=1) / np.sqrt(len(b))),
        "n_b_lower": n_b_lower,
        "sign_test_p_one_sided": float(0.5 ** len(a)) if n_b_lower == len(a) else None,
        "direction_only": True,
    }


def main() -> None:
    rows = []
    for seed in SEEDS:
        arr = _arrangement(seed)
        l2 = _alpha(_model(seed, N_L2, 1), arr, seed)
        l3p = _alpha(_model(seed, N_L3_PARAM, 2), arr, seed)
        l3w = _alpha(_model(seed, N_L3_WIDTH, 2), arr, seed)
        rows.append({
            "seed": seed,
            "L2_N300": l2,
            "L3_N150_param_matched": l3p,
            "L3_N300_width_matched": l3w,
        })

    a2 = np.array([r["L2_N300"]["alpha"] for r in rows])
    a3p = np.array([r["L3_N150_param_matched"]["alpha"] for r in rows])
    a3w = np.array([r["L3_N300_width_matched"]["alpha"] for r in rows])
    param = _pair(a2, a3p)
    width = _pair(a2, a3w)
    depth_at_fixed_n = width
    width_at_fixed_l3 = _pair(a3p, a3w)

    out = {
        "generated_by": "scripts/run_init_alpha_depth.py",
        "question": "is capacity at init already lower at L=3 than L=2, and is that depth or width?",
        "training": False,
        "muP_derivation_used": False,
        "licenses_learning_regime": False,
        "width_rule": "report both: parameter-count matched (L3 N=150) and width matched (L3 N=300). The previous L2 N=300 vs L3 N=150 comparison confounds the two.",
        "L2_N300": {"n_hidden_layers": 1, "N": N_L2, "depth_L": 2,
                    "params_per_module": module_param_count(N_L2, D_AMB, 1)},
        "L3_N150_param_matched": {"n_hidden_layers": 2, "N": N_L3_PARAM, "depth_L": 3,
                                 "params_per_module": module_param_count(N_L3_PARAM, D_AMB, 2)},
        "L3_N300_width_matched": {"n_hidden_layers": 2, "N": N_L3_WIDTH, "depth_L": 3,
                                 "params_per_module": module_param_count(N_L3_WIDTH, D_AMB, 2)},
        "P": P, "d": D_AMB, "M": M, "n_t": N_T, "seeds": list(SEEDS),
        "param_matched_L3_vs_L2": param,
        "width_matched_L3_vs_L2": depth_at_fixed_n,
        "width_at_fixed_L3_N300_vs_N150": width_at_fixed_l3,
        "reading": (
            "Forward-pass α is an initialisation statistic. It licenses no claim "
            "about the learning regime the network ends up in. 8/8 is a sign test "
            "for direction, not magnitude. "
            "Param-matched (N=150) and width-matched (N=300) L=3 are reported separately "
            "so a drop cannot be attributed to both axes at once."
        ),
        "per_seed": rows,
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps({
        "param_matched_L3_vs_L2": param,
        "width_matched_L3_vs_L2": depth_at_fixed_n,
        "width_at_fixed_L3_N300_vs_N150": width_at_fixed_l3,
        "reading": out["reading"],
    }, indent=2))


if __name__ == "__main__":
    main()
