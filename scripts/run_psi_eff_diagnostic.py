"""D3 Ψ_eff check — reclassified from gate to **diagnostic** (`02` §2c).

`replicaMFT`'s `R_M`/`D_M` are the PRX-2018 eq-28/29 functionals, not GLUE's
`R_eff = √(E[c]/E[b−c])` and `D_eff = E[b]/P`, so
`Ψ_eff = α·D_eff/(1 + R_eff⁻²)` computed from replicaMFT outputs is **expected to
disagree** with `E[c]/E[a]`. Running it converts a reasoned expectation into a
recorded measurement.

Also records `res_coeff0` beside both ρ_c conventions, documenting finding F1:
`res_coeff0` is an abs-normalized cosine between raw manifold centers and is
neither `rho_c_glue` nor `rho_c_signed`.

Writes `results/psi_eff_diagnostic.json`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from src.glue import core
from src.manifolds import generator

_VENDOR = Path(__file__).resolve().parents[1] / "third_party" / "replicaMFT"
sys.path.insert(0, str(_VENDOR))
from mftma.manifold_analysis_correlation import manifold_analysis_corr  # noqa: E402

REPLICA_MFT = "replicaMFT@d56eda1d86a71f4f4601ef1acc3254e20259a80c"
CASES = (
    {"P": 8, "d": 200, "D": 4, "R": 1.0, "M": 60},
    {"P": 16, "d": 300, "D": 4, "R": 1.0, "M": 60},
)
N_T = 200


def run_case(cfg: dict, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    arr = generator.make_arrangement(cfg["P"], cfg["d"], cfg["D"], cfg["R"], cfg["M"], rng)
    mans = core.from_arrangement(arr.points)

    alpha_vec, R_M, D_M, res_coeff0, _ = manifold_analysis_corr(mans, 0, N_T)
    alpha_mf = core.harmonic_mean(alpha_vec)  # AGENTS §4: harmonic, never arithmetic
    R_mf, D_mf = float(np.mean(R_M)), float(np.mean(D_M))
    psi_replica = alpha_mf * D_mf / (1.0 + R_mf**-2)

    g = core.glue_measures(mans, np.random.default_rng(seed + 1000), n_t=N_T)

    return {
        "config": cfg,
        "replicaMFT": {
            "estimator": REPLICA_MFT,
            "alpha_mf_harmonic": alpha_mf,
            "alpha_mf_arithmetic": float(np.mean(alpha_vec)),
            "R_M_mean": R_mf,
            "D_M_mean": D_mf,
            "center_cos_abs": float(res_coeff0),
            "psi_eff_via_identity": float(psi_replica),
        },
        "glue_core": {
            "estimator": "glue_core",
            "alpha": g.alpha,
            "R_eff": g.R_eff,
            "D_eff": g.D_eff,
            "Psi_eff_direct": g.Psi_eff,
            "rho_c_glue": g.rho_c_glue,
            "rho_c_signed": g.rho_c_signed,
            "identity_residual": g.identity_residual,
        },
        "psi_relative_gap": float(abs(psi_replica - g.Psi_eff) / g.Psi_eff),
        "verdict": (
            "MISMATCH as expected — R_M/D_M are PRX eq-28/29 functionals, not R_eff/D_eff"
            if abs(psi_replica - g.Psi_eff) / g.Psi_eff > 0.05
            else "AGREEMENT — unexpected; must be explained before use"
        ),
    }


def main() -> None:
    out = {
        "purpose": "02 §2c diagnostic (reclassified from gate 2026-08-11)",
        "expectation": "mismatch",
        "n_t": N_T,
        "cases": [],
    }
    for cfg in CASES:
        row = run_case(cfg)
        out["cases"].append(row)
        r, c = row["replicaMFT"], row["glue_core"]
        print(
            f"P={cfg['P']:<3} psi(replicaMFT)={r['psi_eff_via_identity']:.4f} "
            f"psi(core)={c['Psi_eff_direct']:.4f} gap={row['psi_relative_gap']:.1%}\n"
            f"      alpha  mf={r['alpha_mf_harmonic']:.4f} core={c['alpha']:.4f} | "
            f"R  M={r['R_M_mean']:.3f} eff={c['R_eff']:.3f} | "
            f"D  M={r['D_M_mean']:.3f} eff={c['D_eff']:.3f}\n"
            f"      center_cos_abs={r['center_cos_abs']:.4f} vs "
            f"rho_c_glue={c['rho_c_glue']:.4f} rho_c_signed={c['rho_c_signed']:.4f}",
            flush=True,
        )

    path = Path(__file__).resolve().parents[1] / "results" / "psi_eff_diagnostic.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
