"""Status check for the μP-at-L=3 diagnostic packet.

Reads committed result files and re-runs Gate 1 unit tests. Does not sign
Verification 2. Does not lift A6.

    python scripts/check_mup_l3_validation.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "mup_l3_validation_status.json"

PYTEST = [
    str(ROOT / ".venv" / "bin" / "python"), "-m", "pytest", "-q", "--tb=line",
    "tests/test_models_parameterization.py::test_base_width_equivalence",
    "tests/test_models_parameterization.py::test_base_width_equivalence_l3",
    "tests/test_models_parameterization.py::test_l3_gradients_match_finite_differences",
    "tests/test_models_parameterization.py::test_l3_forward_i6_and_i5_no_training",
    "tests/test_pipeline.py::test_run_arm_refuses_l3",
]


def load(name: str) -> dict | None:
    path = ROOT / "results" / name
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def main() -> None:
    proc = subprocess.run(PYTEST, cwd=ROOT, capture_output=True, text=True)
    gate1_ok = proc.returncode == 0
    files = {
        "hypothesis": load("mup_l3_hypothesis.json"),
        "gate2_l3_original": load("mup_l3_diagnostics.json"),
        "gate2_l2_control": load("mup_l2_gate2_control.json"),
        "operating_point": load("mup_l2_operating_point.json"),
        "frozen_l3_b": load("mup_l3_gate2_frozen.json"),
        "loss_width_a24": load("mup_loss_width.json"),
        "dw_floor_a25": load("mup_dw_floor.json"),
        "seeds_a26": load("mup_loss_width_seeds.json"),
        "n32": load("mup_loss_width_n32.json"),
        "manifold_n32": load("mup_manifold_n32.json"),
        "traj_a29": load("mup_l3_mup_traj.json"),
        "manifold_k512": load("mup_manifold_k512.json"),
        "manifold_n150": load("mup_manifold_n150.json"),
        "manifold_gamma10": load("mup_manifold_gamma10.json"),
        "manifold_sqrtL": load("mup_manifold_sqrtL.json"),
    }
    n32 = files["n32"] or {}
    a26 = files["seeds_a26"] or {}
    status = {
        "generated_by": "scripts/check_mup_l3_validation.py",
        "governs_stream_arms": False,
        "signed_finding": False,
        "verification_2": "unsigned",
        "a6_stream_training": "blocked",
        "gate1_unit_tests": {
            "pass": gate1_ok,
            "returncode": proc.returncode,
            "summary": proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else "",
        },
        "applied_readings": {
            "A21_original_gate2": (files["gate2_l3_original"] or {}).get("reading"),
            "A22_l2_control": (files["gate2_l2_control"] or {}).get("applied_reading"),
            "A23_operating_point": (files["operating_point"] or {}).get("applied_reading"),
            "A24_loss_width_seed0": (files["loss_width_a24"] or {}).get("applied_reading"),
            "A25_dw_floor_seed0": (files["dw_floor_a25"] or {}).get("applied_reading"),
            "A26_unique_n8": a26.get("applied_reading"),
            "n32_rate": n32.get("applied_reading"),
            "A28_manifold_n32": (files["manifold_n32"] or {}).get("applied_reading"),
            "A29_mup_traj": (files["traj_a29"] or {}).get("applied_reading"),
            "A30_manifold_k512": (files["manifold_k512"] or {}).get("applied_reading"),
            "A31_n150": (files["manifold_n150"] or {}).get("applied_reading"),
            "A32_gamma10": (files["manifold_gamma10"] or {}).get("applied_reading"),
            "A33_sqrtL": (files["manifold_sqrtL"] or {}).get("applied_reading"),
        },
        "counts": {
            "A26_L2": f"{a26.get('n_L2_present')}/8",
            "A26_L3": f"{a26.get('n_L3_present')}/8",
            "n32_L2": n32.get("L2"),
            "n32_L3": n32.get("L3"),
            "manifold_n32_L2": (files["manifold_n32"] or {}).get("L2"),
            "manifold_n32_L3": (files["manifold_n32"] or {}).get("L3"),
            "manifold_k512_L2": (files["manifold_k512"] or {}).get("L2"),
            "manifold_k512_L3": (files["manifold_k512"] or {}).get("L3"),
            "manifold_n150_L2": (files["manifold_n150"] or {}).get("L2"),
            "manifold_n150_L3": (files["manifold_n150"] or {}).get("L3"),
            "manifold_gamma10_L2": (files["manifold_gamma10"] or {}).get("L2"),
            "manifold_gamma10_L3": (files["manifold_gamma10"] or {}).get("L3"),
            "manifold_sqrtL_L2": (files["manifold_sqrtL"] or {}).get("L2"),
            "manifold_sqrtL_L3": (files["manifold_sqrtL"] or {}).get("L3"),
        },
        "what_holds": [
            "Gate 1 algebraic pin at N=64, γ₀=1, L=2 and L=3",
            "L=3 analytic W2 grads match finite differences",
            "Stream / run_arm still refuse L=3",
            "A30 k≥24 at L=2 and L=3 on the manifold instrument at K=512",
            "A31 three-width interpolation k≥24 at L=2 and L=3 at K=512, γ₀=1",
            "A32 k≥24 at L=2 and L=3 at γ₀=10 on the same instrument",
        ],
        "what_does_not_hold": [
            "Original Gate 2 check B (argmin η*) at L=3 and at L=2 on the short grid",
            "A26 8/8 replicate of the named-lr0=10 loss-vs-width contrast",
            "A28 k≥24 at L=3 on the manifold instrument at K=128 (L=2 met k≥24)",
            "A33 1/√L at K=128: L=3 still k<24 (17/32; A28 was 19/32)",
        ],
        "seed0_only_and_closed": [
            "A24 table_supported and A25 table_supported_at_floor — seed 0",
        ],
        "continuation_constraints": [
            "A26 instrument_blind stands. n=32 does not retune it to a licence.",
            "A28 l3_problem at K=128 stands. A30 at K=512 does not retune it.",
            "A33 does_not_rescue. Do not adopt 1/√L into parameterization.py.",
            "A31 and A32 do not sign Verification 2 and do not lift A6.",
            "Do not sign Verification 2 on seed-0 contrasts.",
            "Gate 3 still requires Kati to write parameterization-derivation.md.",
            "Do not lift A6. Do not start sequential depth arms.",
        ],
    }
    OUT.write_text(json.dumps(status, indent=2))
    print(json.dumps(status, indent=2))
    if not gate1_ok:
        print(proc.stdout)
        print(proc.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
