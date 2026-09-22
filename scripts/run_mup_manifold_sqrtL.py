"""A28 stress-test: μP η × √(2/L) at K=128.

Pre-commit: `results/mup_manifold_sqrtL_precommit.json` (written first).
Diagnostic only. Does not retune A28. Does not put 1/√L in ScalingConfig.

    python scripts/run_mup_manifold_sqrtL.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

from _par import pin_threads

pin_threads()

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_mup_loss_width_n32 import failure_mode  # noqa: E402
from run_mup_loss_width_seeds import contrast, run_cell  # noqa: E402
from run_mup_manifold_n32 import pack, task_xy  # noqa: E402

PRE = ROOT / "results" / "mup_manifold_sqrtL_precommit.json"
A28 = ROOT / "results" / "mup_manifold_n32.json"
A29 = ROOT / "results" / "mup_l3_mup_traj.json"
OUT = ROOT / "results" / "mup_manifold_sqrtL.json"

L_REF = 2.0


def depth_L(n_hidden: int) -> int:
    return n_hidden + 1


def mup_lr0(lr0: float, n_hidden: int) -> float:
    return lr0 * math.sqrt(L_REF / depth_L(n_hidden))


def prefix_ok(rows: list[dict], a28: dict) -> bool:
    old = {r["seed"]: r for r in a28["per_seed"]}
    for r in rows:
        o = old[r["seed"]]
        if r["L2"]["loss"]["mup"] != o["L2_mup_loss"]:
            return False
        if r["L2"]["ratio_N300_over_N64"]["ntp"] != o["L2_ntp_ratio"]:
            return False
        if r["L3"]["ratio_N300_over_N64"]["ntp"] != o["L3_ntp_ratio"]:
            return False
    return True


def apply_reading(l3_k: int) -> str:
    return "rescues_k24" if l3_k >= 24 else "does_not_rescue"


def main() -> None:
    pre = json.loads(PRE.read_text())
    a28 = json.loads(A28.read_text())
    a29 = json.loads(A29.read_text())
    assert pre["written_before_running"] is True
    assert pre["not_adopted_into_parameterization_py"] is True
    h = pre["held_fixed"]
    P, M, d, D, R = h["P"], h["M"], h["d"], h["D"], h["R"]
    lr0, K, module = h["lr0"], h["K"], h["module"]
    widths = list(h["widths"])
    rows = []
    for seed in range(32):
        X, y = task_xy(seed, P, d, D, R, M)
        per_depth = {}
        for depth, n_hidden in (("L2", 1), ("L3", 2)):
            cells = {}
            for parameterization in ("mup", "ntp"):
                cells[parameterization] = {}
                cell_lr0 = mup_lr0(lr0, n_hidden) if parameterization == "mup" else lr0
                for N in widths:
                    cells[parameterization][str(N)] = run_cell(
                        N, parameterization, cell_lr0, d, seed, X, y, n_hidden, module, K,
                    )
            c = contrast(cells, widths)
            c["failure_mode"] = failure_mode(c)
            c["mup_lr0"] = mup_lr0(lr0, n_hidden)
            per_depth[depth] = c
        rows.append({"seed": seed, "L2": per_depth["L2"], "L3": per_depth["L3"]})

    if not prefix_ok(rows, a28):
        out = {
            "generated_by": "scripts/run_mup_manifold_sqrtL.py",
            "precommit": str(PRE.relative_to(ROOT)),
            "applied_reading": "prefix_mismatch",
            "reading": pre["readings_committed_before_seeing_numbers"]["prefix_mismatch"],
        }
        OUT.write_text(json.dumps(out, indent=2))
        print(json.dumps({"applied_reading": "prefix_mismatch"}, indent=2))
        return

    cohort = set(a29["cohort_seeds"])
    l2, l3 = pack(rows, "L2"), pack(rows, "L3")
    key = apply_reading(l3["k_present"])
    out = {
        "generated_by": "scripts/run_mup_manifold_sqrtL.py",
        "precommit": str(PRE.relative_to(ROOT)),
        "governs_stream_arms": False,
        "signed_finding": False,
        "does_not_reopen_a28": True,
        "not_adopted_into_parameterization_py": True,
        "prefix_matches_a28_control": True,
        "mup_lr0_L2": mup_lr0(lr0, 1),
        "mup_lr0_L3": mup_lr0(lr0, 2),
        "L2": l2,
        "L3": l3,
        "a28_l3_cohort_present_not_a_reading": {
            "n_cohort": len(cohort),
            "n_L3_present": sum(1 for r in rows if r["seed"] in cohort and r["L3"]["present"]),
            "n_L3_mup_does_not_increase": sum(
                1 for r in rows if r["seed"] in cohort and r["L3"]["mup_does_not_increase"]
            ),
        },
        "per_seed": [
            {
                "seed": r["seed"],
                "L2_present": r["L2"]["present"],
                "L3_present": r["L3"]["present"],
                "L2_mode": r["L2"]["failure_mode"],
                "L3_mode": r["L3"]["failure_mode"],
                "L2_mup_ratio": r["L2"]["ratio_N300_over_N64"]["mup"],
                "L3_mup_ratio": r["L3"]["ratio_N300_over_N64"]["mup"],
                "L3_ntp_ratio": r["L3"]["ratio_N300_over_N64"]["ntp"],
            }
            for r in rows
        ],
        "applied_reading": key,
        "reading": pre["readings_committed_before_seeing_numbers"][key],
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps({
        "L2": l2,
        "L3": l3,
        "a28_l3_cohort_present_not_a_reading": out["a28_l3_cohort_present_not_a_reading"],
        "applied_reading": key,
        "reading": out["reading"],
    }, indent=2))


if __name__ == "__main__":
    main()
