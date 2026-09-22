"""A28 manifold loss-vs-width contrast at K=512, n=32.

Pre-commit: `results/mup_manifold_k512_precommit.json` (written first).
Does not retune A28 at K=128. Does not lift A6.

    python scripts/run_mup_manifold_k512.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from _par import pin_threads

pin_threads()

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_mup_loss_width_n32 import failure_mode  # noqa: E402
from run_mup_loss_width_seeds import contrast, run_cell  # noqa: E402
from run_mup_manifold_n32 import apply_reading, pack, task_xy  # noqa: E402

PRE = ROOT / "results" / "mup_manifold_k512_precommit.json"
A29 = ROOT / "results" / "mup_l3_mup_traj.json"
OUT = ROOT / "results" / "mup_manifold_k512.json"


def prefix_ok(rows: list[dict], a29: dict) -> bool:
    old = {r["seed"]: r for r in a29["per_seed"]}
    for r in rows:
        o = old[r["seed"]]
        if r["L2"]["ratio_N300_over_N64"]["mup"] != o["L2_ratio_by_K"]["512"]:
            return False
        if r["L3"]["ratio_N300_over_N64"]["mup"] != o["L3_ratio_by_K"]["512"]:
            return False
    return True


def main() -> None:
    pre = json.loads(PRE.read_text())
    a29 = json.loads(A29.read_text())
    assert pre["written_before_running"] is True
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
                for N in widths:
                    cells[parameterization][str(N)] = run_cell(
                        N, parameterization, lr0, d, seed, X, y, n_hidden, module, K,
                    )
            c = contrast(cells, widths)
            c["failure_mode"] = failure_mode(c)
            per_depth[depth] = c
        rows.append({"seed": seed, "L2": per_depth["L2"], "L3": per_depth["L3"]})

    if not prefix_ok(rows, a29):
        out = {
            "generated_by": "scripts/run_mup_manifold_k512.py",
            "precommit": str(PRE.relative_to(ROOT)),
            "applied_reading": "prefix_mismatch",
            "reading": pre["readings_committed_before_seeing_numbers"]["prefix_mismatch"],
        }
        OUT.write_text(json.dumps(out, indent=2))
        print(json.dumps({"applied_reading": "prefix_mismatch"}, indent=2))
        return

    cohort = set(a29["cohort_seeds"])
    l2, l3 = pack(rows, "L2"), pack(rows, "L3")
    key = apply_reading(l2["k_present"], l3["k_present"])
    out = {
        "generated_by": "scripts/run_mup_manifold_k512.py",
        "precommit": str(PRE.relative_to(ROOT)),
        "governs_stream_arms": False,
        "signed_finding": False,
        "does_not_reopen_a28": True,
        "prefix_mup_ratio_512_matches_a29": True,
        "L2": l2,
        "L3": l3,
        "a28_l3_cohort_present_at_512_not_a_reading": {
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
                "L2_ntp_ratio": r["L2"]["ratio_N300_over_N64"]["ntp"],
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
        "a28_l3_cohort_present_at_512_not_a_reading": out["a28_l3_cohort_present_at_512_not_a_reading"],
        "applied_reading": key,
        "reading": out["reading"],
    }, indent=2))


if __name__ == "__main__":
    main()
