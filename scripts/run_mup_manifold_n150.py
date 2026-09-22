"""Three-width interpolation on the A30 manifold instrument, n=32.

Pre-commit: `results/mup_manifold_n150_precommit.json` (written first).
Does not retune A30. Does not lift A6.

    python scripts/run_mup_manifold_n150.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

from _par import pin_threads

pin_threads()

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_mup_loss_width_n32 import clopper_pearson, np_finite  # noqa: E402
from run_mup_loss_width_seeds import run_cell  # noqa: E402
from run_mup_manifold_n32 import apply_reading, task_xy  # noqa: E402

PRE = ROOT / "results" / "mup_manifold_n150_precommit.json"
A30 = ROOT / "results" / "mup_manifold_k512.json"
OUT = ROOT / "results" / "mup_manifold_n150.json"


def contrast_three(cells: dict) -> dict:
    keys = ("64", "150", "300")
    loss = {p: {k: cells[p][k]["loss"] for k in keys} for p in ("mup", "ntp")}
    mup = loss["mup"]
    ntp = loss["ntp"]
    mup_fin = all(np_finite(mup[k]) for k in keys)
    ntp_fin = all(np_finite(ntp[k]) for k in keys)
    mup_ok = mup_fin and mup["150"] <= mup["64"] and mup["300"] <= mup["150"]
    ntp_ok = ntp_fin and ntp["150"] > ntp["64"] and ntp["300"] > ntp["150"]
    end_mup = np_finite(mup["64"]) and np_finite(mup["300"]) and mup["300"] <= mup["64"]
    end_ntp = np_finite(ntp["64"]) and np_finite(ntp["300"]) and ntp["300"] > ntp["64"]
    if mup_ok and ntp_ok:
        mode = "present"
    elif not (mup_fin and ntp_fin):
        mode = "nonfinite"
    elif (not mup_ok) and (not ntp_ok):
        mode = "both"
    elif not mup_ok:
        mode = "mup_not_monotone"
    else:
        mode = "ntp_not_monotone"
    return {
        "loss": loss,
        "mup_monotone": mup_ok,
        "ntp_monotone": ntp_ok,
        "present": bool(mup_ok and ntp_ok),
        "endpoint_present_not_a_reading": bool(end_mup and end_ntp),
        "failure_mode": mode,
    }


def pack(rows: list[dict], depth: str) -> dict:
    k = sum(1 for r in rows if r[depth]["present"])
    modes = Counter(r[depth]["failure_mode"] for r in rows)
    n_hit = sum(
        1 for r in rows
        if all(r[depth]["loss"]["mup"][w] <= 0.05 for w in ("64", "150", "300"))
    )
    return {
        "k_present": k,
        "n": 32,
        "rate": k / 32,
        "pass_count_threshold": 24,
        "meets_k24": k >= 24,
        "clopper_pearson_95": clopper_pearson(k, 32),
        "failure_modes": dict(modes),
        "n_endpoint_present_not_a_reading": sum(
            1 for r in rows if r[depth]["endpoint_present_not_a_reading"]
        ),
        "n_mup_all_widths_loss_le_0.05_not_a_reading": n_hit,
    }


def prefix_ok(rows: list[dict], a30: dict) -> bool:
    old = {r["seed"]: r for r in a30["per_seed"]}
    for r in rows:
        o = old[r["seed"]]
        l2, l3 = r["L2"]["loss"], r["L3"]["loss"]
        if (l2["mup"]["300"] / l2["mup"]["64"]) != o["L2_mup_ratio"]:
            return False
        if (l2["ntp"]["300"] / l2["ntp"]["64"]) != o["L2_ntp_ratio"]:
            return False
        if (l3["mup"]["300"] / l3["mup"]["64"]) != o["L3_mup_ratio"]:
            return False
        if (l3["ntp"]["300"] / l3["ntp"]["64"]) != o["L3_ntp_ratio"]:
            return False
    return True


def main() -> None:
    pre = json.loads(PRE.read_text())
    a30 = json.loads(A30.read_text())
    assert pre["written_before_running"] is True
    h = pre["held_fixed"]
    P, M, d, D, R = h["P"], h["M"], h["d"], h["D"], h["R"]
    lr0, K, module = h["lr0"], h["K"], h["module"]
    gamma_0 = h["gamma_0"]
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
                        gamma_0,
                    )
            per_depth[depth] = contrast_three(cells)
        rows.append({"seed": seed, "L2": per_depth["L2"], "L3": per_depth["L3"]})

    if not prefix_ok(rows, a30):
        out = {
            "generated_by": "scripts/run_mup_manifold_n150.py",
            "precommit": str(PRE.relative_to(ROOT)),
            "applied_reading": "prefix_mismatch",
            "reading": pre["readings_committed_before_seeing_numbers"]["prefix_mismatch"],
        }
        OUT.write_text(json.dumps(out, indent=2))
        print(json.dumps({"applied_reading": "prefix_mismatch"}, indent=2))
        return

    l2, l3 = pack(rows, "L2"), pack(rows, "L3")
    key = apply_reading(l2["k_present"], l3["k_present"])
    out = {
        "generated_by": "scripts/run_mup_manifold_n150.py",
        "precommit": str(PRE.relative_to(ROOT)),
        "governs_stream_arms": False,
        "signed_finding": False,
        "does_not_reopen_a30": True,
        "prefix_endpoint_ratios_match_a30": True,
        "L2": l2,
        "L3": l3,
        "per_seed": [
            {
                "seed": r["seed"],
                "L2_present": r["L2"]["present"],
                "L3_present": r["L3"]["present"],
                "L2_mode": r["L2"]["failure_mode"],
                "L3_mode": r["L3"]["failure_mode"],
                "L2_mup_loss": r["L2"]["loss"]["mup"],
                "L3_mup_loss": r["L3"]["loss"]["mup"],
                "L2_ntp_loss": r["L2"]["loss"]["ntp"],
                "L3_ntp_loss": r["L3"]["loss"]["ntp"],
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
        "applied_reading": key,
        "reading": out["reading"],
    }, indent=2))


if __name__ == "__main__":
    main()
