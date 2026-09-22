"""Manifold one-task loss-vs-width contrast, n=32.

Pre-commit: `results/mup_manifold_n32_precommit.json` (written first).
Not a stream. Does not lift A6. Does not retune A26/A27.

    python scripts/run_mup_manifold_n32.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

from _par import pin_threads

pin_threads()

from scipy.stats import binomtest  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_mup_loss_width_n32 import clopper_pearson, failure_mode  # noqa: E402
from run_mup_loss_width_seeds import contrast, run_cell  # noqa: E402
from src.manifolds import dichotomies, generator  # noqa: E402
from src.models import paired_init  # noqa: E402
from src.train.loop import flatten_task  # noqa: E402

PRE = ROOT / "results" / "mup_manifold_n32_precommit.json"
OUT = ROOT / "results" / "mup_manifold_n32.json"


def task_xy(seed: int, P: int, d: int, D: int, R: float, M: int):
    streams = paired_init(seed)
    arr = generator.make_arrangement(P, d, D, R, M, streams["data"])
    y = dichotomies.sample_balanced(P, streams["stream"])
    return flatten_task(arr.points, y)


def pack(rows: list[dict], depth: str) -> dict:
    k = sum(1 for r in rows if r[depth]["present"])
    modes = Counter(r[depth]["failure_mode"] for r in rows)
    n_hit = sum(
        1 for r in rows
        if r[depth]["loss"]["mup"]["64"] <= 0.05 and r[depth]["loss"]["mup"]["300"] <= 0.05
    )
    return {
        "k_present": k,
        "n": 32,
        "rate": k / 32,
        "pass_count_threshold": 24,
        "meets_k24": k >= 24,
        "clopper_pearson_95": clopper_pearson(k, 32),
        "failure_modes": dict(modes),
        "n_mup_both_widths_loss_le_0.05_not_a_reading": n_hit,
    }


def apply_reading(l2_k: int, l3_k: int) -> str:
    if l2_k < 24:
        return "instrument_blind"
    if l3_k < 24:
        return "l3_problem"
    return "table_supported"


def main() -> None:
    pre = json.loads(PRE.read_text())
    assert pre["written_before_running"] is True
    assert pre["governs_stream_arms"] is False
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
                    rec = run_cell(
                        N, parameterization, lr0, d, seed, X, y, n_hidden, module, K,
                    )
                    cells[parameterization][str(N)] = rec
            c = contrast(cells, widths)
            c["failure_mode"] = failure_mode(c)
            per_depth[depth] = c
        rows.append({"seed": seed, "L2": per_depth["L2"], "L3": per_depth["L3"]})

    l2, l3 = pack(rows, "L2"), pack(rows, "L3")
    key = apply_reading(l2["k_present"], l3["k_present"])
    out = {
        "generated_by": "scripts/run_mup_manifold_n32.py",
        "precommit": str(PRE.relative_to(ROOT)),
        "governs_stream_arms": False,
        "signed_finding": False,
        "does_not_reopen_a26_a27": True,
        "L2": l2,
        "L3": l3,
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
                "L2_mup_loss": r["L2"]["loss"]["mup"],
                "L3_mup_loss": r["L3"]["loss"]["mup"],
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
