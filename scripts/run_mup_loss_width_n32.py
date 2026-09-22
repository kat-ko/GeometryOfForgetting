"""n=32 rate on the A24 contrast. Does not retune A26.

Pre-commit: `results/mup_loss_width_n32_precommit.json` (written first).

    python scripts/run_mup_loss_width_n32.py
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

from run_mup_loss_width_seeds import _batch, contrast, run_cell  # noqa: E402

PRE = ROOT / "results" / "mup_loss_width_n32_precommit.json"
A26 = ROOT / "results" / "mup_loss_width_seeds.json"
OUT = ROOT / "results" / "mup_loss_width_n32.json"


def clopper_pearson(k: int, n: int) -> list[float]:
    ci = binomtest(k, n).proportion_ci(confidence_level=0.95, method="exact")
    return [float(ci.low), float(ci.high)]


def failure_mode(c: dict) -> str:
    if c["present"]:
        return "present"
    losses = [
        c["loss"]["mup"]["64"], c["loss"]["mup"]["300"],
        c["loss"]["ntp"]["64"], c["loss"]["ntp"]["300"],
    ]
    if not all(np_finite(x) for x in losses):
        return "nonfinite"
    mup_ok, ntp_ok = c["mup_does_not_increase"], c["ntp_increases"]
    if not mup_ok and not ntp_ok:
        return "both"
    if not mup_ok:
        return "mup_increased"
    return "ntp_did_not_increase"


def np_finite(x) -> bool:
    return bool(x == x and x not in (float("inf"), float("-inf")))


def prefix_matches(rows: list[dict], a26: dict) -> bool:
    old = {r["seed"]: r for r in a26["per_seed"]}
    for r in rows:
        if r["seed"] > 7:
            continue
        o = old[r["seed"]]
        for depth in ("L2", "L3"):
            if r[depth]["present"] != o[depth]["present"]:
                return False
            if r[depth]["loss"] != o[depth]["loss"]:
                return False
    return True


def main() -> None:
    pre = json.loads(PRE.read_text())
    a26 = json.loads(A26.read_text())
    assert pre["written_before_running"] is True
    assert pre["governs_stream_arms"] is False
    held = pre["held_fixed"]
    widths = list(held["widths"])
    d, lr0, K, module = held["d"], held["named_lr0"], held["K"], held["module"]
    rows = []
    for seed in range(32):
        X, y = _batch(d, held["batch"], seed + 1)
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
        rows.append({"seed": seed, "data_seed": seed + 1, "L2": per_depth["L2"], "L3": per_depth["L3"]})

    if not prefix_matches(rows, a26):
        out = {
            "generated_by": "scripts/run_mup_loss_width_n32.py",
            "precommit": str(PRE.relative_to(ROOT)),
            "governs_stream_arms": False,
            "applied_reading": "prefix_mismatch",
            "reading": pre["readings_committed_before_seeing_numbers"]["prefix_mismatch"],
        }
        OUT.write_text(json.dumps(out, indent=2))
        print(json.dumps({"applied_reading": "prefix_mismatch"}, indent=2))
        return

    def pack(depth: str) -> dict:
        k = sum(1 for r in rows if r[depth]["present"])
        modes = Counter(r[depth]["failure_mode"] for r in rows)
        return {
            "k_present": k,
            "n": 32,
            "rate": k / 32,
            "clopper_pearson_95": clopper_pearson(k, 32),
            "failure_modes": dict(modes),
        }

    out = {
        "generated_by": "scripts/run_mup_loss_width_n32.py",
        "precommit": str(PRE.relative_to(ROOT)),
        "governs_stream_arms": False,
        "signed_finding": False,
        "does_not_reopen_a26": True,
        "prefix_seeds_0_7_match_a26": True,
        "L2": pack("L2"),
        "L3": pack("L3"),
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
        "applied_reading": "recorded_rate",
        "reading": pre["readings_committed_before_seeing_numbers"]["recorded_rate"],
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps({
        "prefix_ok": True,
        "L2": out["L2"],
        "L3": out["L3"],
        "applied_reading": "recorded_rate",
        "reading": out["reading"],
    }, indent=2))


if __name__ == "__main__":
    main()
