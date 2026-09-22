"""μP loss-vs-width trajectories on the A28 manifold task.

Pre-commit: `results/mup_l3_mup_traj_precommit.json` (written first).
Does not retune A28. Does not lift A6.

    python scripts/run_mup_l3_mup_traj.py
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

from run_mup_dw_floor import snapshot_trajectory  # noqa: E402
from run_mup_loss_width_seeds import _finite  # noqa: E402
from run_mup_manifold_n32 import task_xy  # noqa: E402

PRE = ROOT / "results" / "mup_l3_mup_traj_precommit.json"
A28 = ROOT / "results" / "mup_manifold_n32.json"
OUT = ROOT / "results" / "mup_l3_mup_traj.json"


def loss_at(snaps: list[dict], K: int) -> float:
    for s in snaps:
        if s["K"] == K:
            return s["loss"]
    return float("nan")


def ratio_at(s64: list[dict], s300: list[dict], K: int) -> float:
    a, b = loss_at(s64, K), loss_at(s300, K)
    if not _finite(a) or not _finite(b) or a == 0:
        return float("nan")
    return b / a


def shape_upto_128(s64, s300, Ks) -> str:
    ks = [k for k in Ks if k <= 128]
    rs = [(k, ratio_at(s64, s300, k)) for k in ks]
    if any(not _finite(r) for _, r in rs):
        return "nonfinite"
    r128 = dict(rs)[128]
    earlier = [r for k, r in rs if k < 128]
    if r128 > 1:
        if all(r > 1 for r in earlier):
            return "always_worse"
        if any(r <= 1 for r in earlier):
            return "cross_to_worse"
        return "always_worse"
    if all(r <= 1 for _, r in rs):
        return "always_better"
    if any(r > 1 for r in earlier):
        return "cross_to_better"
    return "always_better"


def rescue_512(s64, s300) -> str:
    r128 = ratio_at(s64, s300, 128)
    r512 = ratio_at(s64, s300, 512)
    if not _finite(r128) or not _finite(r512):
        return "nonfinite"
    if r128 <= 1:
        return "not_in_cohort"
    return "rescued_at_512" if r512 <= 1 else "still_worse_at_512"


def prefix_ok(rows: list[dict], a28: dict) -> bool:
    old = {r["seed"]: r for r in a28["per_seed"]}
    for r in rows:
        o = old[r["seed"]]
        for depth, key in (("L2", "L2_mup_loss"), ("L3", "L3_mup_loss")):
            for N in ("64", "300"):
                a = r[depth]["loss_by_K"][N]["128"]
                b = o[key][N]
                if a != b:
                    return False
    return True


def apply_shape(counts: dict) -> str:
    n_aw = counts.get("always_worse", 0)
    n_ov = counts.get("cross_to_worse", 0)
    if n_aw >= 7:
        return "always_worse_majority"
    if n_ov >= 7:
        return "overshoot_majority"
    return "mixed_shape"


def apply_rescue(counts: dict) -> str:
    n_r = counts.get("rescued_at_512", 0)
    n_s = counts.get("still_worse_at_512", 0)
    if n_r >= 7:
        return "rescues_majority"
    if n_s >= 7:
        return "does_not_rescue_majority"
    return "mixed_rescue"


def main() -> None:
    pre = json.loads(PRE.read_text())
    a28 = json.loads(A28.read_text())
    assert pre["written_before_running"] is True
    h = pre["held_fixed"]
    Ks = list(h["K_sequence"])
    widths = list(h["widths"])
    P, M, d, D, R = 16, 40, 60, 4, 1.0
    lr0, module = 5.0, "A"
    rows = []
    for seed in range(32):
        X, y = task_xy(seed, P, d, D, R, M)
        per_depth = {}
        for depth, n_hidden in (("L2", 1), ("L3", 2)):
            snaps = {}
            for N in widths:
                snaps[str(N)] = snapshot_trajectory(
                    N, "mup", lr0, d, seed, X, y, n_hidden, module, Ks,
                )
            loss_by_K = {
                str(N): {str(k): loss_at(snaps[str(N)], k) for k in Ks} for N in widths
            }
            ratios = {str(k): ratio_at(snaps["64"], snaps["300"], k) for k in Ks}
            per_depth[depth] = {
                "loss_by_K": loss_by_K,
                "ratio_by_K": ratios,
                "shape": shape_upto_128(snaps["64"], snaps["300"], Ks),
                "rescue": rescue_512(snaps["64"], snaps["300"]),
                "weight_change_W_at_128": {
                    str(N): next(s["weight_change_W"] for s in snaps[str(N)] if s["K"] == 128)
                    for N in widths
                },
            }
        rows.append({"seed": seed, "L2": per_depth["L2"], "L3": per_depth["L3"]})

    if not prefix_ok(rows, a28):
        out = {
            "generated_by": "scripts/run_mup_l3_mup_traj.py",
            "precommit": str(PRE.relative_to(ROOT)),
            "applied_reading": "prefix_mismatch",
            "reading": pre["readings_committed_before_seeing_numbers"]["prefix_mismatch"],
        }
        OUT.write_text(json.dumps(out, indent=2))
        print(json.dumps({"applied_reading": "prefix_mismatch"}, indent=2))
        return

    cohort = [r for r in rows if r["L3"]["ratio_by_K"]["128"] > 1]
    shape_counts = Counter(r["L3"]["shape"] for r in cohort)
    rescue_counts = Counter(r["L3"]["rescue"] for r in cohort)
    shape_key = apply_shape(shape_counts)
    rescue_key = apply_rescue(rescue_counts)
    out = {
        "generated_by": "scripts/run_mup_l3_mup_traj.py",
        "precommit": str(PRE.relative_to(ROOT)),
        "governs_stream_arms": False,
        "signed_finding": False,
        "does_not_reopen_a28": True,
        "prefix_K128_matches_a28": True,
        "cohort_n": len(cohort),
        "cohort_seeds": [r["seed"] for r in cohort],
        "L3_cohort_shape_counts": dict(shape_counts),
        "L3_cohort_rescue_counts": dict(rescue_counts),
        "applied_reading": shape_key,
        "reading": pre["readings_committed_before_seeing_numbers"][shape_key],
        "k512": {
            "applied": rescue_key,
            "note": pre["k512_recorded"],
        },
        "L2_fail_shape_counts": dict(Counter(r["L2"]["shape"] for r in rows if r["L2"]["ratio_by_K"]["128"] > 1)),
        "per_seed": [
            {
                "seed": r["seed"],
                "L2_shape": r["L2"]["shape"],
                "L3_shape": r["L3"]["shape"],
                "L3_rescue": r["L3"]["rescue"],
                "L2_ratio_by_K": r["L2"]["ratio_by_K"],
                "L3_ratio_by_K": r["L3"]["ratio_by_K"],
            }
            for r in rows
        ],
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps({
        "cohort_n": out["cohort_n"],
        "cohort_seeds": out["cohort_seeds"],
        "L3_cohort_shape_counts": out["L3_cohort_shape_counts"],
        "L3_cohort_rescue_counts": out["L3_cohort_rescue_counts"],
        "applied_reading": shape_key,
        "k512_applied": rescue_key,
        "L2_fail_shape_counts": out["L2_fail_shape_counts"],
        "reading": out["reading"],
    }, indent=2))


if __name__ == "__main__":
    main()
