"""Corrected-LR vs quadratic, against `results/corrected_lr_precommit.json`.

Does not train. Writes `results/corrected_lr.json`.

Cell: task 0, lag 12, module A. Unique n=8, old arrangements (legacy_seed 0–7).
Composition is Figure 2's three-corner pooled shares (S-HL, S-LH, S-LL) plus the
signed utility term. γ = 1 is the negative control: the two laws are identical.

    python scripts/analyse_corrected_lr.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from audit_unique_n import unique_recs  # noqa: E402
from fig2_gamma_sweep import FORGETTING, MIN_FLOORS, collect  # noqa: E402
from src.analysis import grid as G  # noqa: E402
from src.analysis.attribution import FACTORS  # noqa: E402
from src.models.parameterization import ScalingConfig  # noqa: E402
from src.reservations import reserved_stream_ids  # noqa: E402

PRECOMMIT = ROOT / "results" / "corrected_lr_precommit.json"
ARMS = ROOT / "results" / "corrected_lr"
OUT = ROOT / "results" / "corrected_lr.json"
TASK, LAG, MODULE = 0, 12, "A"
GAMMAS = (0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
CONDITIONS = ("S-HH", "S-HL", "S-LH", "S-LL")
CUT = 1.0
ALPHA = 0.05
TARGET_LOSS = 0.05
N, D, LR0 = 300, 150, 5.0


def _key(spec: dict) -> tuple:
    return (float(spec["gamma_0"]), spec["condition"], int(spec["seed"]))


def _miss_rate(paths: list[Path]) -> dict:
    n = len(paths)
    n_arm_miss, n_task_miss = 0, 0
    n_tasks = 0
    by_g: dict[float, dict] = {g: {"n": 0, "n_miss": 0} for g in GAMMAS}
    for p in paths:
        rec = json.loads(p.read_text())
        g = float(rec["spec"]["gamma_0"])
        by_g[g]["n"] += 1
        tasks = rec.get("tasks") or []
        n_tasks += len(tasks)
        miss = False
        for t in tasks:
            n_task_miss += int(
                (not t.get("converged", True))
                or (t.get("final_loss") is not None and t["final_loss"] > TARGET_LOSS + 1e-12)
            )
            if (not t.get("converged", True)) or (
                t.get("final_loss") is not None and t["final_loss"] > TARGET_LOSS + 1e-12
            ):
                miss = True
        if miss or not rec.get("usable", False):
            n_arm_miss += 1
            by_g[g]["n_miss"] += 1
    return {
        "n_files": n,
        "n_arms_missed": n_arm_miss,
        "arm_miss_rate": n_arm_miss / n if n else None,
        "n_tasks": n_tasks,
        "n_tasks_missed": n_task_miss,
        "by_gamma": {f"{g:g}": v for g, v in by_g.items()},
        "rule": "an arm that misses target_loss on any task is unusable for composition "
                "and is counted here, not dropped silently",
    }


def _rows(recs: list[dict]) -> dict[tuple, dict]:
    out = {}
    for spec, mod, task, lag, att in G.attributions(recs, module=MODULE):
        if task != TASK or lag != LAG:
            continue
        k = _key(spec)
        out[k] = {
            "gamma": k[0], "condition": k[1], "seed": k[2],
            "dlog_alpha": att.dlog_alpha,
            "floors": G.floors(att.dlog_alpha, "alpha"),
            "utility_term": att.terms["utility"],
            "radius_term": att.terms["radius"],
            "dimension_term": att.terms["dimension"],
            "utility_share": att.shares["utility"],
            "radius_share": att.shares["radius"],
            "dimension_share": att.shares["dimension"],
            "gated": abs(G.floors(att.dlog_alpha, "alpha")) >= MIN_FLOORS,
        }
    return out


def _steps(recs: list[dict]) -> dict[tuple, float]:
    out = {}
    for r in recs:
        k = _key(r["spec"])
        tasks = r.get("tasks") or []
        out[k] = float(np.mean([t["steps_taken"] for t in tasks])) if tasks else float("nan")
    return out


def _ttest_rel(a, b) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if np.allclose(a, b):
        return 1.0
    return float(stats.ttest_rel(a, b).pvalue)


def _paired(quad: dict, corr: dict, gammas, conditions) -> dict:
    diffs = defaultdict(list)
    for g in gammas:
        for c in conditions:
            for seed in range(8):
                k = (g, c, seed)
                if k not in quad or k not in corr:
                    continue
                q, o = quad[k], corr[k]
                diffs[g].append({
                    "condition": c, "seed": seed,
                    "d_utility_term": o["utility_term"] - q["utility_term"],
                    "d_utility_share": o["utility_share"] - q["utility_share"],
                    "d_dlog_alpha": o["dlog_alpha"] - q["dlog_alpha"],
                    "d_floors": o["floors"] - q["floors"],
                    "quad_utility_share": q["utility_share"],
                    "corr_utility_share": o["utility_share"],
                })
    out = {}
    for g, rows in diffs.items():
        ut = np.array([r["d_utility_term"] for r in rows])
        us = np.array([r["d_utility_share"] for r in rows])
        da = np.array([r["d_dlog_alpha"] for r in rows])
        t_ut = _ttest_rel(
            [corr[(g, r["condition"], r["seed"])]["utility_term"] for r in rows],
            [quad[(g, r["condition"], r["seed"])]["utility_term"] for r in rows],
        )
        t_us = _ttest_rel(
            [corr[(g, r["condition"], r["seed"])]["utility_share"] for r in rows],
            [quad[(g, r["condition"], r["seed"])]["utility_share"] for r in rows],
        )
        out[g] = {
            "n": len(rows),
            "d_utility_term_mean": float(ut.mean()),
            "d_utility_term_sem": float(ut.std(ddof=1) / np.sqrt(len(ut))),
            "d_utility_share_mean": float(us.mean()),
            "d_utility_share_sem": float(us.std(ddof=1) / np.sqrt(len(us))),
            "d_dlog_alpha_mean": float(da.mean()),
            "d_dlog_alpha_sem": float(da.std(ddof=1) / np.sqrt(len(da))),
            "p_utility_term": t_ut,
            "p_utility_share": t_us,
            "ci95_utility_term": [float(ut.mean() - 1.96 * ut.std(ddof=1) / np.sqrt(len(ut))),
                                  float(ut.mean() + 1.96 * ut.std(ddof=1) / np.sqrt(len(ut)))],
            "ci95_utility_share": [float(us.mean() - 1.96 * us.std(ddof=1) / np.sqrt(len(us))),
                                   float(us.mean() + 1.96 * us.std(ddof=1) / np.sqrt(len(us)))],
        }
    return out


def _pool(recs: list[dict], conditions: tuple[str, ...]) -> dict:
    data = collect(recs, conditions, LAG, only_task=TASK)
    return {f"{g:g}": {
        "n": v["n"],
        "dlog_alpha": v["dlog_alpha"],
        "floors_signed": v["floors_signed"],
        "shares": {k: v["shares"][k] for k in FACTORS},
        "terms": v["terms"],
        "sign_positive_fraction": v["sign_positive_fraction"],
    } for g, v in data.items()}


def _lr(gamma: float, scaling: str) -> float:
    return ScalingConfig(N=N, d=D, gamma_0=gamma, lr0=LR0, lr_scaling=scaling).lr


def _changed(cell: dict, ctrl: dict) -> bool:
    """A γ differs from quadratic beyond the γ=1 re-run envelope."""
    if cell["n"] == 0:
        return False
    lo, hi = cell["ci95_utility_term"]
    term_excludes_0 = (hi < 0) or (lo > 0)
    share_beyond_ctrl = abs(cell["d_utility_share_mean"]) > (
        abs(ctrl["d_utility_share_mean"]) + 2 * cell["d_utility_share_sem"]
        + 2 * ctrl["d_utility_share_sem"]
    )
    return bool(term_excludes_0 and share_beyond_ctrl)


def _reading(miss: dict, paired_three: dict) -> dict:
    pre = json.loads(PRECOMMIT.read_text())["readings_committed_before_seeing_numbers"]
    ctrl = paired_three[1.0]
    lazy = any(_changed(paired_three[g], ctrl) for g in (0.03, 0.1, 0.3))
    rich = any(_changed(paired_three[g], ctrl) for g in (3.0, 10.0))
    ctrl_bad = (ctrl["ci95_utility_term"][1] < 0) or (ctrl["ci95_utility_term"][0] > 0)
    high_miss = miss["arm_miss_rate"] is not None and miss["arm_miss_rate"] >= 0.25

    if ctrl_bad:
        key = "composition_changes_at_gamma_eq_1"
        primary = None
        stop = True
    elif lazy and rich:
        key = "composition_changes_throughout_including_lazy"
        primary = None
        stop = False
    elif rich and not lazy:
        key = "composition_changes_at_gamma_gt_1_only"
        primary = "corrected"
        stop = False
    else:
        key = "composition_unchanged_at_gamma_gt_1"
        primary = "quadratic"
        stop = False

    return {
        "key": key,
        "text": pre[key],
        "primary_law_for_subsequent_axes": primary,
        "stop_and_diagnose": stop,
        "high_miss_rate": high_miss,
        "high_miss_text": pre["high_miss_rate_at_rich_gamma"] if high_miss else None,
        "lazy_changed": lazy,
        "rich_changed": rich,
        "gamma_1_ci_excludes_0": ctrl_bad,
        "operationalization": (
            "γ changed if the 95% CI of the paired (corrected − quadratic) signed "
            "utility term excludes 0 AND |Δ utility share| exceeds the γ=1 |Δ share| "
            "plus two SEMs of each. Three-corner pool, unique n=8, task 0 lag 12. "
            "γ=1 is the re-run envelope: the laws are identical there."
        ),
    }


def main() -> None:
    pre = json.loads(PRECOMMIT.read_text())
    if not pre.get("written_before_running"):
        sys.exit("pre-commit missing")

    paths = sorted(ARMS.glob("*.json"))
    if len(paths) != 192:
        sys.exit(f"expected 192 corrected-LR files, got {len(paths)}")
    reserved = reserved_stream_ids()
    for p in paths:
        sid = json.loads(p.read_text())["spec"]["stream_id"]
        if sid in reserved:
            sys.exit(f"reserved stream id {sid} in {p.name}")

    miss = _miss_rate(paths)
    corr_recs = unique_recs(G.load(arms_dir=ARMS))
    quad_recs = unique_recs(G.load())
    if len(corr_recs) != 192 or len(quad_recs) != 192:
        sys.exit(f"unique n mismatch: corrected {len(corr_recs)}, quadratic {len(quad_recs)}")

    corr_rows, quad_rows = _rows(corr_recs), _rows(quad_recs)
    missing = [k for k in quad_rows if k not in corr_rows]
    if missing:
        sys.exit(f"unpaired keys: {missing[:5]}…")

    three_q = [r for r in quad_recs if r["spec"]["condition"] in FORGETTING]
    three_c = [r for r in corr_recs if r["spec"]["condition"] in FORGETTING]
    paired_three = _paired(quad_rows, corr_rows, GAMMAS, FORGETTING)
    paired_four = _paired(quad_rows, corr_rows, GAMMAS, CONDITIONS)

    steps_q, steps_c = _steps(quad_recs), _steps(corr_recs)
    steps_by_g = {}
    for g in GAMMAS:
        qs = [steps_q[k] for k in steps_q if k[0] == g]
        cs = [steps_c[k] for k in steps_c if k[0] == g]
        steps_by_g[f"{g:g}"] = {
            "quadratic_mean": float(np.mean(qs)),
            "corrected_mean": float(np.mean(cs)),
            "ratio": float(np.mean(cs) / np.mean(qs)) if np.mean(qs) else None,
            "lr_quadratic": _lr(g, "quadratic"),
            "lr_corrected": _lr(g, "corrected"),
            "lr_ratio_quad_over_corr": _lr(g, "quadratic") / _lr(g, "corrected"),
        }

    reading = _reading(miss, paired_three)
    q10 = _pool(three_q, FORGETTING)["10"]
    c10 = _pool(three_c, FORGETTING)["10"]
    q3 = _pool(three_q, FORGETTING)["3"]
    c3 = _pool(three_c, FORGETTING)["3"]
    out = {
        "generated_by": "scripts/analyse_corrected_lr.py",
        "precommit": str(PRECOMMIT.relative_to(ROOT)),
        "question": pre["question"],
        "cell": {"task": TASK, "lag": LAG, "module": MODULE, "unique_n": 8},
        "population": "old arrangements, legacy_seed 0–7; not the reserved set",
        "n_corrected": len(corr_recs),
        "n_quadratic": len(quad_recs),
        "misses": miss,
        "lr_note": (
            "At L=2, corrected is η ∝ γ¹ against quadratic η ∝ γ², so the rich-regime "
            "ratio is γ, not γ². At γ=10 that is 10× smaller, not 100× (A4's 100× is a slip)."
        ),
        "steps_and_lr": steps_by_g,
        "composition_at_gamma_gt_1": {
            "3": {
                "utility_share_quadratic": q3["shares"]["utility"],
                "utility_share_corrected": c3["shares"]["utility"],
                "floors_quadratic": q3["floors_signed"],
                "floors_corrected": c3["floors_signed"],
            },
            "10": {
                "utility_share_quadratic": q10["shares"]["utility"],
                "utility_share_corrected": c10["shares"]["utility"],
                "floors_quadratic": q10["floors_signed"],
                "floors_corrected": c10["floors_signed"],
            },
            "paper_gamma_step_utility_share_1_to_10": (
                q10["shares"]["utility"] - _pool(three_q, FORGETTING)["1"]["shares"]["utility"]
            ),
        },
        "three_corner_pooled": {
            "quadratic": _pool(three_q, FORGETTING),
            "corrected": _pool(three_c, FORGETTING),
        },
        "paired_corrected_minus_quadratic": {
            "three_corner": {f"{g:g}": paired_three[g] for g in GAMMAS},
            "four_corner": {f"{g:g}": paired_four[g] for g in GAMMAS},
        },
        "negative_control_gamma_1": {
            "rule": pre["negative_control"]["rule"],
            "three_corner": paired_three[1.0],
        },
        "reading": reading,
    }
    OUT.write_text(json.dumps(out, indent=2, allow_nan=False))
    print(json.dumps({
        "n": len(corr_recs),
        "arm_miss_rate": miss["arm_miss_rate"],
        "gamma_1_d_share": paired_three[1.0]["d_utility_share_mean"],
        "gamma_1_p_term": paired_three[1.0]["p_utility_term"],
        "gamma_10_d_share": paired_three[10.0]["d_utility_share_mean"],
        "gamma_10_p_term": paired_three[10.0]["p_utility_term"],
        "reading": reading["key"],
        "primary_law": reading["primary_law_for_subsequent_axes"],
        "wrote": str(OUT),
    }, indent=2))


if __name__ == "__main__":
    main()
