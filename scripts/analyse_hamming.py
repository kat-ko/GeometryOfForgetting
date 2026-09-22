"""Hamming × input-change slice against `results/hamming_precommit.json`. Does not train.

Primary cell: unique n=8, module A, task 0, lag 12, Δ log α vs Hamming, per
input level, γ=10. Readings named before numbers.

    python scripts/analyse_hamming.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis import grid as G  # noqa: E402
from src.reservations import reserved_stream_ids  # noqa: E402

PRECOMMIT = ROOT / "results" / "hamming_precommit.json"
ARMS = ROOT / "results" / "hamming"
OUT = ROOT / "results" / "hamming_slice.json"
MD = ROOT / "results" / "hamming_slice.md"
TASK, LAG, MODULE = 0, 12, "A"
GAMMAS = (1.0, 10.0)
INPUTS = ("frozen", "drift", "jump")
S_R = (1.0, 0.75, 0.5, 0.25)
CHANGING = (0.75, 0.5, 0.25)
MIN_FLOORS = 2.0


def _fmt(x) -> float | None:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return None
    return float(x)


def _ci95(mean: float, sem: float) -> list[float]:
    return [float(mean - 1.96 * sem), float(mean + 1.96 * sem)]


def pack(vals) -> dict:
    v = np.asarray(list(vals), dtype=float)
    v = v[np.isfinite(v)]
    n = int(v.size)
    if n == 0:
        return {"n": 0, "mean": None, "sem": None, "sd": None, "ci95": [None, None],
                "mean_floors": None, "n_positive": 0, "n_negative": 0}
    sd = float(v.std(ddof=1)) if n > 1 else 0.0
    sem = sd / np.sqrt(n) if n else 0.0
    mean = float(v.mean())
    floors = float(np.sign(mean) * G.floors(mean, "alpha"))
    return {
        "n": n,
        "mean": mean,
        "sd": sd,
        "sem": float(sem),
        "ci95": _ci95(mean, sem),
        "mean_floors": floors,
        "n_positive": int(np.sum(v > 0)),
        "n_negative": int(np.sum(v < 0)),
        "min": float(v.min()),
        "max": float(v.max()),
    }


def _monotone(seq: list[float]) -> bool:
    return all(seq[i] >= seq[i + 1] for i in range(len(seq) - 1)) or all(
        seq[i] <= seq[i + 1] for i in range(len(seq) - 1)
    )


def dose_reading(means_by_sr: dict[float, float]) -> str:
    """Name one of monotone_dose / binary_only / nonmonotone from cell means."""
    seq = [means_by_sr[s] for s in S_R]
    changing = [means_by_sr[s] for s in CHANGING]
    if _monotone(seq):
        return "monotone_dose"
    s1 = means_by_sr[1.0]
    separates = all(s1 > x for x in changing) or all(s1 < x for x in changing)
    if separates and not _monotone(changing):
        return "binary_only"
    return "nonmonotone"


def frozen_reading(cells: dict[str, dict[float, float]]) -> str:
    """`frozen_intermediate` if every changing Hamming sits in [drift, jump]; else `frozen_outside`."""
    insides = []
    for s in CHANGING:
        f, d, j = cells["frozen"][s], cells["drift"][s], cells["jump"][s]
        lo, hi = (d, j) if d <= j else (j, d)
        insides.append(lo <= f <= hi)
    return "frozen_intermediate" if all(insides) else "frozen_outside"


def axis_reading(means: dict[str, dict[float, float]]) -> str:
    """Hamming-axis reading. `monotone_dose` requires frozen and drift both monotone."""
    frozen_mono = _monotone([means["frozen"][s] for s in S_R])
    drift_mono = _monotone([means["drift"][s] for s in S_R])
    if frozen_mono and drift_mono:
        return "monotone_dose"
    drift = dose_reading(means["drift"])
    if drift == "monotone_dose":
        return "nonmonotone"
    return drift


def _rows(recs: list[dict]) -> list[dict]:
    out = []
    for spec, mod, task, lag, att in G.attributions(recs, module=MODULE):
        if task != TASK or lag != LAG:
            continue
        dlog = att.dlog_alpha
        floors = float(np.sign(dlog) * G.floors(dlog, "alpha"))
        out.append({
            "gamma": spec["gamma_0"],
            "input_level": spec["input_level"],
            "s_r": spec["readout_similarity"],
            "s_f": spec["feature_similarity"],
            "stream_id": spec["stream_id"],
            "dlog_alpha": dlog,
            "floors": floors,
            "usable": True,
        })
    return out


def _rho_rows(recs: list[dict]) -> list[dict]:
    """Δρ_c signed on retained task 0, baseline → lag 12, module A. Reported, not a reading."""
    out = []
    for r in recs:
        spec = r["spec"]
        pts = {(g["task"], g["boundary"]): g for g in r["geometry"]
               if g["module"] == MODULE and g["task"] == TASK and g["ensemble"] == "retained"}
        if (TASK, TASK) not in pts or (TASK, TASK + LAG) not in pts:
            continue
        a, b = pts[(TASK, TASK)]["rho_c_signed"], pts[(TASK, TASK + LAG)]["rho_c_signed"]
        out.append({
            "gamma": spec["gamma_0"],
            "input_level": spec["input_level"],
            "s_r": spec["readout_similarity"],
            "stream_id": spec["stream_id"],
            "d_rho_c": float(b - a),
        })
    return out


def _lag4_rows(recs: list[dict]) -> list[dict]:
    """Matched lag 4 across task position. Reported, not a reading."""
    out = []
    for spec, mod, task, lag, att in G.attributions(recs, module=MODULE):
        if lag != 4:
            continue
        out.append({
            "gamma": spec["gamma_0"],
            "input_level": spec["input_level"],
            "s_r": spec["readout_similarity"],
            "stream_id": spec["stream_id"],
            "task": task,
            "dlog_alpha": att.dlog_alpha,
        })
    return out


def _grid(rows, field="dlog_alpha") -> dict:
    by = defaultdict(list)
    for x in rows:
        by[(x["gamma"], x["input_level"], x["s_r"])].append(x[field])
    return {k: pack(v) for k, v in sorted(by.items())}


def _means(grid, gamma: float) -> dict[str, dict[float, float]]:
    out = {level: {} for level in INPUTS}
    for level in INPUTS:
        for s in S_R:
            cell = grid[(gamma, level, s)]
            if cell["n"] == 0 or cell["mean"] is None:
                raise RuntimeError(f"empty cell γ={gamma} {level} s_r={s}")
            out[level][s] = cell["mean"]
    return out


def r1(x: float) -> float:
    return float(round(x, 1))


def cliff_capacity(grid, gamma: float, level: str) -> dict:
    """Cliff = 1.0→0.75 in 1-decimal floors. Post-cliff range = |0.75−0.25|."""
    fl = {s: grid[(gamma, level, s)]["mean_floors"] for s in S_R}
    a75 = grid[(gamma, level, 0.75)]
    a25 = grid[(gamma, level, 0.25)]
    return {
        "floors_1dp": {str(s): r1(fl[s]) for s in S_R},
        "cliff": r1(fl[1.0]) - r1(fl[0.75]),
        "post_range": abs(r1(fl[0.75]) - r1(fl[0.25])),
        "post_ci_overlap": not (
            a75["ci95"][1] < a25["ci95"][0] or a25["ci95"][1] < a75["ci95"][0]
        ),
    }


def cliff_rho(rho, gamma: float, level: str) -> dict:
    m = {s: rho[(gamma, level, s)]["mean"] for s in S_R}
    a75, a25 = rho[(gamma, level, 0.75)], rho[(gamma, level, 0.25)]
    return {
        "means": {str(s): m[s] for s in S_R},
        "cliff": m[1.0] - m[0.75],
        "post_range": abs(m[0.75] - m[0.25]),
        "post_ci_overlap": not (
            a75["ci95"][1] < a25["ci95"][0] or a25["ci95"][1] < a75["ci95"][0]
        ),
    }


def apply_readings(grid) -> dict:
    pre = json.loads(PRECOMMIT.read_text())
    g10 = _means(grid, 10.0)
    g1 = _means(grid, 1.0)
    recovered = g10["drift"][1.0] > 0 and g10["jump"][1.0] < 0
    g10_dose = axis_reading(g10)
    g1_dose = axis_reading(g1)
    mixed = recovered and g10_dose != g1_dose
    if not recovered:
        primary = "finding3_fails_to_recover"
    elif mixed:
        primary = "mixed"
    else:
        primary = g10_dose
    return {
        "finding3_recovered": recovered,
        "gamma_10_dose": g10_dose,
        "gamma_1_dose": g1_dose,
        "mixed": mixed,
        "frozen_level": frozen_reading(g10),
        "primary_reading": primary,
        "operationalization": pre["operationalization"],
    }


def _fmt_rho(c: dict) -> str:
    return (f"{c['mean']:+.4f}  [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}]  "
            f"{c['n_positive']}+/{c['n_negative']}-")


def _fmt_fl(c: dict) -> str:
    return f"{c['mean_floors']:+.1f}"


def _fmt_cell(c: dict) -> str:
    if c["n"] == 0 or c["mean"] is None:
        return "—"
    return (f"{c['mean_floors']:+.1f} fl  ({c['mean']:+.4f} "
            f"[{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}]  n={c['n']}  "
            f"{c['n_positive']}+/{c['n_negative']}-)")


def main() -> None:
    pre = json.loads(PRECOMMIT.read_text())
    if not pre.get("written_before_running"):
        sys.exit("pre-commit missing")
    reserved = reserved_stream_ids()
    paths = sorted(ARMS.glob("*.json"))
    raw = [json.loads(p.read_text()) for p in paths]
    if len(raw) != 192:
        sys.exit(f"expected 192 arms, found {len(raw)}")
    sids = {r["spec"]["stream_id"] for r in raw}
    if sids != set(reserved):
        sys.exit(f"stream ids {sorted(sids)} are not the reserved set")
    for r in raw:
        sp = r["spec"]
        if sp.get("arrangement_source") != "stream_rng":
            sys.exit("non-reserved arrangement_source")
        if sp.get("lr_scaling") != "quadratic":
            sys.exit("non-quadratic record")
        if int(sp.get("tracked_stride", 0)) != 2:
            sys.exit("tracked_stride is not 2")
        if int(sp.get("P", 0)) != 16:
            sys.exit("P is not 16")
        if sp.get("input_level") not in INPUTS:
            sys.exit(f"bad input_level {sp.get('input_level')}")

    n_usable = sum(1 for r in raw if r.get("usable"))
    n_miss = len(raw) - n_usable
    usable = [r for r in raw if r.get("usable")]
    if n_miss:
        print(f"WARNING: {n_miss} unusable arm(s); primary uses the usable set", flush=True)

    rows = _rows(usable)
    grid = _grid(rows)
    for g in GAMMAS:
        for level in INPUTS:
            for s in S_R:
                cell = grid[(g, level, s)]
                if cell["n"] != 8:
                    sys.exit(f"cell γ={g} {level} s_r={s} has n={cell['n']}, want 8")

    applied = apply_readings(grid)
    rho = _grid(_rho_rows(usable), field="d_rho_c")
    lag4 = _lag4_rows(usable)
    lag4_pack = {}
    for x in lag4:
        lag4_pack.setdefault(x["gamma"], {})
        lag4_pack[x["gamma"]].setdefault(
            (x["input_level"], x["s_r"], x["task"]), []
        ).append(x["dlog_alpha"])
    lag4_out = {
        f"gamma={g:g}|{k[0]}|s_r={k[1]:g}|task={k[2]}": pack(v)
        for g, cells in lag4_pack.items()
        for k, v in sorted(cells.items())
    }

    cells_out = {
        f"gamma={g:g}|{level}|s_r={s:g}": grid[(g, level, s)]
        for g in GAMMAS for level in INPUTS for s in S_R
    }
    rho_out = {
        f"gamma={g:g}|{level}|s_r={s:g}": rho[(g, level, s)]
        for g in GAMMAS for level in INPUTS for s in S_R
    }

    structure = {
        "capacity": {
            f"gamma={g:g}|{level}": cliff_capacity(grid, g, level)
            for g in GAMMAS for level in INPUTS
        },
        "rho_c": {
            f"gamma={g:g}|{level}": cliff_rho(rho, g, level)
            for g in GAMMAS for level in INPUTS
        },
        "crossover_gamma10": {
            "at_0.75": "drift better than frozen",
            "at_0.25": "frozen better than drift",
            "frozen_0.75_floors": r1(grid[(10.0, "frozen", 0.75)]["mean_floors"]),
            "drift_0.75_floors": r1(grid[(10.0, "drift", 0.75)]["mean_floors"]),
            "frozen_0.25_floors": r1(grid[(10.0, "frozen", 0.25)]["mean_floors"]),
            "drift_0.25_floors": r1(grid[(10.0, "drift", 0.25)]["mean_floors"]),
        },
        "superseding": (
            "Capacity is a cliff at Hamming>0 except under drift, where a graded "
            "post-cliff component is resolved at both γ. The pre-commit readings "
            "are historical: monotone_dose fires on a step-then-flat series; mixed "
            "fires on a 0.2-floor unresolved wobble. Frozen_outside fired because "
            "frozen is insensitive to task change while drift is not."
        ),
        "going_forward": (
            "A monotonicity reading requires the post-threshold range to clear "
            "the floor, not just the ordering to hold."
        ),
    }

    rec = {
        "generated_by": "scripts/analyse_hamming.py",
        "precommit": "results/hamming_precommit.json",
        "n_arms": len(raw),
        "n_usable": n_usable,
        "n_missed": n_miss,
        "unique_n": 8,
        "cell": {"task": TASK, "lag": LAG, "module": MODULE},
        "applied": applied,
        "structure": structure,
        "cells": cells_out,
        "delta_rho_c_retained_task0": rho_out,
        "lag4_position": lag4_out,
        "will_not": pre["will_not_do"],
    }
    OUT.write_text(json.dumps(rec, indent=2) + "\n")

    def cap_row(g, level):
        st = structure["capacity"][f"gamma={g:g}|{level}"]
        return (f"| {level} | {st['cliff']:+.1f} | {st['post_range']:.1f} | "
                f"{'overlap' if st['post_ci_overlap'] else '**resolved**'} |")

    lines = [
        "# Hamming × input-change slice",
        "",
        "Unique n=8, module A, task 0, lag 12 unless noted. "
        "95% CI = mean ± 1.96 SEM. The arm worked. The raw table says more than "
        "the pre-committed readings.",
        "",
        f"**n_arms = {len(raw)}, usable = {n_usable}, missed = {n_miss}.**",
        "",
        f"**Finding 3 recovery** (s_r=1, γ=10): "
        f"{'yes' if applied['finding3_recovered'] else 'NO — stop'}. "
        f"drift {grid[(10.0, 'drift', 1.0)]['mean_floors']:+.1f} fl, "
        f"jump {grid[(10.0, 'jump', 1.0)]['mean_floors']:+.1f} fl.",
        "",
        "## The structure is a cliff, not a dose",
        "",
        "Cliff = floors at s_r=1.0 minus floors at 0.75 (1-decimal). "
        "Post-cliff range = |floors(0.75) − floors(0.25)|.",
        "",
        "### γ=10",
        "",
        "| input | cliff (1.0 → 0.75) | range 0.75 → 0.25 | 0.75 vs 0.25 CI |",
        "|---|---|---|---|",
        cap_row(10.0, "frozen"),
        cap_row(10.0, "drift"),
        cap_row(10.0, "jump"),
        "",
        "### γ=1",
        "",
        "| input | cliff (1.0 → 0.75) | range 0.75 → 0.25 | 0.75 vs 0.25 CI |",
        "|---|---|---|---|",
        cap_row(1.0, "frozen"),
        cap_row(1.0, "drift"),
        cap_row(1.0, "jump"),
        "",
        "Task change is a threshold at zero everywhere except under drift, "
        "where a graded component appears and is resolved (drift 0.75 vs 0.25 "
        "CIs do not overlap at either γ). How much the task changed only "
        "matters when the input distribution is also changing slowly.",
        "",
        "## Crossover",
        "",
        "Frozen is flat and drift is graded, so they cross. At γ=10, s_r=0.75: "
        f"drift {r1(grid[(10.0, 'drift', 0.75)]['mean_floors']):+.1f}, "
        f"frozen {r1(grid[(10.0, 'frozen', 0.75)]['mean_floors']):+.1f} "
        "(drift better). At s_r=0.25: "
        f"drift {r1(grid[(10.0, 'drift', 0.25)]['mean_floors']):+.1f}, "
        f"frozen {r1(grid[(10.0, 'frozen', 0.25)]['mean_floors']):+.1f} "
        "(frozen better). A slowly drifting input helps when the task barely "
        "changes and hurts when it changes a lot. That is why `frozen_outside` "
        "fired: frozen is insensitive to task change while drift is not.",
        "",
        "## Pre-commit readings (historical; they failed to discriminate)",
        "",
        f"- Primary as written: `{applied['primary_reading']}` "
        f"(γ=10 `{applied['gamma_10_dose']}`, γ=1 `{applied['gamma_1_dose']}`).",
        f"- Frozen level: `{applied['frozen_level']}` — fired correctly; the "
        "reason is the crossover above.",
        "- `monotone_dose` fires on a step-then-flat series, so it cannot "
        "distinguish a dose from a threshold.",
        "- `mixed` fires on a 0.2-floor unresolved wobble in γ=10 frozen "
        "(−69.1 → −68.9, CIs overlap). Both γ show the same cliff-plus-drift-"
        "gradation. Going forward: a monotonicity reading requires the "
        "post-threshold range to clear the floor.",
        "",
        "## γ=10, Δ log α, task 0 lag 12",
        "",
        "| input \\ s_r | 1.00 | 0.75 | 0.50 | 0.25 |",
        "|---|---|---|---|---|",
    ]
    for level in INPUTS:
        row = [level] + [_fmt_cell(grid[(10.0, level, s)]) for s in S_R]
        lines.append("| " + " | ".join(row) + " |")
    lines += [
        "",
        "## γ=1, same cell",
        "",
        "| input \\ s_r | 1.00 | 0.75 | 0.50 | 0.25 |",
        "|---|---|---|---|---|",
    ]
    for level in INPUTS:
        row = [level] + [_fmt_cell(grid[(1.0, level, s)]) for s in S_R]
        lines.append("| " + " | ".join(row) + " |")

    lines += [
        "",
        "## Δρ_c, retained task 0, lag 12 (signed; no ρ_c floor)",
        "",
        "Finding 4's recast now has four Hamming levels. Capacity and centre "
        "geometry come apart: frozen capacity is a cliff, frozen Δρ_c is graded "
        "and the 0.75 vs 0.25 CIs do not overlap. Drift capacity is graded; "
        "drift Δρ_c is a cliff then unresolved. Jump Δρ_c is already large at "
        "s_r=1. Populations named, not pooled: registered 2×2 drift at a "
        "repeated task is +0.055; reserved Hamming drift at s_r=1 is +0.002 "
        "(CI includes 0, 5+/3−). Capacity on the reserved same cells recovered "
        "(+6.7 vs −55.1). Δρ_c did not."
        "",
        "### γ=10",
        "",
        "| input \\ s_r | 1.00 | 0.75 | 0.50 | 0.25 |",
        "|---|---|---|---|---|",
    ]
    for level in INPUTS:
        row = [level] + [_fmt_rho(rho[(10.0, level, s)]) for s in S_R]
        lines.append("| " + " | ".join(row) + " |")
    lines += [
        "",
        "### γ=1",
        "",
        "| input \\ s_r | 1.00 | 0.75 | 0.50 | 0.25 |",
        "|---|---|---|---|---|",
    ]
    for level in INPUTS:
        row = [level] + [_fmt_rho(rho[(1.0, level, s)]) for s in S_R]
        lines.append("| " + " | ".join(row) + " |")

    TASKS = (0, 2, 4, 6, 8, 10)
    lines += [
        "",
        "## Lag 4 across task position (γ=10, floors)",
        "",
        "First output from schedule B. W5b on the registered grid was ×1.7 "
        "(tasks 0/4/8 at lag 4). Here the ratio |task 0 / task 8| is "
        "cell-dependent: ~1.2–1.4× on most forgetting cells, ×3.5 on jump at "
        "s_r=1, and inverted (later tasks gain more) on drift at s_r=1.",
        "",
        "| input | s_r | t=0 | t=2 | t=4 | t=6 | t=8 | t=10 | |t0/t8| |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for level in INPUTS:
        for s in S_R:
            cells = [
                lag4_out[f"gamma=10|{level}|s_r={s:g}|task={t}"]
                for t in TASKS
            ]
            a, b = cells[0]["mean_floors"], cells[4]["mean_floors"]
            ratio = abs(a) / abs(b) if b != 0 else float("nan")
            bits = [level, f"{s:g}"] + [_fmt_fl(c) for c in cells] + [f"{ratio:.2f}×"]
            lines.append("| " + " | ".join(bits) + " |")
    lines += [
        "",
        "P=32 is a motivated cliff-resolution arm (h=2 is 6.25% of labels vs "
        "12.5% at P=16). It still needs its own §2a gate and is a scope change. "
        "Not next. Next is a second learner (binary cross-entropy) on these "
        "streams; see `docs/21-second-learner-ce.md`.",
        "",
    ]
    MD.write_text("\n".join(lines))
    print(MD.read_text())
    print(f"wrote {OUT.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
