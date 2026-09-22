"""Not-tuned reserved duplicate against `results/nottuned_precommit.json`. Does not train.

Cell: task 0, lag 12, module A. Unique n=8 reserved stream_ids, seed=0.
Composition object: γ=1→10 three-corner utility-share step.
Interaction object: signed utility, 6 γ × 4 conditions.

    python scripts/analyse_nottuned.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.analysis import anova  # noqa: E402
from src.analysis import grid as G  # noqa: E402
from src.reservations import reserved_stream_ids  # noqa: E402
from test_interaction_gamma_condition import (  # noqa: E402
    ALPHA,
    CONDITIONS,
    MIN_FLOORS,
    N_PERM,
    PERM_SEED,
    _licensed_gammas,
    _permute_F,
    _reading,
)

PRECOMMIT = ROOT / "results" / "nottuned_precommit.json"
ARMS = ROOT / "results" / "nottuned"
OUT = ROOT / "results" / "nottuned_slice.json"
CORRECTED = ROOT / "results" / "corrected_lr.json"
TASK, LAG, MODULE = 0, 12, "A"
GAMMAS = (0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
FORGETTING = ("S-HL", "S-LH", "S-LL")


def _rows(recs: list[dict]) -> list[dict]:
    out = []
    for spec, mod, task, lag, att in G.attributions(recs, module=MODULE):
        if task != TASK or lag != LAG:
            continue
        floors = G.floors(att.dlog_alpha, "alpha")
        out.append({
            "gamma": spec["gamma_0"],
            "condition": spec["condition"],
            "stream_id": spec["stream_id"],
            "dlog_alpha": att.dlog_alpha,
            "utility_term": att.terms["utility"],
            "utility_share": att.shares["utility"],
            "floors": floors,
            "gated": abs(floors) >= MIN_FLOORS,
        })
    return out


def _cell(rows, g, conditions) -> dict:
    xs = [x for x in rows if x["gamma"] == g and x["condition"] in conditions]
    by = defaultdict(list)
    for x in xs:
        by[x["stream_id"]].append(x)
    shares, terms, floors = [], [], []
    for sid, vs in by.items():
        shares.append(float(np.mean([v["utility_share"] for v in vs])))
        terms.append(float(np.mean([v["utility_term"] for v in vs])))
        floors.append(float(np.mean([v["floors"] for v in vs])))
    n = len(shares)
    if n == 0:
        return {"n": 0}
    return {
        "n": n,
        "utility_share_mean": float(np.mean(shares)),
        "utility_share_sem": float(np.std(shares, ddof=1) / np.sqrt(n)) if n > 1 else 0.0,
        "utility_term_mean": float(np.mean(terms)),
        "utility_term_sem": float(np.std(terms, ddof=1) / np.sqrt(n)) if n > 1 else 0.0,
        "floors_mean": float(np.mean(floors)),
        "n_gated": int(sum(abs(f) >= MIN_FLOORS for f in floors)),
    }


def _grid_aligned(rows, field, gammas, conditions, units) -> np.ndarray:
    """(n_γ, n_condition, n_unit) with the same stream_id on the last axis."""
    by = {(r["gamma"], r["condition"], r["stream_id"]): r[field] for r in rows}
    data = np.empty((len(gammas), len(conditions), len(units)), dtype=np.float64)
    for i, g in enumerate(gammas):
        for j, c in enumerate(conditions):
            for k, u in enumerate(units):
                data[i, j, k] = by[(g, c, u)]
    return data


def main() -> None:
    pre = json.loads(PRECOMMIT.read_text())
    reserved = reserved_stream_ids()
    units = tuple(sorted(reserved))
    paths = sorted(ARMS.glob("*.json"))
    raw = [json.loads(p.read_text()) for p in paths]
    sids = {r["spec"]["stream_id"] for r in raw}
    if sids != set(reserved):
        sys.exit(f"stream ids {sorted(sids)} are not the reserved set")
    if any(r["spec"].get("manifold_kind") != "spherical" for r in raw):
        sys.exit("non-spherical record in the not-tuned directory")
    if any("readout_rank" in r["spec"] for r in raw):
        sys.exit("rank field in the not-tuned directory; this duplicate is Q = I")
    if any(r["spec"].get("lr_scaling") != "quadratic" for r in raw):
        sys.exit("non-quadratic record in the not-tuned directory")

    miss_by = {}
    n_miss_arm = 0
    for r in raw:
        g = r["spec"]["gamma_0"]
        k = f"{g:g}"
        miss_by.setdefault(k, {"n": 0, "n_miss": 0})
        miss_by[k]["n"] += 1
        if not r.get("usable"):
            miss_by[k]["n_miss"] += 1
            n_miss_arm += 1

    recs = G.load(arms_dir=ARMS)
    rows = _rows(recs)
    n_expected = len(GAMMAS) * len(CONDITIONS) * len(units)
    if len(rows) != n_expected:
        sys.exit(f"expected {n_expected} unique rows; got {len(rows)}")

    cells = {}
    for g in GAMMAS:
        cells[f"{g:g}"] = {
            "three_corner": _cell(rows, g, FORGETTING),
            "all_four": _cell(rows, g, CONDITIONS),
        }
    step = (cells["10"]["three_corner"]["utility_share_mean"]
            - cells["1"]["three_corner"]["utility_share_mean"])
    old = json.loads(CORRECTED.read_text())
    old_step = old["composition_at_gamma_gt_1"]["paper_gamma_step_utility_share_1_to_10"]
    old_share_1 = old["three_corner_pooled"]["quadratic"]["1"]["shares"]["utility"]
    old_share_10 = old["three_corner_pooled"]["quadratic"]["10"]["shares"]["utility"]

    signed = _grid_aligned(rows, "utility_term", GAMMAS, CONDITIONS, units)
    signed_anova = anova.two_way_balanced(signed)
    rng = np.random.default_rng(PERM_SEED)
    n_perm_exceeded, p_perm = _permute_F(signed, signed_anova["F_interaction"], rng)

    licensed = _licensed_gammas(rows, list(GAMMAS), CONDITIONS)
    share_licensed = len(licensed) >= 2
    p_share = None
    if share_licensed:
        gated = [r for r in rows if r["gamma"] in licensed and r["gated"]]
        y = [r["utility_share"] for r in gated]
        row = [r["gamma"] for r in gated]
        col = [r["condition"] for r in gated]
        share_anova = anova.two_way_interaction_ols(y, row, col, licensed, CONDITIONS)
        p_share = share_anova["p_interaction"]

    op = pre["operationalization"]
    readings = pre["readings_committed_before_seeing_numbers"]
    comp_reproduces = step >= 0.5 * old_step
    comp_vanishes = abs(step) < 0.25 * old_step
    inter_reproduces = signed_anova["p_interaction"] < ALPHA
    if comp_reproduces and inter_reproduces:
        key = "headline_reproduces_on_reserved"
    elif comp_vanishes or not inter_reproduces:
        key = "headline_does_not_reproduce"
    else:
        key = "mixed"

    walls = [r.get("wall_seconds") for r in raw if r.get("wall_seconds") is not None]
    mtimes = [p.stat().st_mtime for p in paths]
    out = {
        "generated_by": "scripts/analyse_nottuned.py",
        "precommit": "results/nottuned_precommit.json",
        "n_files": len(raw),
        "n_usable": sum(1 for r in raw if r.get("usable")),
        "n_arms_missed": n_miss_arm,
        "misses": miss_by,
        "miss_interpretation": pre["misses"]["interpretation"],
        "cell": {"task": TASK, "lag": LAG, "module": MODULE,
                 "unique_n": 8, "unit": "reserved stream_id, seed=0"},
        "three_corner": FORGETTING,
        "cells": cells,
        "gamma_1_to_10_share_step": step,
        "old_arrangements_quadratic": {
            "share_gamma_1": old_share_1,
            "share_gamma_10": old_share_10,
            "step": old_step,
            "source": "results/corrected_lr.json",
        },
        "signed_utility_interaction": {
            "F_interaction": signed_anova["F_interaction"],
            "p_interaction": signed_anova["p_interaction"],
            "df_interaction": signed_anova["df_interaction"],
            "df_error": signed_anova["df_error"],
            "p_interaction_permutation": p_perm,
            "n_perm": N_PERM,
            "n_perm_exceeded": n_perm_exceeded,
            "reading": _reading(signed_anova["p_interaction"], True, "signed"),
        },
        "utility_share_gated": {
            "licensed_gammas": licensed,
            "licensed": share_licensed,
            "p_interaction": p_share,
        },
        "wall_seconds_mtime_span": float(max(mtimes) - min(mtimes)) if mtimes else None,
        "wall_seconds_serial_sum": float(sum(walls)) if walls else None,
        "reading": {
            "key": key,
            "text": readings[key],
            "composition_reproduces": comp_reproduces,
            "composition_vanishes": comp_vanishes,
            "interaction_reproduces": inter_reproduces,
            "operationalization": op,
        },
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps({
        "n_files": out["n_files"],
        "n_usable": out["n_usable"],
        "n_arms_missed": n_miss_arm,
        "misses": miss_by,
        "cells_three_corner": {k: v["three_corner"] for k, v in cells.items()},
        "gamma_step": step,
        "old_step": old_step,
        "F_interaction": signed_anova["F_interaction"],
        "p_interaction": signed_anova["p_interaction"],
        "p_perm": p_perm,
        "reading": out["reading"]["key"],
    }, indent=2))


if __name__ == "__main__":
    main()
