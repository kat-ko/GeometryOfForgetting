"""Rank first slice against `results/rank_precommit.json`. Does not train.

Cell: task 0, lag 12, module A. Unique n=8 reserved stream_ids, seed=0.
Composition claims licensed at γ ∈ {1, 10} only. γ=0.03 is excluded, not a control.

    python scripts/analyse_rank.py
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

PRECOMMIT = ROOT / "results" / "rank_precommit.json"
ARMS = ROOT / "results" / "rank"
OUT = ROOT / "results" / "rank_slice.json"
TASK, LAG, MODULE = 0, 12, "A"
LICENSED = (1.0, 10.0)
RANKS = (4, 16, 300)
FORGETTING = ("S-HL", "S-LH", "S-LL")
MIN_FLOORS = 2.0


def _rows(recs: list[dict]) -> list[dict]:
    out = []
    for spec, mod, task, lag, att in G.attributions(recs, module=MODULE):
        if task != TASK or lag != LAG:
            continue
        floors = G.floors(att.dlog_alpha, "alpha")
        out.append({
            "gamma": spec["gamma_0"],
            "rank": spec["readout_rank"],
            "condition": spec["condition"],
            "stream_id": spec["stream_id"],
            "utility_term": att.terms["utility"],
            "utility_share": att.shares["utility"],
            "floors": floors,
            "gated": abs(floors) >= MIN_FLOORS,
        })
    return out


def _cell(rows, g, r, conditions) -> dict:
    xs = [x for x in rows if x["gamma"] == g and x["rank"] == r
          and x["condition"] in conditions]
    by = defaultdict(list)
    for x in xs:
        by[x["stream_id"]].append(x)
    # one value per unique stream: mean over the named conditions present
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


def main() -> None:
    pre = json.loads(PRECOMMIT.read_text())
    reserved = reserved_stream_ids()
    paths = sorted(ARMS.glob("*.json"))
    raw = [json.loads(p.read_text()) for p in paths]
    if any(not r["spec"].get("q_nested") for r in raw):
        sys.exit("un-nested record in the slice directory")
    sids = {r["spec"]["stream_id"] for r in raw}
    if sids != set(reserved):
        sys.exit(f"stream ids {sorted(sids)} are not the reserved set")

    miss_by = {}
    n_miss_arm = 0
    for r in raw:
        g, rk = r["spec"]["gamma_0"], r["spec"]["readout_rank"]
        k = f"{g:g}:{int(rk)}"
        miss_by.setdefault(k, {"n": 0, "n_miss": 0})
        miss_by[k]["n"] += 1
        if not r.get("usable"):
            miss_by[k]["n_miss"] += 1
            n_miss_arm += 1

    recs = G.load(arms_dir=ARMS)
    recs = [r for r in recs if r["spec"].get("q_nested") and r["spec"]["gamma_0"] in LICENSED]
    rows = _rows(recs)

    cells = {}
    for g in LICENSED:
        for rk in RANKS:
            cells[f"{g:g}:{rk}"] = {
                "three_corner": _cell(rows, g, rk, FORGETTING),
                "all_four": _cell(rows, g, rk, ("S-HH",) + FORGETTING),
            }

    # Compare r=4 vs r=N at each licensed γ (three-corner share).
    deltas = {}
    for g in LICENSED:
        a = cells[f"{g:g}:4"]["three_corner"]
        b = cells[f"{g:g}:300"]["three_corner"]
        deltas[f"{g:g}:r4_minus_rN"] = {
            "d_utility_share": a["utility_share_mean"] - b["utility_share_mean"],
            "d_utility_term": a["utility_term_mean"] - b["utility_term_mean"],
            "share_4": a["utility_share_mean"],
            "share_N": b["utility_share_mean"],
        }
    step = (cells["10:300"]["three_corner"]["utility_share_mean"]
            - cells["1:300"]["three_corner"]["utility_share_mean"])
    # Rank effect vs the γ=1→10 step at unconstrained readout.
    rank_at_10 = abs(deltas["10:r4_minus_rN"]["d_utility_share"])
    rank_at_1 = abs(deltas["1:r4_minus_rN"]["d_utility_share"])
    gamma_step = abs(step)

    if rank_at_10 < 0.5 * gamma_step and rank_at_1 < 0.5 * gamma_step:
        key = "reorganisation_tracks_gamma_regardless_of_rank"
        text = pre["readings_committed_before_seeing_numbers"][key]
    elif rank_at_10 >= gamma_step and cells["10:4"]["three_corner"]["utility_share_mean"] < cells["10:300"]["three_corner"]["utility_share_mean"]:
        key = "reorganisation_reproduces_under_rank_at_fixed_gamma"
        text = pre["readings_committed_before_seeing_numbers"][key]
    else:
        key = "mixed"
        text = pre["readings_committed_before_seeing_numbers"][key]

    out = {
        "generated_by": "scripts/analyse_rank.py",
        "precommit": "results/rank_precommit.json",
        "n_files": len(raw),
        "n_usable": sum(1 for r in raw if r.get("usable")),
        "n_arms_missed": n_miss_arm,
        "misses": miss_by,
        "miss_interpretation": pre["misses"]["interpretation"],
        "gamma_0.03": pre["design"]["first_slice"]["gamma_0.03"],
        "licensed_gammas": list(LICENSED),
        "cell": {"task": TASK, "lag": LAG, "module": MODULE,
                 "unique_n": 8, "unit": "reserved stream_id, seed=0"},
        "three_corner": FORGETTING,
        "cells": cells,
        "gamma_1_to_10_share_step_at_rN": step,
        "rank_deltas_three_corner": deltas,
        "reading": {
            "key": key,
            "text": text,
            "operationalization": (
                "Three-corner utility share, unique n=8, task 0 lag 12 module A. "
                "Rank effect = |share(r=4) − share(r=N)|. γ-step = |share(γ=10, r=N) − share(γ=1, r=N)|. "
                "Tracks γ if rank effect < half the γ-step at both licensed γ. "
                "Dimensionality if the rank effect at γ=10 is at least the γ-step. Else mixed."
            ),
        },
        "wall_seconds_slice": 16614.381955385208,
        "n_workers": 192,
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps({
        "n_files": out["n_files"],
        "n_usable": out["n_usable"],
        "n_arms_missed": n_miss_arm,
        "misses": miss_by,
        "cells_three_corner": {k: v["three_corner"] for k, v in cells.items()},
        "rank_deltas_three_corner": deltas,
        "gamma_step_rN": step,
        "reading": out["reading"],
    }, indent=2))


if __name__ == "__main__":
    main()
