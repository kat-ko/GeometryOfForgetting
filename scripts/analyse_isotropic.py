"""Isotropic-Gaussian control against `results/isotropic_precommit.json`. Does not train.

Cell: task 0, lag 12, module A. Unique n=8 reserved stream_ids, seed=0.
Licensed at γ ∈ {1, 10}. No rank. Comparator is spherical reserved r=N.

    python scripts/analyse_isotropic.py
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

PRECOMMIT = ROOT / "results" / "isotropic_precommit.json"
ARMS = ROOT / "results" / "isotropic"
OUT = ROOT / "results" / "isotropic_slice.json"
RANK_SLICE = ROOT / "results" / "rank_slice.json"
TASK, LAG, MODULE = 0, 12, "A"
LICENSED = (1.0, 10.0)
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
            "condition": spec["condition"],
            "stream_id": spec["stream_id"],
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


def main() -> None:
    pre = json.loads(PRECOMMIT.read_text())
    reserved = reserved_stream_ids()
    paths = sorted(ARMS.glob("*.json"))
    raw = [json.loads(p.read_text()) for p in paths]
    sids = {r["spec"]["stream_id"] for r in raw}
    if sids != set(reserved):
        sys.exit(f"stream ids {sorted(sids)} are not the reserved set")
    if any(r["spec"].get("manifold_kind") != "isotropic_gaussian" for r in raw):
        sys.exit("non-isotropic record in the isotropic directory")
    if any("readout_rank" in r["spec"] for r in raw):
        sys.exit("rank field in the isotropic directory; this control is unconstrained")

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
    recs = [r for r in recs if r["spec"]["gamma_0"] in LICENSED]
    rows = _rows(recs)

    cells = {}
    for g in LICENSED:
        cells[f"{g:g}"] = {
            "three_corner": _cell(rows, g, FORGETTING),
            "all_four": _cell(rows, g, ("S-HH",) + FORGETTING),
        }

    step = (cells["10"]["three_corner"]["utility_share_mean"]
            - cells["1"]["three_corner"]["utility_share_mean"])
    rank = json.loads(RANK_SLICE.read_text())
    spherical = rank["gamma_1_to_10_share_step_at_rN"]
    readings = pre["readings_committed_before_seeing_numbers"]
    # Locked before the 64-arm numbers: half the spherical step is still a
    # reorganisation; a quarter is gone. In between is mixed.
    if step >= 0.5 * spherical:
        key = "gamma_reorganisation_reproduces"
    elif abs(step) < 0.25 * spherical:
        key = "gamma_reorganisation_vanishes"
    else:
        key = "mixed"

    walls = [r.get("wall_seconds") for r in raw if r.get("wall_seconds") is not None]
    mtimes = [p.stat().st_mtime for p in paths]
    out = {
        "generated_by": "scripts/analyse_isotropic.py",
        "precommit": "results/isotropic_precommit.json",
        "n_files": len(raw),
        "n_usable": sum(1 for r in raw if r.get("usable")),
        "n_arms_missed": n_miss_arm,
        "misses": miss_by,
        "miss_interpretation": pre["misses"]["interpretation"],
        "licensed_gammas": list(LICENSED),
        "cell": {"task": TASK, "lag": LAG, "module": MODULE,
                 "unique_n": 8, "unit": "reserved stream_id, seed=0"},
        "three_corner": FORGETTING,
        "cells": cells,
        "gamma_1_to_10_share_step": step,
        "spherical_reserved_rN_share_step": spherical,
        "wall_seconds_mtime_span": float(max(mtimes) - min(mtimes)) if mtimes else None,
        "wall_seconds_serial_sum": float(sum(walls)) if walls else None,
        "n_workers": 192,
        "runner_summary_missing": (
            "Parent `run_isotropic.py` was interrupted after all 64 files were written. "
            "Wall is the arm-file mtime span, not a runner-printed wall."
        ),
        "reading": {
            "key": key,
            "text": readings[key],
            "operationalization": (
                "Three-corner utility share, unique n=8, task 0 lag 12 module A. "
                "Step = share(γ=10) − share(γ=1). Comparator is the spherical reserved "
                "r=N step. Reproduces if step ≥ half the spherical step. Vanishes if "
                "|step| < a quarter of the spherical step. Else mixed."
            ),
            "from_precommit": pre["operationalization"],
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
        "spherical_step": spherical,
        "reading": out["reading"],
    }, indent=2))


if __name__ == "__main__":
    main()
