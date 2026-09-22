"""Scope granularity slice against `results/scope_precommit.json`. Does not train.

Cell: dichotomy 0, lag 12 from own-group baseline, module A.
K=1 recovery against not-tuned; primary is paired K=4 − K=1 at γ=10.

    python scripts/analyse_scope.py
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

PRECOMMIT = ROOT / "results" / "scope_precommit.json"
ARMS = ROOT / "results" / "scope"
OUT = ROOT / "results" / "scope_slice.json"
NOTTUNED = ROOT / "results" / "nottuned_slice.json"
TASK, LAG, MODULE = 0, 12, "A"
LICENSED = (1.0, 10.0)
KS = (1, 2, 4)
FORGETTING = ("S-HL", "S-LH", "S-LL")
MIN_FLOORS = 2.0
RECOVERY_TOL = 0.02


def _rows(recs: list[dict]) -> list[dict]:
    out = []
    for spec, mod, task, lag, att in G.attributions(recs, module=MODULE):
        if task != TASK or lag != LAG:
            continue
        floors = G.floors(att.dlog_alpha, "alpha")
        out.append({
            "gamma": spec["gamma_0"],
            "K": spec["scope_K"],
            "condition": spec["condition"],
            "stream_id": spec["stream_id"],
            "utility_term": att.terms["utility"],
            "utility_share": att.shares["utility"],
            "floors": floors,
            "gated": abs(floors) >= MIN_FLOORS,
        })
    return out


def _cell(rows, g, k, conditions) -> dict:
    xs = [x for x in rows if x["gamma"] == g and x["K"] == k
          and x["condition"] in conditions]
    by = defaultdict(list)
    for x in xs:
        by[x["stream_id"]].append(x)
    shares = []
    for sid, vs in by.items():
        shares.append(float(np.mean([v["utility_share"] for v in vs])))
    n = len(shares)
    if n == 0:
        return {"n": 0}
    sem = float(np.std(shares, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    return {
        "n": n,
        "utility_share_mean": float(np.mean(shares)),
        "utility_share_sem": sem,
        "n_gated": int(sum(
            abs(np.mean([v["floors"] for v in vs])) >= MIN_FLOORS
            for vs in by.values())),
        "per_stream": {str(sid): float(np.mean([v["utility_share"] for v in vs]))
                       for sid, vs in sorted(by.items())},
    }


def _paired_delta(rows, g) -> dict:
    a = _cell(rows, g, 4, FORGETTING)
    b = _cell(rows, g, 1, FORGETTING)
    if a.get("n", 0) == 0 or b.get("n", 0) == 0:
        return {"n": 0}
    sids = sorted(set(a["per_stream"]) & set(b["per_stream"]))
    ds = [a["per_stream"][s] - b["per_stream"][s] for s in sids]
    n = len(ds)
    mean = float(np.mean(ds))
    sem = float(np.std(ds, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    lo, hi = mean - 1.96 * sem, mean + 1.96 * sem
    return {
        "n": n,
        "d_share_mean": mean,
        "d_share_sem": sem,
        "ci95": [lo, hi],
        "excludes_0": bool(lo > 0 or hi < 0),
        "share_K4": a["utility_share_mean"],
        "share_K1": b["utility_share_mean"],
    }


def main() -> None:
    pre = json.loads(PRECOMMIT.read_text())
    reserved = reserved_stream_ids()
    paths = sorted(ARMS.glob("*.json"))
    raw = [json.loads(p.read_text()) for p in paths]
    sids = {r["spec"]["stream_id"] for r in raw}
    if sids != set(reserved):
        sys.exit(f"stream ids {sorted(sids)} are not the reserved set")
    if any(r["spec"].get("scope_K") not in KS for r in raw):
        sys.exit("unexpected scope_K in the scope directory")

    miss_by = {}
    n_miss_arm = 0
    for r in raw:
        k = f"{r['spec']['gamma_0']:g}:{int(r['spec']['scope_K'])}"
        miss_by.setdefault(k, {"n": 0, "n_miss": 0})
        miss_by[k]["n"] += 1
        if not r.get("usable"):
            miss_by[k]["n_miss"] += 1
            n_miss_arm += 1

    recs = G.load(arms_dir=ARMS)
    rows = _rows(recs)
    cells = {}
    for g in LICENSED:
        for k in KS:
            cells[f"{g:g}:{k}"] = {
                "three_corner": _cell(rows, g, k, FORGETTING),
                "all_four": _cell(rows, g, k, ("S-HH",) + FORGETTING),
            }

    nottuned = json.loads(NOTTUNED.read_text())
    nt10 = nottuned["cells"]["10"]["three_corner"]["utility_share_mean"]
    k1_10 = cells["10:1"]["three_corner"].get("utility_share_mean")
    recovery_ok = (
        k1_10 is not None and abs(k1_10 - nt10) <= RECOVERY_TOL
    )
    d10 = _paired_delta(rows, 10.0)
    d1 = _paired_delta(rows, 1.0)
    readings = pre["readings_committed_before_seeing_numbers"]
    if not recovery_ok:
        key = "K1_does_not_recover_nottuned"
    elif d10.get("excludes_0"):
        if d1.get("excludes_0") and (d10["d_share_mean"] > 0) != (d1["d_share_mean"] > 0):
            key = "mixed"
        else:
            key = "forgetting_differs_by_partition"
    elif d1.get("excludes_0"):
        key = "mixed"
    else:
        key = "no_difference"

    walls = [r.get("wall_seconds") for r in raw if r.get("wall_seconds") is not None]
    mtimes = [p.stat().st_mtime for p in paths]
    out = {
        "generated_by": "scripts/analyse_scope.py",
        "precommit": "results/scope_precommit.json",
        "n_files": len(raw),
        "n_usable": sum(1 for r in raw if r.get("usable")),
        "n_arms_missed": n_miss_arm,
        "misses": miss_by,
        "cell": {"task": TASK, "lag": LAG, "module": MODULE,
                 "unique_n": 8, "unit": "reserved stream_id, seed=0"},
        "three_corner": FORGETTING,
        "cells": cells,
        "K1_recovery": {
            "nottuned_gamma_10": nt10,
            "scope_K1_gamma_10": k1_10,
            "tol": RECOVERY_TOL,
            "ok": recovery_ok,
        },
        "paired_K4_minus_K1": {"1": d1, "10": d10},
        "wall_seconds_mtime_span": float(max(mtimes) - min(mtimes)) if mtimes else None,
        "wall_seconds_serial_sum": float(sum(walls)) if walls else None,
        "reading": {
            "key": key,
            "text": readings[key],
            "operationalization": pre["operationalization"],
        },
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps({
        "n_files": out["n_files"],
        "n_usable": out["n_usable"],
        "n_arms_missed": n_miss_arm,
        "K1_recovery": out["K1_recovery"],
        "paired_K4_minus_K1": out["paired_K4_minus_K1"],
        "cells_three_corner": {k: v["three_corner"] for k, v in cells.items()},
        "reading": out["reading"]["key"],
    }, indent=2))


if __name__ == "__main__":
    main()
