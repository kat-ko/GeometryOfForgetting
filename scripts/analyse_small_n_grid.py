"""Score the small-N 2×2 against `results/small_n_grid_precommit.json`.

    python scripts/analyse_small_n_grid.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PRE = json.loads((ROOT / "results" / "small_n_grid_precommit.json").read_text())
PRE16 = ROOT / "results" / "small_n16_precommit.json"
DIR = ROOT / "results" / "small_n_grid"
OUT = ROOT / "results" / "small_n_grid.json"
CORNERS = ("S-HH", "S-HL", "S-LH", "S-LL")
CF_ORDER = ("S-HL", "S-LL", "S-LH", "S-HH")


def past_task_acc(rec: dict) -> float:
    acc = np.asarray(rec["accuracy_matrix"], dtype=float)
    return float(np.nanmean(acc[-1, :-1]))


def dlog_alpha(rec: dict, *, lag: int = 15) -> float | None:
    for a in rec.get("attribution", []):
        if a["module"] == "A" and a["task"] == 0 and a["lag"] == lag:
            return float(a["dlog_alpha"])
    return None


def delta_rho(rec: dict) -> float | None:
    g0 = g15 = None
    for g in rec["geometry"]:
        if g["module"] == "A" and g["ensemble"] == "retained" and g["task"] == 0:
            if g["boundary"] == 0:
                g0 = float(g["rho_c_signed"])
            if g["boundary"] == 15:
                g15 = float(g["rho_c_signed"])
    if g0 is None or g15 is None:
        return None
    return g15 - g0


def d_eff_generic_b0(rec: dict) -> float | None:
    for g in rec["geometry"]:
        if (g["module"] == "A" and g["ensemble"] == "generic"
                and g["task"] is None and g["boundary"] == 0):
            return float(g["D_eff"])
    return None


def alpha_at(rec: dict, *, ensemble: str, task, boundary: int) -> float | None:
    for g in rec["geometry"]:
        if (g["module"] == "A" and g["ensemble"] == ensemble
                and g["task"] == task and g["boundary"] == boundary):
            return float(g["alpha"])
    return None


def summarise(rec: dict) -> dict:
    return {
        "N": rec["spec"]["N"],
        "gamma_0": rec["spec"]["gamma_0"],
        "condition": rec["spec"]["condition"],
        "seed": rec["spec"]["seed"],
        "usable": bool(rec.get("usable")),
        "CF": float(rec["forgetting"]["CF"]),
        "past_task_acc": past_task_acc(rec),
        "dlog_alpha": dlog_alpha(rec),
        "delta_rho_c": delta_rho(rec),
        "D_eff_generic_b0": d_eff_generic_b0(rec),
        "alpha_retained_b0": alpha_at(rec, ensemble="retained", task=0, boundary=0),
        "alpha_retained_b15": alpha_at(rec, ensemble="retained", task=0, boundary=15),
        "alpha_generic_b0": alpha_at(rec, ensemble="generic", task=None, boundary=0),
    }


def cell_stats(rows: list[dict]) -> dict:
    def arr(key):
        return np.array([r[key] for r in rows if r[key] is not None], dtype=float)

    cf, past, dlog, drho, deff = (arr(k) for k in (
        "CF", "past_task_acc", "dlog_alpha", "delta_rho_c", "D_eff_generic_b0"))
    a0, a15, ag = (arr(k) for k in (
        "alpha_retained_b0", "alpha_retained_b15", "alpha_generic_b0"))
    out = {
        "n": len(rows),
        "n_usable": int(sum(r["usable"] for r in rows)),
        "mean_CF": float(np.mean(cf)),
        "min_CF": float(np.min(cf)),
        "max_CF": float(np.max(cf)),
        "mean_past_task_acc": float(np.mean(past)),
        "mean_dlog_alpha": float(np.mean(dlog)) if len(dlog) else None,
        "mean_delta_rho_c": float(np.mean(drho)) if len(drho) else None,
        "mean_D_eff_generic_b0": float(np.mean(deff)) if len(deff) else None,
        "mean_alpha_retained_b0": float(np.mean(a0)) if len(a0) else None,
        "mean_alpha_retained_b15": float(np.mean(a15)) if len(a15) else None,
        "mean_alpha_generic_b0": float(np.mean(ag)) if len(ag) else None,
    }
    return out


def _cells(grouped: dict) -> dict:
    return {k: cell_stats(v) for k, v in grouped.items()}


def cf_order(cells_at_N_g: dict) -> list[str]:
    return sorted(CORNERS, key=lambda c: -cells_at_N_g[c]["mean_CF"])


def rich_cf_verdict(cells_g10: dict) -> str:
    hh, hl = cells_g10["S-HH"]["mean_CF"], cells_g10["S-HL"]["mean_CF"]
    order = cf_order(cells_g10)
    if hh <= 0.01 and hl >= 0.25 and order == list(CF_ORDER):
        return "survives"
    if hh >= 0.05 or hl < 0.20 or order != list(CF_ORDER):
        return "breaks"
    return "intermediate"


def glue_sign_verdict(cells_g10: dict) -> str:
    signs = {c: cells_g10[c]["mean_dlog_alpha"] for c in CORNERS}
    if any(v is None for v in signs.values()):
        return "missing"
    hh_ok = signs["S-HH"] > 0
    loss_ok = all(signs[c] < 0 for c in ("S-HL", "S-LH", "S-LL"))
    if hh_ok and loss_ok:
        return "match"
    return "break"


def verdict(cells: dict) -> dict:
    widths = sorted({k[0] for k in cells})
    out = {}
    for N in widths:
        g10 = {c: cells[(N, 10.0, c)] for c in CORNERS if (N, 10.0, c) in cells}
        g003 = {c: cells[(N, 0.03, c)] for c in CORNERS if (N, 0.03, c) in cells}
        if len(g10) < 4 or len(g003) < 4:
            continue
        out[f"N{N}"] = {
            "rich_CF_2x2": rich_cf_verdict(g10),
            "glue_signs_gamma10": glue_sign_verdict(g10),
            "rich_CF_order": cf_order(g10),
            "packing_D_eff_g10_HH": g10["S-HH"]["mean_D_eff_generic_b0"],
            "lazy_S-HH_mean_CF": g003["S-HH"]["mean_CF"],
            "rich_S-HH_n_usable": g10["S-HH"]["n_usable"],
            "rich_fit_breaks": bool(g10["S-HH"]["n_usable"] < 4),
            "mean_alpha_retained_b0_HH_g10": g10["S-HH"]["mean_alpha_retained_b0"],
            "mean_alpha_retained_b15_HH_g10": g10["S-HH"]["mean_alpha_retained_b15"],
        }
    if "N32" in out:
        out["lazy_S-HH_CF_break_confirmed"] = bool(out["N32"]["lazy_S-HH_mean_CF"] >= 0.05)
        out["N32_packing"] = bool((out["N32"]["packing_D_eff_g10_HH"] or 99) < 2.0)
    if "N16" in out:
        out["N16_packing"] = bool((out["N16"]["packing_D_eff_g10_HH"] or 99) < 2.0)
    return out


def main() -> None:
    recs = [json.loads(p.read_text()) for p in sorted(DIR.glob("N*.json"))]
    if not recs:
        raise SystemExit(f"no arms in {DIR}")
    arms = [summarise(r) for r in recs]
    grouped: dict[tuple, list] = defaultdict(list)
    for a in arms:
        grouped[(a["N"], a["gamma_0"], a["condition"])].append(a)
    cells = _cells(grouped)
    cells_out = {
        f"N{n}_g{g:g}_{c}": cells[(n, g, c)]
        for n, g, c in sorted(cells)
    }
    v = verdict(cells)
    extra_pre = json.loads(PRE16.read_text()) if PRE16.is_file() else None
    payload = {
        "generated_by": "scripts/analyse_small_n_grid.py",
        "precommit": PRE,
        "precommit_N16": extra_pre,
        "n_arms": len(arms),
        "n_cells": len(cells),
        "cells": cells_out,
        "verdict": v,
    }
    OUT.write_text(json.dumps(payload, indent=2))
    print(json.dumps({"n_arms": len(arms), "verdict": v}, indent=2))
    print("\nγ=10 mean CF / Δlogα / retained α")
    widths = sorted({k[0] for k in cells})
    for N in widths:
        print(f"  N={N}")
        for c in CORNERS:
            cell = cells[(N, 10.0, c)]
            print(
                f"    {c} CF={cell['mean_CF']:+.4f}  "
                f"Δlogα={cell['mean_dlog_alpha']:+.3f}  "
                f"α0={cell['mean_alpha_retained_b0']:.3f}  "
                f"α15={cell['mean_alpha_retained_b15']:.3f}  "
                f"usable={cell['n_usable']}/{cell['n']}"
            )


if __name__ == "__main__":
    main()
