"""γ × condition interaction on channel composition, unique n=8.

Model and unit: `results/interaction_gamma_condition_precommit.json`.
Writes `results/interaction_gamma_condition.json`. Does not train.

    python scripts/test_interaction_gamma_condition.py
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

from audit_unique_n import unique_recs  # noqa: E402
from src.analysis import anova  # noqa: E402
from src.analysis import attribution as ATT  # noqa: E402
from src.analysis import grid as G  # noqa: E402

PRECOMMIT = ROOT / "results" / "interaction_gamma_condition_precommit.json"
OUT = ROOT / "results" / "interaction_gamma_condition.json"
MIN_FLOORS = 2.0
N_PERM = 999
PERM_SEED = 20260901
TASK, LAG, MODULE = 0, 12, "A"
CONDITIONS = ("S-HH", "S-HL", "S-LH", "S-LL")
ALPHA = 0.05


def _rows(recs: list[dict]) -> list[dict]:
    out = []
    for spec, mod, task, lag, att in G.attributions(recs, module=MODULE):
        if task != TASK or lag != LAG:
            continue
        floors = G.floors(att.dlog_alpha, "alpha")
        out.append({
            "gamma": spec["gamma_0"],
            "condition": spec["condition"],
            "seed": spec["seed"],
            "dlog_alpha": att.dlog_alpha,
            "floors": floors,
            "utility_term": att.terms["utility"],
            "radius_term": att.terms["radius"],
            "dimension_term": att.terms["dimension"],
            "utility_share": att.shares["utility"],
            "gated": abs(floors) >= MIN_FLOORS,
        })
    return out


def _grid(rows, field, gammas, conditions) -> np.ndarray:
    by = defaultdict(list)
    for r in rows:
        by[(r["gamma"], r["condition"])].append(r[field])
    y, row, col = [], [], []
    for g in gammas:
        for c in conditions:
            vals = by[(g, c)]
            y.extend(vals)
            row.extend([g] * len(vals))
            col.extend([c] * len(vals))
    return anova.fill_balanced(y, row, col, gammas, conditions)


def _licensed_gammas(rows, gammas, conditions) -> list[float]:
    by = defaultdict(list)
    for r in rows:
        by[(r["gamma"], r["condition"])].append(r["gated"])
    keep = []
    for g in gammas:
        ns = [int(np.sum(by[(g, c)])) for c in conditions]
        if min(ns) >= 6:
            keep.append(g)
    return keep


def _permute_F(data: np.ndarray, F_obs: float, rng: np.random.Generator) -> tuple[int, float]:
    count = 0
    n_g, n_c, n_s = data.shape
    for _ in range(N_PERM):
        shuffled = data.copy()
        for g in range(n_g):
            for s in range(n_s):
                shuffled[g, :, s] = rng.permutation(data[g, :, s])
        F = anova.two_way_balanced(shuffled)["F_interaction"]
        if F >= F_obs:
            count += 1
    return count, (count + 1) / (N_PERM + 1)


def _cell_means(data: np.ndarray, gammas, conditions, field: str) -> dict:
    out = {}
    for i, g in enumerate(gammas):
        for j, c in enumerate(conditions):
            v = data[i, j]
            out[f"{g:g}:{c}"] = {
                "mean": float(np.mean(v)), "sem": float(np.std(v, ddof=1) / np.sqrt(len(v))),
                "n": int(len(v)),
            }
    return out


def _reading(p: float, licensed: bool, which: str) -> str:
    if which == "share" and not licensed:
        return "share ANOVA not licensed; not substituted with an ungated share test"
    if p < ALPHA:
        return ("Channel composition's γ-gradient differs by condition. "
                "The headline is an interaction, not a γ-main-effect pooled over corners.")
    return ("No detectable γ × condition interaction on the utility term at unique n=8. "
            "The composition shift with γ is not licensed as corner-specific from this test."
            if which == "signed" else
            "No detectable γ × condition interaction on gated utility share at unique n=8.")


def main() -> None:
    pre = json.loads(PRECOMMIT.read_text())
    if not pre.get("written_before_running"):
        sys.exit("pre-commit missing or not marked written_before_running")

    recs = unique_recs(G.load())
    rows = _rows(recs)
    gammas = list(G.REGISTERED_GAMMAS)
    n_expected = len(gammas) * len(CONDITIONS) * 8
    if len(rows) != n_expected:
        sys.exit(f"expected {n_expected} unique (γ, condition, seed) rows; got {len(rows)}")

    signed = _grid(rows, "utility_term", gammas, CONDITIONS)
    signed_anova = anova.two_way_balanced(signed)
    rng = np.random.default_rng(PERM_SEED)
    n_perm_exceeded, p_perm = _permute_F(signed, signed_anova["F_interaction"], rng)

    licensed = _licensed_gammas(rows, gammas, CONDITIONS)
    share_licensed = len(licensed) >= 2
    share_anova = None
    p_perm_share = None
    share_cells = None
    share_n_per_cell = None
    if share_licensed:
        gated_rows = [r for r in rows if r["gamma"] in licensed and r["gated"]]
        share_n_per_cell = {
            f"{g:g}:{c}": int(sum(1 for r in gated_rows
                                  if r["gamma"] == g and r["condition"] == c))
            for g in licensed for c in CONDITIONS
        }
        y = [r["utility_share"] for r in gated_rows]
        row = [r["gamma"] for r in gated_rows]
        col = [r["condition"] for r in gated_rows]
        if set(share_n_per_cell.values()) == {8}:
            share = anova.fill_balanced(y, row, col, licensed, CONDITIONS)
            share_anova = anova.two_way_balanced(share)
            _, p_perm_share = _permute_F(
                share, share_anova["F_interaction"], np.random.default_rng(PERM_SEED))
            share_cells = _cell_means(share, licensed, CONDITIONS, "utility_share")
        else:
            share_anova = anova.two_way_interaction_ols(
                y, row, col, licensed, CONDITIONS)
            share_cells = {}
            by = defaultdict(list)
            for r in gated_rows:
                by[(r["gamma"], r["condition"])].append(r["utility_share"])
            for g in licensed:
                for c in CONDITIONS:
                    v = np.asarray(by[(g, c)])
                    share_cells[f"{g:g}:{c}"] = {
                        "mean": float(np.mean(v)),
                        "sem": float(np.std(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0,
                        "n": int(len(v)),
                    }

    out = {
        "generated_by": "scripts/test_interaction_gamma_condition.py",
        "precommit": str(PRECOMMIT.relative_to(ROOT)),
        "n_unique_draws": len(rows),
        "n_files_refused_as_unit": "copies dropped by unique_recs; unit is seed",
        "design": {
            "n_gamma": len(gammas),
            "n_condition": len(CONDITIONS),
            "n_rep": 8,
            "n_cells": len(gammas) * len(CONDITIONS),
            "n_obs": len(rows),
            "gammas": list(gammas),
            "conditions": list(CONDITIONS),
            "df_interaction": (len(gammas) - 1) * (len(CONDITIONS) - 1),
            "df_error": len(gammas) * len(CONDITIONS) * (8 - 1),
            "note": "Registered γ grid is 6 levels. F(15, 168) is the 6×4 two-way interaction on 192 unique (γ, condition, seed) draws — one measurement per arm (task 0, lag 12, module A), not a one-way over cells and not within-run points as replicates."
        },
        "signed_utility_term": {
            **{k: signed_anova[k] for k in (
                "F_row", "p_row", "F_col", "p_col",
                "F_interaction", "p_interaction",
                "df_row", "df_col", "df_interaction", "df_error")},
            "p_interaction_permutation": p_perm,
            "n_perm": N_PERM,
            "n_perm_exceeded": n_perm_exceeded,
            "p_interaction_permutation_note": (
                "no permutation of 999 exceeded observed; 0.001 is the (0+1)/(999+1) floor"
                if n_perm_exceeded == 0 else
                f"{n_perm_exceeded} of {N_PERM} permutations exceeded observed"
            ),
            "cell_means": _cell_means(signed, gammas, CONDITIONS, "utility_term"),
            "reading": _reading(signed_anova["p_interaction"], True, "signed"),
        },
        "utility_share_gated": {
            "licensed_gammas": licensed,
            "licensed": share_licensed,
            "min_floors": MIN_FLOORS,
            "n_min_per_cell": 6,
            "n_per_cell": share_n_per_cell,
            "anova": (None if share_anova is None else {
                k: share_anova[k] for k in share_anova
                if k in ("F_row", "p_row", "F_col", "p_col",
                         "F_interaction", "p_interaction",
                         "df_interaction", "df_error", "n")
            }),
            "p_interaction_permutation": p_perm_share,
            "cell_means": share_cells,
            "reading": _reading(
                share_anova["p_interaction"] if share_anova else 1.0,
                share_licensed, "share"),
        },
        "alpha": ALPHA,
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps({
        "n": len(rows),
        "F_interaction_signed": signed_anova["F_interaction"],
        "p_signed": signed_anova["p_interaction"],
        "p_perm": p_perm,
        "licensed_gammas": licensed,
        "share_licensed": share_licensed,
        "p_share": None if share_anova is None else share_anova["p_interaction"],
        "reading_signed": out["signed_utility_term"]["reading"],
        "wrote": str(OUT),
    }, indent=2))


if __name__ == "__main__":
    main()
