"""Recompute every SEM, CI and sign-test p on the registered grid at unique n=8.

`stream_id` was stored and never read, so 40 files per (γ, condition) cell are 8 unique
(seed-tied arrangement + init + dichotomy) triples, each written five times. Point
estimates are invariant to the copies. SEMs, CIs and sign tests are not.

Writes `results/unique_n8.json`. Does not train.

    python scripts/audit_unique_n.py
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

from fig4_corners_rho_c import CORNERS, effects, per_arm, summarize  # noqa: E402
from src.analysis import grid as G  # noqa: E402
from src.analysis.attribution import min_dlog_R_for  # noqa: E402
from src.analysis.ledger import BENIGN, THREE, UNIQUE_SEEDS_PER_CELL  # noqa: E402


def unique_recs(recs: list[dict]) -> list[dict]:
    """One file per (γ, condition, seed, N). stream_id is unused on the registered grid."""
    keep = {}
    for r in recs:
        s = r["spec"]
        keep.setdefault((s["gamma_0"], s["condition"], s["seed"], s.get("N", 300)), r)
    return list(keep.values())


def unique_obs(rows: list[dict]) -> list[dict]:
    keep = {}
    for r in rows:
        keep.setdefault(
            (r["gamma"], r["condition"], r["seed"], r.get("N", 300), r["task"], r["lag"]), r)
    return list(keep.values())


def sign_row(traj_all, traj_u, corner: str) -> dict:
    a, u = summarize(traj_all)[corner], summarize(traj_u)[corner]
    return {
        "corner": corner,
        "n_files": a["n_arms"], "n_unique": u["n_arms"],
        "delta_mean": u["delta_mean"],  # copies don't change the mean
        "sem_files": a["delta_sem"], "sem_unique": u["delta_sem"],
        "n_declining_files": int(round(a["fraction_declining"] * a["n_arms"])),
        "n_declining_unique": int(round(u["fraction_declining"] * u["n_arms"])),
        "p_files": a["sign_test_p"], "p_unique": u["sign_test_p"],
        "spearman_median": u["spearman_median"],
    }


def status_sign(p: float, alpha: float = 0.05) -> str:
    return "rejects" if p < alpha else "does not reject"


def main() -> None:
    recs = G.load()
    recs_u = unique_recs(recs)
    n_files, n_unique = len(recs), len(recs_u)
    copies = n_files / n_unique if n_unique else None

    out: dict = {
        "generated_by": "scripts/audit_unique_n.py",
        "question": "which paper conclusions change status when the grid is read at unique n=8",
        "n_files": n_files, "n_unique": n_unique, "copies_per_unique": copies,
        "unique_seeds_per_cell": UNIQUE_SEEDS_PER_CELL,
        "note": "point estimates unchanged; SEMs/CIs/sign tests recomputed on one file per seed",
    }

    # --- Figure 4 Δρ_c sign tests at γ=10 and γ=3 --------------------------------
    fig4 = {}
    for g in (3.0, 10.0):
        t_all, t_u = per_arm(recs, g), per_arm(recs_u, g)
        cells = {c: sign_row(t_all, t_u, c) for c in CORNERS if c in t_all}
        mean = {c: cells[c]["delta_mean"] for c in cells}
        sem_u = {c: cells[c]["sem_unique"] for c in cells}
        eff = effects(mean)
        # interaction SEM: half the RSS of four independent cell SEMs
        half_u = 0.5 * float(np.sqrt(sum(sem_u[c] ** 2 for c in CORNERS)))
        half_f = 0.5 * float(np.sqrt(sum(cells[c]["sem_files"] ** 2 for c in CORNERS)))
        fig4[f"{g:g}"] = {
            "cells": cells,
            "effects": eff,
            "interaction_sem_files": half_f,
            "interaction_sem_unique": half_u,
            "interaction_over_sem_files": abs(eff["interaction"]) / half_f,
            "interaction_over_sem_unique": abs(eff["interaction"]) / half_u,
        }
    out["fig4"] = fig4

    # --- γ=5 probe: 16 files = 4 unique × 4 copies --------------------------------
    g5_dir = ROOT / "results" / "gamma_5"
    g5 = unique_recs(G.load(arms_dir=g5_dir)) if g5_dir.exists() else []
    g5_all = G.load(arms_dir=g5_dir) if g5_dir.exists() else []
    if g5_all:
        t_all, t_u = per_arm(g5_all, 5.0), per_arm(g5, 5.0)
        out["gamma5"] = {c: sign_row(t_all, t_u, c) for c in CORNERS if c in t_all}

    # --- S-HH radius term, lag 12, vs 3σ gate and vs SEM8 -------------------------
    radius = {}
    for r in recs_u:
        if r["spec"]["condition"] != BENIGN:
            continue
        g = r["spec"]["gamma_0"]
        hits = [x for x in r["attribution"]
                if x["module"] == "A" and x["task"] == 0 and x["lag"] == 12]
        if hits:
            radius.setdefault(g, []).append(hits[0]["terms"]["radius"])
    radius_rows = {}
    for g, vals in sorted(radius.items()):
        arr = np.array(vals)
        mean = float(arr.mean())
        sem = float(arr.std(ddof=1) / np.sqrt(len(arr)))
        gate = min_dlog_R_for(g)
        fl = float(np.sign(mean) * G.floors(mean, "R_eff", gamma=g))
        radius_rows[f"{g:g}"] = {
            "n_unique": len(arr), "mean": mean, "sem_unique": sem,
            "gate_3sigma": gate, "R_eff_floors": fl,
            "mean_over_sem": abs(mean) / sem if sem else None,
            "clears_3sigma_gate": abs(mean) >= gate,
            "resolvable_as_mean_3sem": abs(mean) >= 3 * sem,
        }
    out["shh_radius"] = radius_rows

    # --- variance decomposition at γ=10, three-corner, unique seeds --------------
    # Import the same r2 used in the published table.
    sys.path.insert(0, str(ROOT / "scripts"))
    from audit_propagation import observations as ap_obs, sel, variance_decomposition  # noqa: E402
    rows = ap_obs(recs)
    rows_u = unique_obs(rows)
    at10 = sel(rows, conditions=THREE, gamma=10.0)
    at10_u = sel(rows_u, conditions=THREE, gamma=10.0)
    out["variance_gamma10_three"] = {
        "files": variance_decomposition(at10),
        "unique": variance_decomposition(at10_u),
    }

    # --- status changes: sign tests that reject at n_files but not at n_unique, ---
    # and the other way; plus "still rejects but p moves orders of magnitude".
    changes = []
    for g, block in fig4.items():
        for c, cell in block["cells"].items():
            a, b = status_sign(cell["p_files"]), status_sign(cell["p_unique"])
            changes.append({
                "quantity": f"γ={g} {c} Δρ_c sign test",
                "files": f"{cell['n_declining_files']}/{cell['n_files']} p={cell['p_files']:.4g}",
                "unique": f"{cell['n_declining_unique']}/{cell['n_unique']} p={cell['p_unique']:.4g}",
                "status_files": a, "status_unique": b,
                "changed": a != b,
            })
        for label, zf, zu in (
            ("interaction / SEM", block["interaction_over_sem_files"],
             block["interaction_over_sem_unique"]),
        ):
            a = "resolved" if zf >= 3 else "unresolved"
            b = "resolved" if zu >= 3 else "unresolved"
            changes.append({
                "quantity": f"γ={g} Figure 4 {label}",
                "files": f"{zf:.2f} SEM", "unique": f"{zu:.2f} SEM",
                "status_files": a, "status_unique": b, "changed": a != b,
            })
    if "gamma5" in out:
        cell = out["gamma5"]["S-HL"]
        a, b = status_sign(cell["p_files"]), status_sign(cell["p_unique"])
        changes.append({
            "quantity": "γ=5 S-HL sign test",
            "files": f"{cell['n_declining_files']}/{cell['n_files']} p={cell['p_files']:.4g}",
            "unique": f"{cell['n_declining_unique']}/{cell['n_unique']} p={cell['p_unique']:.4g}",
            "status_files": a, "status_unique": b, "changed": a != b,
        })
    for g, row in radius_rows.items():
        if float(g) < 1:
            continue
        a = "clears 3σ gate" if row["clears_3sigma_gate"] else "below 3σ gate"
        b = "mean |z|≥3" if row["resolvable_as_mean_3sem"] else "mean |z|<3"
        changes.append({
            "quantity": f"S-HH radius at γ={g}",
            "files": a, "unique": f"{b} ({row['mean_over_sem']:.2f} SEM8)",
            "status_files": a, "status_unique": b,
            "changed": False,  # gate unchanged; this is the other reading
            "note": "gate kept; mean reading added",
        })
    # --- peak CI at unique n=8 (paired over seeds; stream_id is unused) ----------
    from tier2_backward_transfer import N_BOOT, SEED, _vertex  # noqa: E402
    by_seed: dict = {}
    for r in recs_u:
        if r["spec"]["condition"] != BENIGN:
            continue
        g = r["spec"]["gamma_0"]
        hits = [x for x in r["attribution"]
                if x["module"] == "A" and x["task"] == 0 and x["lag"] == 12]
        if hits:
            by_seed.setdefault(r["spec"]["seed"], {})[g] = hits[0]["dlog_alpha"]
    gammas_pk = sorted({g for d in by_seed.values() for g in d})
    complete = [s for s, d in by_seed.items() if set(d) == set(gammas_pk)]
    lg = np.log10(gammas_pk)
    rng = np.random.default_rng(SEED)
    vertices = []
    for _ in range(N_BOOT):
        draw = [by_seed[complete[i]] for i in rng.integers(0, len(complete), len(complete))]
        m = np.array([np.mean([d[g] for d in draw]) for g in gammas_pk])
        k = int(np.argmax(m))
        if 0 < k < len(gammas_pk) - 1:
            vertices.append(_vertex(lg[k - 1:k + 2], m[k - 1:k + 2]))
    ci_u = [float(10 ** np.percentile(vertices, 2.5)),
            float(10 ** np.percentile(vertices, 97.5))] if vertices else None
    out["peak_lag12"] = {
        "n_units_unique": len(complete),
        "ci_unique": ci_u,
        "ci_files": [1.1766417413775898, 1.2823275614362841],
        "n_units_files": 40,
        "ci_width_in_steps_unique": (
            float(np.log10(ci_u[1] / ci_u[0]) / np.log10(gammas_pk[1] / gammas_pk[0]))
            if ci_u else None),
    }
    vf, vu = out["variance_gamma10_three"]["files"], out["variance_gamma10_three"]["unique"]
    pk = out["peak_lag12"]
    changes.append({
        "quantity": "S-HH peak paired-bootstrap CI at lag 12",
        "files": (f"[{pk['ci_files'][0]:.2f}, {pk['ci_files'][1]:.2f}] n=40"),
        "unique": (f"[{pk['ci_unique'][0]:.2f}, {pk['ci_unique'][1]:.2f}] n=8"
                   if pk["ci_unique"] else "n/a"),
        "status_files": "quoted (peak already declined as a location)",
        "status_unique": "wider; still beside the point (peak moves with lag)",
        "changed": False,
        "note": "interpolated peak 1.23 is a mean and is unchanged",
    })
    changes.append({
        "quantity": "§B stream R² at γ=10 (three-corner)",
        "files": f"{vf['stream alone']:.4g} (n={vf['n']})",
        "unique": f"{vu['stream alone']:.4g} (n={vu['n']})",
        "status_files": "quoted as empty",
        "status_unique": "tautological: stream_id did not vary the stream",
        "changed": True,
        "note": "stream dummy is a label on identical copies",
    })
    out["status_changes"] = changes
    out["n_status_changes"] = sum(1 for c in changes if c.get("changed"))

    dest = ROOT / "results" / "unique_n8.json"
    dest.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {dest.relative_to(ROOT)}")
    print(f"  {n_files} files -> {n_unique} unique ({copies:.2f} copies)")
    print(f"  {out['n_status_changes']} status changes of {len(changes)} quantities\n")
    print(f"{'quantity':<42} {'files':<28} {'unique':<28} {'moved?'}")
    for c in changes:
        print(f"{c['quantity']:<42} {c['files']:<28} {c['unique']:<28} "
              f"{'YES' if c.get('changed') else 'no'}")


if __name__ == "__main__":
    main()
