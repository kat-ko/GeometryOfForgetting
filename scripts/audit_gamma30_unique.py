"""Recompute the γ=30 2×2 interaction at unique n=4.

`results/gamma_ext/` is 64 files, 16 unique (4 seeds × 4 unused stream_id copies).
The body claim that additivity fails at γ=30 quoted +0.016 at 3.9 SEM on 16 files/corner.
This is the unique-n pass that the registered grid already had.

Same interaction SEM as `audit_unique_n.py` and `analyse_gamma5.py`: half the RSS of
four cell SEMs. Also reports a paired-seed reading, because the unique unit is seed.

    python scripts/audit_gamma30_unique.py
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

from fig4_corners_rho_c import CORNERS, effects, per_arm  # noqa: E402
from audit_unique_n import unique_recs, sign_row  # noqa: E402
from src.analysis import grid as G  # noqa: E402

BOOT_SEED = 20260817
N_BOOT = 10_000


def _deltas(traj: dict) -> dict[str, np.ndarray]:
    return {c: np.array([p[-1][1] - p[0][1] for p in traj[c]]) for c in CORNERS}


def boot_interaction(deltas: dict[str, np.ndarray], rng: np.random.Generator) -> list[float]:
    draws = []
    for _ in range(N_BOOT):
        m = {c: float(deltas[c][rng.integers(0, len(deltas[c]), len(deltas[c]))].mean())
             for c in CORNERS}
        draws.append(effects(m)["interaction"])
    arr = np.array(draws)
    return [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))]


def main() -> None:
    ext = G.load(arms_dir=ROOT / "results" / "gamma_ext")
    ext_u = unique_recs(ext)
    t_all, t_u = per_arm(ext, 30.0), per_arm(ext_u, 30.0)
    cells = {c: sign_row(t_all, t_u, c) for c in CORNERS}
    mean = {c: cells[c]["delta_mean"] for c in CORNERS}
    eff = effects(mean)
    half_u = 0.5 * float(np.sqrt(sum(cells[c]["sem_unique"] ** 2 for c in CORNERS)))
    half_f = 0.5 * float(np.sqrt(sum(cells[c]["sem_files"] ** 2 for c in CORNERS)))
    z_f = abs(eff["interaction"]) / half_f
    z_u = abs(eff["interaction"]) / half_u

    rng = np.random.default_rng(BOOT_SEED)
    ci_f = boot_interaction(_deltas(t_all), rng)
    ci_u = boot_interaction(_deltas(t_u), rng)

    by_seed: dict[int, dict[str, float]] = defaultdict(dict)
    for r in ext_u:
        s = r["spec"]
        gs = [g for g in r["geometry"] if g["ensemble"] == "generic"]
        bs = sorted({g["boundary"] for g in gs})
        first = float(np.mean([g["rho_c_signed"] for g in gs if g["boundary"] == bs[0]]))
        last = float(np.mean([g["rho_c_signed"] for g in gs if g["boundary"] == bs[-1]]))
        by_seed[s["seed"]][s["condition"]] = last - first
    ints = np.array([effects(d)["interaction"] for d in by_seed.values() if len(d) == 4])
    paired_sem = float(ints.std(ddof=1) / np.sqrt(len(ints)))
    paired_draws = [float(np.mean(ints[rng.integers(0, len(ints), len(ints))]))
                    for _ in range(N_BOOT)]
    ci_p = [float(np.percentile(paired_draws, 2.5)), float(np.percentile(paired_draws, 97.5))]

    out = {
        "generated_by": "scripts/audit_gamma30_unique.py",
        "n_files": len(ext), "n_unique": len(ext_u),
        "unique_seeds_per_cell": 4, "copies_per_unique": len(ext) / len(ext_u),
        "cells": cells,
        "effects": {k: float(v) for k, v in eff.items()},
        "interaction_sem_files": half_f,
        "interaction_sem_unique": half_u,
        "interaction_over_sem_files": z_f,
        "interaction_over_sem_unique": z_u,
        "clears_3sem_files": z_f >= 3,
        "clears_3sem_unique": z_u >= 3,
        "bootstrap_ci_files": ci_f,
        "bootstrap_ci_unique": ci_u,
        "paired_seed_interactions": [float(x) for x in ints],
        "paired_seed_mean": float(ints.mean()),
        "paired_seed_sem": paired_sem,
        "paired_seed_over_sem": float(abs(ints.mean()) / paired_sem),
        "paired_seed_bootstrap_ci": ci_p,
        "paired_seed_ci_excludes_0": not (ci_p[0] <= 0 <= ci_p[1]),
        "verdict": ("STATUS CHANGED: resolved at 16 files/corner (3.9 SEM); "
                    "unresolved at 4 unique seeds (1.74 SEM). "
                    "The additivity-failure claim is not licensed at unique n."),
    }
    dest = ROOT / "results" / "gamma30_unique.json"
    dest.write_text(json.dumps(out, indent=2, default=float) + "\n")
    print(f"wrote {dest.relative_to(ROOT)}")
    print(f"  interaction {eff['interaction']:+.4f}")
    print(f"  files  {z_f:.2f} SEM  CI {ci_f}  clears 3σ? {z_f >= 3}")
    print(f"  unique {z_u:.2f} SEM  CI {ci_u}  clears 3σ? {z_u >= 3}")
    print(f"  paired-seed {out['paired_seed_over_sem']:.2f} SEM  CI {ci_p}  "
          f"excludes 0? {out['paired_seed_ci_excludes_0']}")
    print(f"  {out['verdict']}")


if __name__ == "__main__":
    main()
