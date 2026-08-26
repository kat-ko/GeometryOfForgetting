"""Score S-HH N=32/100 against `results/shh_width_precommit.json`.

    python scripts/analyse_shh_width.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PRE = json.loads((ROOT / "results" / "shh_width_precommit.json").read_text())
DIR = ROOT / "results" / "shh_width"
OUT = ROOT / "results" / "shh_width.json"


def past_task_acc(rec: dict) -> float:
    acc = np.asarray(rec["accuracy_matrix"], dtype=float)
    return float(np.nanmean(acc[-1, :-1]))


def summarise(rec: dict) -> dict:
    tasks = rec["tasks"]
    return {
        "N": rec["spec"]["N"],
        "gamma_0": rec["spec"]["gamma_0"],
        "seed": rec["spec"]["seed"],
        "usable": bool(rec.get("usable")),
        "n_converged": int(sum(t["converged"] for t in tasks)),
        "task0_converged": bool(tasks[0]["converged"]),
        "task0_final_loss": float(tasks[0]["final_loss"]),
        "CF": float(rec["forgetting"]["CF"]),
        "past_task_acc": past_task_acc(rec),
        "final_mean_accuracy": rec["forgetting"].get("final_mean_accuracy"),
        "diag_acc_mean": float(np.nanmean(np.diag(np.asarray(rec["accuracy_matrix"], float)))),
        "weight_change_A_end": float(rec["manipulation_checks"][-1]["weight_change"]["A"]),
    }


def cell_stats(rows: list[dict]) -> dict:
    cfs = np.array([r["CF"] for r in rows], dtype=float)
    past = np.array([r["past_task_acc"] for r in rows], dtype=float)
    return {
        "n": len(rows),
        "mean_CF": float(np.mean(cfs)),
        "min_CF": float(np.min(cfs)),
        "max_CF": float(np.max(cfs)),
        "mean_past_task_acc": float(np.mean(past)),
        "min_past_task_acc": float(np.min(past)),
        "max_past_task_acc": float(np.max(past)),
        "n_usable": int(sum(r["usable"] for r in rows)),
        "per_seed": rows,
    }


def verdict(cells: dict) -> dict:
    def get(N, g):
        return cells.get((N, g))

    n32_lazy, n32_rich = get(32, 0.03), get(32, 10.0)
    n100_lazy, n100_rich = get(100, 0.03), get(100, 10.0)

    def survives(cell):
        if cell is None or cell["n"] < 4:
            return False
        return (
            cell["mean_CF"] <= 0.01
            and cell["mean_past_task_acc"] >= 0.99
            and cell["max_CF"] <= 0.02
        )

    def breaks(cell):
        if cell is None or cell["n"] < 4:
            return False
        return cell["mean_CF"] >= 0.05 or cell["mean_past_task_acc"] < 0.95

    n32_survives = bool(survives(n32_lazy) and survives(n32_rich))
    n32_breaks = bool(breaks(n32_lazy) or breaks(n32_rich))
    if n32_survives and n32_breaks:
        label = "contradictory_rules"
    elif n32_survives:
        label = "split_survives"
    elif n32_breaks:
        label = "split_breaks"
    else:
        label = "intermediate"
    return {
        "label": label,
        "n32_survives": n32_survives,
        "n32_breaks": n32_breaks,
        "n100_survives": bool(survives(n100_lazy) and survives(n100_rich)),
        "n100_breaks": bool(breaks(n100_lazy) or breaks(n100_rich)),
        "expand": label == "split_breaks",
    }


def main() -> None:
    recs = [json.loads(p.read_text()) for p in sorted(DIR.glob("N*.json"))]
    if not recs:
        raise SystemExit(f"no arms in {DIR}")
    arms = [summarise(r) for r in recs]
    grouped: dict[tuple, list] = defaultdict(list)
    for a in arms:
        grouped[(a["N"], a["gamma_0"])].append(a)
    cells = {k: cell_stats(v) for k, v in grouped.items()}
    cells_out = {f"N{k[0]}_g{k[1]:g}": v for k, v in sorted(cells.items())}
    # drop per_seed duplication of huge? per_seed is small dicts, ok
    v = verdict(cells)
    payload = {
        "generated_by": "scripts/analyse_shh_width.py",
        "precommit": PRE,
        "n_arms": len(arms),
        "cells": cells_out,
        "verdict": v,
        "reading": (
            "Sampling unit is init, one arrangement. split_survives → stop. "
            "split_breaks → sequel paragraph, then more arrangements, not a GLUE grid."
        ),
    }
    OUT.write_text(json.dumps(payload, indent=2))
    print(json.dumps({"verdict": v, "cells": {
        k: {kk: vv for kk, vv in cell.items() if kk != "per_seed"}
        for k, cell in cells_out.items()
    }}, indent=2))


if __name__ == "__main__":
    main()
