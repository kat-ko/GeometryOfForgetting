"""Score the width smoke against its pre-commit and plot ReLU clouds.

Reads `results/width_smoke_precommit.json` and `results/width_smoke/`. Writes
`results/width_smoke.json` and `results/width_smoke/fig_*.png`. Does not train.

    python scripts/analyse_width_smoke.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PRE = json.loads((ROOT / "results" / "width_smoke_precommit.json").read_text())
DIR = ROOT / "results" / "width_smoke"
OUT = ROOT / "results" / "width_smoke.json"
ALPHA_REF = 0.33
ALPHA_TOL = 0.15


def _geom(rec: dict, *, module: str, ensemble: str, task, boundary: int) -> dict | None:
    for g in rec["geometry"]:
        if (g["module"] == module and g["ensemble"] == ensemble
                and g["task"] == task and g["boundary"] == boundary):
            return g
    return None


def _dlog_alpha(rec: dict, *, lag: int = 15) -> float | None:
    for a in rec.get("attribution", []):
        if a["module"] == "A" and a["task"] == 0 and a["lag"] == lag:
            return float(a["dlog_alpha"])
    return None


def _rho_c(rec: dict, boundary: int) -> float | None:
    g = _geom(rec, module="A", ensemble="retained", task=0, boundary=boundary)
    return None if g is None else float(g["rho_c_signed"])


def summarise_arm(rec: dict) -> dict:
    tasks = rec["tasks"]
    g0 = _geom(rec, module="A", ensemble="generic", task=None, boundary=0)
    g15 = _geom(rec, module="A", ensemble="generic", task=None, boundary=15)
    acc = np.asarray(rec["accuracy_matrix"], dtype=float)
    rho0, rho15 = _rho_c(rec, 0), _rho_c(rec, 15)
    return {
        "N": rec["spec"]["N"],
        "gamma_0": rec["spec"]["gamma_0"],
        "usable": bool(rec.get("usable")),
        "n_converged": int(sum(t["converged"] for t in tasks)),
        "n_tasks": len(tasks),
        "task0_converged": bool(tasks[0]["converged"]),
        "task0_final_loss": float(tasks[0]["final_loss"]),
        "task0_steps": int(tasks[0]["steps_taken"]),
        "task0_train_accuracy": float(tasks[0]["train_accuracy"]),
        "mean_final_loss": float(np.mean([t["final_loss"] for t in tasks])),
        "mean_steps": float(np.mean([t["steps_taken"] for t in tasks])),
        "mean_train_accuracy": float(np.mean([t["train_accuracy"] for t in tasks])),
        "final_mean_accuracy": rec.get("forgetting", {}).get("final_mean_accuracy"),
        "CF": rec.get("forgetting", {}).get("CF"),
        "alpha_generic_b0": None if g0 is None else float(g0["alpha"]),
        "alpha_generic_b15": None if g15 is None else float(g15["alpha"]),
        "D_eff_generic_b0": None if g0 is None else float(g0["D_eff"]),
        "D_eff_generic_b15": None if g15 is None else float(g15["D_eff"]),
        "identity_max": float(max(abs(g["identity_residual"]) for g in rec["geometry"])),
        "dlog_alpha_task0_lag15": _dlog_alpha(rec),
        "delta_rho_c_task0": None if None in (rho0, rho15) else float(rho15 - rho0),
        "weight_change_A_end": float(rec["manipulation_checks"][-1]["weight_change"]["A"]),
        "diag_acc_mean": float(np.nanmean(np.diag(acc))),
        "strictly_lower_acc_mean": float(np.nanmean(np.tril(acc, k=-1))),
        "wall_seconds": rec.get("wall_seconds"),
    }


def pca2(H: np.ndarray) -> np.ndarray:
    """(P, M, N) -> (P, M, 2) per-snapshot PCA."""
    P, M, N = H.shape
    X = H.reshape(P * M, N).astype(np.float64)
    X = X - X.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(X, full_matrices=False)
    k = min(2, vt.shape[0])
    Z = X @ vt[:k].T
    if k == 1:
        Z = np.hstack([Z, np.zeros((Z.shape[0], 1))])
    return Z.reshape(P, M, 2)


def plot_init_curve(curve: dict, path: Path) -> None:
    ns = [r["N"] for r in curve["by_N"]]
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.plot(ns, [r["A"]["median_mse"] for r in curve["by_N"]], "o-",
            color="#1d4ed8", label="module A, median MSE")
    ax.plot(ns, [r["AB"]["median_mse"] for r in curve["by_N"]], "s--",
            color="#64748b", label="A+B concatenated")
    ax.axhline(0.05, color="#b45309", ls=":", lw=1.2, label="train target 0.05")
    ax.axvline(53, color="#9f1239", ls="--", lw=1, label="N≈53 (P/N=α_c)")
    ax.axvline(80, color="#a16207", ls="--", lw=1, label="N=80 (packing)")
    ax.set_xscale("log")
    ax.set_xticks(ns, [str(n) for n in ns])
    ax.set_xlabel("hidden width N")
    ax.set_ylabel("exact linear-readout MSE at init")
    ax.set_title("Frozen-W readout floor vs width (32 random dichotomies + y0)")
    ax.legend(fontsize=8, frameon=False)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_clouds(dir_: Path, path: Path) -> dict:
    files = sorted(dir_.glob("N32__*.clouds.npz"))
    info = {"n_files": len(files), "init_i6_max_abs_diff": None}
    if not files:
        return info
    fig, axes = plt.subplots(len(files), 3, figsize=(9.2, 2.8 * len(files)),
                             squeeze=False)
    inits = []
    cmap = plt.get_cmap("tab20")
    snapshots = ("init", "t0", "t15")
    titles = ("init (before SGD)", "after task 0", "after task 15 (task-0 manifolds)")
    for i, f in enumerate(files):
        z = np.load(f)
        g = float(z["gamma_0"])
        inits.append(np.asarray(z["init"]))
        for j, key in enumerate(snapshots):
            ax = axes[i, j]
            if key not in z.files:
                ax.set_axis_off()
                continue
            H = np.asarray(z[key])
            Z = pca2(H)
            P = Z.shape[0]
            for mu in range(P):
                ax.scatter(Z[mu, :, 0], Z[mu, :, 1], s=6, alpha=0.55,
                           color=cmap(mu % 20), linewidths=0)
                ax.scatter(Z[mu, :, 0].mean(), Z[mu, :, 1].mean(), s=36,
                           color=cmap(mu % 20), edgecolors="k", linewidths=0.4, zorder=3)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal", adjustable="datalim")
            if i == 0:
                ax.set_title(titles[j], fontsize=10)
            if j == 0:
                ax.set_ylabel(f"N=32  γ={g:g}", fontsize=10)
    if len(inits) == 2:
        info["init_i6_max_abs_diff"] = float(np.max(np.abs(inits[0] - inits[1])))
    fig.suptitle("Module-A ReLU clouds, 16 manifolds (colour) × 150 points; 2D PCA per panel",
                 fontsize=11, y=1.01)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return info


def score(curve: dict, arms: list[dict]) -> dict:
    by = {(a["N"], a["gamma_0"]): a for a in arms}
    a32 = curve["by_N"][0]["A"] if curve["by_N"][0]["N"] == 32 else next(
        r["A"] for r in curve["by_N"] if r["N"] == 32)
    a100 = next(r["A"] for r in curve["by_N"] if r["N"] == 100)

    def arm(N, g):
        return by.get((N, g))

    n100_lazy, n100_rich = arm(100, 0.03), arm(100, 10.0)
    n32_lazy, n32_rich = arm(32, 0.03), arm(32, 10.0)

    def alpha_ok(a):
        if a is None or a["alpha_generic_b0"] is None:
            return False
        return abs(a["alpha_generic_b0"] - ALPHA_REF) / ALPHA_REF <= ALPHA_TOL

    def identity_ok(a):
        return a is not None and a["identity_max"] < 1e-10

    n100_go = bool(
        n100_lazy and n100_rich
        and n100_lazy["usable"] and n100_rich["usable"]
        and alpha_ok(n100_lazy) and alpha_ok(n100_rich)
        and identity_ok(n100_lazy) and identity_ok(n100_rich)
    )
    n100_stop = bool(
        n100_lazy and (not n100_lazy["usable"] or not alpha_ok(n100_lazy)
                       or not identity_ok(n100_lazy))
    )
    n100_go_failure = []
    if n100_lazy is None or n100_rich is None:
        n100_go_failure.append("missing arm")
    else:
        if not n100_lazy["usable"]:
            n100_go_failure.append("lazy missed 0.05")
        if not n100_rich["usable"]:
            n100_go_failure.append("rich missed 0.05")
        if n100_lazy["alpha_generic_b0"] is not None and not alpha_ok(n100_lazy):
            n100_go_failure.append(
                f"lazy α_generic@b0={n100_lazy['alpha_generic_b0']:.3f} outside 15% of 0.33")
        if n100_rich["alpha_generic_b0"] is not None and not alpha_ok(n100_rich):
            n100_go_failure.append(
                f"rich α_generic@b0={n100_rich['alpha_generic_b0']:.3f} outside 15% of 0.33 "
                "(boundary 0 is after task 0; H2a already says rich generic α rises)"
            )
        if not identity_ok(n100_lazy) or not identity_ok(n100_rich):
            n100_go_failure.append("identity residual")
    n32_go = bool(
        (a32["median_mse"] >= 0.05 or a32["frac_perfect_sign"] <= 0.5)
        and n32_rich is not None and n32_rich["task0_converged"]
    )
    n32_broken = bool(n32_rich is not None and not n32_rich["task0_converged"])
    return {
        "n100_go_small_robustness": n100_go,
        "n100_stop_robustness": n100_stop and not n100_go,
        "n32_go_capacity_limited": n32_go,
        "n32_stop_broken": n32_broken,
        "n32_lazy_fail_is_predicted": bool(
            n32_lazy is not None and not n32_lazy["usable"]
        ),
        "n100_go_failure": n100_go_failure,
        "init_A_N32_median_mse": a32["median_mse"],
        "init_A_N32_frac_perfect_sign": a32["frac_perfect_sign"],
        "init_A_N100_median_mse": a100["median_mse"],
        "init_A_N100_frac_perfect_sign": a100["frac_perfect_sign"],
    }


def main() -> None:
    if not (DIR / "init_readout.json").is_file():
        raise SystemExit("run scripts/run_width_smoke.py --init-only first")
    curve = json.loads((DIR / "init_readout.json").read_text())
    recs = []
    for p in sorted(DIR.glob("N*.json")):
        recs.append(json.loads(p.read_text()))
    arms = [summarise_arm(r) for r in recs]
    verdict = score(curve, arms) if recs else {
        "pending_full_arms": True,
        "n100_go_small_robustness": None,
        "n32_go_capacity_limited": None,
    }
    if recs:
        # re-score with full rules; init-only still informs n32_go via curve
        pass
    else:
        a32 = next(r["A"] for r in curve["by_N"] if r["N"] == 32)
        verdict["n32_init_suggests_capacity_limited"] = bool(
            a32["median_mse"] >= 0.05 or a32["frac_perfect_sign"] <= 0.5
        )
    plot_init_curve(curve, DIR / "fig_init_readout.png")
    clouds = plot_clouds(DIR, DIR / "fig_clouds_N32.png")
    payload = {
        "generated_by": "scripts/analyse_width_smoke.py",
        "precommit": PRE,
        "init_readout": {
            "by_N": [
                {
                    "N": r["N"], "P_over_N": r["P_over_N"],
                    "packing": r["packing_P_Dp1_over_N"],
                    "A": {k: r["A"][k] for k in (
                        "median_mse", "p90_mse", "frac_mse_le_target",
                        "frac_perfect_sign", "y0_mse", "y0_sign_accuracy", "rank",
                    )},
                    "AB": {k: r["AB"][k] for k in (
                        "median_mse", "frac_mse_le_target", "frac_perfect_sign",
                    )},
                }
                for r in curve["by_N"]
            ]
        },
        "arms": arms,
        "clouds": clouds,
        "verdict": verdict,
        "reading": (
            "Smoke, n=1 per cell. GO/STOP are whether a larger sweep is licensed, "
            "not findings about γ or forgetting."
        ),
    }
    OUT.write_text(json.dumps(payload, indent=2))
    print(json.dumps({"verdict": verdict, "n_arms": len(arms),
                      "clouds": clouds, "wrote": str(OUT.relative_to(ROOT))},
                     indent=2))


if __name__ == "__main__":
    main()
