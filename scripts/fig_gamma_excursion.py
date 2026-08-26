"""Figure: the γ manipulation measured in noise-floor units, on the paper's own measures.

Capacity-at-init flatness shows γ and `a` are orthogonal; this shows the γ knob
*does something*, quantified on `(α, D_eff, R_eff, ρ_c)` in units of each measure's
own Monte-Carlo noise floor. That is a stronger demonstration than capacity-at-init
alone, because it is expressed in the same currency as every effect the paper will
claim: an excursion of 65 floors at γ=10 against 0.2 at γ=0.03 says the manipulation
moves the measured quantities by tens of times the resolution limit at one end and
not at all at the other.

Reads `results/timewarp.json`; writes a tidy CSV and a two-panel PDF/PNG so the
numbers can be replotted without re-running anything.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.analysis import timewarp  # noqa: E402
from src.analysis.trajectories import CHANNELS, GeometryTrajectory  # noqa: E402

LABEL = {"alpha": r"$\alpha$", "D_eff": r"$D_{\mathrm{eff}}$",
         "R_eff": r"$R_{\mathrm{eff}}$", "rho_c_glue": r"$\rho_c$"}
ROOT = Path(__file__).resolve().parents[1]


def channel_movement(t: GeometryTrajectory, channel: str) -> float:
    """|log change| from first to last checkpoint, in units of the noise floor."""
    v = t.channels[channel]
    return float(abs(np.log(v[-1] / v[0])) / np.log1p(timewarp.NOISE_FLOOR_CV[channel]))


def main() -> None:
    data = json.loads((ROOT / "results" / "timewarp.json").read_text())
    trajs = [GeometryTrajectory.from_dict(t) for t in data["trajectories"]]
    gammas = sorted({t.gamma for t in trajs})

    rows = []
    for t in trajs:
        rows.append({"gamma": t.gamma, "seed": t.seed,
                     "excursion_total": t.excursion_in_floors(timewarp.NOISE_FLOOR_CV),
                     **{f"movement_{c}": channel_movement(t, c) for c in CHANNELS}})

    out_csv = ROOT / "results" / "fig_gamma_excursion.csv"
    with out_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    def agg(key):
        m = np.array([[r[key] for r in rows if r["gamma"] == g] for g in gammas])
        return m.mean(axis=1), m.std(axis=1, ddof=1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.6))

    mean, sd = agg("excursion_total")
    ax1.errorbar(gammas, mean, yerr=sd, marker="o", color="k", capsize=3, lw=1.6)
    ax1.axhline(1.0, ls=":", c="grey", lw=1)
    ax1.axhline(3.0, ls="--", c="crimson", lw=1)
    ax1.text(gammas[0], 3.4, "vacuity threshold", color="crimson", fontsize=8)
    ax1.text(gammas[0], 1.1, "noise floor", color="grey", fontsize=8)
    ax1.set(xscale="log", yscale="log", xlabel=r"richness $\gamma_0$",
            ylabel="geometric excursion (noise floors)",
            title="the $\\gamma$ knob, in the paper's own units")
    for g, m in zip(gammas, mean):
        ax1.annotate(f"{m:.1f}", (g, m), textcoords="offset points",
                     xytext=(6, -10), fontsize=8)

    for c in CHANNELS:
        m, s = agg(f"movement_{c}")
        ax2.errorbar(gammas, m, yerr=s, marker="o", capsize=3, lw=1.5, label=LABEL[c])
    ax2.axhline(1.0, ls=":", c="grey", lw=1)
    ax2.set(xscale="log", yscale="log", xlabel=r"richness $\gamma_0$",
            ylabel="movement (noise floors)", title="per measure")
    ax2.legend(frameon=False, fontsize=9, ncol=2)

    for ax in (ax1, ax2):
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    figdir = ROOT / "results" / "figures"
    figdir.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(figdir / f"gamma_excursion.{ext}", dpi=200, bbox_inches="tight")

    print(f"wrote {out_csv}")
    print(f"wrote {figdir}/gamma_excursion.{{pdf,png}}")
    print("\n  gamma   excursion (floors)   " +
          "  ".join(f"{c:>11}" for c in CHANNELS))
    tot, tots = agg("excursion_total")
    per = {c: agg(f"movement_{c}") for c in CHANNELS}
    for i, g in enumerate(gammas):
        cells = "  ".join(f"{per[c][0][i]:11.1f}" for c in CHANNELS)
        print(f"  {g:<7} {tot[i]:8.1f} +- {tots[i]:<7.1f} {cells}")


if __name__ == "__main__":
    main()
