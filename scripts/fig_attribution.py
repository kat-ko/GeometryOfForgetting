"""Draft Figure 2 — forgetting attributed to the three capacity factors, lazy vs rich.

The point of drawing this early, on pilot data, is that the encoding is where the
mistakes live: signed contributions do not stack the way unsigned ones do, and a
decomposition whose factors oppose each other can show a large `Δ log α` made of two
larger cancelling terms. Both are invisible in a table and obvious in a plot.

Top row: signed log-space contributions per lag, stacked above and below zero, with
the exact total `Δ log α` overlaid as a line. Because the identity closes exactly, the
line must land on the net top of the stack — that is a visual check of the arithmetic,
not decoration.

Bottom row: the center-collapse share of the radius channel, which is what makes
`00` §8's joint-reporting requirement quantitative.

Usage: `python scripts/fig_attribution.py [results/phase1_summary_pilot.json]`
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.analysis import attribution as AT  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FACTORS = ("utility", "radius", "dimension")
COLOR = {"utility": "#4C72B0", "radius": "#DD8452", "dimension": "#55A868"}
PRETTY = {"utility": r"$\Psi_{\mathrm{eff}}$ (utility)",
          "radius": r"$1+R_{\mathrm{eff}}^{-2}$ (radius)",
          "dimension": r"$-D_{\mathrm{eff}}$ (dimension)"}


def load(path: Path) -> dict:
    rows = {}
    for key, t in json.loads(path.read_text())["attribution"].items():
        gamma, a, cond, module, lag = key.split("|")
        rows.setdefault((float(gamma), cond, module), {})[int(lag)] = t
    return rows


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else \
        ROOT / "results" / "phase1_summary_pilot.json"
    rows = load(path)
    module = "A"
    gammas = sorted({g for g, _, m in rows if m == module})
    conds = sorted({c for _, c, m in rows if m == module})
    cond = "S-HL" if "S-HL" in conds else conds[0]

    fig, axes = plt.subplots(2, len(gammas), figsize=(4.4 * len(gammas), 6.4),
                             squeeze=False, sharey="row")

    shared_span = max((abs(t["dlog_alpha"]) for by in rows.values() for t in by.values()),
                      default=0.0)

    for j, g in enumerate(gammas):
        by_lag = rows.get((g, cond, module), {})
        lags = sorted(by_lag)
        ax = axes[0][j]
        pos = np.zeros(len(lags))
        neg = np.zeros(len(lags))
        for f in FACTORS:
            v = np.array([by_lag[l]["terms"][f] for l in lags])
            err = np.array([by_lag[l]["term_sem"][f] for l in lags])
            base = np.where(v >= 0, pos, neg)
            ax.bar(range(len(lags)), v, bottom=base, color=COLOR[f], width=0.62,
                   label=PRETTY[f] if j == 0 else None, edgecolor="white", linewidth=0.6)
            ax.errorbar(range(len(lags)), base + v, yerr=err, fmt="none",
                        ecolor="0.25", elinewidth=0.8, capsize=2)
            pos = pos + np.where(v >= 0, v, 0)
            neg = neg + np.where(v < 0, v, 0)

        total = np.array([by_lag[l]["dlog_alpha"] for l in lags])
        ax.plot(range(len(lags)), total, "o-", color="k", lw=1.8, ms=5,
                label=r"$\Delta\log\alpha$ (exact)" if j == 0 else None)
        ax.axhline(0, color="0.3", lw=0.8)
        ax.set(title=f"$\\gamma_0$ = {g:g}   ({cond}, module {module})",
               xlabel="boundaries since the task was learned")
        ax.set_xticks(range(len(lags)), [str(l) for l in lags])
        if j == 0:
            ax.set_ylabel(r"contribution to $\Delta\log\alpha_{\mathrm{retained}}$")
            # Below the axes: every term here is negative, so any in-axes corner the
            # legend could take is occupied by a bar.
            ax.legend(frameon=False, fontsize=8, loc="upper center",
                      bbox_to_anchor=(0.5, -0.16), ncol=4, columnspacing=1.2,
                      handlelength=1.4)

        # A shared y-axis is what makes the γ contrast legible, but it flattens the
        # lazy arm to a line and hides its internal structure. Inset it at native
        # scale when its excursion is a small fraction of the shared range.
        span = max(abs(np.concatenate([total, [0]])))
        if shared_span and span < 0.15 * shared_span:
            ins = ax.inset_axes((0.12, 0.12, 0.5, 0.45))
            ipos, ineg = np.zeros(len(lags)), np.zeros(len(lags))
            for f in FACTORS:
                v = np.array([by_lag[l]["terms"][f] for l in lags])
                base = np.where(v >= 0, ipos, ineg)
                ins.bar(range(len(lags)), v, bottom=base, color=COLOR[f], width=0.62,
                        edgecolor="white", linewidth=0.4)
                ipos = ipos + np.where(v >= 0, v, 0)
                ineg = ineg + np.where(v < 0, v, 0)
            ins.plot(range(len(lags)), total, "o-", color="k", lw=1.2, ms=3)
            ins.axhline(0, color="0.3", lw=0.6)
            ins.set_xticks([])
            ins.tick_params(labelsize=6)
            ins.set_title(f"native scale (×{shared_span / span:.0f})", fontsize=6.5)
            ins.spines[["top", "right"]].set_visible(False)

        # cancellation: how much of the total motion cancels between factors
        for i, l in enumerate(lags):
            motion = sum(abs(by_lag[l]["terms"][f]) for f in FACTORS)
            if motion > 0 and 1 - abs(total[i]) / motion > 0.25:
                ax.text(i, total[i], "  opposing", fontsize=7, color="crimson",
                        va="center")

        axc = axes[1][j]
        frac = [by_lag[l]["center_attributable_median"] for l in lags]
        shown = [(i, f) for i, f in enumerate(frac) if f is not None]
        if shown:
            axc.bar([i for i, _ in shown], [f for _, f in shown], width=0.62,
                    color="#8172B3")
        for i, f in enumerate(frac):
            if f is None:
                axc.text(i, 0.02, "n/a", ha="center", fontsize=7, color="0.4")
        axc.axhline(1.0, ls="--", color="crimson", lw=1)
        axc.axhline(0, color="0.3", lw=0.8)
        axc.set(xlabel="boundaries since the task was learned",
                ylim=(0, max(1.15, *(f for _, f in shown or [(0, 0)]))))
        axc.set_xticks(range(len(lags)), [str(l) for l in lags])
        if j == 0:
            axc.set_ylabel(r"share of $\Delta\log R_{\mathrm{eff}}$" "\n"
                           r"attributable to $\rho_c$")
            axc.text(0.02, 0.90, "1.0 = radius change is entirely center collapse",
                     transform=axc.transAxes, fontsize=7, color="crimson")
            axc.text(0.02, 0.80, f"calibration on {AT.RHO_R_CONVENTION}, "
                     f"valid for $\\rho_c\\in$[{AT.RHO_FIT_RANGE[0]:.2f}, "
                     f"{AT.RHO_FIT_RANGE[1]:.2f}]",
                     transform=axc.transAxes, fontsize=6.5, color="0.35")

    for ax in axes.ravel():
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Forgetting decomposes exactly: "
                 r"$\Delta\log\alpha=\Delta\log\Psi_{\mathrm{eff}}"
                 r"+\Delta\log(1+R_{\mathrm{eff}}^{-2})-\Delta\log D_{\mathrm{eff}}$",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    figdir = ROOT / "results" / "figures"
    figdir.mkdir(exist_ok=True)
    stem = "attribution_pilot" if "pilot" in path.name else "attribution"
    for ext in ("pdf", "png"):
        fig.savefig(figdir / f"{stem}.{ext}", dpi=200, bbox_inches="tight")
    print(f"wrote {figdir}/{stem}.{{pdf,png}}")

    for g in gammas:
        by_lag = rows.get((g, cond, module), {})
        for l in sorted(by_lag):
            t = by_lag[l]
            terms = "  ".join(f"{f[:3]} {t['terms'][f]:+.4f}" for f in FACTORS)
            print(f"  gamma {g:<6g} lag {l:<3d} dlogA {t['dlog_alpha']:+.4f}  {terms}"
                  f"  dom {t['dominant']:<9s} n={t['n']}")


if __name__ == "__main__":
    main()
