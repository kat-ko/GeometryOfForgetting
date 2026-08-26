"""Width robustness in Figure 2's own form, so the comparison is visual, not tabular.

Same decomposition, same noise-floor gate, same lag as `fig2_gamma_sweep.py` — the only
change is that `N` becomes a third line rather than a fixed design choice. Load `P/N` spans
0.107 → 0.053 → 0.027, a 4× range around the design point.

**Like-for-like N=300.** The width arm ran 2 streams × 2 seeds, so the N=300 comparison is
restricted to the same subset rather than using all 40 arms per cell. Otherwise the middle
line would be an average over 10× more arms than its neighbours, and any difference could be
sampling rather than width. The full-grid value is written to the JSON as a check.

**Lag 12 is task 0 only** — no lag-12 comparison exists for any later task (see W5 in
`results/LOG.md`) — so this figure, like Figure 2, is a task-0 figure and shows the largest
forgetting in the grid rather than its average.

    python scripts/fig_width_invariance.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis import figstyle  # noqa: E402

figstyle.apply()
from src.analysis import grid as G  # noqa: E402
from src.analysis.attribution import FACTORS, attribution_table  # noqa: E402

MIN_FLOORS = 2.0
WIDTHS = (150, 300, 600)
GAMMAS = (0.03, 1.0, 10.0)
COLOR = {150: "#5B8FF9", 300: "#333333", 600: "#E8684A"}
FIGDIR = ROOT / "figures"


def result_set_sha(recs: list[dict]) -> str:
    """Hash of the arms behind the figure, so a PDF is traceable to its inputs."""
    h = hashlib.sha256()
    for r in sorted(recs, key=lambda r: r["key"]):
        h.update(r["key"].encode())
        h.update(str(r["code"].get("modules", {})).encode())
    return h.hexdigest()[:12]


def collect(lag: int) -> tuple[dict, dict, dict]:
    """`(N, γ) -> attribution table`, on matched streams and seeds.

    This figure is the only one that reads two result sets — the width arms and a matched subset
    of the registered grid — so it stamps both. A single `result_set_sha` would name one of its two
    inputs and imply it had named them all.
    """
    width = G.load(arms_dir=ROOT / "results" / "width")
    grid = G.load()
    matched = [r for r in grid if r["spec"]["stream_id"] < 2 and r["spec"]["seed"] < 2
               and r["spec"]["gamma_0"] in GAMMAS]
    full = [r for r in grid if r["spec"]["gamma_0"] in GAMMAS]

    def tab(recs):
        out = defaultdict(list)
        for spec, _m, _t, lg, att in G.attributions(recs, module="A"):
            if lg == lag:
                out[(spec.get("N", 300), spec["gamma_0"])].append(att)
        return {k: attribution_table(v) for k, v in out.items()}

    stamps = {"width_arms_sha": result_set_sha(width),
              "grid_matched_subset_sha": result_set_sha(matched),
              "grid_full_sha": result_set_sha(full),
              "n_arms": len(width) + len(matched)}
    return tab(width + matched), tab(full), stamps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lag", type=int, default=12)
    args = ap.parse_args()

    data, full, stamps = collect(args.lag)
    if not data:
        sys.exit("no width arms on disk")

    fig, axes = plt.subplots(2, 2, figsize=figstyle.figsize(3.5))
    fig.suptitle(figstyle.wrap(
        "The γ result is width-invariant; the one exception is the split *within* "
        f"the two non-utility channels   (lag {args.lag}, task 0, module A)"),
        fontsize=figstyle.fs(11))
    x = np.log10(GAMMAS)

    # (a) magnitude — the headline invariance
    ax = axes[0][0]
    for n in WIDTHS:
        fl = [G.floors(data[(n, g)]["dlog_alpha"], "alpha") for g in GAMMAS]
        ax.plot(x, fl, "o-", color=COLOR[n], lw=1.8, ms=5, label=f"N={n}  (P/N={16 / n:.3f})")
    ax.axhspan(-MIN_FLOORS, MIN_FLOORS, color="0.88", zorder=0)
    ax.text(x[0], MIN_FLOORS * 1.4, f"±{MIN_FLOORS:g} floors: below resolution",
            fontsize=figstyle.fs(7),
            color="0.35")
    gmax = max(GAMMAS)
    hi_fl = [G.floors(data[(n, gmax)]["dlog_alpha"], "alpha") for n in WIDTHS]
    spread = (max(hi_fl) - min(hi_fl)) / abs(np.mean(hi_fl))
    ax.set(ylabel=r"$\Delta\log\alpha$ (noise floors)", xlabel=r"$\gamma_0$",
           title=f"(a) how much is lost — {100 * spread:.1f}% spread at γ={gmax:g}")
    ax.legend(frameon=False, fontsize=figstyle.fs(8))

    # (b) composition at each width, gated exactly as Figure 2 gates it
    ax = axes[0][1]
    w = 0.24
    for i, n in enumerate(WIDTHS):
        for j, g in enumerate(GAMMAS):
            t = data[(n, g)]
            if abs(G.floors(t["dlog_alpha"], "alpha")) < MIN_FLOORS:
                ax.text(j + (i - 1) * w, 0.5, "n/r", ha="center", va="center",
                        fontsize=figstyle.fs(6),
                        color="0.45", rotation=90)
                continue
            bottom = 0.0
            for f in FACTORS:
                v = t["shares"][f]
                ax.bar(j + (i - 1) * w, v, bottom=bottom, width=w * 0.92,
                       color=G.FACTOR_COLORS[f], edgecolor="white", linewidth=0.5)
                bottom += v
    for f in FACTORS:
        ax.bar([], [], color=G.FACTOR_COLORS[f], label=G.FACTOR_LABELS[f])
    ax.set(xticks=range(len(GAMMAS)),
           xticklabels=[f"γ={g:g}\nN=150 · 300 · 600" for g in GAMMAS], ylim=(0, 1),
           ylabel="share of total motion", title="(b) which channel carries it, per width")
    ax.legend(frameon=False, fontsize=figstyle.fs(7.5), loc="upper center",
              bbox_to_anchor=(0.5, -0.14),
              ncol=3)

    # (c) the invariant: utility share
    ax = axes[1][0]
    for n in WIDTHS:
        v = [data[(n, g)]["shares"]["utility"] for g in GAMMAS]
        ok = [abs(G.floors(data[(n, g)]["dlog_alpha"], "alpha")) >= MIN_FLOORS for g in GAMMAS]
        ax.plot(np.array(x)[ok], np.array(v)[ok], "o-", color=COLOR[n], lw=1.8, ms=5,
                label=f"N={n}")
    uti = [data[(n, gmax)]["shares"]["utility"] for n in WIDTHS]
    ax.set(ylabel="utility share of total motion", xlabel=r"$\gamma_0$", ylim=(0, 0.6),
           title=f"(c) the utility share is width-invariant "
                 f"({min(uti):.3f}–{max(uti):.3f} at γ={gmax:g})")
    ax.legend(frameon=False, fontsize=figstyle.fs(8))

    # (d) the one real N-dependence, and the fact that it cancels
    ax = axes[1][1]
    g = max(GAMMAS)
    rad = [data[(n, g)]["shares"]["radius"] for n in WIDTHS]
    dim = [data[(n, g)]["shares"]["dimension"] for n in WIDTHS]
    xs = np.arange(len(WIDTHS))
    ax.plot(xs, rad, "o-", color=G.FACTOR_COLORS["radius"], lw=1.8, ms=6, label="radius")
    ax.plot(xs, dim, "o-", color=G.FACTOR_COLORS["dimension"], lw=1.8, ms=6, label="dimension")
    ax.plot(xs, np.add(rad, dim), "s--", color="0.35", lw=1.6, ms=5,
            label="their sum (stable)")
    for xx, s in zip(xs, np.add(rad, dim)):
        ax.annotate(f"{s:.3f}", (xx, s), textcoords="offset points", xytext=(0, 7),
                    ha="center", fontsize=figstyle.fs(7), color="0.35")
    ax.set(xticks=xs, xticklabels=[f"N={n}" for n in WIDTHS], ylim=(0, 0.72),
           ylabel="share of total motion",
           title=f"(d) the exception: the radius/dimension split moves (γ={g:g})")
    ax.annotate("the utility/non-utility division is width-invariant;\n"
                "what moves is how the non-utility part is spent", xy=(0.03, 0.06),
                xycoords="axes fraction", fontsize=figstyle.fs(7), color="0.35")
    ax.legend(frameon=False, fontsize=figstyle.fs(8), loc="upper right")

    for a in (axes[0][0], axes[1][0]):
        a.set_xticks(x, [f"{gg:g}" for gg in GAMMAS])
    for a in axes.ravel():
        a.spines[["top", "right"]].set_visible(False)
    # layout: constrained_layout, set in src/analysis/figstyle.apply()

    FIGDIR.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256(str(sorted(data)).encode()).hexdigest()[:12]
    stem = f"fig_width_invariance__lag{args.lag}__{h}"
    for ext in ("pdf", "png"):
        fig.savefig(FIGDIR / f"{stem}.{ext}", dpi=170)
    out = {"generated_by": "scripts/fig_width_invariance.py", "lag": args.lag,
           "result_set_sha": stamps["width_arms_sha"], "n_arms": stamps["n_arms"],
           "result_sets": {k: v for k, v in stamps.items() if k != "n_arms"},
           "min_floors": MIN_FLOORS, "matched_subset": "streams 0-1, seeds 0-1",
           "note": "lag 12 exists only for task 0; this is a task-0 figure",
           "cells": {f"N{n}_g{g:g}": {"dlog_alpha": data[(n, g)]["dlog_alpha"],
                                      "floors": G.floors(data[(n, g)]["dlog_alpha"], "alpha"),
                                      "shares": data[(n, g)]["shares"],
                                      "n": data[(n, g)]["n"]}
                     for n in WIDTHS for g in GAMMAS},
           "n300_full_grid_check": {f"g{g:g}": {"dlog_alpha": full[(300, g)]["dlog_alpha"],
                                                "shares": full[(300, g)]["shares"],
                                                "n": full[(300, g)]["n"]} for g in GAMMAS}}
    (FIGDIR / f"{stem}.json").write_text(json.dumps(out, indent=2))
    print(f"wrote figures/{stem}.{{pdf,png,json}}\n")

    print(f"{'N':>5} {'γ':>6} {'floors':>7} {'uti':>7} {'rad':>7} {'dim':>7} {'rad+dim':>8} "
          f"{'n':>5}")
    for n in WIDTHS:
        for g in GAMMAS:
            s = data[(n, g)]["shares"]
            print(f"{n:>5} {g:>6g} {G.floors(data[(n, g)]['dlog_alpha'], 'alpha'):>7.1f} "
                  f"{s['utility']:>7.3f} {s['radius']:>7.3f} {s['dimension']:>7.3f} "
                  f"{s['radius'] + s['dimension']:>8.3f} {data[(n, g)]['n']:>5}")
    print("\nmatched-subset check against the full N=300 grid:")
    for g in GAMMAS:
        a, b = data[(300, g)], full[(300, g)]
        print(f"  γ={g:<5g} matched {a['dlog_alpha']:+.4f} (n={a['n']:>4})   "
              f"full {b['dlog_alpha']:+.4f} (n={b['n']:>4})   "
              f"utility share {a['shares']['utility']:.3f} vs {b['shares']['utility']:.3f}")


if __name__ == "__main__":
    main()
