"""Figure 4 — center geometry across the Hiratani 2×2, and the death of H1d and H6.

Two registered predictions are tested here and both fail, in informative ways:

* **H1d** predicted that richer training *decorrelates* task centers. Three of the four
  corners converge instead; only `S-HL` decorrelates.
* **H6** predicted the largest |Δρ_c| at `S-HL`. The largest is at `S-LH`, twice the size.

What replaces them is a consistent corner ordering: readout similarity drives center
convergence, and the one corner with high feature but low readout similarity — the same
stimuli under different rules — moves the other way.

**The gate that makes this honest.** ρ_c has no measured Monte-Carlo floor, so the lazy
arms (γ = 0.03, where the representation barely moves) supply an empirical baseline: they
drift by +0.005 to +0.010 over the stream, common to all four corners and non-monotone
across blocks. That band is drawn on every panel. It matters twice: it is what makes
`S-LL` at γ = 10 (+0.011) an unresolved non-effect rather than a small convergence, and it
is what makes `S-HL` (−0.055) notable beyond its size, since it moves *against* the drift.
Because the baseline is common to the corners, the between-corner contrasts — which are the
actual claims — are immune to it.

    python scripts/fig4_corners_rho_c.py [--gamma 10]
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
from scipy import stats  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis import figstyle  # noqa: E402

figstyle.apply()
from src.analysis import grid as G  # noqa: E402

FIGDIR = ROOT / "figures"
CORNERS = ("S-HH", "S-HL", "S-LH", "S-LL")
# (feature similarity, readout similarity), src/manifolds/streams.py HIRATANI
CORNER_LABEL = {"S-HH": "S-HH  feat↑ read↑", "S-HL": "S-HL  feat↑ read↓  (catastrophic)",
                "S-LH": "S-LH  feat↓ read↑", "S-LL": "S-LL  feat↓ read↓"}
CORNER_COLOR = {"S-HH": "#4C72B0", "S-HL": "#C44E52",
                "S-LH": "#55A868", "S-LL": "#8172B3"}
BASELINE_GAMMA = 0.03


def result_set_sha(recs: list[dict]) -> str:
    h = hashlib.sha256()
    for r in sorted(recs, key=lambda r: r["key"]):
        h.update(r["key"].encode())
        h.update(str(r["code"].get("modules", {})).encode())
    return h.hexdigest()[:12]


def per_arm(recs: list[dict], gamma: float) -> dict[str, list[list[tuple[int, float]]]]:
    """`corner -> [[(block, ρ_c), ...], ...]`, one trajectory per arm.

    Per-arm rather than pooled because monotonicity is an arm-level property: a pooled mean
    can rise smoothly while no individual arm does.
    """
    out: dict[str, list] = defaultdict(list)
    for r in recs:
        if r["spec"]["gamma_0"] != gamma:
            continue
        pts = sorted((g["boundary"], g["rho_c_signed"]) for g in r["geometry"]
                     if g["ensemble"] == "generic")
        if len(pts) >= 4:
            out[r["spec"]["condition"]].append(pts)
    return out


def effects(mean: dict[str, float]) -> dict[str, float]:
    """The 2×2 read as two main effects plus an interaction, on `Δρ_c`.

    Half-differences, so each number is what flipping one factor does. This is the
    quantitative form of the corner pattern: an ordering merely ranks the cells, whereas
    additivity says the two similarity axes act separately and predicts the fourth cell
    from the other three.
    """
    hh, hl, lh, ll = (mean[c] for c in CORNERS)
    return {"readout": ((hh + lh) - (hl + ll)) / 2,
            "feature": ((hh + hl) - (lh + ll)) / 2,
            "interaction": ((hh - lh) - (hl - ll)) / 2,
            "additive_prediction_SHH": lh + hl - ll,
            "measured_SHH": hh}


def summarize(traj: dict[str, list]) -> dict:
    d = {}
    for c, v in traj.items():
        delta = [p[-1][1] - p[0][1] for p in v]
        rs = [stats.spearmanr([b for b, _ in p], [x for _, x in p]).statistic for p in v]
        rs = [x for x in rs if np.isfinite(x)]
        neg = sum(1 for x in rs if x < 0)
        d[c] = {
            "rho_c_first": float(np.mean([p[0][1] for p in v])),
            "rho_c_last": float(np.mean([p[-1][1] for p in v])),
            "delta_mean": float(np.mean(delta)),
            "delta_sem": float(np.std(delta, ddof=1) / np.sqrt(len(delta))),
            "spearman_median": float(np.median(rs)),
            "spearman_all": [float(x) for x in rs],
            "fraction_declining": neg / len(rs),
            "sign_test_p": float(stats.binomtest(neg, len(rs), 0.5).pvalue),
            "n_arms": len(v),
        }
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gamma", type=float, default=10.0)
    args = ap.parse_args()

    recs = G.load()
    if not recs:
        sys.exit("no arms on disk")

    rich = summarize(per_arm(recs, args.gamma))
    base = summarize(per_arm(recs, BASELINE_GAMMA))
    lo = min(base[c]["delta_mean"] for c in CORNERS)
    hi = max(base[c]["delta_mean"] for c in CORNERS)

    # Δρ_c against γ, for the emergence panel, and the effect decomposition per γ.
    gs = G.gammas(recs)
    sweep = {c: [] for c in CORNERS}
    eff_by_gamma = {}
    for g in gs:
        s = summarize(per_arm(recs, g))
        for c in CORNERS:
            sweep[c].append((s[c]["delta_mean"], s[c]["delta_sem"]) if c in s else (np.nan, 0))
        if all(c in s for c in CORNERS):
            eff_by_gamma[g] = effects({c: s[c]["delta_mean"] for c in CORNERS})

    # The same decomposition against width, using the W1 arms.
    eff_by_width = {}
    wrecs = G.load(arms_dir=ROOT / "results" / "width")
    for n, src in ((150, wrecs), (300, recs), (600, wrecs)):
        s = summarize(per_arm([r for r in src if r["spec"]["N"] == n], args.gamma))
        if all(c in s for c in CORNERS):
            eff_by_width[n] = effects({c: s[c]["delta_mean"] for c in CORNERS})
            eff_by_width[n]["n_arms_per_corner"] = min(s[c]["n_arms"] for c in CORNERS)

    fig, axes = plt.subplots(2, 3, figsize=figstyle.figsize(3.15))
    fig.suptitle(figstyle.wrap("Center correlation moves in opposite directions across the 2×2, "
                               "and the two similarity axes act almost separately"),
                 fontsize=figstyle.fs(12))

    # (a) trajectories over the stream
    ax = axes[0][0]
    tr = per_arm(recs, args.gamma)
    blocks = sorted({b for v in tr.values() for p in v for b, _ in p})
    for c in CORNERS:
        m = [np.mean([dict(p)[b] for p in tr[c]]) for b in blocks]
        e = [np.std([dict(p)[b] for p in tr[c]], ddof=1) / np.sqrt(len(tr[c]))
             for b in blocks]
        ax.errorbar(blocks, m, yerr=e, fmt="o-", color=CORNER_COLOR[c], lw=1.8, ms=4,
                    capsize=2, label=CORNER_LABEL[c])
    ax.set(xlabel="task boundary (block)", ylabel=r"$\rho_c$ (signed)",
           title=figstyle.wrap(f"(a) within-stream center correlation, "
                               f"$\\gamma_0$={args.gamma:g}", 30))
    ax.legend(frameon=False, fontsize=figstyle.fs(7.5), loc="upper left")
    ax.annotate("all four corners start together;\nthe split is training, not initialization",
                xy=(0.40, 0.06), xycoords="axes fraction", fontsize=figstyle.fs(7), color="0.35")

    # (b) Δρ_c across the whole sweep, with the lazy baseline band
    ax = axes[0][1]
    x = np.log10(gs)
    ax.axhspan(lo, hi, color="0.85", zorder=0)
    ax.text(x[0], hi * 1.9, "lazy-arm drift: unresolved", fontsize=figstyle.fs(7), color="0.35")
    for c in CORNERS:
        v = np.array([s[0] for s in sweep[c]])
        e = np.array([s[1] for s in sweep[c]])
        ax.errorbar(x, v, yerr=e, fmt="o-", color=CORNER_COLOR[c], lw=1.5, ms=4, capsize=2,
                    label=c)
    ax.axhline(0, color="0.3", lw=0.9)
    ax.set(xlabel=r"$\gamma_0$", ylabel=r"$\Delta\rho_c$ over the stream",
           title=figstyle.wrap("(b) the ordering is a rich-regime effect", 30))
    ax.set_xticks(x, [f"{g:g}" for g in gs], fontsize=figstyle.fs(7))
    ax.legend(frameon=False, fontsize=figstyle.fs(7.5), ncol=2)

    # (c) monotonicity: is the S-HL decline progressive, or an endpoint difference?
    #
    # This is what licenses comparison with practice-related decorrelation in humans,
    # which is a within-condition claim over time rather than a between-condition one.
    ax = axes[0][2]
    for i, c in enumerate(CORNERS):
        v = rich[c]["spearman_all"]
        ax.scatter(np.random.default_rng(i).normal(i, 0.07, len(v)), v, s=9,
                   color=CORNER_COLOR[c], alpha=0.55, edgecolors="none")
        ax.plot([i - 0.25, i + 0.25], [np.median(v)] * 2, color="k", lw=2)
        ax.text(i, 1.28, f"{rich[c]['fraction_declining'] * 100:.0f}%\ndeclining",
                ha="center", fontsize=figstyle.fs(7), color="0.3")
    ax.axhline(0, color="0.3", lw=0.9)
    # Rotated: 'S-HH S-HL S-LH S-LL' runs together in a 1.8 inch panel.
    ax.tick_params(axis='x', labelrotation=45)
    ax.set(xticks=range(4), xticklabels=CORNERS, ylim=(-1.45, 1.5),
           ylabel=r"per-arm Spearman($\rho_c$, block)",
           title=figstyle.wrap("(c) is the movement progressive? (bar = median)", 30))

    # (d) registered prediction against outcome
    ax = axes[1][0]
    v = [rich[c]["delta_mean"] for c in CORNERS]
    e = [rich[c]["delta_sem"] for c in CORNERS]
    ax.bar(range(4), v, yerr=e, width=0.6, color=[CORNER_COLOR[c] for c in CORNERS],
           capsize=3)
    ax.axhspan(lo, hi, color="0.85", zorder=0)
    ax.axhline(0, color="0.3", lw=0.9)
    imax = int(np.argmax(np.abs(v)))
    ax.set_ylim(min(v) - 0.062, max(v) + 0.052)
    ax.annotate("H6 predicted the\nlargest |Δ| here", xy=(1, v[1] - e[1]),
                xytext=(-0.42, -0.100), fontsize=figstyle.fs(7.5), color="crimson",
                arrowprops=dict(arrowstyle="->", color="crimson", lw=1))
    ax.annotate(f"largest |Δ| is {CORNERS[imax]},\n2× larger", xy=(imax, v[imax] + e[imax]),
                xytext=(1.70, 0.140), fontsize=figstyle.fs(7.5), color="crimson",
                arrowprops=dict(arrowstyle="->", color="crimson", lw=1))
    ax.annotate("H1d predicted decorrelation everywhere:\n3 of 4 converge instead",
                xy=(0.03, 0.90), xycoords="axes fraction",
                fontsize=figstyle.fs(7.5), color="crimson",
                va="top")
    # Rotated: 'S-HH S-HL S-LH S-LL' runs together in a 1.8 inch panel.
    ax.tick_params(axis='x', labelrotation=45)
    ax.set(xticks=range(4), xticklabels=CORNERS,
           ylabel=r"$\Delta\rho_c$ over the stream",
           title=figstyle.wrap(f"(d) H1d and H6 both fail "
                               f"($\\gamma_0$={args.gamma:g})", 30))

    # (e) the two similarity axes, separated
    ax = axes[1][1]
    xe = np.log10(sorted(eff_by_gamma))
    for k, col, lab in (("readout", "#2A6F97", "readout similarity → convergence"),
                        ("feature", "#B5651D", "feature similarity → decorrelation"),
                        ("interaction", "0.55", "interaction")):
        ax.plot(xe, [eff_by_gamma[g][k] for g in sorted(eff_by_gamma)], "o-", color=col,
                lw=1.8, ms=4.5, label=lab)
    ax.axhspan(-(hi - lo), hi - lo, color="0.85", zorder=0)
    ax.axhline(0, color="0.3", lw=0.9)
    ax.set(xlabel=r"$\gamma_0$", ylabel=r"effect on $\Delta\rho_c$ (half-difference)",
           title=figstyle.wrap("(e) richness sets the gain, the 2×2 sets the sign", 30))
    ax.set_xticks(xe, [f"{g:g}" for g in sorted(eff_by_gamma)], fontsize=figstyle.fs(7))
    ax.legend(frameon=False, fontsize=figstyle.fs(7.5), loc="center left")

    # (f) and the same decomposition across width
    ax = axes[1][2]
    ns = sorted(eff_by_width)
    w = 0.26
    for i, (k, col) in enumerate((("readout", "#2A6F97"), ("feature", "#B5651D"),
                                 ("interaction", "0.55"))):
        ax.bar(np.arange(len(ns)) + (i - 1) * w, [eff_by_width[n][k] for n in ns], width=w,
               color=col, label=k)
    ax.axhspan(-(hi - lo), hi - lo, color="0.85", zorder=0)
    ax.axhline(0, color="0.3", lw=0.9)
    ax.set(xticks=range(len(ns)),
           xticklabels=[f"N={n}\n({eff_by_width[n]['n_arms_per_corner']}/corner)"
                        for n in ns],
           ylabel=r"effect on $\Delta\rho_c$",
           title=figstyle.wrap(f"(f) both main effects survive 4× load "
                               f"($\\gamma_0$={args.gamma:g})", 30))
    ax.set_ylim(top=max(eff_by_width[n]["readout"] for n in ns) * 1.45)
    ax.legend(frameon=False, fontsize=figstyle.fs(7.5), ncol=3, loc="upper center")
    ax.annotate("interaction is at the resolution limit,\nand noisiest where arms are fewest",
                xy=(0.02, 0.86), xycoords="axes fraction",
                fontsize=figstyle.fs(7), color="0.4", va="top")

    for a in axes.ravel():
        a.spines[["top", "right"]].set_visible(False)
    # layout: constrained_layout, set in src/analysis/figstyle.apply()

    FIGDIR.mkdir(parents=True, exist_ok=True)
    sha = result_set_sha(recs)
    stem = f"fig4_corners_rho_c__gamma{args.gamma:g}__{len(recs)}arms__{sha}"
    for ext in ("pdf", "png"):
        fig.savefig(FIGDIR / f"{stem}.{ext}", dpi=170)
    (FIGDIR / f"{stem}.json").write_text(json.dumps(
        {"generated_by": "scripts/fig4_corners_rho_c.py", "n_arms": len(recs),
         "result_set_sha": sha, "gamma": args.gamma,
         "convention": "rho_c_signed, generic ensemble",
         "baseline": {"gamma": BASELINE_GAMMA, "band": [lo, hi], "per_corner": base},
         "rich": rich,
         "effects_by_gamma": eff_by_gamma,
         "effects_by_width": eff_by_width}, indent=2))
    print(f"wrote figures/{stem}.{{pdf,png,json}}\n")

    print(f"γ={args.gamma:g}   lazy baseline band [{lo:+.5f}, {hi:+.5f}]")
    print(f"{'corner':>7} {'ρ_c first':>10} {'ρ_c last':>9} {'Δρ_c':>9} {'±sem':>8} "
          f"{'×baseline':>10} {'median ρ_S':>11} {'%decl':>6} {'p':>9}  n")
    for c in CORNERS:
        d = rich[c]
        ratio = d["delta_mean"] / (hi if d["delta_mean"] > 0 else -lo)
        print(f"{c:>7} {d['rho_c_first']:>10.4f} {d['rho_c_last']:>9.4f} "
              f"{d['delta_mean']:>+9.4f} {d['delta_sem']:>8.4f} {ratio:>+10.1f} "
              f"{d['spearman_median']:>11.3f} {d['fraction_declining'] * 100:>5.0f}% "
              f"{d['sign_test_p']:>9.1e}  {d['n_arms']}")

    e10 = eff_by_gamma[args.gamma]
    print(f"\n2×2 as two main effects: readout {e10['readout']:+.4f}, "
          f"feature {e10['feature']:+.4f}, interaction {e10['interaction']:+.4f}")
    print(f"  additive model predicts S-HH = {e10['additive_prediction_SHH']:+.4f}, "
          f"measured {e10['measured_SHH']:+.4f}")


if __name__ == "__main__":
    main()
