"""Figure 2 — how forgetting decomposes across the richness sweep.

    Δ log α = Δ log Ψ_eff + Δ log(1 + R_eff⁻²) − Δ log D_eff

Four panels: the magnitude of forgetting per γ in noise-floor units; the channel
composition where it is resolvable; the signed utility term, whose zero crossing is the
qualitative change in the sweep; and the center-collapse share of the radius channel.

**Why the default is three conditions and not four.** `S-HH` — high feature *and* high readout
similarity, registered in `01` §3.1 as the benign corner — does not forget. Its retained capacity
**rises** (Δ log α = +0.11 to +0.22 at every γ) and its behavioral forgetting is zero
(−0.0005 to +0.0001), both as predicted for that corner. Pooling it with the three conditions
that do forget averages a gain against three losses, which showed up three ways: panel (a)'s
axis was untrue of a quarter of what it pooled; the γ = 0.3 utility share read 0.101 rather than
0.189 because `S-HH`'s utility term enters with the opposite sign; and at γ = 1 the pooled
utility share (0.284) fell *below all four* corner values through the same cancellation. So the
default is `--condition forgetting`, and `--condition S-HH` draws the benign corner on its own.
Separating them makes the trend cleaner as well as truer: every γ becomes resolvable, and no term
crosses zero anywhere in the range.

**What the default lag means.** `--lag 12` is not one lag among many pooled over tasks: in a
16-task stream with boundaries at 0, 4, 8, 12, 15, a lag of 12 exists **only for task 0**
(verified: 960 of 960 records). So this figure is a task-0 figure by construction, immune to
the lag/task-position confound of W5 — and, because task 0 is the most-forgotten task in the
stream, it reports the largest forgetting in the grid rather than its average. Both belong in
the caption.

**The gate that makes this honest.** Channel shares are `|term| / Σ|term|`, which is
defined whether or not anything moved — so at γ ≤ 0.1, where total capacity change is
0.06–0.48 noise floors, a naive plot shows confident-looking shares attributing a change
that did not happen. Worse, the pooled utility share at γ = 0.3 collapses to 0.02 not
because utility is inactive but because its term is *crossing zero* there, so positive and
negative arms cancel in the mean. Shares are therefore drawn only where |Δ log α| clears
`MIN_FLOORS`, and the sign structure gets its own panel rather than being hidden inside an
absolute-value ratio.

**Panel (d)'s gate varies by bar.** Its denominator is floored at three times the `R_eff` noise
floor, which the per-γ measurement null showed to be a function of γ, not a constant (CV 0.28% to
1.13%). Each bar is therefore gated at its own threshold and the coverage percentages are not
comparable across bars, so both the coverage and the threshold are printed. Panels (a)–(c) do
not use this floor and are unaffected by it.

    python scripts/fig2_gamma_sweep.py [--condition forgetting|pooled|S-HH|...] [--lag 12]
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
from src.analysis import attribution as ATT  # noqa: E402
from src.analysis.attribution import FACTORS, attribution_table  # noqa: E402

# Below this, |Δ log α| is not distinguishable from Monte-Carlo noise and there is
# nothing to attribute. Two floors is a deliberately loose bar.
MIN_FLOORS = 2.0
# W4: a CV from n=4 measurement seeds has relative SE 1/√(2(n−1)) ≈ 41%.
# A bar that clears ±2 floors by less than that is drawn hatched: the margin sits
# inside the floor's own uncertainty. The gate itself is not moved.
FLOOR_N_SEEDS = 4
FLOOR_REL_SE = 1.0 / np.sqrt(2 * (FLOOR_N_SEEDS - 1))
FIGDIR = ROOT / "figures"

# The three conditions that lose retained capacity. `S-HH` is excluded by name rather than by a
# sign test on the data, so which conditions the figure pools is a stated choice and not an
# outcome of the numbers it then reports.
FORGETTING = ("S-HL", "S-LH", "S-LL")


def result_set_sha(recs: list[dict]) -> str:
    """Hash of the arms behind the figure, so a PDF is traceable to its inputs."""
    h = hashlib.sha256()
    for r in sorted(recs, key=lambda r: r["key"]):
        h.update(r["key"].encode())
        h.update(str(r["code"].get("modules", {})).encode())
    return h.hexdigest()[:12]


def collect(recs: list[dict], conditions: tuple[str, ...] | None, lag: int | None,
            only_task: int | None = None) -> dict:
    """Per-γ pooled attribution at matched lag, plus the noise-floor gate.

    Matched lag matters: pooling every lag together mixes lag-1 comparisons (little
    forgetting yet) with lag-15 (much), so a γ difference in composition could be a
    difference in *how far through the stream* the average comparison sits.

    `only_task` matters for the same reason one step in. Every lag below 15 is available to
    several tasks, and task position has its own effect on forgetting — larger than the lag
    effect, as it turns out — so a lag comparison pooled over tasks varies two things at once.
    Fixing the task makes a change in lag a change in lag.
    """
    atts = defaultdict(list)
    for spec, _mod, task, lg, att in G.attributions(recs, module="A"):
        if conditions and spec["condition"] not in conditions:
            continue
        if lag is not None and lg != lag:
            continue
        if only_task is not None and task != only_task:
            continue
        atts[spec["gamma_0"]].append(att)

    out = {}
    for g, v in sorted(atts.items()):
        t = attribution_table(v)
        signed = {f: float(np.mean([a.terms[f] for a in v])) for f in FACTORS}
        sem = {f: float(np.std([a.terms[f] for a in v], ddof=1) / np.sqrt(len(v)))
               for f in FACTORS}
        flips = {f: float(np.mean([a.terms[f] > 0 for a in v])) for f in FACTORS}
        cen = [a.center["attributable_fraction"] for a in v
               if a.center.get("attributable_fraction") is not None]
        out[g] = {
            "dlog_alpha": t["dlog_alpha"],
            "floors": G.floors(t["dlog_alpha"], "alpha"),
            # `G.floors` is a magnitude. Panel (a) needs the sign, or a condition that *gains*
            # retained capacity plots as though it had lost that much.
            "floors_signed": float(np.sign(t["dlog_alpha"])
                                   * G.floors(t["dlog_alpha"], "alpha")),
            "terms": signed, "term_sem": sem, "sign_positive_fraction": flips,
            "shares": t["shares"], "n": t["n"],
            "cancellation_median": float(np.median([a.meta["cancellation"] for a in v])),
            "center_share_median": float(np.median(cen)) if cen else None,
            "center_coverage": len(cen) / len(v),
            "R_eff_floor_cv": ATT.R_EFF_FLOOR_CV.get(g, max(ATT.R_EFF_FLOOR_CV.values())),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", default="forgetting",
                    help="'forgetting' (the three that lose capacity), 'pooled' (all four, "
                         "which mixes signs), or one condition name")
    ap.add_argument("--lag", type=int, default=12)
    ap.add_argument("--task", type=int, default=None,
                    help="restrict to one task position, so a lag change is only a lag change")
    args = ap.parse_args()
    conds = ({"pooled": None, "forgetting": FORGETTING}
             .get(args.condition, (args.condition,)))

    recs = G.load()
    if not recs:
        sys.exit("no arms on disk")
    data = collect(recs, conds, args.lag, args.task)
    gs = sorted(data)
    x = np.log10(gs)
    resolved = [g for g in gs if abs(data[g]["floors"]) >= MIN_FLOORS]
    # Clears the two-floor gate, but a 1σ downward revision of the floor would
    # put it back under. Drawn hatched, like panel (d)'s sub-gate bars. Gate unchanged.
    marginal = [g for g in resolved
                if (abs(data[g]["floors"]) - MIN_FLOORS)
                < FLOOR_REL_SE * abs(data[g]["floors"])]

    fig, axes = plt.subplots(2, 2, figsize=figstyle.figsize(3.25))
    lab = ("pooled over the 2×2, mixing signs" if conds is None else
           "S-HL, S-LH, S-LL pooled" if conds == FORGETTING else args.condition)
    tasks = sorted({t for spec, _m, t, lg, _a in G.attributions(recs, module="A")
                    if lg == args.lag and (not conds or spec["condition"] in conds)
                    and (args.task is None or t == args.task)})
    tlab = f"task {tasks[0]}" if len(tasks) == 1 else f"tasks {tasks} pooled"
    gains = all(data[g]["dlog_alpha"] > 0 for g in resolved) if resolved else False
    head = ("Retained capacity *rises* in the benign corner" if gains else
            "Forgetting decomposes exactly; the channel mix shifts with richness")
    fig.suptitle(figstyle.wrap(f"{head}  ({lab}, lag {args.lag}, {tlab}, module A)"),
                 fontsize=figstyle.fs(11))

    # (a) magnitude, in noise floors — signed, so a gain cannot read as a loss
    ax = axes[0][0]
    fl = [data[g]["floors_signed"] for g in gs]
    ax.plot(x, fl, "o-", color="k", lw=1.8, ms=5)
    ax.axhspan(-MIN_FLOORS, MIN_FLOORS, color="0.88", zorder=0)
    ax.axhline(0.0, color="0.6", lw=0.8, zorder=0)
    ax.text(x[0], MIN_FLOORS * 1.3, f"±{MIN_FLOORS:g} floors: below resolution",
            fontsize=figstyle.fs(7), color="0.35")
    ax.set(ylabel=r"$\Delta\log\alpha$ (noise floors)", xlabel=r"$\gamma_0$",
           title=f"(a) how much retained capacity is {'gained' if gains else 'lost'}")
    for g, f in zip(gs, fl):
        ax.annotate(f"{f:.0f}", (np.log10(g), f), textcoords="offset points",
                    xytext=(4, -9), fontsize=figstyle.fs(7))

    # (b) composition, only where resolvable
    ax = axes[0][1]
    labeled = False
    for g in resolved:
        bottom = 0.0
        hatch = "///" if g in marginal else None
        for f in FACTORS:
            kw = dict(width=0.22, color=G.FACTOR_COLORS[f], edgecolor="white",
                      linewidth=0.6, hatch=hatch)
            if not labeled and hatch is None:
                kw["label"] = G.FACTOR_LABELS[f]
            ax.bar(np.log10(g), data[g]["shares"][f], bottom=bottom, **kw)
            bottom += data[g]["shares"][f]
        if hatch is None:
            labeled = True
        if g in marginal:
            ax.text(np.log10(g), 1.02, "marginal", ha="center", va="bottom",
                    fontsize=figstyle.fs(6.0), color="0.35")
    for g in gs:
        if g not in resolved:
            ax.text(np.log10(g), 0.5, "not\nresolvable", ha="center", va="center",
                    fontsize=figstyle.fs(7), color="0.4")
    ax.set(ylabel="share of total motion  $|term| / \\Sigma|term|$", xlabel=r"$\gamma_0$",
           title="(b) which channel carries it",
           ylim=(0, 1.08 if marginal else 1))
    # -0.28 rather than -0.16: at print size the x-label needs the room, and a legend that sits on
    # top of the axis label is the kind of thing only a compile shows.
    ax.legend(frameon=False, fontsize=figstyle.fs(8), loc="upper center",
              bbox_to_anchor=(0.5, -0.28),
              ncol=3)

    # (c) signed terms, with unresolved signs marked as such
    #
    # A term whose sign flips on about half the arms has no sign, and drawing it as a
    # filled point next to a term that is consistently negative invites reading a
    # "positive contribution at low γ" that is really noise about zero. Points whose
    # positive fraction sits inside SIGN_AMBIGUOUS are drawn hollow.
    ax = axes[1][0]
    SIGN_AMBIGUOUS = (0.35, 0.65)
    for f in FACTORS:
        v = np.array([data[g]["terms"][f] for g in gs])
        e = np.array([data[g]["term_sem"][f] for g in gs])
        frac = np.array([data[g]["sign_positive_fraction"][f] for g in gs])
        amb = (frac > SIGN_AMBIGUOUS[0]) & (frac < SIGN_AMBIGUOUS[1])
        ax.errorbar(x, v, yerr=e, fmt="-", color=G.FACTOR_COLORS[f], lw=1.5,
                    capsize=2, label=G.FACTOR_LABELS[f])
        ax.plot(x[~amb], v[~amb], "o", color=G.FACTOR_COLORS[f], ms=5)
        ax.plot(x[amb], v[amb], "o", mfc="white", mec=G.FACTOR_COLORS[f], ms=5, mew=1.3)
    ax.axhline(0, color="0.3", lw=0.9)
    resolved_uti = [g for g in gs
                    if not (SIGN_AMBIGUOUS[0]
                            < data[g]["sign_positive_fraction"]["utility"]
                            < SIGN_AMBIGUOUS[1]) and data[g]["terms"]["utility"] < 0]
    if resolved_uti and resolved_uti[0] != gs[0]:
        prev = gs[gs.index(resolved_uti[0]) - 1]
        ax.axvspan(np.log10(prev), np.log10(resolved_uti[0]), color="crimson", alpha=0.10)
        # Left-anchored at the band's left edge. Centring it on a band that sits against the axis
        # spills the label off the canvas once the panel is only 3 inches wide.
        ax.annotate("$\\Psi_{\\mathrm{eff}}$ becomes resolvable\nand negative in here",
                    xy=(np.log10(prev), 0.70),
                    xycoords=("data", "axes fraction"), ha="left", fontsize=figstyle.fs(7),
                    color="crimson")
    ax.plot([], [], "o", mfc="white", mec="0.4", ms=5, label="sign not resolved")
    ax.set(ylabel=r"signed contribution to $\Delta\log\alpha$", xlabel=r"$\gamma_0$",
           title="(c) signed terms — hollow = sign not resolved")
    # Pinned rather than "best": the shaded band carries an annotation, and automatic placement put
    # the legend underneath it once the panel shrank to print size.
    ax.legend(frameon=False, fontsize=figstyle.fs(7.5), loc="lower left")

    # (d) center-collapse share of the radius channel.
    # Each bar is gated by the `R_eff` noise floor measured at *its own* γ, so the coverage
    # percentages are not comparable across bars; the threshold is printed under each.
    ax = axes[1][1]
    cs = [(np.log10(g), data[g]["center_share_median"], data[g]["center_coverage"],
           data[g]["R_eff_floor_cv"], g in resolved)
          for g in gs if data[g]["center_share_median"] is not None]
    if cs:
        ax.bar([c[0] for c in cs], [c[1] for c in cs], width=0.22,
               color=["#8172B3" if c[4] else "white" for c in cs],
               edgecolor="#8172B3", hatch=[None if c[4] else "///" for c in cs])
        # Staggered heights. The two-line label is wider than the gap between adjacent bars at
        # print size, so neighbours at a common height run into each other.
        for i, (xx, yy, covf, cv, _res) in enumerate(cs):
            lift = 0.04 if i % 2 == 0 else 0.14
            ax.text(xx, yy + (lift if yy >= 0 else -0.12),
                    f"{100 * covf:.0f}%\ngate {100 * ATT.FLOOR_GATE_K * cv:.2f}%", ha="center",
                    fontsize=figstyle.fs(6.0), color="0.35", linespacing=1.4)
    ax.axhline(1.0, ls="--", color="crimson", lw=1)
    # A *negative* share means ρ_c and the radius moved in opposite directions, so the
    # conversion explains none of the radius change. It is shown rather than clipped, because
    # clipping it to the axis floor would render a sign disagreement as a small positive share.
    lo = min([c[1] for c in cs], default=0.0)
    if lo < 0:
        ax.axhline(0.0, color="0.6", lw=0.8)
        ax.annotate("below 0: $\\rho_c$ and $R_{\\mathrm{eff}}$ moved in opposite directions",
                    xy=(0.02, 0.02), xycoords="axes fraction",
                    fontsize=figstyle.fs(6.0), color="0.35")
    ax.set(ylabel=r"share of $\Delta\log R_{\mathrm{eff}}$ from $\rho_c$",
           xlabel=r"$\gamma_0$", title="(d) is the radius channel center collapse?",
           ylim=(min(0.0, lo * 1.35) - 0.05, 1.15))
    # Wrapped narrow and pinned to the top-left corner: the bars and their coverage labels rise to
    # the right, so a note that spans the panel width runs into them at print size.
    note = ("1.0 = entirely center collapse\n"
            f"% = coverage, but each bar is\n"
            f"gated at {ATT.FLOOR_GATE_K:g}× its own measured\n"
            "$R_{\\mathrm{eff}}$ floor — so coverage is\n"
            "not comparable across bars")
    if any(not c[4] for c in cs):
        note += "\nhatched = no resolvable\nforgetting to attribute"
    ax.text(0.02, 0.98, note, transform=ax.transAxes, fontsize=figstyle.fs(6.0), color="crimson",
            va="top", linespacing=1.5)

    for ax in axes.ravel():
        ax.set_xticks(x, [f"{g:g}" for g in gs])
        ax.spines[["top", "right"]].set_visible(False)
    # layout: constrained_layout, set in src/analysis/figstyle.apply()

    FIGDIR.mkdir(parents=True, exist_ok=True)
    sha = result_set_sha(recs)
    tag = f"__task{args.task}" if args.task is not None else ""
    stem = f"fig2_gamma_sweep__{args.condition}__lag{args.lag}{tag}__{len(recs)}arms__{sha}"
    for ext in ("pdf", "png"):
        fig.savefig(FIGDIR / f"{stem}.{ext}", dpi=170,
                    metadata={"Creator": "scripts/fig2_gamma_sweep.py"} if ext == "pdf"
                    else None)
    (FIGDIR / f"{stem}.json").write_text(json.dumps(
        {"generated_by": "scripts/fig2_gamma_sweep.py", "n_arms": len(recs),
         "result_set_sha": sha, "condition": args.condition, "lag": args.lag,
         "only_task": args.task,
         "min_floors": MIN_FLOORS, "resolved_gammas": resolved,
         "marginal_gammas": marginal, "floor_rel_se": FLOOR_REL_SE,
         "coverage_note": G.coverage_note(recs), "data": data}, indent=2))
    print(f"wrote figures/{stem}.{{pdf,png,json}}\n")

    print(f"{'γ':>6}  {'floors':>7}  {'Δlogα':>9}  {'uti':>8} {'dim':>8} {'rad':>8}  "
          f"{'uti>0':>6}  {'center':>6}  n")
    for g in gs:
        d = data[g]
        mark = "" if g in resolved else "  (below floor)"
        cen = d["center_share_median"]
        print(f"{g:>6g}  {d['floors']:>7.1f}  {d['dlog_alpha']:>+9.4f}  "
              f"{d['terms']['utility']:>+8.4f} {d['terms']['dimension']:>+8.4f} "
              f"{d['terms']['radius']:>+8.4f}  "
              f"{d['sign_positive_fraction']['utility']:>6.2f}  "
              f"{'   n/a' if cen is None else f'{cen:>6.2f}'}  {d['n']}{mark}")


if __name__ == "__main__":
    main()
