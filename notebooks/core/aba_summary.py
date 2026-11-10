"""docs/30: setting cards and figures from the two FULL A–B–A runs (not hashed, no training).

    python notebooks/core/aba_summary.py            # figures + setting cards into docs/30
    python notebooks/core/aba_summary.py --check    # only re-derive docs/28 §9.3 and compare

Reads results/aba/FULL_2528909f2feaf3f3.jsonl (seeds 0-7, job 4349) and
results/aba/FULL_4e110835b8fc2863.jsonl (seeds 8-15, job 4396), copied from `real`.
Each run is computed on its own; nothing is pooled. Every figure shows both runs side by side.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parents[2]
RUNS = {  # label -> (checkpoint, seeds, job)
    "seeds 0–7": (REPO / "results/aba/FULL_2528909f2feaf3f3.jsonl", range(0, 8), "4349"),
    "seeds 8–15": (REPO / "results/aba/FULL_4e110835b8fc2863.jsonl", range(8, 16), "4396"),
}
GAMMAS = (0.1, 1.0, 10.0)
HS = (0, 4, 8)
SFS = (1.0, 0.9, 0.1)
SF_NAME = {1.0: "frozen", 0.9: "drift", 0.1: "jump"}
PHASES = ("init", "A1", "B", "A2")
T = stats.t.ppf(1 - 0.05 / 12, 7)          # docs/28 §5 erratum: 3.636
FIG = REPO / "docs" / "figures"
DOC = REPO / "docs" / "30-aba-summary.md"
# Categorical slots 1-3 of the dataviz reference palette (validated all-pairs, light mode).
GCOL = {0.1: "#2a78d6", 1.0: "#eb6834", 10.0: "#1baf7a"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def load() -> dict[str, dict]:
    """run -> {(seed, gamma, h, s_f): row}, design arms only (noise 'copy'), the run's seeds only."""
    out = {}
    for run, (path, seeds, _) in RUNS.items():
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        by = {}
        for r in rows:
            s = r["spec"]
            if s["seed"] in seeds and s.get("noise", "copy") == "copy":
                assert r["usable"], r["key"]
                by[(s["seed"], s["gamma"], s["hamming"], s["s_f"])] = r
        assert len(by) == 8 * 27, (run, len(by))
        out[run] = by
    return out


def ci(x) -> tuple[float, float, float]:
    x = np.asarray(x, float)
    m, se = x.mean(), x.std(ddof=1) / math.sqrt(len(x))
    return m, m - T * se, m + T * se


def level(r, ph):
    return math.log(r["geometry"][f"A@{ph}"]["alpha"])


def gain(r):
    return r["backward"]["init_to_A1"]["total"]


def frac_removed(r):
    return -r["backward"]["A1_to_B"]["total"] / gain(r)


def restoration(r):
    return (level(r, "A2") - level(r, "init")) / (level(r, "A1") - level(r, "init"))


def share(r, c):
    b = r["backward"]["A1_to_B"]
    return b[c] / b["total"]


def seeds_of(run):
    return list(RUNS[run][1])


# --------------------------------------------------------------------------- docs/28 §9.3 check

def fmt(v, nd):
    return f"{v:+.{nd}f}"


def fci(t, nd):
    m, lo, hi = t
    return f"{fmt(m, nd)} [{fmt(lo, nd)}, {fmt(hi, nd)}]"


def derive_93(data) -> dict[str, list[str]]:
    """run -> the statistic strings of docs/28 §9.3, in table order (H2 rows are gamma-free)."""
    out = {}
    for run, A in data.items():
        S = seeds_of(run)
        v = []
        for sf in (1.0, 0.9):
            v.append(fci(ci([frac_removed(A[s, 10.0, 4, sf]) - frac_removed(A[s, 0.1, 4, sf]) for s in S]), 3))
        g = {gm: [gain(A[s, gm, 0, 1.0]) for s in S] for gm in GAMMAS}
        v.append(fci(ci(np.subtract(g[1.0], g[0.1])), 3))
        v.append(fci(ci(np.subtract(g[10.0], g[1.0])), 3))
        for gm in GAMMAS:
            v.append(fci(ci([A[s, gm, 0, 0.9]["backward"]["A1_to_B"]["total"] for s in S]), 4))
        v.append(fci(ci([A[s, 10.0, 8, 1.0]["behaviour"]["FT_beh"] - A[s, 10.0, 8, 0.1]["behaviour"]["FT_beh"]
                         for s in S]), 3))
        for h, sf in ((4, 1.0), (4, 0.9), (8, 1.0), (8, 0.9)):
            v.append(fci(ci([share(A[s, 10.0, h, sf], "utility") + share(A[s, 10.0, h, sf], "dimension")
                             for s in S]), 3))
        for gm in (0.1, 1.0):
            for h, sf in ((4, 1.0), (4, 0.9), (8, 1.0), (8, 0.9)):
                v.append("; ".join(f"{c[0]} " + fci(ci([share(A[s, gm, h, sf], c) for s in S]), 2)
                                   for c in ("utility", "radius", "dimension")))
        for gm in GAMMAS:
            for h, sf in ((4, 1.0), (4, 0.9), (8, 1.0), (8, 0.9)):
                v.append(fci(ci([restoration(A[s, gm, h, sf]) for s in S]), 3))
        out[run] = v
    return out


def check_93(data) -> None:
    text = (REPO / "docs/28-aba-full-precommit.md").read_text(encoding="utf-8")
    sec = text.split("### 9.3 Every stratum", 1)[1]
    rows = [ln for ln in sec.splitlines() if ln.startswith("| H")]
    der = derive_93(data)
    assert len(rows) == len(der["seeds 0–7"]) == 32, (len(rows), len(der["seeds 0–7"]))
    bad = 0
    for i, ln in enumerate(rows):
        cells = [c.strip() for c in ln.replace("−", "-").split("|")[1:-1]]
        for col, run in ((2, "seeds 0–7"), (4, "seeds 8–15")):
            want = der[run][i].replace("−", "-")
            # docs/28 prints "-0.417" (not "+") for negatives and "+" for positives; same as fmt
            if want not in cells[col].replace("+-", "-"):
                bad += 1
                print(f"MISMATCH row {i} {run}: derived {want!r} vs doc {cells[col]!r}")
    if bad:
        sys.exit(f"{bad} mismatches against docs/28 §9.3")
    print(f"docs/28 §9.3 re-derived from the checkpoints: 32 rows × 2 runs match")


# --------------------------------------------------------------------------- setting cards

def setting_cards(data) -> str:
    out = ["Exposure is a property of the setting **as the learner meets it**: training stops at matched loss "
           "(≤ 0.05), so each learner sets its own number of updates per phase. The step counts below are that "
           "exposure. Realised relations depend on (h, s_f) and the seed only, never on γ0 (checked: identical in "
           "all three γ0 arms of a seed).", ""]
    for h in HS:
        for sf in SFS:
            out.append(f"#### Card: h = {h}, s_f = {sf} ({SF_NAME[sf]})")
            out.append("")
            out.append(f"Nominal: Hamming(y_A, y_B) = {h} of P = 16; input-centre change s_f = {sf}.")
            out.append("")
            out.append("| run | seed | realised s_r(y_A, y_B) | realised ρ_in(A, B) |")
            out.append("|---|---|---|---|")
            for run, A in data.items():
                for s in seeds_of(run):
                    rel = [A[s, gm, h, sf]["relations"] for gm in GAMMAS]
                    assert len({(x["s_r_AB"], x["rho_in_AB"]) for x in rel}) == 1, (run, s, h, sf)
                    out.append(f"| {run} | {s} | {rel[0]['s_r_AB']:+.3f} | {rel[0]['rho_in_AB']:.4f} |")
            out.append("")
            out.append("Exposure: updates per phase, median [min, max] over the run's 8 seeds.")
            out.append("")
            out.append("| run | γ0 | A1 | B | A2 | fresh B |")
            out.append("|---|---|---|---|---|---|")
            for run, A in data.items():
                for gm in GAMMAS:
                    cells = []
                    for ph in ("A1", "B", "A2", "freshB"):
                        st = [A[s, gm, h, sf]["behaviour"]["steps"][ph] for s in seeds_of(run)]
                        cells.append(f"{int(np.median(st))} [{min(st)}, {max(st)}]")
                    out.append(f"| {run} | {gm:g} | " + " | ".join(cells) + " |")
            out.append("")
    return "\n".join(out)


# --------------------------------------------------------------------------- figures

def _style(ax):
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)


def _pt(ax, x, vals, col, filled, dots=True):
    """Seeds as small dots, mean as a marker with the 99.17% t-interval."""
    m, lo, hi = ci(vals)
    if dots:
        ax.scatter(np.full(len(vals), x) + np.linspace(-0.04, 0.04, len(vals)), vals, s=6,
                   color=col, alpha=0.35, lw=0, zorder=2)
    ax.plot([x, x], [lo, hi], color=col, lw=2, solid_capstyle="round", zorder=3)
    ax.scatter([x], [m], s=34, zorder=4, color=col if filled else "white", edgecolors=col, linewidths=1.6)


RUN_STYLE = {"seeds 0–7": (False, -0.12), "seeds 8–15": (True, 0.12)}   # hollow = original, filled = replication


def _legend(fig, extra=()):
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", ls="", color=GCOL[g], label=f"γ0 = {g:g}", markersize=7) for g in GAMMAS]
    h += [Line2D([], [], marker="o", ls="", color=INK2, markerfacecolor="white", label="seeds 0–7 (original)"),
          Line2D([], [], marker="o", ls="", color=INK2, label="seeds 8–15 (replication, confirmatory)")]
    h += list(extra)
    fig.legend(handles=h, loc="lower center", ncol=len(h), frameon=False, fontsize=8, labelcolor=INK)


def fig_levels(data, plt):
    fig, axes = plt.subplots(3, 3, figsize=(11, 9), sharey=True)
    for i, h in enumerate(HS):
        for j, sf in enumerate(SFS):
            ax = axes[i, j]
            _style(ax)
            for run, A in data.items():
                filled, off = RUN_STYLE[run]
                for k, gm in enumerate(GAMMAS):
                    o = off + (k - 1) * 0.06
                    means = []
                    for p, ph in enumerate(PHASES):
                        vals = [level(A[s, gm, h, sf], ph) for s in seeds_of(run)]
                        _pt(ax, p + o, vals, GCOL[gm], filled, dots=False)
                        means.append(np.mean(vals))
                    ax.plot(np.arange(4) + o, means, color=GCOL[gm], lw=1, alpha=0.6,
                            ls="-" if filled else ":", zorder=1)
            ax.set_xticks(range(4), ["init", "after A", "after B", "after A again"])
            ax.set_title(f"h = {h}, s_f = {sf:g} ({SF_NAME[sf]})", fontsize=9, color=INK)
            if j == 0:
                ax.set_ylabel("log α_A (retained capacity of A)", fontsize=8, color=INK2)
    fig.suptitle("Capacity of task A at each phase boundary, per configuration (mean, 99.17% t-interval, n = 8)",
                 fontsize=11, color=INK)
    _legend(fig)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    return fig


def fig_ratios(data, plt):
    configs = [(h, sf) for h in (4, 8) for sf in (1.0, 0.9, 0.1)]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    panels = [("gain init → A: Δlog α_A", lambda r: gain(r), "init→A gain (identical in every cell of a seed)"),
              ("fraction of the gain removed by B", frac_removed, None),
              ("restoration ρ after relearning A", restoration, None)]
    for ax, (ylab, f, note) in zip(axes, panels):
        _style(ax)
        cfgs = [(0, 1.0)] if note else configs
        for c, (h, sf) in enumerate(cfgs):
            for run, A in data.items():
                filled, off = RUN_STYLE[run]
                for k, gm in enumerate(GAMMAS):
                    x = c + off + (k - 1) * 0.07
                    _pt(ax, x, [f(A[s, gm, h, sf]) for s in seeds_of(run)], GCOL[gm], filled, dots=False)
        if note:
            ax.set_xticks([0], ["(every configuration)"])
            ax.set_xlim(-0.6, 0.6)
        else:
            ax.set_xticks(range(len(configs)), [f"h {h}\n{SF_NAME[sf]}" for h, sf in configs], fontsize=8)
        ax.set_ylabel(ylab, fontsize=8, color=INK2)
        if f is restoration:
            ax.axhline(0.9, color=INK2, lw=1, ls="--")
            ax.text(-0.45, 0.9, "H6 threshold 0.9", fontsize=7, color=INK2, ha="left", va="top")
            ax.axhline(1.0, color=GRID, lw=1)
    fig.suptitle("Gain, removal and restoration of A's capacity (H4/H8 configurations; the ratios need a gain)",
                 fontsize=11, color=INK)
    _legend(fig)
    fig.tight_layout(rect=(0, 0.07, 1, 0.94))
    return fig


def fig_shares(data, plt):
    from matplotlib.lines import Line2D
    strata = [(4, 1.0), (4, 0.9), (8, 1.0), (8, 0.9)]
    marks = {"utility": "o", "radius": "s", "dimension": "^"}
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.2), sharey=True)
    for ax, (h, sf) in zip(axes, strata):
        _style(ax)
        for k, gm in enumerate(GAMMAS):
            for ci_, c in enumerate(marks):
                for run, A in data.items():
                    filled, off = RUN_STYLE[run]
                    x = k + (ci_ - 1) * 0.28 + off * 0.5
                    m, lo, hi = ci([share(A[s, gm, h, sf], c) for s in seeds_of(run)])
                    ax.plot([x, x], [lo, hi], color=GCOL[gm], lw=2, solid_capstyle="round")
                    ax.scatter([x], [m], marker=marks[c], s=32, color=GCOL[gm] if filled else "white",
                               edgecolors=GCOL[gm], linewidths=1.4, zorder=4)
        ax.axhline(0.5, color=INK2, lw=1, ls="--")
        ax.axhline(0, color=GRID, lw=1)
        ax.set_xticks(range(3), [f"γ0 {g:g}" for g in GAMMAS])
        ax.set_title(f"h = {h}, s_f = {sf:g} ({SF_NAME[sf]})", fontsize=9, color=INK)
    axes[0].set_ylabel("share of Δlog α_A(A → B)", fontsize=8, color=INK2)
    extra = [Line2D([], [], marker=m, ls="", color=INK2, label=c) for c, m in marks.items()]
    fig.suptitle("Channel shares of the loss B causes to A (dashed: H5b bound 0.5 at γ0 = 0.1)",
                 fontsize=11, color=INK)
    _legend(fig, extra)
    fig.tight_layout(rect=(0, 0.07, 1, 0.94))
    return fig


def fig_h4(data, plt):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [3, 1.2]})
    ax = axes[0]
    _style(ax)
    configs = [(h, sf) for h in HS for sf in SFS]
    for c, (h, sf) in enumerate(configs):
        for run, A in data.items():
            filled, off = RUN_STYLE[run]
            for k, gm in enumerate(GAMMAS):
                _pt(ax, c + off + (k - 1) * 0.07, [A[s, gm, h, sf]["behaviour"]["FT_beh"] for s in seeds_of(run)],
                    GCOL[gm], filled, dots=False)
    ax.axhline(0, color=INK2, lw=1)
    ax.set_xticks(range(len(configs)), [f"h {h}\n{SF_NAME[sf]}" for h, sf in configs], fontsize=8)
    ax.set_ylabel("FT_beh = log((1 + steps fresh B) / (1 + steps B))", fontsize=8, color=INK2)
    ax.set_title("Behavioural forward transfer per configuration", fontsize=9, color=INK)
    ax = axes[1]
    _style(ax)
    for k, gm in enumerate(GAMMAS):
        for run, A in data.items():
            filled, off = RUN_STYLE[run]
            _pt(ax, k + off, [A[s, gm, 8, 1.0]["behaviour"]["FT_beh"] - A[s, gm, 8, 0.1]["behaviour"]["FT_beh"]
                              for s in seeds_of(run)], GCOL[gm], filled)
    ax.axhspan(-math.log(1.25), math.log(1.25), color=GRID, alpha=0.6, lw=0)
    ax.axhline(0, color=INK2, lw=1)
    ax.set_xticks(range(3), [f"γ0 {g:g}" for g in GAMMAS])
    ax.set_title("H4: FT_beh(h 8 frozen) − FT_beh(h 8 jump)\n(read at γ0 = 10; band ±log 1.25)", fontsize=9,
                 color=INK)
    _legend(fig)
    fig.tight_layout(rect=(0, 0.07, 1, 0.97))
    return fig


def fig_acc(data, plt):
    fig, axes = plt.subplots(3, 3, figsize=(11, 9), sharex=True, sharey=True)
    pm = {"A1": "o", "B": "s", "A2": "^"}
    for i, h in enumerate(HS):
        for j, sf in enumerate(SFS):
            ax = axes[i, j]
            _style(ax)
            for run, A in data.items():
                filled, _ = RUN_STYLE[run]
                for gm in GAMMAS:
                    for ph, mk in pm.items():
                        xs = [level(A[s, gm, h, sf], ph) for s in seeds_of(run)]
                        ys = [A[s, gm, h, sf]["bottleneck"][ph]["acc"] for s in seeds_of(run)]
                        ax.scatter(xs, ys, marker=mk, s=16, color=GCOL[gm] if filled else "white",
                                   edgecolors=GCOL[gm], linewidths=0.9, alpha=0.85)
            ax.set_title(f"h = {h}, s_f = {sf:g} ({SF_NAME[sf]})", fontsize=9, color=INK)
            if j == 0:
                ax.set_ylabel("bottleneck probe accuracy on A (n = 16)", fontsize=8, color=INK2)
            if i == 2:
                ax.set_xlabel("log α_A", fontsize=8, color=INK2)
    from matplotlib.lines import Line2D
    extra = [Line2D([], [], marker=m, ls="", color=INK2, label=f"after {p}") for p, m in pm.items()]
    fig.suptitle("Accuracy vs capacity: bottleneck probe on A against log α_A, one dot per seed and phase",
                 fontsize=11, color=INK)
    _legend(fig, extra)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    return fig


def write_generated(doc: Path, tag: str, body: str) -> None:
    text = doc.read_text(encoding="utf-8")
    a, b = f"<!-- BEGIN GENERATED: {tag} -->", f"<!-- END GENERATED: {tag} -->"
    pre, rest = text.split(a, 1)
    _, post = rest.split(b, 1)
    doc.write_text(f"{pre}{a}\n{body}\n{b}{post}", encoding="utf-8")


# --------------------------------------------------------------------------- readings (port of notebook cell 43)
# H1, H2, H5a and H6 only (the docs/31 ablation reads these). Same rules as docs/28 §3–§4.

def excl0(t):
    return t[1] > 0 or t[2] < 0


def res_sign(t, floor):
    return int(np.sign(t[0])) if excl0(t) and abs(t[0]) >= 2 * floor else 0


def dir_reading(sg, want=+1):
    return "held" if sg == want else "not held" if sg == -want else "unresolved at floor"


def readings(A, seeds, FL) -> list[dict]:
    """Per stratum: hypothesis, stratum, statistic string, mean, sign (resolved, H1/H2), reading."""
    G0, G1, G2 = GAMMAS
    out = []
    gain_res = {}
    for g in GAMMAS:
        gs = [gain(A[s, g, 0, 1.0]) for s in seeds]
        for h in HS:
            for sf in SFS:
                assert all(abs(gain(A[s, g, h, sf]) - v) < 1e-12 for s, v in zip(seeds, gs))
        gain_res[g] = res_sign(ci(gs), FL[g]) > 0
    for sf in (1.0, 0.9):
        v = [frac_removed(A[s, G2, 4, sf]) - frac_removed(A[s, G0, 4, sf]) for s in seeds]
        t = ci(v)
        read = gain_res[G0] and gain_res[G2]
        sg = int(np.sign(t[0])) if excl0(t) else 0
        out.append({"hyp": "H1", "stratum": f"h 4, s_f {sf:g} ({SF_NAME[sf]})",
                    "stat": "d = f(10) − f(0.1) = " + fci(t, 3),
                    "mean": t[0], "sign": sg, "reading": dir_reading(sg) if read else "not read",
                    "vals": v, "sesoi": 0.10, "centre": 0.0})
    gg = {g: [gain(A[s, g, 0, 1.0]) for s in seeds] for g in GAMMAS}
    for (ga, gb), name in (((G1, G0), "d₁ = g(1) − g(0.1)"), ((G2, G1), "d₂ = g(10) − g(1)")):
        v = list(np.subtract(gg[ga], gg[gb]))
        t = ci(v)
        sg = res_sign(t, max(FL[ga], FL[gb]))
        out.append({"hyp": "H2", "stratum": name, "stat": fci(t, 3), "mean": t[0], "sign": sg,
                    "reading": dir_reading(sg), "vals": v, "sesoi": 0.05, "centre": 0.0})
    for h in (4, 8):
        for sf in (1.0, 0.9):
            tot = ci([A[s, G2, h, sf]["backward"]["A1_to_B"]["total"] for s in seeds])
            read = res_sign(tot, FL[G2]) != 0
            v = [share(A[s, G2, h, sf], "utility") + share(A[s, G2, h, sf], "dimension") for s in seeds]
            t = ci(v)
            r = "held" if t[1] >= 0.8 else "not held" if t[2] < 0.8 else "unresolved at floor"
            out.append({"hyp": "H5a", "stratum": f"γ0 10, h {h}, s_f {sf:g} ({SF_NAME[sf]})",
                        "stat": "u + d share " + fci(t, 3), "mean": t[0], "sign": None,
                        "reading": r if read else "not read", "vals": v, "sesoi": 0.10, "centre": 0.8})
    for g in GAMMAS:
        for h in (4, 8):
            for sf in (1.0, 0.9):
                v = [restoration(A[s, g, h, sf]) for s in seeds]
                t = ci(v)
                r = "held" if t[1] >= 0.9 else "not held" if t[2] < 0.9 else "unresolved at floor"
                out.append({"hyp": "H6", "stratum": f"γ0 {g:g}, h {h}, s_f {sf:g} ({SF_NAME[sf]})",
                            "stat": "ρ = " + fci(t, 3), "mean": t[0], "sign": None,
                            "reading": r if gain_res[g] else "not read", "vals": v, "sesoi": 0.10,
                            "centre": 0.9})
    return out


# docs/28 §9.2 floors (printed to 4 decimals by each run's notebook, cell 14)
FLOORS_928 = {"seeds 0–7": {0.1: 0.0103, 1.0: 0.0169, 10.0: 0.0293},
              "seeds 8–15": {0.1: 0.0248, 1.0: 0.0354, 10.0: 0.0428}}


def check_readings(data) -> None:
    """The port must give docs/28 §9.3's readings for H1, H2, H5a, H6 in both criterion runs."""
    text = (REPO / "docs/28-aba-full-precommit.md").read_text(encoding="utf-8")
    rows = [ln for ln in text.split("### 9.3 Every stratum", 1)[1].splitlines() if ln.startswith("| H")]
    doc = [[c.strip() for c in ln.split("|")[1:-1]] for ln in rows]
    want = {k: [r for r in doc if r[0] == k] for k in ("H1", "H2", "H5a", "H6")}
    for run, col in (("seeds 0–7", 3), ("seeds 8–15", 5)):
        got = readings(data[run], seeds_of(run), FLOORS_928[run])
        for hyp, drows in want.items():
            mine = [r["reading"] for r in got if r["hyp"] == hyp]
            theirs = [r[col] for r in drows]
            assert mine == theirs, (run, hyp, mine, theirs)
    print("readings port reproduces docs/28 §9.3 for H1, H2, H5a, H6 in both runs "
          f"({sum(len(v) for v in want.values())} strata × 2)")


def compare(crit: dict, fix: dict) -> str:
    """docs/31 pre-committed reading of one stratum."""
    if fix["reading"] == "not read" or crit["reading"] == "not read":
        return "not read"
    if crit["sign"] is not None:                       # H1, H2: resolved sign of the contrast
        if fix["sign"] == 0:
            return "unresolved"
        return "same sign" if fix["sign"] == crit["sign"] else "**sign flip**"
    if fix["reading"] == "unresolved at floor":        # H5a, H6: side of the threshold
        return "unresolved"
    if {fix["reading"], crit["reading"]} == {"held", "not held"}:
        return "**flip**"
    return "same reading" if fix["reading"] == crit["reading"] else "unresolved"


T4 = stats.t.ppf(1 - 0.05 / 8, 7)          # docs/29 S0: 98.75% two-sided (family of four), df 7


def category(vals, sesoi: float, centre: float = 0.0, direction: int = +1) -> str:
    """Kati's four categories (Lakens), on the 98.75% t-interval of (vals − centre); the
    predicted direction is `direction`. Descriptive column in docs/31, the reading in docs/29."""
    x = np.asarray(vals, float) - centre
    m, se = x.mean(), x.std(ddof=1) / math.sqrt(len(x))
    lo, hi = m - T4 * se, m + T4 * se
    if (direction > 0 and lo > sesoi) or (direction < 0 and hi < -sesoi):
        return "meaningful"
    if (direction > 0 and hi < -sesoi) or (direction < 0 and lo > sesoi):
        return "meaningful, opposite"
    if -sesoi < lo and hi < sesoi:
        return "equivalent"
    if lo > 0 or hi < 0:
        return "nonzero, possibly trivial"
    return "inconclusive"


def _check_category() -> None:
    base = np.array([-1.0, 1.0] * 4) * 0.01
    assert category(1.0 + base, 0.1) == "meaningful"
    assert category(-1.0 + base, 0.1) == "meaningful, opposite"
    assert category(base, 0.1) == "equivalent"
    assert category(0.1 + base, 0.1) == "nonzero, possibly trivial"
    assert category(np.array([-1.0, 1.0] * 4), 0.1) == "inconclusive"
    assert category(0.9 + 1.0 + base, 0.1, centre=0.9) == "meaningful"


def load_fixed(path: Path, seeds) -> dict:
    by = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        s = r["spec"]
        if s["seed"] in seeds and s.get("noise", "copy") == "copy":
            assert r["usable"] and r["behaviour"].get("stopping") == "fixed", r["key"]
            by[(s["seed"], s["gamma"], s["hamming"], s["s_f"])] = r
    assert len(by) == 8 * 27, len(by)
    return by


def ablation_tables(data, fixed_path: Path, floors_path: Path) -> str:
    run = "seeds 8–15"
    seeds = seeds_of(run)
    F = load_fixed(fixed_path, seeds)
    fj = json.loads(floors_path.read_text(encoding="utf-8"))
    FLF = {float(g): v["floor"] for g, v in fj["floors"].items()}
    budget = next(iter(F.values()))["config"]["fixed_steps"]
    crit = readings(data[run], seeds, FLOORS_928[run])
    fix = readings(F, seeds, FLF)
    out = [f"Fixed budget (A1, B, A2) = {tuple(budget)}; fresh B takes the B budget. Seeds 8–15. "
           f"Fixed-run checkpoint `{fixed_path.name}`, floors `{floors_path.name}`.", "",
           "**Readings per stratum.** Statistic and reading: docs/28 rules, 99.17% t-intervals, n = 8 (the "
           "pre-committed reading). Category: Kati's four categories (docs/29 S0) on the **98.75%** t-interval, "
           "descriptive here; SESOI 0.10 for H1 (d_f), 0.05 for H2 (Δlog α), and 0.10 around the threshold for "
           "H5a (u + d share − 0.8) and H6 (ρ − 0.9), predicted direction > 0.", "",
           "| hypothesis | stratum | criterion: statistic | criterion reading | fixed budget: statistic "
           "| fixed reading | docs/31 reading | category, criterion | category, fixed |",
           "|---|---|---|---|---|---|---|---|---|"]
    for c, f in zip(crit, fix):
        assert (c["hyp"], c["stratum"]) == (f["hyp"], f["stratum"])
        cc = category(c["vals"], c["sesoi"], c["centre"])
        cf = category(f["vals"], f["sesoi"], f["centre"])
        out.append(f"| {c['hyp']} | {c['stratum']} | {c['stat']} | {c['reading']} | {f['stat']} "
                   f"| {f['reading']} | {compare(c, f)} | {cc} | {cf} |")
    out += ["", "**Floors** (SD of log α_A over K = 8 estimator seeds, seed 8, representation after A).", "",
            "| γ0 | criterion (docs/28 §9.2) | fixed budget |", "|---|---|---|"]
    out += [f"| {g:g} | {FLOORS_928[run][g]:.4f} | {FLF[g]:.4f} |" for g in GAMMAS]
    out += ["", "**Final training loss per phase** under the fixed budget (criterion target 0.05; "
            "median [min, max] over seeds). Exposure is matched; fit is not.", "",
            "| configuration | γ0 | A1 | B | A2 | fresh B |", "|---|---|---|---|---|---|"]
    for h in HS:
        for sf in SFS:
            for g in GAMMAS:
                cells = []
                for ph in ("A1", "B", "A2", "freshB"):
                    v = [F[s, g, h, sf]["behaviour"]["final_loss"][ph] for s in seeds]
                    cells.append(f"{np.median(v):.2e} [{min(v):.1e}, {max(v):.1e}]")
                out.append(f"| h {h}, s_f {sf:g} ({SF_NAME[sf]}) | {g:g} | " + " | ".join(cells) + " |")
    return "\n".join(out)


# --------------------------------------------------------------------------- A–B–A figure set (draft, F1–F4)

H_CFGS = [(4, 1.0), (4, 0.9), (8, 1.0), (8, 0.9)]          # the H1/H5/H6 configurations
CH_COL = {"utility": "#6250d6", "dimension": "#eda100", "radius": "#e87ba4"}   # slots 7, 4, 5 (validated)
RUNS_32 = (REPO / "results/aba/FULL_635c4e3bf8cc6b96.jsonl", REPO / "results/aba/FULL_635c4e3bf8cc6b96_fixed.jsonl")


def figF1(data, plt):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), gridspec_kw={"width_ratios": [1, 2, 2]})
    ax = axes[0]
    _style(ax)
    for k, gm in enumerate(GAMMAS):
        for run, A in data.items():
            filled, off = RUN_STYLE[run]
            _pt(ax, k + off, [gain(A[s, gm, 0, 1.0]) for s in seeds_of(run)], GCOL[gm], filled, dots=False)
    ax.set_xticks(range(3), [f"γ0 {g:g}" for g in GAMMAS])
    ax.set_ylabel("Δlog α_A, init → after A", fontsize=8, color=INK2)
    ax.set_title("a  Gain (H2)", fontsize=9, color=INK, loc="left")
    for ax, f, ttl, ylab in ((axes[1], frac_removed, "b  Fraction of the gain removed by B (H1)", "fraction removed"),
                             (axes[2], restoration, "c  Restoration after relearning A (H6)", "restoration ρ")):
        _style(ax)
        for c, (h, sf) in enumerate(H_CFGS):
            for run, A in data.items():
                filled, off = RUN_STYLE[run]
                for k, gm in enumerate(GAMMAS):
                    _pt(ax, c + off + (k - 1) * 0.07, [f(A[s, gm, h, sf]) for s in seeds_of(run)],
                        GCOL[gm], filled, dots=False)
        ax.set_xticks(range(4), [f"h {h}, {SF_NAME[sf]}" for h, sf in H_CFGS], fontsize=8)
        ax.set_ylabel(ylab, fontsize=8, color=INK2)
        ax.set_title(ttl, fontsize=9, color=INK, loc="left")
    axes[2].axhline(0.9, color=INK2, lw=1, ls="--")
    axes[2].text(3.45, 0.9, "0.9", fontsize=7, color=INK2, va="bottom", ha="right")
    fig.suptitle("A–B–A at matched current-task fit: mean and 99.17% t-interval, n = 8 per run", fontsize=10,
                 color=INK)
    _legend(fig)
    fig.tight_layout(rect=(0, 0.07, 1, 0.94))
    return fig


def figF2(data, plt):
    from matplotlib.patches import Patch
    fig, axes = plt.subplots(1, 4, figsize=(13, 4.4), sharey=True)
    w = 0.36
    for ax, (h, sf) in zip(axes, H_CFGS):
        _style(ax)
        for k, gm in enumerate(GAMMAS):
            for run, A in data.items():
                filled, off = RUN_STYLE[run]
                x = k + (w / 2 + 0.02) * (1 if filled else -1)
                bottom = 0.0
                for ch, col in CH_COL.items():
                    v = float(np.mean([share(A[s, gm, h, sf], ch) for s in seeds_of(run)]))
                    ax.bar(x, v, w, bottom=bottom, color=col if filled else "white", edgecolor=col,
                           linewidth=1.2, hatch=None if filled else "////")
                    if v >= 0.12:
                        ax.text(x, bottom + v / 2, f"{v:.2f}", ha="center", va="center", fontsize=6.5, color=INK)
                    bottom += v
        ax.axhline(0.5, color=INK2, lw=1, ls="--")
        ax.set_xticks(range(3), [f"γ0 {g:g}" for g in GAMMAS])
        ax.set_title(f"h {h}, {SF_NAME[sf]}", fontsize=9, color=INK)
    axes[0].set_ylabel("share of Δlog α_A (after A → after B)", fontsize=8, color=INK2)
    h = [Patch(facecolor=c, edgecolor=c, label=ch) for ch, c in CH_COL.items()]
    h += [Patch(facecolor="white", edgecolor=INK2, hatch="////", label="left bar: seeds 0–7 (original)"),
          Patch(facecolor=INK2, edgecolor=INK2, label="right bar: seeds 8–15 (replication)")]
    fig.legend(handles=h, loc="lower center", ncol=len(h), frameon=False, fontsize=8, labelcolor=INK)
    fig.suptitle("Channel shares of the loss B causes to A (mean over 8 seeds; residual ≈ 0 not drawn; dashed: 0.5, the H5b bound)",
                 fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0.07, 1, 0.94))
    return fig


def figF3(data, plt):
    from matplotlib.lines import Line2D
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), sharex=True, sharey=True)
    pm = {"init": "D", "A1": "o", "B": "s", "A2": "^"}
    for ax, h in zip(axes, HS):
        _style(ax)
        for run, A in data.items():
            filled, _ = RUN_STYLE[run]
            for gm in GAMMAS:
                for ph, mk in pm.items():
                    xs = [level(A[s, gm, h, 1.0], ph) for s in seeds_of(run)]
                    ys = [A[s, gm, h, 1.0]["bottleneck"][ph]["acc"] for s in seeds_of(run)]
                    ax.scatter(xs, ys, marker=mk, s=18, color=GCOL[gm] if filled else "white",
                               edgecolors=GCOL[gm], linewidths=0.9, alpha=0.85)
        ax.set_title(f"h {h}, frozen input (s_f 1)", fontsize=9, color=INK)
        ax.set_xlabel("log α_A", fontsize=8, color=INK2)
    axes[0].set_ylabel("bottleneck-probe accuracy on A (n = 16)", fontsize=8, color=INK2)
    extra = [Line2D([], [], marker=m, ls="", color=INK2, label="init" if p == "init" else f"after {p}")
             for p, m in pm.items()]
    fig.suptitle("Accuracy vs capacity at frozen input: one marker per seed and phase (init is the same W(0) for every γ0, so its "
                 "markers coincide)", fontsize=10, color=INK)
    _legend(fig, extra)
    fig.tight_layout(rect=(0, 0.07, 1, 0.94))
    return fig


def figF4(cells, plt):
    from matplotlib.lines import Line2D
    S = list(SEEDS_32)

    def f(c, g, sf):
        return np.array([frac_removed(cells[c][s, g, sf]) for s in S])

    def pt(ax, x, v, col, filled, mk="o"):
        m, lo, hi = ci_at(v, T8)
        ax.plot([x, x], [lo, hi], color=col, lw=2, solid_capstyle="round", zorder=3)
        ax.scatter([x], [m], marker=mk, s=36, zorder=4, color=col if filled else "white", edgecolors=col,
                   linewidths=1.6)
        return m, lo, hi

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    ax = axes[0]
    _style(ax)
    ax.axhspan(-0.10, 0.10, color=GRID, alpha=0.6, lw=0)
    ax.axhline(0, color=INK2, lw=1)
    for c, cell in enumerate(CELLS_32):
        for sf, off, filled in ((0.9, 0.1, True), (1.0, -0.1, False)):
            pt(ax, c + off, f(cell, 10.0, sf) - f(cell, 0.1, sf), INK, filled)
    ax.set_xticks(range(4), [f"A {c.split(',')[0]}, B {c.split(',')[1]}" for c in CELLS_32], fontsize=8)
    ax.set_ylabel("d = f(γ0 10) − f(γ0 0.1)", fontsize=8, color=INK2)
    ax.set_title("a  H1 contrast under each stopping combination (h 4)", fontsize=9, color=INK, loc="left")
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=INK, label="h 4 drift"),
                       Line2D([], [], marker="o", ls="", color=INK, markerfacecolor="white", label="h 4 frozen")],
              frameon=False, fontsize=8, loc="upper right")
    ax = axes[1]
    _style(ax)
    ax.axhspan(-0.10, 0.10, color=GRID, alpha=0.6, lw=0)
    ax.axhline(0, color=INK2, lw=1)
    for i, (sf, g) in enumerate(STRATA_32):
        EA = 0.5 * ((f("Fx,Cr", g, sf) - f("Cr,Cr", g, sf)) + (f("Fx,Fx", g, sf) - f("Cr,Fx", g, sf)))
        EB = 0.5 * ((f("Cr,Fx", g, sf) - f("Cr,Cr", g, sf)) + (f("Fx,Fx", g, sf) - f("Fx,Cr", g, sf)))
        for E, off, mk, d in ((EA, -0.15, "o", -1 if g == 10.0 else 0), (EB, 0.15, "s", 0)):
            m, lo, hi = pt(ax, i + off, E, GCOL[g], True, mk)
            cat = category_at(E, 0.10, T8, d).replace("nonzero, possibly trivial", "nonzero,\npossibly trivial")
            ax.text(i + off, hi + 0.012, cat, ha="center", va="bottom", fontsize=6, color=INK2)
    ax.set_xticks(range(4), [f"γ0 {g:g}, h 4 {SF_NAME[sf]}" for sf, g in STRATA_32], fontsize=8)
    ax.set_ylabel("main effect on fraction removed", fontsize=8, color=INK2)
    ax.set_title("b  Over-training A (E_A, ●) and extra B updates (E_B, ■)", fontsize=9, color=INK, loc="left")
    ax.set_ylim(top=ax.get_ylim()[1] + 0.06)
    fig.suptitle("Protocol robustness (docs/31–32; seeds 8–15 only): mean and 99.375% t-interval, n = 8; "
                 "grey band = SESOI ±0.10", fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0.0, 1, 0.94))
    return fig


# --------------------------------------------------------------------------- docs/32: stopping 2x2

T8 = stats.t.ppf(1 - 0.05 / 16, 7)          # docs/32 §4: 99.375% two-sided (family of eight), df 7
CELLS_32 = ("Cr,Cr", "Fx,Cr", "Cr,Fx", "Fx,Fx")
STRATA_32 = [(1.0, 0.1), (1.0, 10.0), (0.9, 0.1), (0.9, 10.0)]     # (s_f, γ0), all at h 4
SEEDS_32 = range(8, 16)


def ci_at(x, tcrit) -> tuple[float, float, float]:
    x = np.asarray(x, float)
    m, se = x.mean(), x.std(ddof=1) / math.sqrt(len(x))
    return m, m - tcrit * se, m + tcrit * se


def category_at(vals, sesoi: float, tcrit: float, direction: int = 0) -> str:
    """Four categories (docs/29 S0) on the t-interval at `tcrit`. direction 0 = two-sided
    ("meaningful +" / "meaningful −"); ±1 = predicted direction ("meaningful" / "meaningful, opposite")."""
    _, lo, hi = ci_at(vals, tcrit)
    if lo > sesoi or hi < -sesoi:
        if direction == 0:
            return "meaningful +" if lo > sesoi else "meaningful −"
        return "meaningful" if (lo > sesoi) == (direction > 0) else "meaningful, opposite"
    if -sesoi < lo and hi < sesoi:
        return "equivalent"
    if lo > 0 or hi < 0:
        return "nonzero, possibly trivial"
    return "inconclusive"


def _check_category_at() -> None:
    base = np.array([-1.0, 1.0] * 4) * 0.01
    assert category_at(-1.0 + base, 0.1, T8, -1) == "meaningful"
    assert category_at(1.0 + base, 0.1, T8, -1) == "meaningful, opposite"
    assert category_at(1.0 + base, 0.1, T8) == "meaningful +"
    assert category_at(-1.0 + base, 0.1, T8) == "meaningful −"
    assert category_at(base, 0.1, T8) == "equivalent"
    assert category_at(0.1 + base, 0.1, T8) == "nonzero, possibly trivial"
    assert category_at(np.array([-1.0, 1.0] * 4), 0.1, T8) == "inconclusive"
    for v in (1.0 + base, -1.0 + base, base, 0.1 + base, np.array([-1.0, 1.0] * 4)):
        assert category_at(v, 0.1, T4, +1) == category(v, 0.1)    # docs/31's helper at the same t


def load_2x2(crit_path: Path, fixed_path: Path) -> dict:
    """cell -> {(seed, γ0, s_f): row}, selected by the cell's exact config dict (docs/32 §2)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import aba_cluster as C
    import aba_lib as L
    keys = {s.key for s in C.SPECS_2X2}
    out = {}
    for cell in CELLS_32:
        want = json.dumps(C.cell_cfg(L.ABAConfig.make(False), cell).key_fields(), sort_keys=True)
        path = crit_path if cell == "Cr,Cr" else fixed_path
        by = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r["key"] in keys and json.dumps(r["config"], sort_keys=True) == want:
                assert r["usable"], (cell, r["key"])
                s = r["spec"]
                by[(s["seed"], s["gamma"], s["s_f"])] = r
        assert len(by) == 32, (cell, len(by))
        out[cell] = by
    return out


def _fmt_ci(v, nd=3) -> str:
    m, lo, hi = ci_at(v, T8)
    return f"{m:+.{nd}f} [{lo:+.{nd}f}, {hi:+.{nd}f}]"


def twobytwo_tables(cells: dict) -> str:
    S = list(SEEDS_32)

    def f(cell, g, sf):
        return np.array([frac_removed(cells[cell][s, g, sf]) for s in S])

    def col(cell, g, sf, fn):
        return np.array([fn(cells[cell][s, g, sf]) for s in S])

    out = [f"Cells (A rule, B rule): Cr = criterion, Fx = fixed (A1 2300, B 2400; A2 2000 with Fx A). "
           f"h 4, seeds 8–15, n = 8 per stratum. Intervals: 99.375% t, critical value {T8:.3f} (df 7). "
           "Category: four categories, SESOI 0.10 (fraction removed), predicted direction only for E_A at "
           "γ0 10 (< 0); all other tests two-sided.", "",
           "**Main effects (the family of eight).**", "",
           "| stratum | effect | mean [99.375%] | seeds < 0 | ratio read | category |",
           "|---|---|---|---|---|---|"]
    verdict, accounts = {}, {}
    for sf, g in STRATA_32:
        read = all(category_at(col(c, g, sf, gain), 0.05, T8, +1) == "meaningful" for c in CELLS_32)
        EA = 0.5 * ((f("Fx,Cr", g, sf) - f("Cr,Cr", g, sf)) + (f("Fx,Fx", g, sf) - f("Cr,Fx", g, sf)))
        EB = 0.5 * ((f("Cr,Fx", g, sf) - f("Cr,Cr", g, sf)) + (f("Fx,Fx", g, sf) - f("Fx,Cr", g, sf)))
        for name, E, d in (("E_A (over-training A)", EA, -1 if g == 10.0 else 0),
                           ("E_B (over-exposure B)", EB, 0)):
            cat = category_at(E, 0.10, T8, d) if read else "not read"
            verdict[sf, g, name[:3]] = cat
            out.append(f"| h 4, s_f {sf:g} ({SF_NAME[sf]}), γ0 {g:g} | {name} | {_fmt_ci(E)} "
                       f"| {int((E < 0).sum())}/8 | {'yes' if read else 'no'} | {cat} |")
    out += ["", "**Committed prediction (docs/32 §5):** E_A < 0 at γ0 10 in both h 4 strata."]
    ea10 = [verdict[sf, 10.0, "E_A"] for sf in (1.0, 0.9)]
    if all(c == "meaningful" for c in ea10):
        pv = "**held** (meaningful < 0 in both strata)"
    elif any(c == "meaningful, opposite" for c in ea10):
        pv = "**not held** (meaningful > 0 in at least one stratum)"
    elif all(c == "equivalent" for c in ea10):
        pv = "**null** (equivalent in both strata)"
    else:
        pv = "neither held, not held nor null: " + "; ".join(
            f"{SF_NAME[sf]} {c}" for sf, c in zip((1.0, 0.9), ea10))
    out.append(f"Reading: {pv}.")
    out += ["", "**Accounts (docs/32 §5), per h 4 stratum:**", ""]
    for sf in (1.0, 0.9):
        ea, eb10, eb01 = verdict[sf, 10.0, "E_A"], verdict[sf, 10.0, "E_B"], verdict[sf, 0.1, "E_B"]
        if ea == "meaningful" and eb10 != "meaningful −":
            acc = "consolidation account"
        elif ea == "equivalent" and (eb10 == "meaningful −" or eb01 == "meaningful +"):
            acc = "exposure account"
        else:
            acc = "other pattern (reported as is)"
        out.append(f"- s_f {sf:g} ({SF_NAME[sf]}): E_A(10) {ea}; E_B(10) {eb10}; E_B(0.1) {eb01} → {acc}.")
    out += ["", "**Per cell (descriptive).** f = fraction removed; G = Δlog α_A(init → A1); "
            "L = Δlog α_A(A1 → B). Mean [99.375%].", "",
            "| stratum | cell | f | G | L |", "|---|---|---|---|---|"]
    for sf, g in STRATA_32:
        for c in CELLS_32:
            out.append(f"| h 4, s_f {sf:g}, γ0 {g:g} | ({c}) | {_fmt_ci(f(c, g, sf))} "
                       f"| {_fmt_ci(col(c, g, sf, gain))} "
                       f"| {_fmt_ci(col(c, g, sf, lambda r: r['backward']['A1_to_B']['total']))} |")
    out += ["", "**Interaction and simple effects (descriptive).** Interaction = ((Fx,Fx) − (Fx,Cr)) − "
            "((Cr,Fx) − (Cr,Cr)).", "",
            "| stratum | A at B Cr | A at B Fx | B at A Cr | B at A Fx | interaction |", "|---|---|---|---|---|---|"]
    for sf, g in STRATA_32:
        cc, fc, cf, ff = (f(c, g, sf) for c in CELLS_32)
        out.append(f"| h 4, s_f {sf:g}, γ0 {g:g} | {_fmt_ci(fc - cc)} | {_fmt_ci(ff - cf)} "
                   f"| {_fmt_ci(cf - cc)} | {_fmt_ci(ff - fc)} | {_fmt_ci((ff - fc) - (cf - cc))} |")
    out += ["", "**H1 contrast per cell (descriptive):** d = f(γ0 10) − f(γ0 0.1).", "",
            "| s_f | " + " | ".join(f"({c})" for c in CELLS_32) + " |", "|---|---|---|---|---|"]
    for sf in (1.0, 0.9):
        out.append(f"| {sf:g} ({SF_NAME[sf]}) | " + " | ".join(
            _fmt_ci(f(c, 10.0, sf) - f(c, 0.1, sf)) for c in CELLS_32) + " |")
    out += ["", "**Setting card per cell:** steps and final loss per phase, median [min, max] over seeds. "
            "Final losses are recorded only in the cells with a fixed phase (`behaviour.final_loss`).", "",
            "| stratum | cell | steps A1 | steps B | final loss A1 | final loss B |", "|---|---|---|---|---|---|"]
    for sf, g in STRATA_32:
        for c in CELLS_32:
            rs = [cells[c][s, g, sf] for s in S]

            def med(v, fmt):
                return f"{np.median(v):{fmt}} [{min(v):{fmt}}, {max(v):{fmt}}]"
            st = [med([r["behaviour"]["steps"][p] for r in rs], ".0f") for p in ("A1", "B")]
            if "final_loss" in rs[0]["behaviour"]:
                fl = [med([r["behaviour"]["final_loss"][p] for r in rs], ".1e") for p in ("A1", "B")]
            else:
                fl = ["≤ 0.05 (criterion)"] * 2
            out.append(f"| h 4, s_f {sf:g}, γ0 {g:g} | ({c}) | {st[0]} | {st[1]} | {fl[0]} | {fl[1]} |")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--ablation", nargs=2, metavar=("FIXED_JSONL", "FLOORS_JSON"),
                    help="docs/31: write the ablation results into docs/31")
    ap.add_argument("--twobytwo", nargs=2, metavar=("CRIT_JSONL", "FIXED_JSONL"),
                    help="docs/32: write the 2x2 results into docs/32")
    ap.add_argument("--twobytwo-dry", nargs=2, metavar=("CRIT_JSONL", "FIXED_JSONL"),
                    help="docs/32 format dry run: (Cr,Cr)/(Fx,Fx) from the given files stand in for "
                         "all four cells; printed, never written")
    ap.add_argument("--figset", action="store_true", help="draft A–B–A figure set F1–F4 into docs/figures")
    a = ap.parse_args()
    data = load()
    check_93(data)
    check_readings(data)
    _check_category()
    _check_category_at()
    if a.check:
        return
    if a.twobytwo:
        doc = REPO / "docs" / "32-aba-stopping-2x2-precommit.md"
        write_generated(doc, "2x2 results", twobytwo_tables(load_2x2(Path(a.twobytwo[0]), Path(a.twobytwo[1]))))
        print("2x2 results written into", doc)
        return
    if a.figset:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": "#fcfcfb",
                             "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb"})
        FIG.mkdir(parents=True, exist_ok=True)
        cells = load_2x2(*RUNS_32)
        for name, fig in (("aba_F1_gain_removal_restoration", figF1(data, plt)),
                          ("aba_F2_channel_shares", figF2(data, plt)),
                          ("aba_F3_accuracy_vs_capacity_frozen", figF3(data, plt)),
                          ("aba_F4_protocol_robustness", figF4(cells, plt))):
            fig.savefig(FIG / f"{name}.png", dpi=130)
            plt.close(fig)
            print("wrote", FIG / f"{name}.png")
        return
    if a.twobytwo_dry:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import aba_cluster as C
        import aba_lib as L
        stand = {}
        for cell, path in (("Cr,Cr", a.twobytwo_dry[0]), ("Fx,Fx", a.twobytwo_dry[1])):
            want = json.dumps(C.cell_cfg(L.ABAConfig.make(False), cell).key_fields(), sort_keys=True)
            keys = {s.key for s in C.SPECS_2X2}
            stand[cell] = {(r["spec"]["seed"], r["spec"]["gamma"], r["spec"]["s_f"]): r
                           for r in map(json.loads, Path(path).read_text(encoding="utf-8").splitlines())
                           if r["key"] in keys and json.dumps(r["config"], sort_keys=True) == want}
        print(twobytwo_tables({"Cr,Cr": stand["Cr,Cr"], "Fx,Cr": stand["Fx,Fx"],
                               "Cr,Fx": stand["Cr,Cr"], "Fx,Fx": stand["Fx,Fx"]}))
        return
    if a.ablation:
        doc = REPO / "docs" / "31-aba-stopping-ablation-precommit.md"
        write_generated(doc, "ablation results", ablation_tables(data, Path(a.ablation[0]), Path(a.ablation[1])))
        print("ablation results written into", doc)
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": "#fcfcfb",
                         "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb"})
    FIG.mkdir(parents=True, exist_ok=True)
    for name, f in (("aba_levels", fig_levels), ("aba_ratios", fig_ratios), ("aba_shares", fig_shares),
                    ("aba_ft", fig_h4), ("aba_acc_vs_capacity", fig_acc)):
        fig = f(data, plt)
        fig.savefig(FIG / f"{name}.png", dpi=130)
        plt.close(fig)
        print("wrote", FIG / f"{name}.png")
    if DOC.exists():
        write_generated(DOC, "setting cards", setting_cards(data))
        print("setting cards written into", DOC)


if __name__ == "__main__":
    main()
