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
        t = ci([frac_removed(A[s, G2, 4, sf]) - frac_removed(A[s, G0, 4, sf]) for s in seeds])
        read = gain_res[G0] and gain_res[G2]
        sg = int(np.sign(t[0])) if excl0(t) else 0
        out.append({"hyp": "H1", "stratum": f"h 4, s_f {sf:g} ({SF_NAME[sf]})",
                    "stat": "d = f(10) − f(0.1) = " + fci(t, 3),
                    "mean": t[0], "sign": sg, "reading": dir_reading(sg) if read else "not read"})
    gg = {g: [gain(A[s, g, 0, 1.0]) for s in seeds] for g in GAMMAS}
    for (ga, gb), name in (((G1, G0), "d₁ = g(1) − g(0.1)"), ((G2, G1), "d₂ = g(10) − g(1)")):
        t = ci(np.subtract(gg[ga], gg[gb]))
        sg = res_sign(t, max(FL[ga], FL[gb]))
        out.append({"hyp": "H2", "stratum": name, "stat": fci(t, 3), "mean": t[0], "sign": sg,
                    "reading": dir_reading(sg)})
    for h in (4, 8):
        for sf in (1.0, 0.9):
            tot = ci([A[s, G2, h, sf]["backward"]["A1_to_B"]["total"] for s in seeds])
            read = res_sign(tot, FL[G2]) != 0
            t = ci([share(A[s, G2, h, sf], "utility") + share(A[s, G2, h, sf], "dimension") for s in seeds])
            r = "held" if t[1] >= 0.8 else "not held" if t[2] < 0.8 else "unresolved at floor"
            out.append({"hyp": "H5a", "stratum": f"γ0 10, h {h}, s_f {sf:g} ({SF_NAME[sf]})",
                        "stat": "u + d share " + fci(t, 3), "mean": t[0], "sign": None,
                        "reading": r if read else "not read"})
    for g in GAMMAS:
        for h in (4, 8):
            for sf in (1.0, 0.9):
                t = ci([restoration(A[s, g, h, sf]) for s in seeds])
                r = "held" if t[1] >= 0.9 else "not held" if t[2] < 0.9 else "unresolved at floor"
                out.append({"hyp": "H6", "stratum": f"γ0 {g:g}, h {h}, s_f {sf:g} ({SF_NAME[sf]})",
                            "stat": "ρ = " + fci(t, 3), "mean": t[0], "sign": None,
                            "reading": r if gain_res[g] else "not read"})
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
           "**Readings per stratum** (docs/28 rules; 99.17% t-intervals, n = 8).", "",
           "| hypothesis | stratum | criterion: statistic | criterion reading | fixed budget: statistic "
           "| fixed reading | docs/31 reading |",
           "|---|---|---|---|---|---|---|"]
    for c, f in zip(crit, fix):
        assert (c["hyp"], c["stratum"]) == (f["hyp"], f["stratum"])
        out.append(f"| {c['hyp']} | {c['stratum']} | {c['stat']} | {c['reading']} | {f['stat']} "
                   f"| {f['reading']} | {compare(c, f)} |")
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--ablation", nargs=2, metavar=("FIXED_JSONL", "FLOORS_JSON"),
                    help="docs/31: write the ablation results into docs/31")
    a = ap.parse_args()
    data = load()
    check_93(data)
    check_readings(data)
    if a.check:
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
