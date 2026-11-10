"""Build notebooks/core/aba_similarity_map.ipynb (docs/34; docs/25 §7 format). Load-only notebook.

    python notebooks/core/_build_aba_similarity_map.py

The expectations (docs/34 §6, amended 2026-11-10) and the reading rules are written into the notebook
here and committed before the Phase 1 job is submitted.
"""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf

CORE = Path(__file__).resolve().parent
sys.path.insert(0, str(CORE))
import aba_map_analysis as A  # noqa: E402

OUT = CORE / "aba_similarity_map.ipynb"

EXPECTATIONS = """\
## 2. Pre-commit: expectations (docs/34 §6, quoted verbatim; written and committed before submission)

Source: [`docs/34-aba-similarity-map.md`](../../docs/34-aba-similarity-map.md), ratified by Kati
2026-11-10, with the dated amendments of the same day (before any map arm). EXPLORATORY: no
confirmatory claims; nothing enters a Bonferroni family. These are directions, not thresholds:

- **E1 (Hiratani corners).** Forgetting (f and bottleneck) is largest at high input similarity × H8,
  and smallest at low input similarity, at every γ₀.
- **E2 (shape in s_f).** At H4 and H8, f is monotonic in s_f, *or* peaked at intermediate s_f. Record
  which. A peak would match Lee et al. and Hiratani.
- **E3 (shape in H).** At s_f ∈ {1, 0.9}, f rises from H0 to H8 (a dose). Behaviour at H12 and H16
  falls below chance on A while geometry recovers towards H0.
- **E4 (H16).** At γ₀ 0.1, A's capacity is unchanged at H16 (readout flip only). At γ₀ 10, record
  whether capacity is lost (feature rebuilding).
- **E5 (Menghi tension).** At intermediate s_f (0.4–0.6) with H0, forward transfer stays positive. If
  it turns negative, that matches Menghi's partly-similar-input interference.
- **E6 (richness).** The ordering across similarity cells is the same at every γ₀; only size and
  channel change.
- **E7 (restoration; amended 2026-11-10).** Restoration is complete at high input similarity and falls
  below 0.9 as input similarity drops, at γ₀ ≥ 1 (cf. docs/33 limits: ρ = 0.862 at h 8 jump, γ₀ 10).
"""

CELLS = [
    ("md", """\
# A–B–A similarity map (docs/34, Phase 1) — EXPLORATORY

**Question.** How do forgetting and transfer depend on label and input similarity?

**Slot.** `docs/24` §4 core, controlled paradigm (A–B–A), exploratory extension (`docs/34`). The stream
is parked until this map is reported (`docs/34` §8).

**Design (Phase 1, amended 2026-11-10).** Learner, data, training and measurement exactly as docs/28 §2
(1M, N = 300, μP, matched-loss stopping). γ₀ ∈ {0.1, 1, 10}; seeds 16–23 (fresh). Two slices:
- the H sweep, H ∈ {0, 2, 4, 6, 8, 12, 16}, at s_f ∈ {1, 0.9};
- the s_f sweep, s_f ∈ {1, 0.95, 0.9, 0.8, 0.6, 0.4, 0.2, 0.1}, at H ∈ {0, 4, 8}.

That is 32 configurations × 3 γ₀ × 8 seeds = 768 arms, plus one fresh-noise identity arm per γ₀.
"""),
    ("md", EXPECTATIONS),
    ("md", "### Reading rules (fixed before the run)\n\n" + A.READING_RULES),
    ("md", "## 3. Setup"),
    ("code", """\
import os, sys, json, math
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
import numpy as np
import matplotlib.pyplot as plt
import aba_lib as L
import aba_map_analysis as A
plt.rcParams.update({"figure.dpi": 80, "font.family": "DejaVu Sans", "figure.facecolor": "#fcfcfb",
                     "axes.facecolor": "#fcfcfb", "axes.spines.top": False, "axes.spines.right": False})
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
LEVEL_COL = ["#2a78d6", "#eb6834", "#1baf7a"]      # validated categorical slots 1-3 (slice levels)
GCOL = {0.1: "#2a78d6", 1.0: "#eb6834", 10.0: "#1baf7a"}

cfg = A.map_cfg()
MAP_JSONL = L.REPO / "results" / "aba" / f"FULL_{L.lib_hash()}.jsonl"
HAVE = MAP_JSONL.exists()
SEEDS = list(cfg.seeds)
print(f"lib hash {L.lib_hash()}; stamp now {L.git_stamp()}")
print(f"checkpoint {MAP_JSONL.name}: {'present' if HAVE else 'NOT YET RUN'}")
print(json.dumps({k: v for k, v in cfg.key_fields().items()}, indent=None)[:600])
print(f"γ0 {cfg.gammas}; H {cfg.hammings}; s_f {cfg.s_fs}; seeds {SEEDS[0]}–{SEEDS[-1]} (n = {len(SEEDS)}); "
      f"sampling unit: one (arrangement, init) draw per seed, paired across cells and γ0")
print(f"Phase 1 arms: {len(L.phase1_specs(cfg))} ({len(A.phase1_cells(cfg))} configurations)")
"""),
    ("md", "## 4. Load and validate (must pass)"),
    ("code", """\
if HAVE:
    D, F, ROWS = A.load_map(MAP_JSONL, cfg)
    CHECKS = A.validate(D, F, ROWS, SEEDS, cfg)
    for name, ok, det in CHECKS:
        print(f"{'PASS' if ok else 'FAIL'}  {name}  {det}")
    assert all(ok for _, ok, _ in CHECKS), "validation failed: no readings"
    T = A.table(D, SEEDS, cfg)
"""),
    ("md", f"""\
## 5. Results (EXPLORATORY)

Mean and 95% t-interval over 8 seeds, per configuration; signed values only; nothing pooled across
configurations. {A.EXPOSURE_NOTE}

**Profiles over the nominal axes.** One figure per quantity. Columns are γ₀. Row 1: the H sweep (lines
s_f 1, 0.9); row 2: updates on B for the same cells; row 3: the s_f sweep (lines H 0, 4, 8); row 4:
updates on B for the same cells. (With two slices, profiles are the map; a 2-D heatmap needs Phase 2.)
"""),
    ("code", """\
def profile(q):
    lab = A.QUANTS[q][0]
    fig, axes = plt.subplots(4, 3, figsize=(11, 10), sharex="row")
    for j, g in enumerate(A.GAMMAS):
        for row, (xs, lines, key) in enumerate([(cfg.hammings, A.H_SWEEP_SF, lambda x, l: (x, l)),
                                                (cfg.s_fs, A.SF_SWEEP_H, lambda x, l: (l, x))]):
            for k, qq in enumerate((q, "stepsB")):
                ax = axes[2 * row + k, j]
                for c, lev in enumerate(lines):
                    m = [T[qq, g, *key(x, lev)] if qq != "stepsB" else A.steps_range(D, g, *key(x, lev), SEEDS)
                         for x in xs]
                    ax.errorbar(xs, [v[0] for v in m], yerr=[[v[0] - v[1] for v in m], [v[2] - v[0] for v in m]],
                                color=LEVEL_COL[c], lw=1.5, marker="o", ms=4, capsize=0,
                                label=(f"s_f {lev:g}" if row == 0 else f"H {lev}"))
                if qq == "stepsB":
                    ax.set_yscale("symlog", linthresh=10)
                    ax.set_ylim(bottom=0)
                ax.grid(color=GRID, lw=0.6)
                if j == 0:
                    ax.set_ylabel(lab if qq == q else "updates on B (median, range)", fontsize=8, color=INK2)
                if 2 * row + k in (0, 2):
                    ax.set_title(f"γ0 {g:g}" + (" — H sweep" if row == 0 else " — s_f sweep"), fontsize=9)
                if k == 1:
                    ax.set_xlabel("H (Hamming, of 16)" if row == 0 else "s_f (input-centre relation)", fontsize=8)
                if j == 2 and k == 0:
                    ax.legend(frameon=False, fontsize=7)
    fig.suptitle(f"{lab}: EXPLORATORY, mean ± 95% CI, n = 8 per cell. Updates on B (exposure; median [min, max]) "
                 "below each map.",
                 fontsize=10)
    fig.tight_layout()
    plt.show()

if HAVE:
    for q in ("f", "dA", "rho", "u", "r", "d", "ftgeo", "ftbeh_spec", "bott", "accA", "drhoc", "savings"):
        profile(q)
"""),
    ("md", f"""\
**Against measured similarity (docs/34 §4).** One point per configuration and γ₀: x is the mean
measured similarity over seeds, y the mean ± 95% CI of the quantity. Circles: on the H sweep; squares: s_f
sweep only. Updates on B as median [min, max]. FT_beh specific is relative to H8 × s_f 0.1 of the same seed and γ₀. {A.EXPOSURE_NOTE}
"""),
    ("code", """\
def measured(qs=("f", "dA", "rho", "ftbeh", "bott", "stepsB")):
    fig, axes = plt.subplots(len(qs), 3, figsize=(11, 2.6 * len(qs)), sharex="col")
    for i, q in enumerate(qs):
        for j, (mk, mlab) in enumerate(A.MEASURED.items()):
            ax = axes[i, j]
            for (h, sf) in A.phase1_cells(cfg):
                for g in A.GAMMAS:
                    x = np.mean([D[s, g, h, sf]["relations"][mk] for s in SEEDS])
                    m, lo, hi = T[q, g, h, sf] if q != "stepsB" else A.steps_range(D, g, h, sf, SEEDS)
                    marker = "o" if sf in A.H_SWEEP_SF else "s"     # circle: on the H sweep; square: s_f sweep only
                    ax.plot([x, x], [lo, hi], color=GCOL[g], lw=1, alpha=0.6)
                    ax.scatter([x], [m], color=GCOL[g], s=14, marker=marker)
            if q == "stepsB":
                ax.set_yscale("symlog", linthresh=10)
                ax.set_ylim(bottom=0)
            ax.grid(color=GRID, lw=0.6)
            if j == 0:
                ax.set_ylabel(A.QUANTS[q][0] + (" (median [min, max])" if q == "stepsB" else ""), fontsize=7, color=INK2)
            if i == len(qs) - 1:
                ax.set_xlabel(mlab, fontsize=8)
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], marker="o", ls="", color=GCOL[g], label=f"γ0 {g:g}") for g in A.GAMMAS],
               loc="lower center", ncol=3, frameon=False, fontsize=8)
    fig.suptitle("Against measured similarity: EXPLORATORY, mean ± 95% CI per configuration (updates on B in the last row)",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    plt.show()

if HAVE:
    measured()
"""),
    ("md", """\
**Third independent sample of the docs/28 configurations (amendment b).** h 4 and h 8 × s_f {1, 0.9,
0.1}, seeds 16–23, beside seeds 8–15 (job 4396) and seeds 0–7 (job 4349). Unpooled; 95% intervals here
(the docs/28 readings used 99.17%). Not a reading of any docs/28 hypothesis. EXPLORATORY.
"""),
    ("code", """\
import aba_summary as S
if HAVE:
    PRIOR = S.load()
    hdr = "| H | s_f | γ0 | quantity | seeds 16–23 (map) | seeds 8–15 | seeds 0–7 |"
    lines = [hdr, "|---|---|---|---|---|---|---|"]
    fn = {"f": S.frac_removed, "dA": lambda r: r["backward"]["A1_to_B"]["total"], "rho": S.restoration,
          "ftbeh": lambda r: r["behaviour"]["FT_beh"]}
    for h in (4, 8):
        for sf in (1.0, 0.9, 0.1):
            for g in A.GAMMAS:
                for q in ("f", "dA", "rho", "ftbeh"):
                    cells = [A._f(T[q, g, h, sf])]
                    for run in ("seeds 8–15", "seeds 0–7"):
                        P = PRIOR[run]
                        cells.append(A._f(A.ci([fn[q](P[s, g, h, sf]) for s in S.seeds_of(run)])))
                    lines.append(f"| {h} | {sf:g} | {g:g} | {q} | " + " | ".join(cells) + " |")
    from IPython.display import Markdown, display
    display(Markdown("\\n".join(lines)))
"""),
    ("md", f"""\
**Setting cards (docs/29 §2a, per configuration).** Nominal H and s_f; realised s_r (exact), ρ_in,
task-vector cosines (inputs, h₀) as median [min, max] over seeds; updates per phase per γ₀ (median
[min, max]). {A.EXPOSURE_NOTE}
"""),
    ("code", """\
def mmm(v, fmt):
    return f"{np.median(v):{fmt}} [{min(v):{fmt}}, {max(v):{fmt}}]"

if HAVE:
    lines = ["| H | s_f | s_r | ρ_in | taskvec (inputs) | taskvec (h₀) | γ0 | A1 | B | A2 | fresh B |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for h, sf in A.phase1_cells(cfg):
        rel = [D[s, 0.1, h, sf]["relations"] for s in SEEDS]
        for g in A.GAMMAS:
            st = [D[s, g, h, sf]["behaviour"]["steps"] for s in SEEDS]
            lines.append(f"| {h} | {sf:g} | {rel[0]['s_r_AB']:+.3f} | {mmm([r['rho_in_AB'] for r in rel], '.3f')} | "
                         f"{mmm([r['taskvec_cos_input'] for r in rel], '+.3f')} | {mmm([r['taskvec_cos_h0'] for r in rel], '+.3f')} | "
                         f"{g:g} | " + " | ".join(mmm([x[p] for x in st], '.0f') for p in ("A1", "B", "A2", "freshB")) + " |")
    from IPython.display import Markdown, display
    display(Markdown("\\n".join(lines)))
"""),
    ("md", "## 6. Expectation readings (rules fixed in §2; EXPLORATORY)"),
    ("code", """\
if HAVE:
    READ = A.read_expectations(D, SEEDS, cfg)
    for e, (verdict, det) in READ.items():
        print(f"{e}: {verdict}")
        for d in det:
            print("    " + d)
"""),
    ("md", """\
## 7. Limits

- EXPLORATORY: 95% intervals for description, no family, no confirmatory language.
- One learner (1M, μP, full-batch GD) at matched current-task fit. Under matched-loss stopping a similar
  B needs fewer updates, so every similarity effect includes its effect on exposure.
- Two slices only (Phase 1). Interactions between H and s_f away from the slices need Phase 2.
- Our s_f moves the input centres; it is not the literature's input-feature similarity. Our H changes
  which directions separate the classes as well as the readout. Both axes are therefore also shown
  against measured similarity (task-vector cosine, realised ρ_in).
- The h 4/h 8 × s_f {1, 0.9, 0.1} cells are a third sample of docs/28 configurations, reported beside
  docs/28 and docs/30, unpooled, and not read against the docs/28 hypotheses.
"""),
]


def build() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "gof-venv", "display_name": "gof-venv", "language": "python"}
    nb.cells = [nbf.v4.new_markdown_cell(src) if kind == "md" else nbf.v4.new_code_cell(src)
                for kind, src in CELLS]
    return nb


if __name__ == "__main__":
    nbf.write(build(), OUT)
    print("wrote", OUT)
