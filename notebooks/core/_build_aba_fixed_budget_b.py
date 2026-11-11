"""Build notebooks/core/aba_fixed_budget_b.ipynb (docs/36; docs/25 §7 format). Load-only notebook.

    python notebooks/core/_build_aba_fixed_budget_b.py

The predictions (docs/36 §5) and the reading rules are written into the notebook here and committed
before the Phase 1b job is submitted.
"""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf

CORE = Path(__file__).resolve().parent
sys.path.insert(0, str(CORE))
import aba_fixedb_analysis as FB  # noqa: E402

OUT = CORE / "aba_fixed_budget_b.ipynb"

PREDICTIONS = """\
## 2. Pre-commit: predictions (docs/36 §5; committed before submission)

Source: [`docs/36-aba-fixed-budget-b-precommit.md`](../../docs/36-aba-fixed-budget-b-precommit.md), Kati
2026-11-11. EXPLORATORY (docs/34).

- **(i) The H4–H12 gap in f shrinks under equal exposure on B.** H4 and H12 are independent draws within a
  seed; an H12 task is an H4 task with a flipped readout only in distribution. A distributional comparison
  paired by seed. Null: the gap is unchanged, so exposure does not explain it.
- **(ii) Shape survives.** Within each γ₀, the rank order of the 11 cells by f and the sign of each resolved
  adjacent contrast are unchanged. Null: a resolved adjacent contrast reverses.
- **(iii) The H0-drift gain is not reversed by extra exposure.** At H0 × s_f 0.9, Δlog α_A(A1 → B) stays ≥ 0.
  Null: extra exposure turns the gain into a loss.

**By design** (docs/36 §3): A1 is the job 4651 computation, so the gain G and every forward quantity are
unchanged (FT_beh_hit = job 4651 FT_beh). This arm changes backward quantities only.
"""

CELLS = [
    ("md", """\
# A–B–A similarity map, Phase 1b: fixed budget on phase B per γ₀ (docs/36) — EXPLORATORY

**Question.** Does the shape of the forgetting map survive when every cell gets the same number of updates
on B within each γ₀?

**Slot.** `docs/24` §4 core, controlled paradigm (A–B–A), exploratory extension (`docs/34` Phase 1b, `docs/36`).
Paper backlog B1, extreme cases run 1.

**Design.** As docs/34 Phase 1 (job 4651; 1M, N = 300, μP, seeds 16–23), except: phase B and fresh B run a
fixed budget per γ₀ (γ₀ 0.1: 3200, γ₀ 1: 600, γ₀ 10: 100 updates; the job 4651 maximum rounded up to the next
100); A1 and A2 stay at criterion. 11 cells: s_f 1 × H {0, 2, 4, 8, 12, 16}; H0 × s_f {0.9, 0.4, 0.1};
s_f 0.1 × H {4, 8}. 264 arms, each paired with its job 4651 row (Cr).
"""),
    ("md", PREDICTIONS),
    ("md", "### Reading rules (fixed before the run)\n\n" + FB.READING_RULES_1B),
    ("md", "## 3. Setup"),
    ("code", """\
import os, sys, json
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
import numpy as np
import matplotlib.pyplot as plt
import aba_lib as L
import aba_map_analysis as A
import aba_fixedb_analysis as FB
plt.rcParams.update({"figure.dpi": 80, "font.family": "DejaVu Sans", "figure.facecolor": "#fcfcfb",
                     "axes.facecolor": "#fcfcfb", "axes.spines.top": False, "axes.spines.right": False})
INK2, GRID = "#52514e", "#e4e3df"
GCOL = {0.1: "#2a78d6", 1.0: "#eb6834", 10.0: "#1baf7a"}

base = L.ABAConfig.make(False)
CFGS = FB.fixed_cfgs(base)
FX_PATH = L.cache_path(CFGS[0.1])
HAVE = FX_PATH.exists() and FB.MAP_JSONL.exists()
SEEDS = list(L.map_config(base).seeds)
print(f"lib hash {L.lib_hash()}; stamp now {L.git_stamp()}")
print(f"Fx checkpoint {FX_PATH.name}: {'present' if FX_PATH.exists() else 'NOT YET RUN'}; "
      f"Cr (job 4651) {FB.MAP_JSONL.name}: {'present' if FB.MAP_JSONL.exists() else 'MISSING'}")
print(f"budgets {L.PHASE1B_BUDGET}; cells {len(FB.CELLS)}; seeds {SEEDS[0]}–{SEEDS[-1]} (n = {len(SEEDS)})")
"""),
    ("md", "## 4. Load and integrity checks (docs/36 §2; must pass)"),
    ("code", """\
if HAVE:
    FXD, FXROWS = FB.load_fixed(base, FX_PATH)
    CR = FB.load_criterion(FB.MAP_JSONL, base)
    CHECKS = FB.integrity((FXD, FXROWS), CR, base)
    for name, ok, det in CHECKS:
        print(f"{'PASS' if ok else 'FAIL'}  {name}  {det}")
    assert all(ok for _, ok, _ in CHECKS), "integrity failed: no readings"
    T = FB.side_by_side(FXD, CR, SEEDS)
"""),
    ("md", """\
## 5. Results (EXPLORATORY)

**Example plot (paper-backlog B1).** f along the H sweep at s_f 1, criterion (Cr, job 4651, solid) against
fixed budget on B (Fx, dashed), one column per γ₀; below, updates on B and the weight change during B (r_B).
Equal updates are not equal movement once a learner is past the criterion. Mean ± 95% CI, n = 8.
"""),
    ("code", """\
def sweep_plot(cells, xs, xlab, title):
    fig, axes = plt.subplots(3, 3, figsize=(11, 8), sharex=True)
    for j, g in enumerate(FB.GAMMAS):
        for i, q in enumerate(("f", "stepsB", "rB")):
            ax = axes[i, j]
            for k, ls, lab in ((0, "-", "criterion (job 4651)"), (1, "--", f"fixed B = {L.PHASE1B_BUDGET[g]}")):
                m = [T[q, g, c][k] for c in cells]
                ax.errorbar(xs, [v[0] for v in m], yerr=[[v[0] - v[1] for v in m], [v[2] - v[0] for v in m]],
                            color=GCOL[g], ls=ls, lw=1.5, marker="o", ms=4, capsize=0, label=lab)
            if q == "stepsB":
                ax.set_yscale("symlog", linthresh=10)
                ax.set_ylim(bottom=0)
            ax.grid(color=GRID, lw=0.6)
            if j == 0:
                ax.set_ylabel(FB.QUANTS[q][0], fontsize=8, color=INK2)
            if i == 0:
                ax.set_title(f"γ0 {g:g}", fontsize=9)
                ax.legend(frameon=False, fontsize=7)
            if i == 2:
                ax.set_xlabel(xlab, fontsize=8)
    fig.suptitle(title + ": EXPLORATORY, mean ± 95% CI, n = 8 per cell", fontsize=10)
    fig.tight_layout()
    plt.show()

if HAVE:
    sweep_plot(FB.H_SWEEP, [h for h, _ in FB.H_SWEEP], "H (Hamming, of 16) at s_f 1",
               "Fraction of A's gain removed by B, criterion vs fixed budget on B")
    sweep_plot([(0, 1.0), (0, 0.9), (0, 0.4), (0, 0.1)], [1.0, 0.9, 0.4, 0.1], "s_f at H0",
               "H0 input axis, criterion vs fixed budget on B")
"""),
    ("md", """\
**Per cell, Cr beside Fx (docs/36 §5b, descriptive).** Mean [95% CI] under Cr and Fx, and the paired
difference Fx − Cr. Includes measured exposure (r_B, ΔK_u(A1 → B)) and the first step at criterion on B.
"""),
    ("code", """\
if HAVE:
    from IPython.display import Markdown, display
    for q in ("f", "dA", "bott", "rB", "dKu", "stepsB", "hitB", "u", "r", "d", "rho", "savings", "accA", "drhoc"):
        lines = [f"**{FB.QUANTS[q][0]}**", "", "| H | s_f | γ0 | Cr | Fx | Fx − Cr |", "|---|---|---|---|---|---|"]
        for (h, sf) in FB.CELLS:
            for g in FB.GAMMAS:
                c, x, d = T[q, g, (h, sf)]
                lines.append(f"| {h} | {sf:g} | {g:g} | {A._f(c)} | {A._f(x)} | {A._f(d)} |")
        display(Markdown("\\n".join(lines)))
"""),
    ("md", """\
**Setting card.** Final loss of B and fresh B under Fx, median [min, max] over seeds; first step at
criterion per phase (Fx), which equals the job 4651 step count for B and fresh B (integrity check 2).
"""),
    ("code", """\
def mmm(v, fmt):
    return f"{np.median(v):{fmt}} [{min(v):{fmt}}, {max(v):{fmt}}]"

if HAVE:
    from IPython.display import Markdown, display
    lines = ["| H | s_f | γ0 | budget | final loss B | final loss fresh B | first_hit A1 | B | A2 | fresh B |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for (h, sf) in FB.CELLS:
        for g in FB.GAMMAS:
            rs = [FXD[s, g, h, sf]["behaviour"] for s in SEEDS]
            lines.append(f"| {h} | {sf:g} | {g:g} | {L.PHASE1B_BUDGET[g]} | "
                         f"{mmm([r['final_loss']['B'] for r in rs], '.1e')} | {mmm([r['final_loss']['freshB'] for r in rs], '.1e')} | "
                         + " | ".join(mmm([r['first_hit'][p] for r in rs], '.0f') for p in ("A1", "B", "A2", "freshB")) + " |")
    display(Markdown("\\n".join(lines)))
"""),
    ("md", "## 6. Prediction readings (rules fixed in §2; EXPLORATORY)"),
    ("code", """\
if HAVE:
    READ = FB.read_predictions(FXD, CR, SEEDS)
    for e, (verdict, det) in READ.items():
        print(f"{e}: {verdict}")
        for d in det:
            print("    " + d)
"""),
    ("md", """\
## 7. Limits

- EXPLORATORY: 95% intervals for description, no family, no confirmatory language.
- Extreme cases only (11 cells); the full-grid comparison is on the paper backlog (B1).
- The budget equalises **updates** on B within a γ₀, not **movement**: past the criterion a learner may barely
  move (r_B, ΔK_u are shown). Across γ₀, budgets differ (3200 / 600 / 100); γ₀ comparisons stay at matched fit
  on A.
- A1 and A2 at criterion: over-training of A (paper-backlog B2) is excluded by design; A2 starts from a
  different end of B, so restoration and savings change through B only.
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
