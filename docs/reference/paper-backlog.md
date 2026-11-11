# Paper backlog

Items the final paper must contain that are not yet in final form. Each item states:
- what the paper needs;
- the sources that already exist;
- what is still missing;
- its status.

An item is closed only by a dated line naming where it landed. Items are never deleted.

## B1. Appendix: stopping protocol, matched loss vs fixed budget (Kati, 2026-11-11)

**What the paper needs.** An appendix section that explains why the core results use matched-loss stopping,
shows what changes under a fixed budget per γ0, and contains at least one example plot comparing the two on
the same cells.

**Rationale to write out carefully** (draft points, from the 2026-11-11 discussion):
- **The literature uses fixed budgets.** None of the four reference papers stops on a loss criterion. Items
  marked * were read only in summary, so verify them verbatim before citing.
  - Chou, Le, Wang & Chung (ICML 2025): fixed 10⁵ epochs for the 2-layer nets (App. D.2.1, D.1.4).
    - Training time is equalised across scales by an α⁻² loss rescaling "to ensure common convergence time"
      (D.1.3), not by a stopping rule.
    - Geometry is analysed along the trajectory (50 log-spaced checkpoints, D.1.5).
  - Graldi et al. (ICML 2025): fixed epochs per task (MLP: 1000 epochs, LR 0.25*), cosine schedule restarted
    per task.
    - The LR scales with γ0 through μP (η·γ0²·N, Table 1).
    - Training time itself drives NTK evolution (§4).
  - Hiratani (2024): the analysis is at convergence; the numerics use a fixed 100 iterations per task
    (Fig. 1).
  - Lee, Goldt & Saxe (2021): online SGD with a fixed switch step (for example 5·10⁵); the intent is
    convergence before the switch*, but no criterion is given.
- **Why matched loss is our primary protocol.**
  - Accuracy saturates after one step at every γ0 (docs/01, error registry E1).
  - Steps to criterion differ about 39× across γ0 (docs/15). Any common budget either leaves the lazy learner
    short of the criterion or over-trains the rich one by orders of magnitude.
  - Matched loss compares regimes at equal current-task fit (K9).
  - Caveat to state: loss and geometry run on different clocks (161.6× vs 8.0× time rescaling, docs/01), so
    "equal fit" is not "equal representational change".
- **Why a fixed budget per γ0 is needed as well.**
  - Under matched loss, exposure to B depends on the similarity cell, so within one γ0 a similarity effect
    includes an exposure effect. Phase 1 examples (docs/35 §3):
    - H12 forgets much more than H4 although the two are sign flips of each other, and H12 also gets about
      twice the updates on B.
    - On the H0 input axis, updates on B fall from about 1466 to 0 as s_f → 1 (γ0 0.1).
  - A fixed budget per γ0 equalises exposure across cells within each γ0, while the comparison across γ0 stays
    at matched loss.
  - A common budget across γ0 (the docs/31 design) is the most literal Graldi-style protocol. It is a
    secondary check only, because over-training drives its size (B2).

**Runs.**
- **Interim, before the final paper: extreme cases only**, fixed budget per γ0. Pre-commit and cell choice to
  follow (Phase 1b, docs/34 §5 amendment).
  - **Extreme cases, run 1: docs/36** (pre-committed 2026-11-11). Fixed budget on phase B and fresh B per γ0
    (3200 / 600 / 100); A1 and A2 at criterion; 11 cells × 3 γ0 × seeds 16–23 = 264 arms. Job: TBD. Its §5b
    plot (f along the H sweep, criterion vs fixed, with updates on B and r_B) is the candidate example plot.
- **Final appendix: the full grid**, fixed budget per γ0 on all Phase 1 configurations (32 × 3 γ0 × 8 seeds),
  compared cell by cell with the matched-loss map (job 4651).

**Missing.** The Phase 1b pre-commit, the extreme-case run, the full-grid run, the example plot, verbatim
checks of the * citations, and the appendix text.

**Status:** OPEN. The interim run is not yet pre-committed.

## B2. Appendix: over-training drives the size of the γ0 effect (Kati, 2026-11-11)

**What the paper needs.** The stopping-protocol finding from docs/31–32, reported in the appendix next to B1.
The text is not in final form yet.

**Finding as it stands** (docs/30 §2b–2c; jobs 4458 and 4538; seeds 8–15):
- H1, d = f(10) − f(0.1) at h 4 drift, is sign-robust across all four combinations of the A rule and the B
  rule (criterion or fixed): from +0.333 under (Cr,Cr) to +0.184 under (Fx,Fx), with every interval
  excluding 0.
- At h 4 frozen, d is unresolved under (Fx,Fx): +0.100 [−0.087, +0.288].
- **The size depends on the protocol, through two routes** (docs/32 main effects on f, SESOI 0.10):
  - **γ0 10:** over-training A (loss 0.05 → about 1e-6) lowers f(10): E_A −0.083 (drift) / −0.142 (frozen),
    8/8 seeds each.
  - **γ0 0.1:** extra B updates raise f(0.1): E_B +0.064 / +0.071.
- Kati's pre-committed docs/31 expectation (d keeps its sign or grows) was wrong in direction: d shrank. It
  counted extra exposure on B and ignored over-training of A.
- **Interpretation, not tested:** updates past the criterion barely move a rich learner. At γ0 10, about 100×
  more updates on B left f(10) unchanged at h 4 frozen (E_B equivalent).

**Missing.**
- Final appendix wording.
- A figure: d per (A rule, B rule) cell, with E_A and E_B per γ0.
- A decision on whether the untested interpretation is stated, and in what form.
- Linking to B1 once the fixed-budget-per-γ0 runs exist, since they test the exposure route without
  over-training A.

**Status:** OPEN. Marked for the appendix; the text is not final.
