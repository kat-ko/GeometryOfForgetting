# 09 — Post-Generation Brief: Propagation Audit, Characterisation, and the Results Notebook

```
Context:  11 days to NeurReps (Aug 24 AoE). §5 drafts exist; generation was
          closed and has now reopened narrowly because the sign fix produced a
          new result. ICLR deadline Sept 29.
Priority: Tier 1 blocks the draft. Tier 2 characterises the new result.
          Tier 3 is ICLR lead-time only and must not touch NeurReps artifacts.
```

---

## 0. Feedback

**The plotting bug is the most consequential catch since the joint-QP finding, and
it is a second instance of a pattern worth naming.** Panel (a) plotted
`abs(Δ log α)` under an axis reading "how much retained capacity is lost", so a
capacity *gain* appeared as a positive magnitude indistinguishable from a loss.
This is the same mechanism as the `rho_c` absolute-value convention: a
transformation applied for presentational tidiness erased the distinction being
measured. Twice now, both fixed by keeping signs. Add it to §A.2 as a **separate
family from guard-too-permissive** — those were guards that admitted
out-of-domain values; these are transformations that destroy sign information.
The generalisable point: any absolute value or magnitude applied to a signed
quantity needs a stated justification, because the default outcome is a silent
mixture.

**What it hid is a new result and it changes what the 2×2 section can claim.**
`S-HH` shows backward transfer — retained capacity on task 0 *increases* when
later tasks are similar on both axes, peaking at 11.8 floors at γ=1 and falling
to 6.2 by γ=10. Previously the 2×2 modulated how much was forgotten. Now it
determines *whether anything is forgotten at all*, with one corner improving.

**The three-corner regrouping is justified by removing a sign mixture, not by
removing inconvenient data, and the caption must say so.** Pooling one gain with
three losses is a measurement error. The evidence that it was: γ=0.1 becomes
resolvable at 2.1 floors (was 0.4), the utility term is negative on 100% of arms
for γ ≥ 1 (was 75%), and panel (d) reaches 100% coverage at γ ≥ 1 — better than
the refit's 88% and better than the original 91%. Separating one corner post-hoc
looks like selection unless that reasoning is explicit; state it in the caption
and in §5.1.

**The inventory earned itself on its first run**, and the finding is a repeat.
`S-LH`'s Δρ_c is +0.110, not the +0.092 in the §5.4 draft, because the value came
from the W1 width table's N=300 row — a 4-arm cell. That is the same row and the
same failure as the `S-LL` correction, which was applied to one column only. The
claim survives (0.1098 / 0.0550 is still exactly twice, opposite sign), but that
row has now produced two wrong numbers in drafted text. Fix the row, not the
cells.

**Both requested checks passed cleanly.** Lag-4 companion: shares agree to within
0.032 across lag 12/task 0, lag 4/task 0 and lag 4 pooled, with the `--task` flag
correctly separating pure lag (1.27–1.33×) from task-pooled (1.65–1.78×) — the W5
confound reappearing exactly where predicted. γ=30 capacity pair: generic 0.4159,
retained 1.4175, ratio still climbing to 3.41, both steps resolved, no crossing.
The pairing you propose for §5.4 is right: the ρ_c main effects saturate above
γ=10 while capacity does not, so the two γ=30 results say different things about
what saturates.

---

## Tier 1 — Verification. Blocks the draft. Do first.

### 1.1 Propagation audit of the three-corner change

Every quantity in the paper that pooled over the 2×2 was computed with the sign
mixture in it. Produce a **before/after table** covering at minimum:

- Figure 2's headline magnitudes at every γ. The γ=10 figure was 50.7 floors
  pooled over four corners including a gain; removing `S-HH` should make the
  loss **larger**. Confirm the new value explicitly — it appears in the abstract
  sketch, §5.1 P1, and the width discussion.
- The width figure (47.9 / 48.2 / 47.5 floors at γ=10) and its share columns.
- The W5 lag and task-position series.
- The stratification analysis in §5.3.
- Anything in `07-writeup.md` quoting a pooled number.

Some of these may be unaffected. **None may be assumed so** — state for each
whether it changed, and by how much.

### 1.2 Fix the 4-arm width row at source

All four Δρ_c entries in the W1 N=300 row are 4-arm estimates; the 40-arm values
are +0.0551 / −0.0550 / +0.1098 / +0.0114. Either recompute the row at matched
arm count or mark every cell as 4-arm in the table, the log entry, and any
downstream use. Two wrong numbers from one row is a source problem, not two
transcription problems.

### 1.3 Sign audit across the full grid

Does any other cell show a capacity gain — at any γ, lag, task, or width? Five
minutes on stored data. This tells us whether `S-HH` backward transfer is unique
to that corner or whether gain appears elsewhere in unresolved form, which
changes how §5.1 states it.

---

## Tier 2 — Characterise the new result. Stored data only, no new runs.

### 2.1 Channel composition of the gain

Does a capacity *gain* decompose as the mirror image of a loss, or differently?
Report the three-term decomposition for `S-HH` alongside the loss corners at
matched γ. If gains and losses have different channel signatures that is a
finding for §5.1; if it mirrors, it is a clean consistency check on the identity.
Either outcome is worth a sentence.

### 2.2 Is backward transfer general or a lag-12/task-0 phenomenon?

Run the `S-HH` gain across lags and task positions using the W5 machinery. A
gain that appears only at the maximum-forgetting cell means something different
from one that holds throughout the stream.

### 2.3 The peak location — test, do not frame

The gain peaks at γ=1, and the behavioural optimum was γ ≈ 1–3, not resolvable
further. **Treat any "backward transfer peaks at the behavioural optimum" reading
as a hypothesis to test against the bootstrap CI, not as a frame to adopt.**
Three γ\*-coincidence readings have been proposed in this project and all three
were refuted; the grid resolution is one step, and two quantities landing in the
same step is weak evidence. Report the peak location with its CI and let the text
say only what the CI supports.

---

## Tier 3 — ICLR lead time. Fenced. Must not touch NeurReps artifacts.

Start now for the lead time, not because NeurReps needs it. Results go to a
separate directory; the off-design guard applies; nothing feeds a NeurReps
figure; no NeurReps number changes as a result.

- **Split-CIFAR100 pilot.** Largest ICLR item, longest lead time, and the one
  reviewer objection four pages cannot answer. P=10 changes the estimation
  regime, so the pilot's first job is establishing whether the estimator behaves
  there at all.
- **A re-run retaining activations**, so cross-module CKA is possible in
  September without a second re-run. Scope it as a small arm, not the full grid.

Still deferred, unchanged: tilted-β, depth variation, T=40, heterogeneous
C3/C4, the dichotomy-family probe. All better designed in September with
reviewer comments in hand.

---

## Tier 1.5 — The results notebook

**Build `notebooks/results.ipynb` after the Tier 1 audit and before Tier 2.**
Its purpose is interpretive: every number and figure in one place, at full
precision, with its provenance visible, so that reading the results does not
depend on reconstructing which log entry a value came from. Two of the last
three errors — the 4-arm row and the pooled `S-LL` figure — were errors of
*sourcing*, not of computation, and a notebook that shows each number beside its
source is the direct remedy.

### Requirements

- **Loads from stored results, computes nothing new.** Every number re-read from
  its result file at full precision. If a value cannot be re-derived from stored
  data, the notebook says so rather than hardcoding it.
- **Every table states its n and its arm-selection criteria** in the cell,
  visibly, not in a comment. Given the 4-arm row, this is the single most
  important requirement.
- **Every quantity reported at full precision**, with the paper-quoted rounding
  shown alongside. The inventory compares at paper precision; the notebook is
  where the full value lives.
- **Where a number was superseded, show both** with a one-line note. Specifically:
  `S-LL` +0.0246 → +0.0114 (unresolved); `S-LH` +0.092 → +0.1098; panel (d)'s
  top step +0.004 → −0.010 (both unresolved); the four-corner → three-corner
  magnitudes; cell-mean r² 0.889 → arm-level R² 0.099.
- **Provenance header per section**: result-set hash, git SHA, generating script.
- **Runs top to bottom without manual intervention**, and states its own runtime.

### Suggested structure

1. **Provenance and grid summary** — arms, completeness, identity residual,
   audit status, module hashes.
2. **Noise floors** — registered values, W4 per-γ measured values, the ~2×
   resolution limit of the 4-seed CV design, and which quantities are
   floor-denominated.
3. **Figure 2** — magnitudes, three-term decomposition, signed terms with
   resolution status, panel (d) with per-bar gates and coverage. Both four-corner
   and three-corner groupings, side by side.
4. **`S-HH` backward transfer** — the new result: magnitudes across γ, channel
   composition (Tier 2.1), lag and task dependence (Tier 2.2), peak with CI
   (Tier 2.3).
5. **Figure 3 / capacity pair** — generic and retained across γ including γ=30,
   the surplus, the decay/surplus decorrelation, per-corner breakdown.
6. **Figure 4 / the 2×2** — within-stream ρ_c, per-arm Spearman, the additive
   decomposition with interaction and its CI, γ=30 extension and the saturation
   result.
7. **Width invariance** — matched-lag and matched-sampling table, the
   radius/dimension internal split, the matched-subset diagnostic.
8. **W5 / lag and task position** — the confound, the variance decomposition
   (condition 0.814; +distance+lag 0.943; stream 0.000, seed 0.005).
9. **Stratification** — the three comparisons with CIs, the pooling reversal.
10. **Corrections ledger** — every superseded number with its replacement and
    the reason, in one table. This is the section to check the draft against.

The notebook is not a deliverable for the paper; it is the artifact we read
together to make sure the paper says what the data says.

---

## Order

1. Tier 1 (propagation audit, 4-arm row, sign audit)
2. Results notebook
3. Tier 2 (characterisation, stored data)
4. §5.5 pre-registration table
5. Tier 3 fenced ICLR arms, in the background throughout

Report after Tier 1 and again after the notebook. Do not start Tier 2 before the
propagation audit is reported — if Figure 2's headline number moved, §5.1 needs
rewriting before anything else is characterised.
