# 32 — A–B–A stopping 2×2: over-training A vs over-exposure on B (pre-commit)

**Status: pre-commit, 2026-11-09.**
- The design was decided by Kati on 2026-11-09, after docs/31.
- This document is committed **before** any code for it runs on the cluster. That includes the one timed
  arm.
- Results are added later in §7 and §8. Nothing above §7 changes after the commit, except dated amendments.

## 1. Why

The docs/31 ablation replaced matched-loss stopping with an equal, fixed budget (A1 2300, B 2400, A2 2000).
The H1 contrast d = f(10) − f(0.1) in fraction removed **shrank** at h 4 frozen (+0.332 → +0.100, unresolved)
and at h 4 drift (+0.333 → +0.184). This went against the expected direction.

Both learners moved: f(0.1) rose and f(10) fell. The fixed budget changes two things at once:
- **Over-training of A.** At γ0 10, A is trained to a loss of about 1e-6 instead of 0.05, and the gain after
  A grows (2.294 vs 2.059). At γ0 0.1, A1 runs 2300 updates instead of up to 2266.
- **Over-exposure on B.** B runs 2400 updates instead of about 1500 (γ0 0.1) or about 21 (γ0 10).

docs/31 cannot separate the two. This 2×2 crosses them.

## 2. Design (fixed)

Identical to the docs/28 replication (job 4396) and the docs/31 ablation (job 4458), except for the
per-phase stopping rule.

| item | value |
|---|---|
| factor A | the A phase: **Cr** = criterion (loss ≤ 0.05, checked before each update) or **Fx** = fixed 2300 updates |
| factor B | the B phase: **Cr** = criterion or **Fx** = fixed 2400 updates |
| followers | A2 follows the A rule (Fx: 2000 updates, as in docs/31). Fresh B follows the B rule. Neither is read here. |
| cells | (Cr, Cr), (Fx, Cr), (Cr, Fx), (Fx, Fx), written as (A rule, B rule) |
| strata | h 4 frozen (s_f 1.0) and h 4 drift (s_f 0.9), each at γ0 0.1 and γ0 10: **4 strata** |
| seeds | 8–15, paired across the cells and γ0 (the same `seed_base` draws as jobs 4396 and 4458) |
| arms | 4 cells × 4 strata × 8 seeds = **128**, one commit, one job |
| `ABAConfig` | `stopping = "fixed"`, with `fixed_steps = (A1, B, A2)`, and `None` = criterion for that phase: (Fx, Cr) = (2300, None, 2000); (Cr, Fx) = (None, 2400, None). (Cr, Cr) is `stopping = "criterion"`; (Fx, Fx) is (2300, 2400, 2000). |
| usable arm | every phase is converged (Cr) or ran its full budget with a finite loss (Fx) |

**Integrity check (pre-committed).** The code change (§6) must leave the two existing cells unchanged:
- every (Cr, Cr) row must equal its job 4396 row (`results/aba/FULL_4e110835b8fc2863.jsonl`);
- every (Fx, Fx) row must equal its job 4458 row (`results/aba/FULL_0ee97f82b6bee621_fixed.jsonl`).

Each comparison is bit for bit after JSON, ignoring only `lib_hash`, `stamp` and `timing`. On any difference
no reading is computed, and the difference is reported to Kati.

**[Clarified 2026-11-09, before any cluster run]** The check runs on grime, where the reference rows were
computed. On the laptop the same arm differs from the grime row in the last digits: for example
`backward.A1_to_B.total` −1.2876994931091996 vs −1.2876994931092003 (cross-platform BLAS). On the laptop,
the old and new `aba_lib` gave identical rows for a QUICK criterion arm and a QUICK fixed arm, so the change
itself leaves those paths untouched.

## 3. Quantities

Per arm, from the A–B–A row:
- the gain after A, G = Δlog α_A(init → A1) = `backward.init_to_A1.total`;
- the loss on B, Δlog α_A(A1 → B) = `backward.A1_to_B.total`;
- the **fraction removed**, f = −Δlog α_A(A1 → B) / G. This is docs/28 H1's quantity
  (`aba_summary.frac_removed`).

Per seed s and stratum (h, s_f, γ0), with f(a, b) the fraction removed in cell (A rule a, B rule b):
- **A over-training (main effect):** E_A = ½[(f(Fx, Cr) − f(Cr, Cr)) + (f(Fx, Fx) − f(Cr, Fx))];
- **B over-exposure (main effect):** E_B = ½[(f(Cr, Fx) − f(Cr, Cr)) + (f(Fx, Fx) − f(Fx, Cr))].

Both are paired contrasts per seed (n = 8) within one stratum. They are never pooled across strata or γ0.

## 4. Inference and reading rule

- **Family.** 2 effects × 4 strata = **8 tests**, at Bonferroni **1 − 0.05/8 = 99.375%**. The intervals are
  two-sided t-intervals over the 8 seeds, df 7, with the critical value computed in code (≈ 3.855).
- **Reading.** Kati's four categories (docs/29 S0), SESOI **0.10** in fraction removed:
  - **meaningful**: the whole interval lies beyond ±0.10 in the predicted direction;
  - **nonzero, possibly trivial**: the interval excludes 0 but overlaps ±0.10;
  - **equivalent**: the whole interval lies inside ±0.10;
  - **inconclusive**: otherwise.
  - A test without a committed direction (§5) is read **two-sided**: "meaningful +" or "meaningful −" when
    the whole interval lies beyond +0.10 or below −0.10. The other three categories are as above.
- **Ratio not read.** f is read in a stratum only if G is meaningful (> 0.05 in log α, the docs/29 SESOI)
  in **every** cell of that stratum. Otherwise the stratum's two tests are "not read".
- No bootstrap. A sign count (seeds with E < 0) is reported beside each interval.

## 5. Predictions (stated before running)

**Committed prediction (Kati).** Over-training A **reduces** the fraction removed at γ0 10, through
consolidation:
- **E_A < 0 at γ0 10**, at h 4 frozen and at h 4 drift.
- **Held** if E_A at γ0 10 is meaningful (< −0.10 for the whole interval) in both h 4 strata.
- **Not held** if it is meaningful the opposite way (> +0.10) in either.
- Anything else is reported per stratum by category.

**Null.** E_A at γ0 10 is **equivalent** (inside ±0.10) in both strata. In that case over-training of A does
not account for the docs/31 shrinkage of f(10), and the change in f(10) belongs to exposure on B.

**Not predicted (read two-sided):**
- **E_A at γ0 0.1.** The lazy learner is barely over-trained: A1 runs 2300 updates against a criterion
  maximum of 2266, and its final loss stays near 0.04. So near equivalence is expected, but not committed.
- **E_B at both γ0.** More updates on B could remove more (E_B > 0), especially at γ0 10, where B goes from
  about 21 to 2400 updates. No direction is committed.

**What explains the docs/31 shrinkage (fixed in advance):**
- **Consolidation account.** E_A at γ0 10 is meaningful < 0, and E_B at γ0 10 is not meaningful < 0.
- **Exposure account.** E_A at γ0 10 is equivalent, and E_B at γ0 10 is meaningful < 0 or E_B at γ0 0.1 is
  meaningful > 0.
- **Any other pattern** is reported per stratum as it is, without a summary across strata.

## 5b. Descriptive (not read)

- f, G and Δlog α_A(A1 → B) per cell and stratum, as a mean with its 99.375% interval;
- the interaction, ((Fx, Fx) − (Fx, Cr)) − ((Cr, Fx) − (Cr, Cr)), per stratum;
- the four simple effects per stratum;
- d = f(10) − f(0.1) per cell and h 4 stratum (the H1 contrast under each protocol);
- the setting card per cell: the median [min, max] of steps and of the final loss per phase.

## 6. Execution

**Code changes,** committed after this document and before any cluster run:
- **`aba_lib.train_to_criterion`.** A `fixed_steps` entry of `None` means criterion stopping for that
  phase. No other behaviour changes. `key_fields` stores `None` as JSON null.
- **`aba_cluster.py`.**
  - `--budget` accepts `-` for `None` (e.g. `2300,-,2000`).
  - The `cells` mode runs the 128 arms, one cell after another, each cell into its checkpoint
    (`FULL_<hash>.jsonl` for (Cr, Cr), `FULL_<hash>_fixed.jsonl` for the rest, separated by the config
    dict).
  - The `match` mode runs the §2 integrity check.
- **`aba_summary.py --twobytwo`.** It computes §3–§5b and writes §7 between markers.
- **Tests (toy scale).**
  - A `None` entry gives criterion stopping in that phase only.
  - The all-criterion and all-fixed configs give the same rows as before the change.

**Steps:**
1. **Timing (`sbatch/aba_2x2_timing.sh`).**
   - First, `match` on one (Cr, Cr) arm (seed 8, γ0 10, h 4 drift) against job 4396, and on one (Fx, Fx) arm
     (seed 8, γ0 0.1, h 4 frozen) against job 4458.
   - Then ONE (Fx, Cr) arm (seed 8, γ0 0.1, h 4 frozen), sequential and timed. That is the new code path,
     with the lazy learner's longest training.
2. **Grid (`sbatch/aba_2x2.sh`),** sized from step 1 with the measured 4× parallel slowdown. It is submitted
   only after Kati's go.
   - It runs the 128 arms, then `match` on every (Cr, Cr) and (Fx, Fx) row.
   - Readings are computed locally with `aba_summary.py --twobytwo`.
3. The job id, the commit and the wall time are recorded in §7, and a LOG entry is made.

**Relation to the stream spec.** This change touches `aba_lib`, so it changes `stream_hash` (docs/29 §8).
The stream timing job and the stream grid run on the same commit, at or after this code commit.

## 7. Results

<!-- BEGIN GENERATED: 2x2 results -->
(not yet run)
<!-- END GENERATED: 2x2 results -->

## 8. Interpretation

(after §7, per stratum, against §5)
