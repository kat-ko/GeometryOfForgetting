# 32 — A–B–A stopping 2×2: over-training A vs over-exposure on B (pre-commit)

**Status: pre-commit, 2026-11-09. Results 2026-11-10 (§7, §8): grime job 4538.**
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

**Step 1 done (grime job 4533, commit `3a17d7d`, lib 635c4e3bf8cc6b96, 2026-11-09).**
- **Integrity check.** (Cr,Cr) seed 8, γ0 10, h 4 drift is bit-identical to job 4396. (Fx,Fx) seed 8, γ0 0.1,
  h 4 frozen is bit-identical to job 4458.
- **Timing.** The (Fx,Cr) arm (seed 8, γ0 0.1, h 4 frozen) took 149 s sequential: train 87 s, GLUE 60 s,
  peak RSS 124 MB.
  - Status: A1 fixed, B converged, A2 fixed, fresh B converged.
  - Steps: A1 2300, B 1238, A2 2000, fresh B 1672.
  - Final loss: A1 0.036, B 0.050.
  - (Cr,Cr) took 57 s and (Fx,Fx) 178 s.
- **Projection** for the 128 arms on 32 workers: about 40 min.

## 7. Results

<!-- BEGIN GENERATED: 2x2 results -->
Cells (A rule, B rule): Cr = criterion, Fx = fixed (A1 2300, B 2400; A2 2000 with Fx A). h 4, seeds 8–15, n = 8 per stratum. Intervals: 99.375% t, critical value 3.855 (df 7). Category: four categories, SESOI 0.10 (fraction removed), predicted direction only for E_A at γ0 10 (< 0); all other tests two-sided.

**Main effects (the family of eight).**

| stratum | effect | mean [99.375%] | seeds < 0 | ratio read | category |
|---|---|---|---|---|---|
| h 4, s_f 1 (frozen), γ0 0.1 | E_A (over-training A) | +0.021 [+0.004, +0.039] | 0/8 | yes | equivalent |
| h 4, s_f 1 (frozen), γ0 0.1 | E_B (over-exposure B) | +0.071 [+0.031, +0.112] | 0/8 | yes | nonzero, possibly trivial |
| h 4, s_f 1 (frozen), γ0 10 | E_A (over-training A) | -0.142 [-0.206, -0.078] | 8/8 | yes | nonzero, possibly trivial |
| h 4, s_f 1 (frozen), γ0 10 | E_B (over-exposure B) | +0.003 [-0.047, +0.053] | 5/8 | yes | equivalent |
| h 4, s_f 0.9 (drift), γ0 0.1 | E_A (over-training A) | +0.022 [+0.004, +0.041] | 0/8 | yes | equivalent |
| h 4, s_f 0.9 (drift), γ0 0.1 | E_B (over-exposure B) | +0.064 [+0.026, +0.102] | 0/8 | yes | nonzero, possibly trivial |
| h 4, s_f 0.9 (drift), γ0 10 | E_A (over-training A) | -0.083 [-0.130, -0.036] | 8/8 | yes | nonzero, possibly trivial |
| h 4, s_f 0.9 (drift), γ0 10 | E_B (over-exposure B) | +0.020 [-0.074, +0.114] | 5/8 | yes | inconclusive |

**Committed prediction (docs/32 §5):** E_A < 0 at γ0 10 in both h 4 strata.
Reading: neither held, not held nor null: frozen nonzero, possibly trivial; drift nonzero, possibly trivial.

**Accounts (docs/32 §5), per h 4 stratum:**

- s_f 1 (frozen): E_A(10) nonzero, possibly trivial; E_B(10) equivalent; E_B(0.1) nonzero, possibly trivial → other pattern (reported as is).
- s_f 0.9 (drift): E_A(10) nonzero, possibly trivial; E_B(10) inconclusive; E_B(0.1) nonzero, possibly trivial → other pattern (reported as is).

**Per cell (descriptive).** f = fraction removed; G = Δlog α_A(init → A1); L = Δlog α_A(A1 → B). Mean [99.375%].

| stratum | cell | f | G | L |
|---|---|---|---|---|
| h 4, s_f 1, γ0 0.1 | (Cr,Cr) | +0.387 [+0.267, +0.506] | +0.254 [+0.213, +0.296] | -0.097 [-0.119, -0.075] |
| h 4, s_f 1, γ0 0.1 | (Fx,Cr) | +0.407 [+0.282, +0.532] | +0.280 [+0.253, +0.307] | -0.113 [-0.142, -0.083] |
| h 4, s_f 1, γ0 0.1 | (Cr,Fx) | +0.457 [+0.310, +0.604] | +0.254 [+0.213, +0.296] | -0.115 [-0.145, -0.084] |
| h 4, s_f 1, γ0 0.1 | (Fx,Fx) | +0.480 [+0.330, +0.629] | +0.280 [+0.253, +0.307] | -0.134 [-0.171, -0.096] |
| h 4, s_f 1, γ0 10 | (Cr,Cr) | +0.719 [+0.675, +0.764] | +2.059 [+1.954, +2.164] | -1.481 [-1.581, -1.380] |
| h 4, s_f 1, γ0 10 | (Fx,Cr) | +0.568 [+0.497, +0.639] | +2.294 [+2.225, +2.364] | -1.304 [-1.474, -1.133] |
| h 4, s_f 1, γ0 10 | (Cr,Fx) | +0.713 [+0.670, +0.756] | +2.059 [+1.954, +2.164] | -1.468 [-1.568, -1.368] |
| h 4, s_f 1, γ0 10 | (Fx,Fx) | +0.580 [+0.486, +0.675] | +2.294 [+2.225, +2.364] | -1.332 [-1.561, -1.103] |
| h 4, s_f 0.9, γ0 0.1 | (Cr,Cr) | +0.264 [+0.170, +0.357] | +0.254 [+0.213, +0.296] | -0.066 [-0.084, -0.048] |
| h 4, s_f 0.9, γ0 0.1 | (Fx,Cr) | +0.284 [+0.185, +0.383] | +0.280 [+0.253, +0.307] | -0.079 [-0.102, -0.055] |
| h 4, s_f 0.9, γ0 0.1 | (Cr,Fx) | +0.325 [+0.207, +0.444] | +0.254 [+0.213, +0.296] | -0.082 [-0.107, -0.057] |
| h 4, s_f 0.9, γ0 0.1 | (Fx,Fx) | +0.350 [+0.229, +0.471] | +0.280 [+0.253, +0.307] | -0.097 [-0.128, -0.067] |
| h 4, s_f 0.9, γ0 10 | (Cr,Cr) | +0.597 [+0.534, +0.660] | +2.059 [+1.954, +2.164] | -1.231 [-1.393, -1.068] |
| h 4, s_f 0.9, γ0 10 | (Fx,Cr) | +0.500 [+0.434, +0.565] | +2.294 [+2.225, +2.364] | -1.146 [-1.292, -1.000] |
| h 4, s_f 0.9, γ0 10 | (Cr,Fx) | +0.603 [+0.549, +0.657] | +2.059 [+1.954, +2.164] | -1.242 [-1.367, -1.117] |
| h 4, s_f 0.9, γ0 10 | (Fx,Fx) | +0.534 [+0.479, +0.589] | +2.294 [+2.225, +2.364] | -1.226 [-1.373, -1.079] |

**Interaction and simple effects (descriptive).** Interaction = ((Fx,Fx) − (Fx,Cr)) − ((Cr,Fx) − (Cr,Cr)).

| stratum | A at B Cr | A at B Fx | B at A Cr | B at A Fx | interaction |
|---|---|---|---|---|---|
| h 4, s_f 1, γ0 0.1 | +0.020 [+0.003, +0.037] | +0.023 [+0.004, +0.042] | +0.070 [+0.028, +0.112] | +0.073 [+0.035, +0.111] | +0.003 [-0.002, +0.008] |
| h 4, s_f 1, γ0 10 | -0.151 [-0.211, -0.092] | -0.133 [-0.202, -0.063] | -0.006 [-0.055, +0.042] | +0.012 [-0.041, +0.065] | +0.018 [-0.002, +0.038] |
| h 4, s_f 0.9, γ0 0.1 | +0.020 [+0.003, +0.037] | +0.025 [+0.005, +0.045] | +0.062 [+0.023, +0.101] | +0.066 [+0.030, +0.103] | +0.005 [+0.000, +0.009] |
| h 4, s_f 0.9, γ0 10 | -0.097 [-0.136, -0.058] | -0.069 [-0.135, -0.003] | +0.006 [-0.085, +0.097] | +0.034 [-0.070, +0.138] | +0.028 [-0.027, +0.083] |

**H1 contrast per cell (descriptive):** d = f(γ0 10) − f(γ0 0.1).

| s_f | (Cr,Cr) | (Fx,Cr) | (Cr,Fx) | (Fx,Fx) |
|---|---|---|---|---|
| 1 (frozen) | +0.332 [+0.223, +0.442] | +0.161 [+0.004, +0.319] | +0.256 [+0.122, +0.390] | +0.100 [-0.087, +0.288] |
| 0.9 (drift) | +0.333 [+0.227, +0.440] | +0.216 [+0.113, +0.319] | +0.278 [+0.167, +0.389] | +0.184 [+0.051, +0.318] |

**Setting card per cell:** steps and final loss per phase, median [min, max] over seeds. Final losses are recorded only in the cells with a fixed phase (`behaviour.final_loss`).

| stratum | cell | steps A1 | steps B | final loss A1 | final loss B |
|---|---|---|---|---|---|
| h 4, s_f 1, γ0 0.1 | (Cr,Cr) | 1938 [1761, 2266] | 1494 [1208, 1672] | ≤ 0.05 (criterion) | ≤ 0.05 (criterion) |
| h 4, s_f 1, γ0 0.1 | (Fx,Cr) | 2300 [2300, 2300] | 1465 [1207, 1659] | 3.7e-02 [3.0e-02, 4.9e-02] | 5.0e-02 [5.0e-02, 5.0e-02] |
| h 4, s_f 1, γ0 0.1 | (Cr,Fx) | 1938 [1761, 2266] | 2400 [2400, 2400] | 5.0e-02 [5.0e-02, 5.0e-02] | 2.4e-02 [1.7e-02, 2.9e-02] |
| h 4, s_f 1, γ0 0.1 | (Fx,Fx) | 2300 [2300, 2300] | 2400 [2400, 2400] | 3.7e-02 [3.0e-02, 4.9e-02] | 2.3e-02 [1.7e-02, 2.7e-02] |
| h 4, s_f 1, γ0 10 | (Cr,Cr) | 51 [47, 54] | 22 [15, 28] | ≤ 0.05 (criterion) | ≤ 0.05 (criterion) |
| h 4, s_f 1, γ0 10 | (Fx,Cr) | 2300 [2300, 2300] | 18 [14, 26] | 6.5e-06 [5.5e-06, 7.2e-06] | 4.9e-02 [4.5e-02, 5.0e-02] |
| h 4, s_f 1, γ0 10 | (Cr,Fx) | 51 [47, 54] | 2400 [2400, 2400] | 4.7e-02 [4.4e-02, 5.0e-02] | 6.1e-06 [5.3e-06, 6.6e-06] |
| h 4, s_f 1, γ0 10 | (Fx,Fx) | 2300 [2300, 2300] | 2400 [2400, 2400] | 6.5e-06 [5.5e-06, 7.2e-06] | 5.4e-06 [5.1e-06, 6.5e-06] |
| h 4, s_f 0.9, γ0 0.1 | (Cr,Cr) | 1938 [1761, 2266] | 1410 [1178, 1675] | ≤ 0.05 (criterion) | ≤ 0.05 (criterion) |
| h 4, s_f 0.9, γ0 0.1 | (Fx,Cr) | 2300 [2300, 2300] | 1398 [1175, 1632] | 3.7e-02 [3.0e-02, 4.9e-02] | 5.0e-02 [5.0e-02, 5.0e-02] |
| h 4, s_f 0.9, γ0 0.1 | (Cr,Fx) | 1938 [1761, 2266] | 2400 [2400, 2400] | 5.0e-02 [5.0e-02, 5.0e-02] | 2.1e-02 [2.0e-02, 3.0e-02] |
| h 4, s_f 0.9, γ0 0.1 | (Fx,Fx) | 2300 [2300, 2300] | 2400 [2400, 2400] | 3.7e-02 [3.0e-02, 4.9e-02] | 2.1e-02 [1.9e-02, 2.8e-02] |
| h 4, s_f 0.9, γ0 10 | (Cr,Cr) | 51 [47, 54] | 20 [14, 26] | ≤ 0.05 (criterion) | ≤ 0.05 (criterion) |
| h 4, s_f 0.9, γ0 10 | (Fx,Cr) | 2300 [2300, 2300] | 16 [10, 20] | 6.5e-06 [5.5e-06, 7.2e-06] | 4.7e-02 [4.6e-02, 5.0e-02] |
| h 4, s_f 0.9, γ0 10 | (Cr,Fx) | 51 [47, 54] | 2400 [2400, 2400] | 4.7e-02 [4.4e-02, 5.0e-02] | 6.4e-06 [5.5e-06, 7.3e-06] |
| h 4, s_f 0.9, γ0 10 | (Fx,Fx) | 2300 [2300, 2300] | 2400 [2400, 2400] | 6.5e-06 [5.5e-06, 7.2e-06] | 6.3e-06 [5.7e-06, 7.0e-06] |
<!-- END GENERATED: 2x2 results -->

## 8. Interpretation

**Run.** Grime job 4538, commit `27af9bf`, lib 635c4e3bf8cc6b96. 128/128 arms usable in 2223 s on 32 workers.
Every phase stopped by its cell's rule in every row, including the (Cr,Fx) order:

| cell | A1, A2 | B, fresh B |
|---|---|---|
| (Cr,Cr) | converged | converged |
| (Fx,Cr) | fixed | converged |
| (Cr,Fx) | converged | fixed |
| (Fx,Fx) | fixed | fixed |

**Integrity (§2), checked before any reading.** All 64 reference rows are bit-identical (ignoring
lib_hash, stamp and timing):
- every (Cr,Cr) row against job 4396 (32 rows);
- every (Fx,Fx) row against job 4458 (32 rows).

**Ratio gate.** G is meaningful in every cell of every stratum, so every test is read.

**The committed prediction (E_A < 0 at γ0 10 in both h 4 strata) is neither held, not held, nor null.**

| h 4 stratum | E_A at γ0 10 | seeds < 0 | category |
|---|---|---|---|
| frozen | −0.142 [−0.206, −0.078] | 8/8 | nonzero, possibly trivial |
| drift | −0.083 [−0.130, −0.036] | 8/8 | nonzero, possibly trivial |

- Both intervals exclude 0 on the predicted side, and no seed goes the other way.
- Neither lies wholly beyond the SESOI of 0.10, so by §4 neither is "meaningful".
- By the rule, the prediction is **not held as stated**: the direction is supported and the size is not
  established. Neither "held" nor "null" applies.

**Accounts (§5).** Both strata are "other pattern", because the consolidation account needs E_A(10) to be
meaningful. Reported as it is, per stratum.

| stratum | E_A | E_B | A rule | B rule |
|---|---|---|---|---|
| γ0 10, frozen | −0.142, nonzero, possibly trivial | +0.003 [−0.047, +0.053], equivalent | moves f(10) | does not |
| γ0 10, drift | −0.083, nonzero, possibly trivial | +0.020 [−0.074, +0.114], inconclusive | moves f(10) | unresolved |
| γ0 0.1, frozen | +0.021, equivalent | +0.071 [+0.031, +0.112], nonzero, possibly trivial | does not move f(0.1) | moves f(0.1) up |
| γ0 0.1, drift | +0.022, equivalent | +0.064 [+0.026, +0.102], nonzero, possibly trivial | does not move f(0.1) | moves f(0.1) up |

- **γ0 10, frozen.** Over-training A lowered f(10). Over-exposure on B, about 21 → 2400 updates, did not
  move f(10).
- **γ0 10, drift.** E_A is the same sign as at frozen and smaller. E_B is unresolved.
- **γ0 0.1, both strata.** Over-training A did not move f(0.1). Over-exposure on B, about 1450 → 2400
  updates, raised f(0.1).

**What this says about docs/31 (descriptive).** The docs/31 shrinkage of the H1 contrast splits by learner,
in both strata:
- the fall in f(10) goes with over-training A;
- the rise in f(0.1) goes with over-exposure on B.

The interaction is small in every stratum, from +0.003 to +0.028. By the pre-committed criteria neither
effect reaches the SESOI in a confirmatory test, so this split is a description, not a confirmed account.

The H1 contrast d (descriptive) stays positive in every cell at h 4 drift:

| cell | d at h 4 drift |
|---|---|
| (Cr,Cr) | +0.333 |
| (Fx,Cr) | +0.216 |
| (Cr,Fx) | +0.278 |
| (Fx,Fx) | +0.184 |

At h 4 frozen, d's interval includes 0 only in (Fx,Fx).
