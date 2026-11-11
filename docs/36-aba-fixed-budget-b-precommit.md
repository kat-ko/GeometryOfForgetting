# 36 — A–B–A similarity map: fixed budget on phase B per γ0, extreme cases (pre-commit)

**Status: pre-commit, 2026-11-11. EXPLORATORY (docs/34). Results go in §7 and §8.**
- Design decided by Kati on 2026-11-11, after docs/35. It is docs/34 Phase 1b in the form Kati asked for
  (docs/34 §5 amendment).
- Paper backlog: B1, "extreme cases, run 1". The full-grid version is for the final appendix.
- This document is committed **before** any code for it runs on the cluster, the timed arm included. Nothing
  above §7 changes after the commit, except dated amendments.

## 1. Why

Phase 1 of the map (job 4651, docs/35) uses matched-loss stopping. Under that rule a similar B needs fewer
updates, so **within one γ0 every similarity effect also contains an exposure effect.** Two examples from
docs/35 §3:
- **H12 forgets much more than H4 at s_f 1.**
  - f is 0.70 / 0.82 / 0.79 at H12 against 0.35 / 0.60 / 0.70 at H4 (γ0 0.1 / 1 / 10).
  - H12 also gets about twice the updates on B: medians 2382 vs 1422, 356 vs 197, 44 vs 22.
  - H4 and H12 are **independent label draws for each seed** (`seed_base` draws y_B separately for each H).
    In distribution, an H12 task is an H4 task with a flipped readout, because capacity does not depend on
    the sign of the labels. Seed by seed it is not.
  - Exposure is one candidate explanation of the gap.
- **The H0 input axis.** Updates on B fall from about 1466 (s_f 0.1) to 0 (s_f 1) at γ0 0.1, while f changes
  sign.

A fixed budget **per γ0** on phase B gives every cell the same number of updates on B within each γ0. The
γ0 comparison stays at matched fit on A.
- A1 and A2 stay at criterion. This keeps out the over-training of A that docs/31–32 showed drives the size
  of the γ0 effect (paper-backlog B2).
- Fresh B gets the same budget as B, so the two learners stay comparable.

## 2. Design (fixed)

Identical to docs/34 Phase 1 (job 4651: 1M, N = 300, μP, the map config, seeds 16–23), except for the
stopping rule of phase B.

| item | value |
|---|---|
| phase B and fresh B | **fixed budget per γ0**: exactly b(γ0) updates, whatever the loss |
| A1 and A2 | criterion (loss ≤ 0.05, checked before each update), as in job 4651 |
| budget rule | the largest B or fresh-B criterion step count over all 771 job-4651 arms at that γ0, rounded up to the next multiple of 100 |
| b(γ0) | **γ0 0.1: 3200** (max 3132: seed 19, H16, s_f 1); **γ0 1: 600** (max 552); **γ0 10: 100** (max 99) |
| cells (11) | s_f 1 × H {0, 2, 4, 8, 12, 16}; H0 × s_f {0.9, 0.4, 0.1}; s_f 0.1 × H {4, 8} |
| seeds | 16–23, paired with the job 4651 rows of the same seed, γ0 and cell |
| arms | 11 × 3 γ0 × 8 = **264**, one commit, one job, on grime; no fresh-noise arms |
| `ABAConfig` | `map_config(FULL)` with `stopping = "fixed"`, `fixed_steps = (None, b(γ0), None)`, `gammas = (γ0,)`: one config per γ0. The H set stays the map's, so `probe_hammings` stays in the key, and the seeds' y*, neutral probes and y_B are those of job 4651. |
| usable arm | A1 and A2 converged; B and fresh B ran their full budget with a finite loss and reached the criterion within it (`first_hit` not None) |

**New logging: `first_hit`.** Every phase records the first step at which the loss is ≤ 0.05. It is checked
before each update, as the criterion is, so it is 0 if the phase starts at the criterion.
- In criterion phases, `first_hit` equals the step count.
- Fixed rows also store `FT_beh_hit = log((1 + first_hit freshB)/(1 + first_hit B))`. This keeps FT_beh
  defined. The step-count FT_beh of a fixed row is log(1) = 0 by construction, and is not read.
- Savings, log((1 + steps A1)/(1 + steps A2)), stays defined, because A1 and A2 are at criterion.

**The selected cells.**
- **The H sweep at s_f 1, with H0 as the identity cell.** H0 × s_f 1 gives pure extra training on the task
  just learned, the zero point of the exposure effect. The sweep includes the H4/H12 pair.
- **The H0 input axis at s_f 0.9, 0.4 and 0.1.** This is where exposure varies most.
- **The low-similarity corner at H4 and H8 × s_f 0.1.** These are the docs/28 jump cells.

**Integrity checks (pre-committed; on grime, where job 4651 ran).** By determinism, A1 and the start of B are
the same computation as in job 4651:
1. **A1 identical.** In every row, `behaviour.steps.A1`, `geometry.A@A1` and `backward.init_to_A1` are bit
   for bit equal to the job 4651 row, after JSON.
2. **First hit = job 4651 steps.** For every arm, `first_hit.B` equals the job 4651 `steps.B`, and
   `first_hit.freshB` equals the job 4651 `steps.freshB`.
3. **All usable.** 264/264 rows are usable, with one commit (clean tree) and one lib hash.

Before the grid, the timing job also reproduces two arms bit for bit under the new hash:
- the docs/28 arm (job 4349: seed 0, γ0 10, H8, s_f 0.1);
- a job 4651 arm (seed 16, γ0 0.1, H16, s_f 1).

Added fields are listed and not compared. **If any check fails, no reading is computed and the failure goes
to Kati.**

## 3. Quantities

Per arm, as in docs/35 (`aba_map_analysis`):
- the gain after A, G = Δlog α_A(init → A1);
- the change on B, Δlog α_A(A1 → B);
- the fraction removed, f = −Δlog α_A(A1 → B) / G.

**What this arm changes, by design.**
- **G and every forward quantity are identical to job 4651.** A1 is the same computation. FT_geo is measured
  after A1, and `FT_beh_hit` equals the job 4651 FT_beh (integrity check 2).
- **This arm changes backward quantities only:** f, Δlog α_A(A1 → B), the channel shares, bottleneck
  forgetting, A2 and restoration ρ.
- f differs from job 4651 only through Δlog α_A(A1 → B).

Paired contrasts per seed: X_Fx − X_Cr, where Cr is the job 4651 row of the same seed, γ0 and cell, and Fx is
the new row. Contrasts are never pooled across γ0.

## 4. Inference and reading rule

- **Intervals.** 95% two-sided t-intervals over the 8 seeds, df 7, paired by seed. There is no family:
  docs/34 is exploratory.
- **Disjoint clauses.** Each rule's "held" and "not held" clauses cannot both apply. This avoids the E4
  overlap in docs/35.
- **Combining across γ0.** **Held overall** if held at every γ0. **Not held overall** if not held at any γ0.
  Otherwise unresolved overall, with the per-γ0 readings reported as they are.
- **No bootstrap.** A sign count over seeds is reported beside each interval.

## 5. Predictions (stated before running)

**(i) The H4–H12 gap in f shrinks under equal exposure on B.**
- **Quantity.** Per seed s and γ0, g(s) = f(H12, s_f 1) − f(H4, s_f 1). H4 and H12 are independent draws
  within a seed, so this is a **distributional comparison paired by seed**, not a sign-flip identity. Then
  Δg(s) = g_Fx(s) − g_Cr(s).
- **Held** if Δg's 95% interval is wholly below 0.
- **Not held** if it is wholly above 0 (the gap grows).
- **Unresolved** if it contains 0.
- **Null:** Δg = 0, so equal exposure on B does not change the gap, and exposure does not explain it.
- **Descriptive:** g_Fx with its interval, and the share of the gap closed, 1 − mean g_Fx / mean g_Cr.

**(ii) Shape survives.** Per γ0, over the 11 cells.
- **The 12 adjacent contrasts:**
  - along H at s_f 1: H0→2, 2→4, 4→8, 8→12, 12→16;
  - along s_f at H0: 1→0.9, 0.9→0.4, 0.4→0.1;
  - s_f 1→0.1 at H4 and at H8;
  - along H at s_f 0.1: H0→4, 4→8.
- **Classifying a contrast.** Each contrast is the per-seed difference in f.
  - **Resolved under Cr** if its paired Cr interval excludes 0. Its sign is σ.
  - **Reversed** if its Fx interval lies wholly on the −σ side of 0.
  - **Kept** if its Fx interval lies wholly on the σ side.
  - Contrasts not resolved under Cr are not read.
- **Rank order.** Spearman ρ_s between the 11 cell means of f under Cr and under Fx. Whether the order is
  exactly identical is reported descriptively.
- **Held** at a γ0 if every resolved contrast is kept and ρ_s ≥ 0.9.
- **Not held** if any resolved contrast is reversed, or ρ_s < 0.5.
- **Unresolved** otherwise.
- **Null:** at least one resolved adjacent contrast reverses, so the shape of the map is an exposure artefact.

**(iii) The H0-drift gain is not reversed by extra exposure.**
- **Setting.** At H0 × s_f 0.9, B adds capacity to A under Cr: Δlog α_A(A1 → B) = +0.04 / +0.07 / +0.06.
  Under Fx, B gets 3200 / 600 / 100 updates instead of medians of 398 / 36 / 4.
- **Held** if the Fx interval of Δlog α_A(A1 → B) lies wholly at or above 0.
- **Not held** if it lies wholly below 0.
- **Unresolved** otherwise.
- **Null:** Δlog α_A(A1 → B) < 0, so extra exposure turns the gain into a loss.

## 5b. Descriptive (not read)

Per cell and γ0, Cr beside Fx with the per-seed difference:
- **the docs/35 quantities:** f, Δlog α_A(A1 → B), the channel shares (utility, radius, dimension),
  bottleneck forgetting, the network's accuracy on A at the end of B, Δρ_c, restoration ρ and savings;
- **measured exposure:** the weight change during B, r_B = `richness.r_t.B`, and the kernel change
  ΔK_u(A1 → B) = `richness.B.dK_u_step`. Equal updates are not equal movement once a learner is past the
  criterion;
- **the setting card:** steps and `first_hit` per phase, and the final loss of B and fresh B;
- **the identity cell H0 × s_f 1:** pure extra training on A itself, the zero point of the exposure effect.

**Example plot for the appendix (paper-backlog B1).** f along the H sweep at s_f 1, Cr against Fx, one column
per γ0. Below it, one row with updates on B and one row with r_B, Cr against Fx.

## 6. Execution

**Code changes,** committed after this document and before any cluster run:
- **`aba_lib`.** The lib hash changes.
  - `train_to_criterion` returns `first_hit`.
  - `run_arm` stores `behaviour.first_hit` and, for fixed rows, `behaviour.FT_beh_hit`.
  - `PHASE1B_CELLS`, `PHASE1B_BUDGET`, `phase1b_cfg` and `phase1b_specs` are added.
  - Nothing else changes; `SCHEMA` is unchanged.
- **`aba_cluster.py`.**
  - `map1b` runs the 264 arms.
  - `reproduce --map` checks the job 4651 arm.
  - `check1b` runs integrity checks 1–3.
- **Analysis.** `aba_fixedb_analysis.py` (load only) and the notebook `aba_fixed_budget_b.ipynb`, with these
  predictions and rules written in before submission.
- **The Phase 1 notebook** is pinned to `FULL_e3011fa49aca1994.jsonl`, so the hash change does not orphan it.
- **Tests (toy scale).**
  - `first_hit` equals the step count under criterion.
  - Under `(None, b, None)`, A1 equals the criterion arm, and `first_hit.B` equals the criterion arm's
    `steps.B`.
  - Fresh B is fixed.
  - `phase1b_specs` gives the right counts and budgets.

**Steps:**
1. **Timing** (`sbatch/aba_fixedb_timing.sh`, grime). `reproduce` (docs/28 arm), `reproduce --map` (job 4651
   arm), then ONE fixed-B arm (seed 19, γ0 0.1, H16, s_f 1; B and fresh B 3200 each), sequential and timed. It
   must show `first_hit.B` = 3132.
2. **Grid** (`sbatch/aba_map_phase1b.sh`, grime). It is sized from step 1 with the measured ×4–5 parallel
   slowdown. It runs `map1b` and then `check1b`. Readings are computed locally from the notebook.
3. The job id, the commit and the wall time are recorded in §7, plus a LOG entry and the paper-backlog B1
   line.

**Relation to the stream spec.** This changes `aba_lib` and therefore the stream hash (docs/29 §8). The stream
is parked (docs/34), so nothing else is affected.

## 7. Results

(after the run)

## 8. Interpretation

(after the run; labelled as interpretation)
