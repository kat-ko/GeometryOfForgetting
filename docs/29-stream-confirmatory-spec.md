# 29 — Confirmatory-test specification for the 1M stream grid (DRAFT)

```
Status:  DRAFT, unsigned. Drafted by the agent 2026-11-09 (docs/25 §8 step 6;
         docs/26 A4 amended 2026-11-06). Kati sets every threshold and decides
         every item marked ⟨K…⟩ before signing.
Slot:    docs/24 §4 core (the stream paradigm, second after A–B–A).
Binds:   the grid of docs/25 §1 (1M, N = 300, M = 150, n_t = 200, n = 8).
Gate:    no stream grid runs before Kati signs docs/26 and this document
         (docs/25 §1, amended).
```

## 1. What this commits to

Six confirmatory tests, Holm-corrected as one family:
- C1, C2 and C3 of `docs/24` §3;
- the forward-transfer primary;
- the two richness-axis agreements.

Every other table and figure from the stream grid is exploratory and labelled as such in its caption.
Each test names:
- its primary cell;
- the paired contrast per unit;
- the statistic;
- the criterion;
- the null and the result that would support it.

**What the A–B–A FULL grid changes here.** Job 4349 (seeds 0–7) and its replication on seeds 8–15
(`docs/28` §8–§9) were run before this draft. Its lag-1 results feed the design as prior signal only,
never as evidence:
- **C2.** The "cliff, not dose" reading is not supported at lag 1 (`docs/24` C2, amended 2026-11-09).
  The C2 test below therefore reads the cliff at long lags only.
- **C1.** The γ₀ gap in fraction removed sits at H4. At H8 every γ₀ loses most of what it built. This
  bears on the choice of primary cell for C1, ⟨K1⟩.

## 2. Design (from docs/25 §1; proposals marked)

| item | value |
|---|---|
| learner | 1M: one hidden ReLU layer, N = 300, μP, u(0) = 0; full-batch GD; MSE; stop at loss ≤ 0.05 = 0.1·L(0), with **no update at criterion** (C5, fixed in `src` at `8bc9a49`) |
| data | P = 16, d = 150, D = 4, R = 1, M = 150; `rng_dichotomies` **on**, so task 0 is paired across Hamming levels (error registry E16) |
| stream | T = 16 tasks. Each step changes the labels by a fixed Hamming h and the input by s_f (AR(1) on the centres). |
| factors | γ₀ ∈ {0.03, 0.1, 0.3, 1, 3, 10} × s_f ∈ {1, 0.9, 0.1} × h ∈ {0, 2, 4, 6, 8}. This is 720 arms at n = 8. The fallback (`docs/25` §1) is γ₀ {0.1, 1, 10} × h {0, 2, 4, 8}, 288 arms. ⟨K0: which grid, after timing ONE design-scale arm⟩ |
| unit | one unique (arrangement, init) draw, with fresh ids recorded; paired across γ₀, s_f and h within a draw |
| measurement | GLUE `full_P`, retained fixed-y, at every boundary. Measured on past tasks' manifolds (backward) and on task t+1's manifolds (forward). Analytic NTK on a fixed probe set at every boundary. |
| provenance | clean tree, one commit, the code digest asserted per arm, as in A–B–A |

## 3. Units, floors and the resolution rule (docs/26 A4)

- **Effects** are in log α. Retained capacity of task j after task t is `log α_j(t)`. Its change
  over one boundary is `Δlog α_j(t−1 → t)`.
- **Per-unit summary.** A stream gives many boundaries per unit. They are averaged within the unit
  first (over the boundaries named per test), so each unit contributes one number. Boundaries are
  never treated as units.
- **Floor per γ₀.** SD(log α) over K = 8 estimator seeds, on the trained representation of the first
  draw, at n_t = 200 (same definition as `docs/28` §3). Contrasts across γ₀ use the larger floor.
- **Statistic** (docs/26 A4):
  - a **sign count** over the n units;
  - a **bootstrap CI over units**: percentile, 10 000 resamples, seed fixed in the code;
  - the CI level is set by Holm (§5).
- **Resolved** means the CI excludes 0 **and** \|mean\| ≥ 2 floors.
- **Readings** are as in `docs/28` §3: held, not held, or unresolved at floor. Strata are read one by
  one, never pooled. A test with several primary strata holds only if it holds in each one
  (intersection–union).

## 4. The six tests

### T1 — C1: the learner sets magnitude and channel (an interaction contrast)

- **Claim** (`docs/24` §3). Richness increases forgetting and shifts it from radius to
  alignment/utility. Stated as an interaction: the richness gradient of channel composition differs by
  stream condition.
- **Primary cell** ⟨K1⟩. Lag-1 backward change of task t−1 at h = 8, under frozen input (s_f 1)
  against jump (s_f 0.1), contrasting γ₀ = 10 with γ₀ = 0.1.
- **Per-unit contrast.** For the radius share q = Δ_radius / Δ_total, averaged over boundaries t = 1…15:
  - `I = [q(0.1) − q(10)]_frozen − [q(0.1) − q(10)]_jump`;
  - the magnitude part, read jointly: `m = Δ_total(10) − Δ_total(0.1)` at frozen, h = 8.
- **Read if** Δ_total is resolved in each of the four cells (as the denominator rule in `docs/28` §3).
- **Criterion.**
  - held: m is resolved < 0 **and** I is resolved > 0;
  - not held: either is resolved the other way;
  - otherwise unresolved.
- **Null.** No interaction. Supported if the CI of I lies inside ±⟨K1b⟩.
- **Prior signal (A–B–A, lag 1, γ₀ = 10 vs 0.1).** Radius share at H8 frozen is about 0.06 against
  about 0.38. Under jump, the radius share at γ₀ = 10 is ≈ 0.

### T2 — C2: the stream sets sign

- **Claim.** Transfer to a past task appears only when the task repeats under slowly drifting input.
  The "cliff" part is read at long lags only (amended 2026-11-09).
- **Primary cell** ⟨K2⟩. h = 0 under drift (s_f 0.9), at γ₀ = 10 and γ₀ = 0.1 (two strata). The
  statistic is task 0's retained change from the end of task 0 to the end of the stream:
  `log α_0(15) − log α_0(0)`.
- **Criterion.**
  - held: resolved > 0 in both strata, **and** resolved ≤ 0 at h = 0 under jump in both strata (the
    sign is set by the stream);
  - not held: resolved < 0 at h = 0 drift in either stratum;
  - otherwise unresolved.
- **Cliff, long lags only (secondary inside T2, read with the same CI)** ⟨K2b⟩. At lag ≥ ⟨K2c, e.g.
  8⟩, `Δ(h = 2) − Δ(h = 8)` lies inside the equivalence band ±⟨K2d⟩. A dose at long lags is reported,
  and it does not fail T2's sign reading.
- **Null.** No gain at h = 0 drift. Supported if the CI lies inside ±2 floors.
- **Prior signal.**
  - A–B–A H3 held at lag 1 (+3.4 and +2.3 floors).
  - The early 2M stream S-HH gain is Tier A.
  - Under Adam the gain did not appear (−0.3 floors, CI including 0; `docs/24` §6.1). That is why C2
    is scoped to gradient descent.

### T3 — C3: centre geometry dissociates from capacity

- **Claim.** Input change drives centre convergence. Task change drives graded divergence while
  capacity steps.
- **Primary cell** ⟨K3⟩. γ₀ = 10, at lag 1:
  - (a) **input:** the s_f effect on Δρ_c (signed) at h = 0, jump minus frozen;
  - (b) **task:** the h effect on Δρ_c at frozen input, a monotone trend over h ∈ {2, 4, 6, 8}.
- **Per-unit contrasts.**
  - (a) `Δρ_c(jump) − Δρ_c(frozen)` at h = 0.
  - (b) the slope of Δρ_c on h. Alongside it, the step in Δlog α_total between h = 0 and h = 2.
- **Criterion.**
  - held: (a) is resolved with the sign of convergence; **and** (b) is resolved with the sign of
    divergence, while the capacity step at h 0 → 2 is resolved;
  - not held: either is resolved the other way;
  - otherwise unresolved.
- **Floor for Δρ_c.** There is no capacity floor. ⟨K3b: use the SD of ρ_c over the estimator seeds of
  §3, or treat T3 on the CI alone⟩.
- **Note.** Δρ_c magnitudes stay exploratory at n = 8 (`docs/26` A4). This test reads signs and
  ordering only.

### T4 — Forward-transfer primary

- **Question** (`docs/26` A2). Which channel of the specific geometric FT predicts behavioural FT,
  conditional on the label and input relation?
- **Primary cell** ⟨K4⟩, one named cell: γ₀ = 10, h = 8, s_f 1. This matches A–B–A H4, which held
  at lag 1, FT_beh_spec −0.42 in the original run.
- **Statistics.**
  - Specific FT_geo = Δlog α_{y_{t+1}} − mean_k Δlog α_{y*_k}, against neutral probes y*_k with
    s_r(y*_k, y_t) = 0, as in A–B–A C6.
  - FT_beh = log((1 + steps fresh)/(1 + steps stream)).
  - Both are averaged per unit over boundaries.
- **Criterion** (as `docs/28` H4).
  - held: the specific FT_beh CI excludes 0, the mean is < 0, and \|mean\| ≥ log ⟨K4b, e.g. 1.25⟩;
  - not held: the CI excludes 0 and the mean is > 0.
- **Channel question.** The within-cell association across units between each channel of FT_geo and
  FT_beh is **reported, not read**. n = 8 cannot carry a confirmatory correlation. ⟨K4c: keep it out
  of the family, or make one named channel the read⟩
- **Exclusion.** Only h = 0 with frozen input.

### T5 and T6 — The two richness-axis agreements

`docs/25` §3: every claim is produced twice, against γ₀ and against measured richness r. Agreement
needs a **positive criterion** (docs/26 A4): the same sign, the CI excluding 0 on both axes, and the
same ordering of conditions. Failing to reject a difference does not count as agreement.

- **Measured axis.** r is ΔK_u after task 0, the readout block of the analytic NTK, relative to init.
  ΔK_u is ≈ 0 when training is lazy, and A–B–A expectation 5 showed that the full-NTK change is
  dominated by K_W growth.
- **⟨K5: which two agreements count.⟩** Proposal:
  - **T5: C1 on the measured axis.** Units are binned or ranked by r instead of γ₀, and T1's magnitude
    and interaction contrasts are recomputed. Agreement means T1's reading on r has the same sign and
    the same condition ordering as on γ₀, with both CIs excluding 0.
  - **T6: the forward-transfer primary on the measured axis.** T4's statistic is recomputed against r
    and agreement is read the same way.

  C2 and C3 are stated on the stream and input relations, not on richness. Their γ₀/r comparison is
  therefore reported, not read.
- **How r replaces γ₀** ⟨K5b⟩. Proposal: within each unit, order the γ₀ levels by measured r. The
  contrast "high vs low richness" becomes the arms at the top and bottom r. If the r ordering equals
  the γ₀ ordering in every unit, T5 and T6 reduce to an identity check. That would be reported as
  such and counted as agreement.
- **Null.** The axes disagree: opposite sign, or the ordering differs in a resolved way.

## 5. Multiplicity

- Six tests form one family. **Holm** at family-wise α = 0.05 ⟨K6⟩. The ordered test i uses level
  α/(6 − i + 1).
- With a bootstrap CI this means: each test's p-value is the bootstrap two-sided p of its primary
  contrast. For a test with several strata or parts, it is the **max** over its intersection–union
  parts. Holm then orders the six p-values.
- The CI quoted for a test is at the level Holm gives it.
- **Critical values are computed in the code, not typed** (lesson of the `docs/28` §5 erratum).

## 6. What is not confirmatory

Everything else, including:
- every γ₀ level other than the named strata;
- h ∈ {2, 4, 6} outside T2's long-lag band and T3(b);
- all Δρ_c magnitudes;
- the richness trajectory r_0 … r_15;
- savings;
- zero-shot loss;
- forward transfer outside T4's cell;
- order and spacing (their own pre-commits: `docs/23`, `docs/27`).

## 7. Decisions for Kati before signing

| # | decision |
|---|---|
| K0 | Full grid (720) or fallback (288), after timing ONE design-scale arm with every measurement included (`docs/25` §8 step 7). |
| K1 | T1 primary cell: h = 8 or h = 4. At lag 1 the γ₀ gap in fraction removed was at H4. K1b: the equivalence band for I. |
| K2 | T2 strata and whether the jump control is part of "held". K2b–K2d: the long-lag cliff band, the lag, and whether it stays inside T2. |
| K3 | T3 cells; the floor for Δρ_c (K3b). |
| K4 | T4 cell and threshold (log 1.25 as in `docs/28` H4?). K4c: whether the channel association is read. |
| K5 | Which two agreements count (proposal: C1 and FT on ΔK_u). K5b: how r replaces γ₀ within a unit. |
| K6 | The family-wise α, and Holm vs Bonferroni. Bonferroni is what `docs/28` used for the A–B–A grid. |
| K7 | The bootstrap (percentile, 10 000) against the t-interval used in `docs/28`. docs/26 A4 says bootstrap plus a sign count. |

## 8. Execution (after signing)

1. Implement measurement on task t+1's manifolds for the stream, and the analytic NTK and r_t in
   `src` (`docs/25` §8 step 7).
2. Time ONE design-scale 1M stream arm with every measurement on grime. Size the grid with the
   measured 3–4× parallel slowdown.
3. One commit, clean tree. The readings are computed by code that implements §3–§5 verbatim, and are
   reported as computed.
