# 29 — Confirmatory-test specification for the 1M stream grid (DRAFT)

```
Status:  DRAFT, unsigned. First draft 2026-11-09 (docs/25 §8 step 6);
         redrafted 2026-11-09 after Kati's review (items 3, 4, 8, 9, 11):
         SESOI instead of floors, K0–K7 resolved, readings per
         configuration and per (j, t). Open: the SESOI values and rule
         form (§3, §7), the position rule (K8), stopping (K9).
Slot:    docs/24 §4 core (the stream paradigm, second after A–B–A).
Binds:   the grid of docs/25 §1 (1M, N = 300, M = 150, n_t = 200, n = 8).
Gate:    no stream grid runs before Kati signs docs/26 and this document
         (docs/25 §1, amended). The docs/31 stopping-rule ablation must be
         read before signing (K9).
```

## 1. What this commits to

Six confirmatory tests, **Bonferroni-corrected as one family**. Every interval is a two-sided
t-interval at 1 − 0.05/6 ≈ 99.17% (§5). The six are:
- C1, C2 and C3 of `docs/24` §3;
- the forward-transfer primary;
- two richness-axis agreements.

Every other table and figure from the stream grid is exploratory and labelled as such in its caption.

Each test names:
- its configuration (γ₀, h, s_f);
- its **(j, t) cell**: which past task j is read after which task t;
- the paired contrast per unit;
- the statistic, the criterion, and the null with the result that would support it.

**Setting discipline (Kati, 2026-11-09, items 8, 9 and 11):**
- Every reading is per configuration and per (j, t). There is **no averaging over lags or over
  positions** within a stream, and **no pooling across configurations**.
- "Held in X" means held in X only.
- Every reading is stated against the **realised** label similarity S_r(j, t) and the realised
  input-centre correlation ρ_in(j, t), per seed (§2a). Over a stream they are random walks, not the
  nominal h and s_f.
- Every stream report carries a **setting card** per configuration (§2a), as `docs/30` §4 does for
  A–B–A.

**What the A–B–A FULL runs change here.** These are prior signal only, never evidence. The runs are
job 4349 (seeds 0–7) and its replication, job 4396 on seeds 8–15, which is confirmatory for A–B–A; see
`docs/28` §8–§9 and `docs/30`.
- **C2.** "Cliff, not dose" is not supported at lag 1 (`docs/24` C2, amended). The long-lag cliff is
  moved to exploratory (§6), per K2.
- **C1.** The γ₀ gap in fraction removed held at h 4 frozen and h 4 drift, and not at h 8, where each
  γ₀ removes 0.74–0.91 of A's gain (`docs/30` §1). The C1 primary is therefore at h = 4 (K1).
  **This choice was informed by A–B–A.**
- **Exposure.** Under matched-loss stopping, exposure differs 40–60-fold between γ₀ 0.1 and 10
  (`docs/30` §4). The docs/31 ablation tests whether the A–B–A γ₀ effects survive matched exposure (K9).

## 2. Design (from docs/25 §1)

| item | value |
|---|---|
| learner | 1M: one hidden ReLU layer, N = 300, μP, u(0) = 0; full-batch GD; MSE; stop at loss ≤ 0.05 = 0.1·L(0), with **no update at criterion** (C5, fixed in `src` at `8bc9a49`). Stopping is subject to K9. |
| data | P = 16, d = 150, D = 4, R = 1, M = 150; `rng_dichotomies` on by default (error registry E16), so task 0 is paired across Hamming levels |
| stream | T = 16 tasks. Each step changes the labels by a fixed Hamming h and the input by s_f (AR(1) on the centres). |
| grid (K0) | **Fallback grid:** γ₀ {0.1, 1, 10} × h {0, 2, 4, 8} × s_f {1, 0.9, 0.1} = 36 configurations × n = 8 = **288 arms**. It runs only after ONE design-scale arm with every measurement has been timed on grime, and the grid is sized with the measured 3–4× parallel slowdown. |
| unit | one unique (arrangement, init) draw, with fresh ids recorded; paired across γ₀, s_f and h within a draw |
| measurement | GLUE `full_P`, retained fixed-y, at every boundary. Measured on past tasks' manifolds (backward) and on task t+1's manifolds (forward). Analytic NTK on a fixed probe set at every boundary (ΔK_u, ΔK_W). |
| provenance | clean tree, one commit, the code digest asserted per arm, as in A–B–A |

### 2a. Setting card (required in every stream report)

One card per configuration (γ₀-independent parts once; exposure per γ₀):
- the nominal h and s_f;
- per seed, the realised **S_r(j, t)** = s_r(y_j, y_t) and **ρ_in(j, t)**, the mean cosine of
  matched centres, for every (j, t) that is read, and in full as a supplementary table;
- **exposure per learner**: the updates per task, for each γ₀ and seed. Matched-loss stopping lets
  the learner set its exposure, so the card states it explicitly.

## 3. Units, SESOI and the resolution rule

Replaces the noise-floor rule (Kati, item 3).

- **Effects** are in log α. The retained capacity of task j after task t is `log α_j(t)`.
- **Unit.** One draw contributes **one value per (j, t) cell**. Values from different (j, t) cells
  are never averaged.
- **Interval.** The two-sided t-interval over the n = 8 units at 1 − 0.05/6 (df = 7). The critical
  value is computed in code (`scipy.stats.t.ppf`), never typed (lesson of the `docs/28` §5 erratum).
- **Sign count.** The number of units with the sign of the mean, out of n. It is reported with every
  interval and does not decide a reading (K7).
- **SESOI** (smallest effect size of interest). Fixed here, before any stream data exists.

  | quantity | proposed SESOI | why |
  |---|---|---|
  | Δlog α (capacity) | **0.05** | A 5% change in capacity. Single-draw estimator floors in A–B–A FULL were 0.010–0.043 (`docs/28` §9.2), so 0.05 lies above every measured floor. The A–B–A effects of interest were 0.05–1.7. H3's +0.048 at γ₀ 0.1 (replication) sits just at it. |
  | ρ_c (signed, normalised) | **0.02** | A–B–A Δρ_c(A → B) on seeds 8–15 is ≤ 0.006 in magnitude at γ₀ 0.1 in every configuration. At γ₀ 10 it reaches +0.022 at h 0 jump and +0.061 at h 8 jump. 0.02 separates these. |
  | FT_beh (log step ratio) | **log 1.25 ≈ 0.223** | As `docs/28` H4: a 25% change in steps. |
  | channel share, fraction removed | **0.10** (proposal) | As the `docs/28` H1 null band (±0.1). Used by T1(c) and T6. |

- **Rule form ⟨S0⟩.** The draft rule is a minimal-effect test:
  - **resolved > 0**: the whole interval lies above +SESOI;
  - **resolved < 0**: the whole interval lies below −SESOI;
  - **equivalent** (supports a null of "no effect of interest"): the whole interval lies inside
    ±SESOI;
  - **unresolved**: everything else.

  The alternative mirrors `docs/28`: resolved when the interval excludes 0 **and** |mean| ≥ SESOI.

  On A–B–A H3 (replication, h 0 drift):
  - +0.048 [+0.033, +0.062] at γ₀ 0.1 is unresolved under both forms;
  - +0.070 [+0.049, +0.090] at γ₀ 10 is unresolved under the minimal-effect test, and resolved under the
    mirror form.
- **Floor (diagnostic only).** SD(log α) over K = 8 estimator seeds on **each** of the n draws,
  pooled as the root mean square over draws, at n_t = 200. It is reported beside every capacity
  reading and decides nothing. It replaces the single-draw floor that differed 1.5–2.4× between seed 0
  and seed 8 in A–B–A (`docs/28` §9.1).
- **Ratios** (shares, fraction removed) are read only when their denominator is resolved (§3 rule on
  Δlog α). Otherwise the cell is reported as **not read**.
- **Readings.**
  - **held**: the stated direction is resolved.
  - **not held**: the opposite is resolved.
  - **unresolved**: everything else.
  - A test with several primary cells holds only if it holds in each read cell (intersection–union).
    The number of read cells is reported.

## 4. The six tests

Every (j, t) cell below is a single named cell (K8). "Lag 1 at position p" means j = p − 1 read
after task t = p. The proposed position is p = 8, the middle of the stream. All other positions and
lags are reported per cell in the exploratory tables (§6), each with its own realised S_r and ρ_in.

### T1 — C1: the learner sets magnitude and channel

- **Claim** (`docs/24` §3). Richness increases forgetting, and shifts it from radius to
  alignment/utility.
- **Cell (K1).** h = 4, lag 1 at position p, γ₀ = 10 against γ₀ = 0.1. The parts and their input
  conditions are listed below. **h = 4 was chosen from A–B–A**, where the γ₀ gap in fraction removed
  held at h 4 frozen and h 4 drift, and not at h 8.
- **Per-unit contrasts**, each in one (configuration, j, t) cell:
  - **(a) magnitude.** f = fraction of task j's gain removed by task t:
    d_f = f(10) − f(0.1), at h 4 frozen. SESOI 0.10. Held if resolved > 0.
  - **(b) channel interaction.** The radius share q = Δ_radius / Δ_total at (j, t), with
    `I = [q(0.1) − q(10)]_frozen − [q(0.1) − q(10)]_jump` at h 4. SESOI 0.10. Held if resolved > 0.
  - **(c) utility-share contrast** (from A–B–A H5b, Kati 2026-11-09). The utility share
    u = Δ_utility / Δ_total, with d_u = u(10) − u(0.1) at h 4 frozen. SESOI 0.10. Held if resolved > 0.
    This is the contrast that A–B–A H5b's threshold form only approximated.
- **Read if** Δ_total(j, t) is resolved in each cell that enters a ratio.
- **Criterion.**
  - held: (a), (b) and (c) are each held;
  - not held: any is resolved the other way;
  - otherwise unresolved.
- **Null.** No richness dependence: (a), (b) and (c) each equivalent (inside ±SESOI).
- **Prior signal (A–B–A replication, lag 1).**
  - d_f = +0.332 at h 4 frozen.
  - At h 4 frozen, the utility share is 0.18 at γ₀ 0.1 and 0.50 at γ₀ 10.
  - The radius share at h 4 and h 8, frozen and drift, is 0.38–0.41 at γ₀ 0.1 and 0.07–0.08 at
    γ₀ 10.

### T2 — C2: the stream sets sign

- **Claim.** Transfer to a past task appears when the task repeats under slowly drifting input.
- **Cell (K2).** h = 0 drift (s_f 0.9), **at γ₀ = 0.1 and γ₀ = 10** (two strata). There are two
  candidate statistics:
  - task 0 after the last task: `log α_0(15) − log α_0(0)`, one (j, t) = (0, 15) cell per stratum;
  - lag 1 at position p: `Δlog α_{p−1}(p−1 → p)`.

  ⟨K2a: which one is primary; the draft proposes (0, 15), the claim as stated⟩.
- **Criterion.**
  - held: resolved > 0 in both strata (SESOI 0.05);
  - not held: resolved < 0 in either;
  - otherwise unresolved.
- **Jump control (reported, not part of "held").** The same statistic at h = 0 jump (s_f 0.1), at both
  γ₀, is reported beside T2 with its own interval. It does not enter the reading.
- **Null.** No gain at h = 0 drift: the interval lies inside ±0.05 in both strata.
- **Prior signal.**
  - A–B–A H3 at lag 1 (h 0 drift) gave +0.048 and +0.070 in the replication. Both are positive with
    intervals excluding 0, and both are below the pre-set resolution.
  - Under Adam the gain did not appear (`docs/24` §6.1), which is why C2 is scoped to gradient descent.

### T3 — C3: centre geometry dissociates from capacity

- **Claim.** Input change drives centre convergence. Task change drives graded divergence while
  capacity steps.
- **Cell (K3).** γ₀ = 10, lag 1 at position p, signed normalised ρ_c (`rho_c_signed`). SESOI 0.02.
  - **(a) input.** Δρ_c(jump) − Δρ_c(frozen) at h = 0. Held if resolved with the sign of convergence.
  - **(b) task.** Δρ_c at h 8 frozen minus h 2 frozen. Held if resolved with the sign of divergence,
    while Δlog α_total(h 2) − Δlog α_total(h 0) at frozen is resolved (SESOI 0.05). This replaces the
    slope over h, so the reading stays per configuration pair.
- **Criterion.**
  - held: (a) and (b) are both held;
  - not held: either is resolved the other way;
  - otherwise unresolved.
- **Prior signal (A–B–A replication, γ₀ 10).** Δρ_c(A → B) was +0.022 at h 0 jump and −0.026 at
  h 8 frozen (interval includes 0).

### T4 — Forward-transfer primary (K4: as A–B–A H4)

- **Cell.** γ₀ = 10, forward from task p to task p + 1 at position p.
- **Statistic.** FT_beh_spec = FT_beh(h 8, frozen) − FT_beh(h 8, jump), with
  FT_beh = log((1 + steps fresh)/(1 + steps stream)).
- **Criterion** (as `docs/28` H4).
  - held: resolved < 0 with SESOI log 1.25;
  - not held: resolved > 0;
  - otherwise unresolved.
- **Channels.** The association between the channels of specific FT_geo and FT_beh is reported, not read.
- **Prior signal.** A–B–A H4 gave −0.459 [−0.591, −0.328] (replication).
- **Exposure caveat.** FT_beh is a step ratio, so it is defined only under matched-loss stopping (K9).

### T5 and T6 — The two richness-axis agreements (K5)

`docs/25` §3: every claim is produced twice, against γ₀ and against measured richness r.
- **r** is ΔK_u after the task preceding the cell: the readout block of the analytic NTK, relative to
  init. In A–B–A after A, the median ΔK_u was 0.05 at γ₀ 0.1, 0.69 at 1 and 7.8 at 10.
- **Agreement** needs a positive criterion: the same sign, both intervals resolved, and the same
  ordering of conditions.
- **How r replaces γ₀ ⟨K5b⟩.** Within each unit, the γ₀ arms are ordered by measured r, and "high vs
  low richness" becomes the top-r vs bottom-r arm. If the r ordering equals the γ₀ ordering in every
  unit, T5 and T6 reduce to an identity check. That is reported as such and counted as agreement.

The two tests:
- **T5: C1 against ΔK_u.** T1(a)–(c) are recomputed with top/bottom r in place of γ₀ 10/0.1, in T1's
  cells.
- **T6: the fraction-removed contrast against ΔK_u.** d_f at h 4 frozen, lag 1 at position p, is
  recomputed against r. Its SESOI is 0.10.

**Null.** The axes disagree: opposite resolved signs, or a resolved difference in ordering.

## 5. Multiplicity (K6)

- Six tests form one family: **Bonferroni**, family-wise α = 0.05. Every primary interval is at
  1 − 0.05/6, including the equivalence checks and the parts of T1 and T3.
- Within a test, intersection–union applies and needs no correction.
- Critical values are computed in code.
- No bootstrap (K7). Inference is the t-interval plus a reported sign count.

## 6. What is not confirmatory

Everything else, each reported per configuration and per (j, t), never averaged over lags or
positions. This includes:
- every (j, t) other than the named cells: all positions, all lags;
- **the long-lag cliff**, `Δ(h 2) − Δ(h 8)` at long lags (moved here by K2);
- the jump control of T2, as a named reported control;
- every γ₀ level other than the named strata, and h = 2 outside T3(b);
- all Δρ_c magnitudes outside T3;
- the richness trajectory r_0 … r_15;
- savings and zero-shot loss;
- forward transfer outside T4's cell;
- order and spacing, which have their own pre-commits (`docs/23`, `docs/27`).

## 7. Decisions

**Resolved by Kati, 2026-11-09:**

| # | decision |
|---|---|
| K0 | The fallback grid (288), after timing ONE arm with every measurement, using the measured parallel slowdown. |
| K1 | T1 at h = 4 (informed by A–B–A, stated in T1). |
| K2 | T2 at h = 0 drift, both γ₀; jump as a reported control; the long-lag cliff moves to exploratory. |
| K3 | Signed normalised ρ_c with SESOI. |
| K4 | As A–B–A H4. |
| K5 | T5 = C1 against ΔK_u; T6 = fraction-removed contrast against ΔK_u. |
| K6 | Bonferroni intervals. |
| K7 | t-intervals plus a sign count; no bootstrap. |
| T1 | The utility-share contrast (γ₀ 10 vs 0.1) is added inside T1 as part (c). |

**Open (for Kati before signing):**

| # | decision |
|---|---|
| S0 | SESOI values (§3 table) and the rule form: minimal-effect test (draft) or the `docs/28` mirror. |
| K2a | T2 statistic: task 0 after the stream, (0, 15) (draft), or lag 1 at position p. |
| K5b | How r replaces γ₀ within a unit (draft: top-r vs bottom-r arm). |
| K8 | **Position rule.** Each named cell sits at **one** position. The draft proposes p = 8, the middle of the stream. The alternative is intersection–union over all positions 1…15, which is valid without averaging but has low power at n = 8 per position. |
| K9 | **Stopping.** If the docs/31 ablation finds a sign flip in any stratum, the stream grid needs a stopping decision before this is signed: matched loss, a fixed budget, or both reported. If it does not, matched-loss stopping stands, and exposure is stated on every setting card. |

## 8. Execution (after signing)

1. Implement measurement on task t+1's manifolds for the stream, the analytic NTK and r_t, and the
   realised S_r(j, t) / ρ_in(j, t) per seed, all in `src` (`docs/25` §8 step 7).
2. Time ONE design-scale 1M stream arm with every measurement on grime. Size the 288-arm grid with
   the measured 3–4× parallel slowdown.
3. One commit, clean tree. The readings and setting cards are computed by code that implements §2a–§5
   verbatim, and are reported as computed.
