# 28 — A–B–A FULL run: pre-commit (DRAFT, unsigned)

**Status: DRAFT 2026-11-09, written by the agent for Kati to sign. Not binding until signed.**
Items marked **[K]** are Kati's to decide. No FULL arm runs before the signature.

Signed: ____________________ date: __________

## 1. What this commits to

These are the readings of the FULL A–B–A grid that count as results. Every rule below was fixed
before any FULL number existed. The rest of `notebooks/core/aba_controlled.ipynb` is
observation.

## 2. Design (fixed)

| item | value |
|---|---|
| learner | 1M, N = 300, μP, one hidden ReLU layer, u(0) = 0 (C1) |
| data | P = 16, d = 150, D = 4, R = 1, M = 150 training points per manifold; test points drawn separately |
| training | full-batch GD, lr = 5·γ0²·N/64, stop at loss ≤ 0.05 = 0.1·L(0) (C2), loss checked before each update (C5), at most 20 000 steps per phase |
| factors | γ0 ∈ {0.1, 1, 10} × Hamming(y_A, y_B) ∈ {0, 4, 8} × s_f ∈ {1 (frozen), 0.9 (drift), 0.1 (jump)} |
| seeds | n = 8, paired: within a seed, A, y_A, W(0), the probes and the GLUE stream are shared by every cell and every γ0 |
| controls | identity cell (H0, s_f 1, exact copy, C3); one fresh-noise identity arm per γ0 (seed 0) |
| arms | 8 × 3 × 9 + 3 = 219 |
| capacity | retained (fixed-y) GLUE, full_P, n_t = 200, center_policy "all" |
| probes | bottleneck probe, n = 16 (calibrated, C9), budget 2000 steps (C8); neutral y*_k, K = 4 (C6) |
| provenance | every arm carries commit, dirty = False and the `aba_lib` hash; one commit for the whole grid |

The arms are valid only if every validation check in the notebook passes. If any check fails,
nothing below is read.

## 3. Units, floors and the resolution rule

- **Δlog α_A** is the change in A's retained capacity on the hidden representation.
  - init→A means `backward.init_to_A1`; A→B means `backward.A1_to_B`.
  - log α_A(t) is the level at boundary t ∈ {init, A1, B, A2}.
- **Floor per γ0** is SD(log α_A) over K = 8 estimator seeds, on the representation after A,
  seed 0, at FULL n_t. It is computed in the run itself (`trained_floor`). A contrast between
  two γ0 uses the larger of the two floors.
- **Resolved** means the 95% t-interval over the 8 seeds excludes 0 **and** |mean| ≥ 2 floors.
  Seeds are the unit; a contrast is paired per seed.
- **Ratios.** Fraction removed, channel shares and restoration are dimensionless, so the floor
  rule applies to their parts.
  - A ratio is read only if its numerator and its denominator Δlog α are each resolved.
  - The ratio, or the ratio contrast, is then read on its own 95% t-interval as stated per
    hypothesis.
  - **[K]** Confirm this rule, or set a floor in ratio units.
- **Readings.**
  - **held**: the stated direction is resolved.
  - **not held**: the opposite is resolved, or the threshold is excluded the other way.
  - **unresolved at floor**: everything else.
  - Strata are read one by one, never pooled. A hypothesis with several primary strata holds
    only if it holds in each of them (intersection–union).
- **Null.** Each hypothesis states the null and the result that would support it. "Unresolved"
  does not support the null.

## 4. Hypotheses

Each hypothesis gives the statistic, the primary strata, the reading rule and the null. The prior
signal is from QUICK (n = 3, M = 50, n_t = 50, 11-08/11-09). It is shown for transparency and is
not evidence.

### H1 — B removes a larger fraction of A's learned capacity at γ0 = 10 than at 0.1

- **Statistic.** Per seed, the fraction removed
  f = (log α_A(A1) − log α_A(B)) / (log α_A(A1) − log α_A(init)), and the paired difference
  d = f(γ0 = 10) − f(γ0 = 0.1).
- **Primary strata.** H4 frozen (s_f 1) and H4 drift (s_f 0.9). Two readings.
- **Preconditions.** In each stratum, the init→A gain and the A→B change are resolved at both
  γ0.
- **Rule.**
  - held: the 95% CI of d excludes 0 and d > 0;
  - not held: the CI excludes 0 and d < 0;
  - otherwise unresolved.
- **Null.** The fraction removed does not depend on γ0. It is supported if the 95% CI of d lies
  inside ±0.1 **[K: equivalence band]**.
- **Prior signal.** H4 frozen: 0.76 vs 0.39. H4 drift: 0.65 vs 0.31.

### H2 — The gain from learning A increases monotonically with γ0

- **Statistic.** The init→A gain g(γ0) = Δlog α_A(init→A1), the same in every cell of a seed.
  Paired differences d₁ = g(1) − g(0.1) and d₂ = g(10) − g(1).
- **Rule.**
  - held: d₁ and d₂ are both resolved > 0, each with the larger floor of its pair;
  - not held: either is resolved < 0;
  - otherwise unresolved.
- **Null.** There is no increase at one step or both. A step's null is supported if its CI lies
  inside ±2 floors.
- **Prior signal.** g(0.1) = +0.22, g(10) = +2.02. γ0 = 1 has not been run.

### H3 — Learning B under drift with the same labels (H0, s_f 0.9) increases A's capacity

- **Statistic.** Δlog α_A(A1→B) at H0, s_f 0.9.
- **Primary strata.** γ0 = 0.1 and γ0 = 10, two readings. γ0 = 1 is reported but is not part of
  H3 **[K: include?]**.
- **Rule.**
  - held: resolved > 0;
  - not held: resolved < 0;
  - otherwise unresolved.
- **Null.** No change. Supported if the CI lies inside ±2 floors.
- **Prior signal.** +0.021 (0.5 floors) and +0.045 (0.5 floors). At QUICK both are below
  1 floor. The FULL floor is expected to be smaller (n_t 200 vs 50), but the effect may still not
  reach 2 floors.

### H4 — Specific behavioural forward transfer is negative at H8 frozen, γ0 = 10

- **Statistic.** Per seed, FT_beh_spec = FT_beh(H8, s_f 1) − FT_beh(H8, s_f 0.1), with
  FT_beh = log((1 + steps fresh B)/(1 + steps B)) (C4, C7).
- **Primary stratum.** γ0 = 10, one reading.
- **Rule.**
  - held: the 95% CI excludes 0, the mean is < 0, and |mean| ≥ 2 × the behavioural floor;
  - not held: the CI excludes 0 and the mean is > 0;
  - otherwise unresolved.
- **Behavioural floor [K].** Capacity floors do not apply here. Proposal: the SD over the
  8 seeds of FT_beh in the baseline cell (H8, s_f 0.1, γ0 = 10).
- **Null.** No specific transfer. Supported if the CI lies inside ±2 behavioural floors.
- **Prior signal.** −0.47 [−0.61, −0.33].

### H5 — Channel shares of the A→B loss

- **Statistic.** Per seed, the share of channel c = Δ_c(A1→B) / Δ_total(A1→B), with the exact
  identity total = utility + radius + dimension. Shares can lie outside [0, 1] when channels
  have opposite signs.
- **Primary strata.** H ∈ {4, 8} × s_f ∈ {1, 0.9}: 4 cells per γ0. Each cell is read only if its
  Δ_total is resolved.
- **H5a, γ0 = 10.** Utility + dimension share ≥ 0.8.
  - held: the lower 95% bound is ≥ 0.8;
  - not held: the upper bound is < 0.8;
  - otherwise unresolved.
- **H5b, γ0 = 0.1.** No channel's share is above 0.5.
  - held: every channel's upper 95% bound is ≤ 0.5;
  - not held: some channel's lower bound is > 0.5;
  - otherwise unresolved.
- **Null.**
  - For H5a: the utility + dimension share is < 0.8, supported if the upper bound is < 0.8.
  - For H5b: one channel dominates, supported if some lower bound is > 0.5.
- **Prior signal.**
  - γ0 = 10, H8 frozen: utility 0.55, dimension 0.39, radius 0.06 (as shares of the A→B
    total).
  - γ0 = 0.1, H8 frozen: about one third each.

### H6 — Relearning A restores at least 90% of what A built

- **Statistic.** Per seed, the restoration
  ρ = (log α_A(A2) − log α_A(init)) / (log α_A(A1) − log α_A(init)).
- **Primary strata.** H ∈ {4, 8} × s_f ∈ {1, 0.9}, per γ0: 12 readings. Each is read only if the
  init→A gain and the A→B change are resolved. Other cells are reported but not read.
- **Rule.**
  - held: the lower 95% bound of ρ is ≥ 0.9;
  - not held: the upper bound is < 0.9;
  - otherwise unresolved.
- **Null.** Restoration is < 0.9, supported by "not held".
- **Prior signal.** γ0 = 10: log α_A(A2) is 0.91–0.94 against A1 = 1.00 and init = −1.02, so ρ is
  about 0.96. γ0 = 0.1: A2 is at or above A1.

## 5. Multiplicity [K]

There are 6 hypotheses with 2 + 2 + 2 + 1 + 8 + 12 primary readings. Within a hypothesis,
intersection–union applies, and it needs no correction. Across the 6 hypotheses Kati chooses one
of:
- (a) no correction; each hypothesis is reported on its own;
- (b) Bonferroni-adjusted intervals at 1 − 0.05/6 ≈ 99.2% in place of 95%;
- (c) Holm over the 6, using the per-hypothesis minimum p-value of the paired t-test, alongside
  the floor rule.

## 6. What is not confirmatory

All other tables and figures in the notebook are observations. That includes:
- refit and bottleneck levels;
- phase-to-phase kernel change;
- ρ_c;
- forward transfer outside H4;
- γ0 = 1 outside H2.

## 7. Execution

The run happens only after this document is signed. All code changes land first. After that,
nothing in `aba_lib.py` changes, and any change voids the run.

1. Done: bottleneck n = 16 is calibrated on grime (job 4316), entered into C9 and `ABAConfig`, committed and pushed.
2. Timing on grime sizes the job.
3. One sbatch job, `sbatch/aba_full.sh`, runs the 219 arms, then executes the notebook with
   `ABA_MODE=full` (floors at K = 8).
4. Readings are computed by the notebook's pre-commit cell, which implements §3–§4 verbatim, and
   are reported as computed.
