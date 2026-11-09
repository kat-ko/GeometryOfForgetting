# 28 — A–B–A FULL run: pre-commit (SIGNED)

**Status: SIGNED 2026-11-09.** Drafted by the agent 2026-11-09 and amended per Kati's review the
same day (amendments 1–6, plus the not-read clarification in §3, approved in session). Binding
for the FULL grid.

Signed: __Kati__________________ date: _09.11.2026_________

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
| provenance | every arm carries commit, dirty = False and the `aba_lib` hash (`2528909f2feaf3f3`); one commit for the whole grid |

The arms are valid only if every validation check in the notebook passes. If any check fails,
nothing below is read.

## 3. Units, floors and the resolution rule

- **Δlog α_A** is the change in A's retained capacity on the hidden representation.
  - init→A means `backward.init_to_A1`; A→B means `backward.A1_to_B`.
  - log α_A(t) is the level at boundary t ∈ {init, A1, B, A2}.
- **Floor per γ0** is SD(log α_A) over K = 8 estimator seeds, on the representation after A,
  seed 0, at FULL n_t. It is computed in the run itself (`trained_floor`). A contrast between
  two γ0 uses the larger of the two floors.
- **Intervals.** Every interval in §3–§4 is a two-sided t-interval over the 8 seeds at
  1 − 0.05/6 ≈ 99.17% (§5), with df = n − 1. This includes the resolution checks, the
  denominator checks, the threshold bounds of H5/H6 and the equivalence (null) checks. Seeds are
  the unit; a contrast is paired per seed.
- **Resolved** means the interval excludes 0 **and** |mean| ≥ 2 floors.
- **Ratios.** Fraction removed (H1), channel shares (H5) and restoration (H6) are read when
  their **denominator** is resolved:
  - H1 and H6: the init→A gain is resolved at the γ0 in question (for H1, at both γ0 of the
    contrast);
  - H5: Δ_total(A1→B) of the cell is resolved.
  - The numerator need not be resolved, because a ratio near 0 is a legitimate value.
  - The ratio, or the ratio contrast, is then judged on its own interval as stated per
    hypothesis.
- **Readings.**
  - **held**: the stated direction is resolved, or the threshold is cleared as stated.
  - **not held**: the opposite is resolved, or the threshold is excluded the other way.
  - **unresolved at floor**: everything else.
  - Strata are read one by one, never pooled. A hypothesis with several primary strata holds
    only if it holds in each read stratum (intersection–union).
  - A stratum whose denominator is unresolved is reported as **not read**. It is left out of the
    intersection–union. A hypothesis needs at least one read stratum; if none is read, it is
    unresolved. The number of read strata is always reported.
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
- **Read if** the init→A gain is resolved at both γ0 (§3).
- **Rule.**
  - held: the interval of d excludes 0 and d > 0;
  - not held: the interval excludes 0 and d < 0;
  - otherwise unresolved.
- **Null.** The fraction removed does not depend on γ0. It is supported if the interval of d lies
  inside ±0.1.
- **Prior signal.** H4 frozen: 0.76 vs 0.39. H4 drift: 0.65 vs 0.31.

### H2 — The gain from learning A increases monotonically with γ0

- **Statistic.** The init→A gain g(γ0) = Δlog α_A(init→A1), the same in every cell of a seed.
  Paired differences d₁ = g(1) − g(0.1) and d₂ = g(10) − g(1).
- **Rule.**
  - held: d₁ and d₂ are both resolved > 0, each with the larger floor of its pair;
  - not held: either is resolved < 0;
  - otherwise unresolved.
- **Null.** There is no increase at one step or both. A step's null is supported if its interval
  lies inside ±2 floors.
- **Prior signal.** g(0.1) = +0.22, g(10) = +2.02. γ0 = 1 has not been run.

### H3 — Learning B under drift with the same labels (H0, s_f 0.9) increases A's capacity

- **Statistic.** Δlog α_A(A1→B) at H0, s_f 0.9.
- **Primary strata.** γ0 = 0.1 and γ0 = 10, two readings. γ0 = 1 is reported, not read.
- **Rule.**
  - held: resolved > 0;
  - not held: resolved < 0;
  - otherwise unresolved.
- **Null.** No change. Supported if the interval lies inside ±2 floors.
- **Prior signal.** +0.021 (0.5 floors) and +0.045 (0.5 floors). At QUICK both are below
  1 floor. The FULL floor is expected to be smaller (n_t 200 vs 50), but the effect may still not
  reach 2 floors.

### H4 — Specific behavioural forward transfer is negative at H8 frozen, γ0 = 10

- **Statistic.** Per seed, FT_beh_spec = FT_beh(H8, s_f 1) − FT_beh(H8, s_f 0.1), with
  FT_beh = log((1 + steps fresh B)/(1 + steps B)) (C4, C7).
- **Primary stratum.** γ0 = 10, one reading.
- **Rule.** No floor: training is deterministic given the seed, so the seed SD is already in the
  interval.
  - held: the interval excludes 0, the mean is < 0, and |mean| ≥ log(1.25) ≈ 0.223;
  - not held: the interval excludes 0 and the mean is > 0;
  - otherwise unresolved.
- **Null.** No specific transfer. Supported if the interval lies inside ±log(1.25).
- **Prior signal.** −0.47 [−0.61, −0.33] (95%, n = 3).

### H5 — Channel shares of the A→B loss

- **Statistic.** Per seed, the share of channel c = Δ_c(A1→B) / Δ_total(A1→B), with the exact
  identity total = utility + radius + dimension. Shares can lie outside [0, 1] when channels
  have opposite signs.
- **Primary strata.** H ∈ {4, 8} × s_f ∈ {1, 0.9}: 4 cells at γ0 = 10 (H5a) and 4 at γ0 = 0.1
  (H5b). Each cell is read only if its Δ_total is resolved. γ0 = 1 is reported, not read.
- **H5a, γ0 = 10.** Utility + dimension share ≥ 0.8.
  - held: the lower bound is ≥ 0.8;
  - not held: the upper bound is < 0.8;
  - otherwise unresolved.
- **H5b, γ0 = 0.1.** No channel's share is above 0.5.
  - held: every channel's upper bound is ≤ 0.5;
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
  init→A gain is resolved at that γ0.
- **Rule.**
  - held: the lower bound of ρ is ≥ 0.9;
  - not held: the upper bound is < 0.9;
  - otherwise unresolved.
- **Null.** Restoration is < 0.9, supported by "not held".
- **Prior signal.** γ0 = 10: log α_A(A2) is 0.91–0.94 against A1 = 1.00 and init = −1.02, so ρ is
  about 0.96. γ0 = 0.1: A2 is at or above A1.

## 5. Multiplicity

There are 6 hypotheses with 2 + 2 + 2 + 1 + (4 + 4) + 12 primary readings. Within a hypothesis,
intersection–union applies and needs no correction. Under intersection–union the
hypothesis-level p-value would be the **max** over strata, not the min.

Across the 6 hypotheses: Bonferroni. Every primary interval is a t-interval at
1 − 0.05/6 ≈ 99.17% ~~(with n = 8, t(1 − 0.05/12; df 7) ≈ 4.03)~~ in place of 95%. This applies to
every interval named in §3–§4, including the H5/H6 bounds and the equivalence checks.

> **Erratum 2026-11-09.** §5 stated t ≈ 4.03, which is the 99.75% quantile. The signed rule
> (two-sided 1 − 0.05/6, df = 7) gives t = 3.636, which the readings cell used. Found after the
> FULL results existed. The rule governs.
>
> Consequence: H5b is reported as "held, at the margin (upper bound 0.4992 vs 0.5); would be
> unresolved at 4.03".

## 6. What is not confirmatory

All other tables and figures in the notebook are observations. That includes:
- refit and bottleneck levels;
- phase-to-phase kernel change;
- ρ_c;
- forward transfer outside H4;
- γ0 = 1 outside H2 and H6.

## 7. Execution

The run happens only after this document is signed. All code changes land first. After that,
nothing in `aba_lib.py` changes, and any change voids the run.

1. Done: bottleneck n = 16 is calibrated on grime (job 4316), entered into C9 and `ABAConfig`, committed and pushed.
2. Done: timing on grime (job 4316) sized the job (`sbatch/aba_full.sh`).
3. One sbatch job, `sbatch/aba_full.sh`, runs the 219 arms, then executes the notebook with
   `ABA_MODE=full` (floors at K = 8).
4. Readings are computed by the notebook's pre-commit cell, which implements §3–§4 verbatim, and
   are reported as computed.

## 8. Amendment 2026-11-09: replication on seeds 8–15, same rules

Decided by Kati on 2026-11-09, after the FULL results (job 4349) existed. Recorded here before the
replication was submitted.

**Why.** FULL seeds 0–2 share their `seed_base` draws with the QUICK seeds (n = 3) whose signal
shaped H1–H6 and their thresholds. So 3 of the 8 units are not independent of the hypotheses. The
signed rules are therefore run again, unchanged, on fresh seeds 8–15.

**What is the same.** Everything in §2–§6, including the §5 erratum (t = 3.636, df = 7). That
covers:
- the design and the 219-arm grid;
- the floors (K = 8, n_t = 200);
- the resolution, ratio and not-read rules;
- the hypotheses, strata and thresholds;
- the 99.17% intervals.

The readings cell (notebook §8) is the same code.

**What changes.**
- **Seeds.** `ABAConfig.seed_offset = 8` puts the grid on seeds 8–15.
- **First seed of the run.** Wherever §3 and the notebook say "seed 0", this means the first seed of
  the run, so **seed 8** here. That covers the trained floor, the fresh-noise identity arms and the
  manifold display. The replication then shares no draw with the original run.
- **Code.**
  - `seed_offset` (commit `80044e4`) changes the `aba_lib` hash from `2528909f2feaf3f3` to
    `4e110835b8fc2863`.
  - §7 says any lib change voids the run. That rule is met by a check: grime job 4395 reran the
    FULL arm seed 0, γ0 = 10, H8, s_f 0.1 under the new hash. It is **bit-identical** to job
    4349's row in every field except `lib_hash`, `stamp` and `timing`.
  - At offset 0, `all_specs` gives job 4349's 219 keys in the same order, and `key_fields()` is
    unchanged.
- **Job.** `sbatch/aba_full_rep.sh` runs the grid with `--seed-offset 8` and the notebook with
  `ABA_SEED_OFFSET=8`.

**Reporting.**
- The two runs are reported **side by side** (§9): original (seeds 0–7, job 4349) and
  replication (seeds 8–15).
- They are **not pooled**, and no combined reading is computed.
- Each run is read by §3–§4 on its own.

## 9. Results (2026-11-09): original and replication, side by side

Every value below is parsed from the §8 output of the two executed notebooks, by a script, not
retyped. Each run is read by §3–§4 on its own. The runs are **not pooled** and no combined reading
exists (§8 amendment).

**Runs:**
- **Original:** seeds 0–7, grime job 4349, commit `a7f0398`, lib `2528909f2feaf3f3`. Run record:
  `notebooks/core/runs/aba_full_seeds0-7_job4349.ipynb`.
- **Replication:** seeds 8–15, grime job 4396, commit `8407a7d`, lib `4e110835b8fc2863`. Run record:
  `notebooks/core/runs/aba_full_seeds8-15_job4396.ipynb`.

**Checks in both runs:**
- all validation checks pass;
- 219 of 219 arms are usable, on a clean tree;
- t = 3.636 (df = 7; §5 erratum);
- the floor is computed on the first seed of the run (seed 0 and seed 8 respectively; §8).

### 9.1 Summary

| hypothesis | reading, seeds 0–7 | read strata | reading, seeds 8–15 | read strata |
|---|---|---|---|---|
| H1 | **held** | 2/2 | **held** | 2/2 |
| H2 | **held** | 2/2 | **held** | 2/2 |
| H3 | **held** | 2/2 | **unresolved at floor** | 2/2 |
| H4 | **held** | 1/1 | **held** | 1/1 |
| H5a | **held** | 4/4 | **held** | 4/4 |
| H5b | **held** | 4/4 | **unresolved at floor** | 4/4 |
| H6 | **held** | 12/12 | **held** | 12/12 |

**What did not replicate.** Neither case is a sign reversal; both are resolution or width failures.
- **H3 is unresolved at floor in the replication, in both strata.** Both means are positive and both
  intervals exclude 0:
  - +0.048 [+0.033, +0.062] at γ0 = 0.1;
  - +0.070 [+0.049, +0.090] at γ0 = 10.

  They fall short of the 2-floor criterion, at 1.91 and 1.63 floors. The seed-8 floors are 2.4×
  (γ0 = 0.1) and 1.5× (γ0 = 10) the seed-0 floors, while the means are similar or larger. The
  original run already sat at the margin at γ0 = 10 (+2.26 floors).
- **H5b is unresolved in the replication,** because of its two H4 strata:
  - H4 frozen: the upper bounds are radius 0.51 and dimension 0.48.
  - H4 drift: the upper bounds are radius 0.58 and dimension 0.51.

  Both H8 strata hold again. In the original, H5b held at the margin (upper bound 0.4992 vs 0.5;
  §5 erratum), and it would have been unresolved at t = 4.03.

**What replicated.** H1, H2, H4, H5a and H6 hold again, in every read stratum.

**Floors (observation, not a reading).** The floor is SD(log α_A) over K = 8 estimator seeds on
**one** trained draw. It differed 1.5–2.4× between seed 0 and seed 8 (table 9.2). Since the 2-floor
rule decides H3, a floor estimated on a single draw is a weak point of this design. This is
recorded for the next pre-commit (`docs/29`, ⟨K7⟩); it does not change either reading here.

### 9.2 Floors and the init→A gain (denominators)

| γ0 | floor, seeds 0–7 | floor, seeds 8–15 | init→A gain, seeds 0–7 | init→A gain, seeds 8–15 |
|---|---|---|---|---|
| 0.1 | 0.0103 | 0.0248 | +0.225 [+0.212, +0.239] | +0.254 [+0.215, +0.294] |
| 1 | 0.0169 | 0.0354 | +1.085 [+1.066, +1.104] | +1.116 [+1.049, +1.182] |
| 10 | 0.0293 | 0.0428 | +2.062 [+1.988, +2.137] | +2.059 [+1.960, +2.158] |

### 9.3 Every stratum

Statistics are means with [99.17% CI] over 8 seeds. Strata at γ0 = 1 in H3 and H5 are reported, not
read (§4).

| hypothesis | stratum | seeds 0–7: statistic [99.17% CI] | reading | seeds 8–15: statistic [99.17% CI] | reading |
|---|---|---|---|---|---|
| H1 | H4, s_f 1 | d = f(10) − f(0.1) = +0.277 [+0.154, +0.400] | held | d = f(10) − f(0.1) = +0.332 [+0.229, +0.436] | held |
| H1 | H4, s_f 0.9 | d = f(10) − f(0.1) = +0.313 [+0.225, +0.401] | held | d = f(10) − f(0.1) = +0.333 [+0.233, +0.434] | held |
| H2 | d₁ = g(1) − g(0.1) | +0.859 [+0.837, +0.882] | held | +0.861 [+0.828, +0.894] | held |
| H2 | d₂ = g(10) − g(1) | +0.977 [+0.912, +1.043] | held | +0.944 [+0.878, +1.009] | held |
| H3 | γ0 0.1, H0, s_f 0.9 | +0.0347 [+0.0150, +0.0545] (+3.37 floors) | held | +0.0475 [+0.0328, +0.0621] (+1.91 floors) | unresolved at floor |
| H3 | γ0 1, H0, s_f 0.9 (reported, not read) | +0.0685 [+0.0424, +0.0946] (+4.04 floors) | not read (γ0 = 1) | +0.0783 [+0.0635, +0.0931] (+2.21 floors) | not read (γ0 = 1) |
| H3 | γ0 10, H0, s_f 0.9 | +0.0662 [+0.0412, +0.0912] (+2.26 floors) | held | +0.0696 [+0.0490, +0.0901] (+1.63 floors) | unresolved at floor |
| H4 | γ0 10, H8, s_f 1 | FT_beh_spec = -0.417 [-0.606, -0.229] | held | FT_beh_spec = -0.459 [-0.591, -0.328] | held |
| H5a | H4, s_f 1 | u + d share +0.944 [+0.915, +0.973] | held | u + d share +0.924 [+0.897, +0.950] | held |
| H5a | H4, s_f 0.9 | u + d share +0.952 [+0.923, +0.980] | held | u + d share +0.933 [+0.889, +0.976] | held |
| H5a | H8, s_f 1 | u + d share +0.935 [+0.901, +0.969] | held | u + d share +0.920 [+0.902, +0.938] | held |
| H5a | H8, s_f 0.9 | u + d share +0.936 [+0.905, +0.966] | held | u + d share +0.929 [+0.904, +0.954] | held |
| H5b | H4, s_f 1 | u +0.22 [+0.12, +0.32]; r +0.36 [+0.27, +0.45]; d +0.42 [+0.36, +0.48] | held | u +0.18 [+0.07, +0.29]; r +0.38 [+0.26, +0.51]; d +0.44 [+0.40, +0.48] | unresolved at floor |
| H5b | H4, s_f 0.9 | u +0.20 [+0.09, +0.32]; r +0.38 [+0.26, +0.50]; d +0.42 [+0.35, +0.48] | held | u +0.15 [-0.03, +0.33]; r +0.40 [+0.22, +0.58]; d +0.45 [+0.40, +0.51] | unresolved at floor |
| H5b | H8, s_f 1 | u +0.22 [+0.16, +0.28]; r +0.38 [+0.35, +0.40]; d +0.40 [+0.36, +0.45] | held | u +0.18 [+0.12, +0.23]; r +0.40 [+0.37, +0.43]; d +0.42 [+0.39, +0.46] | held |
| H5b | H8, s_f 0.9 | u +0.21 [+0.15, +0.26]; r +0.38 [+0.35, +0.41]; d +0.41 [+0.36, +0.46] | held | u +0.16 [+0.11, +0.21]; r +0.41 [+0.38, +0.44]; d +0.43 [+0.39, +0.47] | held |
| H5 (γ0 = 1) | H4, s_f 1 | u +0.43 [+0.39, +0.47]; r +0.16 [+0.12, +0.20]; d +0.41 [+0.39, +0.42] | not read (γ0 = 1) | u +0.41 [+0.37, +0.45]; r +0.18 [+0.13, +0.23]; d +0.41 [+0.39, +0.43] | not read (γ0 = 1) |
| H5 (γ0 = 1) | H4, s_f 0.9 | u +0.44 [+0.38, +0.49]; r +0.18 [+0.11, +0.24]; d +0.39 [+0.37, +0.41] | not read (γ0 = 1) | u +0.41 [+0.36, +0.46]; r +0.19 [+0.14, +0.25]; d +0.40 [+0.37, +0.42] | not read (γ0 = 1) |
| H5 (γ0 = 1) | H8, s_f 1 | u +0.42 [+0.39, +0.46]; r +0.17 [+0.13, +0.20]; d +0.41 [+0.40, +0.42] | not read (γ0 = 1) | u +0.39 [+0.37, +0.40]; r +0.20 [+0.19, +0.22]; d +0.41 [+0.40, +0.42] | not read (γ0 = 1) |
| H5 (γ0 = 1) | H8, s_f 0.9 | u +0.41 [+0.38, +0.45]; r +0.17 [+0.14, +0.20]; d +0.41 [+0.40, +0.43] | not read (γ0 = 1) | u +0.38 [+0.35, +0.41]; r +0.20 [+0.18, +0.23]; d +0.42 [+0.40, +0.43] | not read (γ0 = 1) |
| H6 | γ0 0.1, H4, s_f 1 | ρ = +1.198 [+1.149, +1.246] | held | ρ = +1.173 [+1.136, +1.209] | held |
| H6 | γ0 0.1, H4, s_f 0.9 | ρ = +1.197 [+1.157, +1.237] | held | ρ = +1.164 [+1.126, +1.202] | held |
| H6 | γ0 0.1, H8, s_f 1 | ρ = +1.127 [+1.097, +1.157] | held | ρ = +1.125 [+1.095, +1.154] | held |
| H6 | γ0 0.1, H8, s_f 0.9 | ρ = +1.099 [+1.079, +1.119] | held | ρ = +1.079 [+1.045, +1.113] | held |
| H6 | γ0 1, H4, s_f 1 | ρ = +1.005 [+0.982, +1.029] | held | ρ = +1.017 [+0.991, +1.044] | held |
| H6 | γ0 1, H4, s_f 0.9 | ρ = +0.991 [+0.969, +1.013] | held | ρ = +1.006 [+0.977, +1.034] | held |
| H6 | γ0 1, H8, s_f 1 | ρ = +1.011 [+0.989, +1.033] | held | ρ = +1.016 [+0.994, +1.038] | held |
| H6 | γ0 1, H8, s_f 0.9 | ρ = +0.966 [+0.942, +0.989] | held | ρ = +0.969 [+0.944, +0.993] | held |
| H6 | γ0 10, H4, s_f 1 | ρ = +0.963 [+0.937, +0.989] | held | ρ = +0.978 [+0.924, +1.033] | held |
| H6 | γ0 10, H4, s_f 0.9 | ρ = +0.963 [+0.934, +0.992] | held | ρ = +0.978 [+0.930, +1.027] | held |
| H6 | γ0 10, H8, s_f 1 | ρ = +0.967 [+0.932, +1.002] | held | ρ = +0.975 [+0.927, +1.023] | held |
| H6 | γ0 10, H8, s_f 0.9 | ρ = +0.953 [+0.931, +0.976] | held | ρ = +0.959 [+0.916, +1.001] | held |
