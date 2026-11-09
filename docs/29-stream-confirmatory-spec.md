# 29 — Confirmatory-test specification for the 1M stream grid (DRAFT)

```
Status:  DRAFT, unsigned. First draft 2026-11-09 (docs/25 §8 step 6).
         Redrafted twice on 2026-11-09 after Kati's reviews: SESOI with
         four-category reading (S0), four confirmatory tests T1–T4 at
         Bonferroni 98.75%, readings per configuration and per (j, t) at
         p = 8 (K8), T2 = task 0 after the stream (K2a), T5/T6 descriptive.
         K9 (stopping) carried the docs/31 outcome for Kati's decision.
         Amended 2026-11-09 (Kati): K9 decided (matched-loss stopping;
         every γ₀ claim scoped "at matched current-task fit"); T1(a)
         moved from h 4 frozen to h 4 drift.
Slot:    docs/24 §4 core (the stream paradigm, second after A–B–A).
Binds:   the grid of docs/25 §1 (1M, N = 300, M = 150, n_t = 200, n = 8).
Gate:    no stream grid runs before Kati signs docs/26 and this document
         (docs/25 §1, amended).
```

Signed: ______________________ date: ______________

## 1. What this commits to

There are **four confirmatory tests, T1–T4**, one Bonferroni family:
- C1, C2 and C3 of `docs/24` §3;
- the forward-transfer primary.

Every primary interval is a two-sided t-interval over the n = 8 units at **1 − 0.05/4 = 98.75%** (df = 7).
The critical value is computed in code. Every other table and figure from the stream grid is descriptive and
labelled as such in its caption.

**Why T5/T6 are descriptive here (Kati, 2026-11-09).** The two richness-axis agreements were to repeat C1
and the fraction-removed contrast against measured richness ΔK_u instead of γ₀. Within one learner the ΔK_u
ordering matches γ₀ by construction: ΔK_u after A was 0.05, 0.69 and 7.8 at γ₀ 0.1, 1 and 10 in A–B–A, a
factor of about 170. So the agreement tests cannot fail. They are reported descriptively here (§6) and
reserved for the multi-learner extension, where the orderings can disagree.

Each test names:
- its configuration (γ₀, h, s_f);
- its **(j, t) cell**, meaning past task j read after task t;
- the paired contrast per unit;
- the SESOI and the predicted direction;
- the reading rule (§3).

**Setting discipline (Kati, items 8, 9 and 11).**
- Every reading is per configuration and per (j, t). There is **no averaging over lags or over
  positions** within a stream, and **no pooling across configurations**.
- "Held in X" means held in X only.
- Every reading is stated against the **realised** label similarity S_r(j, t) and the **realised**
  input-centre correlation ρ_in(j, t), per seed.
- Every stream report carries a setting card per configuration (§2a).

**What the A–B–A runs contribute.** Prior signal only, never evidence:
- job 4349, seeds 0–7;
- job 4396, seeds 8–15, the confirmatory set;
- job 4458, the stopping-rule ablation;
- see `docs/28` §9, `docs/30` and `docs/31`.

The specific points:
- **C2.** "Cliff, not dose" is not supported at lag 1 (`docs/24` C2, amended). The long-lag cliff is
  descriptive (§6).
- **C1.** The γ₀ gap in fraction removed held at h 4 frozen and h 4 drift, and not at h 8. **T1's choice of
  h = 4 was informed by A–B–A.**
- **Exposure.** Under matched-loss stopping, exposure differs 40–60-fold between γ₀ 0.1 and 10. Under a fixed
  budget (docs/31), the H1 contrast shrank at h 4 frozen (unresolved) and at h 4 drift (still positive).
  Matched-loss stopping stays (K9), so **every γ₀ claim in this document holds at matched current-task fit**,
  and only there. The A–B–A 2×2 that separates over-training of A from over-exposure on B is `docs/32`.

## 2. Design (from docs/25 §1)

| item | value |
|---|---|
| learner | 1M: one hidden ReLU layer, N = 300, μP (`aba_lib.OneModuleNet`), u(0) = 0; full-batch GD; MSE; matched-loss stopping at loss ≤ 0.05 = 0.1·L(0), loss checked before each update, so no update at criterion (C5). See K9. |
| data | P = 16, d = 150, D = 4, R = 1, M = 150 training points per manifold, plus a separate test realisation per task |
| stream | T = 16 tasks from `src.manifolds.streams.make_stream`. Each step changes the labels by a fixed Hamming h and the input centres by AR(1) at s_f. Frozen input (s_f = 1) copies the centres and re-realises the noise; this differs from A–B–A's exact copy (C3). Labels come from the post-E16 default generator. |
| grid (K0) | γ₀ {0.1, 1, 10} × h {0, 2, 4, 8} × s_f {1, 0.9, 0.1} = 36 configurations × n = 8 = **288 arms**. It runs only after ONE design-scale arm with every measurement has been timed on grime (`sbatch/stream_timing.sh`), and the grid is sized with the measured 3–4× parallel slowdown. |
| unit | one draw: the stream (arrangements, labels, test points), W(0) and the probes, keyed by the draw id. Paired across γ₀, h and s_f within a draw. Task 0 is identical across h (E16) and across s_f. |
| provenance | clean tree, one commit; every row carries the git stamp and `stream_hash` (§8) |

### 2b. Measurement (fixed definitions)

The code is `notebooks/core/stream_lib.py` (§8).
- **Backward capacity.** `log α_j(t)` is the retained fixed-y GLUE capacity (`full_P`, n_t = 200) of task
  j's labels on task j's training manifolds, through the hidden representation after task t. It is measured
  for every j ≤ t, and for every j at init. The channel split is utility, radius, dimension and residual, the
  same as `aba_lib.channels`. This is done for every (j, t−1 → t) and every (j, j → t).
- **Fraction removed.** f_j(t) = (log α_j(j) − log α_j(t)) / (log α_j(j) − log α_j(init)).
- **Forward, geometric.** On task t+1's manifolds, FT_geo(t) = Δlog α_{y_{t+1}}(init → t) − mean_k
  Δlog α_{y*_k}(init → t). The four neutral probes y*_k are balanced, satisfy s_r(y*_k, y_t) = 0, and are
  distinct (up to sign) from every stream label and from each other. This is A–B–A C6 with task t as A.
- **Forward, behavioural.** FT_beh(t) = log((1 + steps of a fresh learner on task t+1)/(1 + stream steps
  on task t+1)). The fresh learner starts from the same W(0) and γ₀. There is one fresh learner per
  t = 0…14.
- **Centre geometry.** `rho_c_signed` (normalised, signed) of task j's manifolds after task t, for every
  (j, t).
- **Richness.** The analytic NTK on a fixed probe set (20 points per manifold of task 0) at init and after
  every task: ΔK_u, ΔK_W and the K_W share (`aba_lib.richness_snapshot`).
- **Behaviour.** The test loss and sign accuracy of every task at every recorded step; the steps per task
  (exposure).
- **Relations.** The realised S_r(j, t) = s_r(y_j, y_t), and the realised ρ_in(j, t) = the mean cosine of
  matched centres, for every pair, the diagonal next to the main one included (not the nominal s_f).
- **Floor (diagnostic).** SD(log α_0) over K = 8 estimator keys on the representation after task 0, per draw.

### 2a. Setting card (required in every stream report)

One card per configuration:
- the nominal h and s_f;
- per seed, the realised **S_r(j, t)** and **ρ_in(j, t)** for every (j, t) that is read, and in full as a
  supplementary table;
- **exposure per learner**: the updates per task for each γ₀ and seed, stating that matched-loss stopping lets
  the learner set its exposure;
- the line: **"Streams use the default label generator after E16 (`2bf0629`). They are not comparable draw
  for draw with streams built before it."**

## 3. Units, SESOI and the reading rule (S0, decided)

- **Effects** are in log α, except where a table below names another unit.
- **Unit.** One draw contributes **one value per (j, t) cell**. Values from different (j, t) cells are never
  averaged.
- **Interval.** The two-sided t-interval over the n = 8 units at **98.75%** (df = 7), computed in code. The
  **sign count**, units with the sign of the mean out of n, is reported with every interval and decides
  nothing.
- **SESOI** (decided, Kati 2026-11-09):

  | quantity | SESOI | basis |
  |---|---|---|
  | Δlog α (capacity) | **0.05** | A 5% change in capacity; above every single-draw floor in A–B–A FULL (0.010–0.047) |
  | ρ_c (signed, normalised) | **0.02** | A–B–A Δρ_c(A → B) is ≤ 0.006 at γ₀ 0.1, and 0.022–0.061 at γ₀ 10 under jump |
  | FT_beh (log step ratio) | **log 1.25 ≈ 0.223** | As `docs/28` H4 |
  | channel share, fraction removed | **0.10** | As the `docs/28` H1 null band |

- **Four-category reading** (Lakens' equivalence / minimum-effect framework):

  | category | the 98.75% interval … | reading |
  |---|---|---|
  | **meaningful** | lies entirely beyond ±SESOI **in the predicted direction** | **held** |
  | meaningful, opposite | lies entirely beyond ±SESOI in the opposite direction | **not held** |
  | **nonzero, possibly trivial** | excludes 0 but overlaps the SESOI band | not held, not supporting the null |
  | **equivalent** | lies entirely inside ±SESOI | supports the null |
  | **inconclusive** | none of the above | unresolved |

- **Ratios** (shares, fraction removed) are read only when their denominator Δlog α is meaningful.
  Otherwise the cell is **not read**.
- **Several parts.** A test with several parts holds only if every read part is meaningful
  (intersection–union). It is not held if any part is meaningful in the opposite direction. The number of read
  parts is reported.
- **Floor.** SD(log α) over K = 8 estimator keys per draw, pooled as the root mean square over the n draws. It
  is a diagnostic only, reported beside every capacity reading. It replaces the single-draw floor, which
  differed 1.5–2.4× between seed 0 and seed 8 in A–B–A.

## 4. The four tests

**Position (K8, decided).** Every cell is at the named position **p = 8**:
- "lag 1 at p = 8" means j = 7 read after task t = 8;
- "forward at p = 8" means from task 8 to task 9.

The reasons:
- p = 1 (the first boundary) is covered by A–B–A;
- position effects belong to the order arm (`docs/23`).

All other positions and lags are reported descriptively per (j, t), each with its own realised S_r and ρ_in.

### T1 — C1: the learner sets magnitude and channel

- **Claim** (`docs/24` §3). At matched current-task fit, richness increases forgetting and shifts it from
  radius to alignment/utility.
- **Cell (K1).** h = 4, (j, t) = (7, 8), γ₀ = 10 vs γ₀ = 0.1. h = 4 was chosen from A–B–A, where the γ₀ gap
  in fraction removed held at h 4 frozen and h 4 drift, and not at h 8.
- **Parts.** All are paired per unit, with predicted direction > 0:
  - **(a) magnitude.** d_f = f_7(8; γ₀ 10) − f_7(8; γ₀ 0.1), at **h 4 drift (s_f 0.9)**. SESOI 0.10.
    **[Amended 2026-11-09, Kati]** ~~at h 4 frozen~~. The reason: h 4 frozen is protocol-fragile. Under a
    fixed budget (docs/31) its contrast fell to +0.100 [−0.076, +0.277], unresolved. h 4 drift kept its
    sign under both stopping rules (+0.333 → +0.184). d_f at h 4 frozen is reported (§6), not read.
  - **(b) channel interaction.** q = the radius share of Δlog α_7(7 → 8), and
    `I = [q(0.1) − q(10)]_frozen − [q(0.1) − q(10)]_jump`, at h 4. SESOI 0.10.
  - **(c) utility-share contrast** (from A–B–A H5b). u = the utility share of Δlog α_7(7 → 8), and
    d_u = u(10) − u(0.1), at h 4 frozen. SESOI 0.10.
- **Read if** Δlog α_7(7 → 8) is meaningful in every cell that enters a ratio.
- **Held** if (a), (b) and (c) are each meaningful. **Not held** if any is meaningful the opposite way.
- **Null.** (a), (b) and (c) are each equivalent.
- **Prior signal (A–B–A replication, lag 1 at p = 1).**
  - d_f = +0.333 at h 4 drift, and +0.184 under the fixed budget (same sign, docs/31).
  - d_f = +0.332 at h 4 frozen, and +0.100 [−0.076, +0.277] under the fixed budget (unresolved).
  - At h 4 frozen, the utility share is 0.18 at γ₀ 0.1 and 0.50 at γ₀ 10.
  - The radius share is 0.38–0.41 vs 0.07–0.08 (h 4 and h 8, frozen and drift).

### T2 — C2: the stream sets sign (K2, K2a)

- **Claim.** Transfer to a past task appears when the task repeats under slowly drifting input.
- **Cell.** h = 0 drift (s_f 0.9), **at γ₀ = 0.1 and at γ₀ = 10** (two strata).
- **Statistic.** (j, t) = (0, 15): task 0 after the whole stream, `log α_0(15) − log α_0(0)`. This is
  accumulation. Predicted direction > 0, SESOI 0.05.
- **Held** if it is meaningful in both strata. **Not held** if it is meaningful the opposite way in either.
- **Null.** Equivalent in both strata.
- **Reported beside it, not read:**
  - lag 1 at p = 8, Δlog α_7(7 → 8) at h 0 drift;
  - the jump control, the same statistic at h 0 jump at both γ₀.
- **Prior signal.** A–B–A H3 (h 0 drift, lag 1) gave +0.048 and +0.070 in the replication. Both intervals
  exclude 0, but at SESOI 0.05 they would read "nonzero, possibly trivial". Under Adam the gain did not
  appear, which is why C2 is scoped to gradient descent.

### T3 — C3: centre geometry dissociates from capacity (K3)

- **Claim.** Input change drives centre convergence. Task change drives graded divergence while capacity
  steps.
- **Cell.** γ₀ = 10, (j, t) = (7, 8), signed normalised ρ_c, SESOI 0.02.
  - **(a) input.** Δρ_c,7(7 → 8) at h 0 jump minus at h 0 frozen. The predicted direction is convergence
    (> 0).
  - **(b) task.** Δρ_c,7(7 → 8) at h 8 frozen minus h 2 frozen. The predicted direction is divergence
    (< 0). Read jointly with the capacity step Δlog α_7(7 → 8) at h 2 frozen minus h 0 frozen, which must be
    meaningful (< 0, SESOI 0.05).
- **Held** if (a) and (b) are each meaningful, with (b)'s capacity step meaningful. **Not held** if either is
  meaningful the opposite way.
- **Prior signal (A–B–A replication, γ₀ 10).** Δρ_c(A → B) was +0.022 at h 0 jump and −0.026 at h 8
  frozen (interval includes 0).

### T4 — Forward-transfer primary (K4: as A–B–A H4)

- **Cell.** γ₀ = 10, forward at p = 8 (task 8 → task 9).
- **Statistic.** FT_beh_spec = FT_beh(h 8 frozen) − FT_beh(h 8 jump), paired per unit. Predicted direction
  < 0, SESOI log 1.25.
- **Held** if meaningful. **Not held** if meaningful the opposite way.
- **Null.** Equivalent.
- **Reported, not read.** The association between the channels of FT_geo and FT_beh.
- **Prior signal.** A–B–A H4 gave −0.459 [−0.591, −0.328] (replication).
- **Exposure caveat.** FT_beh is a step ratio, defined under matched-loss stopping (K9).

## 5. Multiplicity

- There are four tests in one family: **Bonferroni**, family-wise α = 0.05, every primary interval at
  1 − 0.05/4 = 98.75%. That includes the equivalence checks and the parts of T1 and T3.
- Within a test, intersection–union applies and needs no correction.
- Critical values are computed in code.
- There is no bootstrap. A sign count is reported.

## 6. Descriptive (not confirmatory)

Reported per configuration and per (j, t), never averaged over lags or positions:
- **T5 and T6, the richness-axis agreements.** C1 and the fraction-removed contrast are recomputed against
  ΔK_u instead of γ₀. Descriptive within one learner (§1), and reserved for the multi-learner extension.
- T1(a) at h 4 frozen, d_f at (7, 8) (reported, not read; amended 2026-11-09);
- every (j, t) other than the named cells: all positions, all lags;
- T2's lag 1 at p = 8 and its jump control;
- the long-lag cliff `Δ(h 2) − Δ(h 8)` at long lags;
- every γ₀ level other than the named strata, and h = 2 outside T3(b);
- Δρ_c magnitudes outside T3;
- the richness trajectory r_0 … r_15;
- savings and zero-shot loss;
- forward transfer outside T4's cell;
- the diagnostic floor;
- order and spacing, which have their own pre-commits (`docs/23`, `docs/27`).

## 7. Decisions

**Resolved by Kati, 2026-11-09:**

| # | decision |
|---|---|
| K0 | The fallback grid (288), after timing ONE arm with every measurement, using the measured parallel slowdown. |
| K1 | T1 at h = 4 (informed by A–B–A, stated in T1). |
| K2 | T2 at h = 0 drift, both γ₀; jump as a reported control; the long-lag cliff is descriptive. |
| K2a | T2 primary = task 0 after the whole stream, (0, 15). Lag 1 at p = 8 is reported. |
| K3 | Signed normalised ρ_c with SESOI 0.02. T3(b) is the h 8 vs h 2 contrast. |
| K4 | As A–B–A H4. |
| K5, K5b | T5/T6 are descriptive for this grid (the ΔK_u ordering equals the γ₀ ordering by construction). |
| K6 | Bonferroni, a family of four: 98.75%. |
| K7 | t-intervals plus a sign count; no bootstrap. |
| K8 | Primary position p = 8, named per test; all positions reported descriptively. |
| S0 | The SESOI values in §3; the four-category reading. |
| T1 | The utility-share contrast (γ₀ 10 vs 0.1) is T1(c). |
| FT | Forward measured against init, with neutral probes at s_r(·, y_t) = 0 (§2b). |
| K9 | Matched-loss stopping stays: no stratum flipped in the docs/31 ablation (job 4458), per its pre-committed rule. Every γ₀ claim is scoped "at matched current-task fit". Exposure is on every card. The disentangling 2×2 is docs/32. |
| T1(a) | The primary cell is h 4 drift, not h 4 frozen (protocol-fragile under docs/31). h 4 frozen is reported, not read. |

Nothing is open for decision.

## 8. Execution (after signing)

**Code (Kati, 2026-11-09).** `notebooks/core/stream_lib.py`, on top of `aba_lib`:
1. `stream_lib` **imports from `aba_lib` only**, with no copied or modified functions. Streams come from
   `src.manifolds.streams.make_stream`.
2. The cache key `stream_hash` hashes `aba_lib.py`, `stream_lib.py` and `src/manifolds/streams.py`, plus
   `generator.py` and `dichotomies.py`, which `make_stream` calls.
3. **Parity first.** One A–B–A arm's exact A and B, run as a T = 2 stream through `stream_lib`, must equal
   `aba_lib.run_arm` bit for bit in every quantity both produce (`stream_cluster.py parity`).
4. Unit tests in `tests/test_stream_lib.py`: (j, t) indexing, realised ρ_in, the fresh-learner baseline,
   cache-key coverage, and parity at QUICK scale.

**Steps.**
1. Kati signs docs/26 and this document.
2. Submit `sbatch/stream_timing.sh`: FULL parity, then ONE design-scale arm with every measurement (γ₀ 0.1,
   h 8, jump), sequential and timed. Size the 288-arm grid from it with the 3–4× parallel slowdown.
3. **`aba_lib` freeze.** docs/32 changes `aba_lib` (a per-phase stopping option), which changes
   `stream_hash`. The stream timing job and the stream grid run on the same commit, at or after the docs/32
   code commit. No edit to any hashed file may land between them.
4. One commit, clean tree. The readings and setting cards are computed by code that implements §2–§5
   verbatim, and are reported as computed.

**Backlog (Kati, 2026-11-09).** After the stream grid, move `aba_lib` + `stream_lib` into `src` as one
package, with one implementation and one bit-for-bit check. Mark the 2M paths legacy.
