# 31 — A–B–A stopping-rule ablation: pre-commit

**Status: committed 2026-11-09, before the ablation job was submitted (`2ea0f16`). Results added 2026-11-09 (§7, §8): grime job 4458.**
- The design and budget were decided by Kati on 2026-11-09 (item 10 of her review, budget rule "max over
  arms").
- The ablation must be read before docs/29 is signed.

## 1. Why

docs/28 trains every phase to matched loss: training stops at loss ≤ 0.05. The learner therefore sets
its own exposure, meaning its number of updates. The setting cards in docs/30 §4 show how different this is.
On seeds 8–15, the median updates are:

| phase | γ0 0.1 | γ0 10 |
|---|---|---|
| A1 | 1938 | 51 |
| B at h 8 frozen | 2137 | 36 |

So every γ0 contrast in docs/28 is also an exposure contrast. This ablation fixes exposure, identical for every
γ0, and asks whether the γ0 effects keep their sign. Fixed training per task is also the standard
continual-learning protocol.

## 2. Design (fixed)

Identical to the docs/28 replication (seeds 8–15, job 4396) except for stopping.

| item | value |
|---|---|
| stopping | `ABAConfig.stopping = "fixed"`: exactly the budget of full-batch GD updates per phase, whatever the loss. The loss is checked before each update, as in C5. |
| budget | **A1 2300, B 2400, A2 2000**; fresh B takes the B budget. The same for every γ0. |
| budget rule | The largest criterion step count of any replication arm, rounded up. Measured maxima: A1 2266, B 2373, A2 1973, fresh B 2365. So every learner reaches loss ≤ 0.05 in every phase, and the ratios stay readable. |
| design, seeds | The same 219 specs: γ0 {0.1, 1, 10} × h {0, 4, 8} × s_f {1, 0.9, 0.1} × seeds 8–15, plus 3 fresh-noise arms at seed 8. |
| everything else | As docs/28 §2: data, learner, lr, capacity measurement, probes, n_t = 200. |
| floors | Recomputed under the fixed budget, on seed 8, K = 8 estimator seeds, after A (`aba_cluster.py floors`). The representation after A is not the criterion one. |
| code | `aba_lib` with the `stopping` option (commit `2245806`, lib `0ee97f82b6bee621`). Criterion mode is unchanged: the job-4349 bit-identity check is rerun (§6). |
| usable arm | Every phase ran its full budget with a finite loss. |

**Accepted trade-off.** Exposure is matched; fit is not. Rich learners train roughly 50–100× past criterion,
so their final loss is far below 0.05. The final loss per γ0 and phase is reported (§5) so that this is
visible.

## 3. Readings

- H1, H2, H5a and H6 are read per stratum, by the docs/28 §3–§4 rules: t = 3.636, the 2-floor resolution, and
  the not-read rule for ratios.
- The fixed-budget floors are used.
- The same code reads both runs: `aba_summary.readings`. It is checked to reproduce every docs/28 §9.3 reading
  of these four hypotheses in both criterion runs.
- **Not read:**
  - H4 and anything step-based (FT_beh, savings): under equal budgets these are ratios of equal numbers.
  - H3 and H5b: not part of the question.

## 4. Pre-committed reading

Each stratum is compared with the same stratum of the criterion replication (seeds 8–15, docs/28 §9.3).

- **H1 and H2**, which are contrasts:
  - **same sign**: the fixed-budget contrast is resolved with the criterion sign;
  - **sign flip**: it is resolved with the opposite sign;
  - **unresolved**: otherwise.
- **H5a and H6**, which are thresholds:
  - **same reading**: the same side of the threshold (held / held, or not held / not held);
  - **flip**: held ↔ not held;
  - **unresolved**: the fixed-budget reading is unresolved.
- A stratum not read in either run is reported as **not read**.
- **Interpretation, per stratum only:**
  - Same sign or same reading: the γ0 effect in that stratum is **not driven by exposure**.
  - A sign flip or flip in **any** stratum is reported as a finding: the effect in that stratum depends on
    exposure.
  - Unresolved is neither agreement nor a flip.
- Strata are not pooled, and no cross-stratum summary is formed.

**Expected direction (stated before running).**
- Under matched-loss stopping, the lazy learner (γ0 0.1) got 60–70× more updates on B than the rich one (h 4 frozen 1493 vs 21; h 8 frozen 2137 vs 36)
  (γ0 10). Yet B removed a smaller share of A's gain from the lazy learner: h 4 frozen 0.39 vs 0.72, h 4 drift
  0.26 vs 0.60.
- So exposure plausibly worked *against* the γ0 effect. With equal budgets the rich learner gets far more
  updates on B.
- The H1 difference d = f(10) − f(0.1) at h 4 frozen and h 4 drift is therefore expected to **keep its sign or
  grow**.
- This makes the ablation a robustness check rather than a likely result-changer.

## 5. Reported alongside

The final training loss per configuration, γ0 and phase (A1, B, A2, fresh B), as median [min, max] over
seeds, beside the criterion target of 0.05. Also the fixed-budget floors next to the criterion floors.

## 6. Execution

1. **Check job** (`sbatch/aba_check.sh`, grime job 4453):
   - the criterion seed-0 arm under lib `0ee97f82b6bee621` must be bit-identical to job 4349's row;
   - then one fixed-budget arm (seed 8, γ0 0.1, h 8, frozen) is timed sequentially to size the job.
   - If the arm is not bit-identical, nothing runs.
   - **Result (2026-11-09):** **bit-identical** (17 fields, ignoring lib_hash, stamp and timing).
     The fixed arm took 184 s sequential (train 112 s, GLUE 66 s), peak RSS 124 MB. Its final losses
     were A1 0.036, B 0.047, A2 0.040 and fresh B 0.047, so even the slowest learner ends below 0.05.
     The ablation job is sized from this: about 740 s per arm in parallel (×4), about 85 min for the
     grid, `--time 4 h`.
2. **Ablation job** (`sbatch/aba_ablation.sh`, grime job 4458, commit `2ea0f16`, clean; 219/219 arms usable, every phase status "fixed" with a finite loss; grid wall 5213 s on 32 workers, floors 126 s):
   - the grid with `--seed-offset 8 --stopping fixed --budget 2300,2400,2000`, into
     `results/aba/FULL_0ee97f82b6bee621_fixed.jsonl`;
   - then `floors` with the same flags.
   - The notebook is not executed: its identity and validation checks assume criterion stopping.
3. Copy back the checkpoint and floors. `python notebooks/core/aba_summary.py --ablation <jsonl> <floors>`
   writes §7. The readings are reported as computed.

## 7. Results

<!-- BEGIN GENERATED: ablation results -->
Fixed budget (A1, B, A2) = (2300, 2400, 2000); fresh B takes the B budget. Seeds 8–15. Fixed-run checkpoint `FULL_0ee97f82b6bee621_fixed.jsonl`, floors `floors_0ee97f82b6bee621_fixed_seed8.json`.

**Readings per stratum.** Statistic and reading: docs/28 rules, 99.17% t-intervals, n = 8 (the pre-committed reading). Category: Kati's four categories (docs/29 S0) on the **98.75%** t-interval, descriptive here; SESOI 0.10 for H1 (d_f), 0.05 for H2 (Δlog α), and 0.10 around the threshold for H5a (u + d share − 0.8) and H6 (ρ − 0.9), predicted direction > 0.

| hypothesis | stratum | criterion: statistic | criterion reading | fixed budget: statistic | fixed reading | docs/31 reading | category, criterion | category, fixed |
|---|---|---|---|---|---|---|---|---|
| H1 | h 4, s_f 1 (frozen) | d = f(10) − f(0.1) = +0.332 [+0.229, +0.436] | held | d = f(10) − f(0.1) = +0.100 [-0.076, +0.277] | unresolved at floor | unresolved | meaningful | inconclusive |
| H1 | h 4, s_f 0.9 (drift) | d = f(10) − f(0.1) = +0.333 [+0.233, +0.434] | held | d = f(10) − f(0.1) = +0.184 [+0.058, +0.310] | held | same sign | meaningful | nonzero, possibly trivial |
| H2 | d₁ = g(1) − g(0.1) | +0.861 [+0.828, +0.894] | held | +1.084 [+1.040, +1.127] | held | same sign | meaningful | meaningful |
| H2 | d₂ = g(10) − g(1) | +0.944 [+0.878, +1.009] | held | +0.930 [+0.903, +0.958] | held | same sign | meaningful | meaningful |
| H5a | γ0 10, h 4, s_f 1 (frozen) | u + d share +0.924 [+0.897, +0.950] | held | u + d share +0.981 [+0.955, +1.008] | held | same reading | nonzero, possibly trivial | meaningful |
| H5a | γ0 10, h 4, s_f 0.9 (drift) | u + d share +0.933 [+0.889, +0.976] | held | u + d share +0.968 [+0.930, +1.007] | held | same reading | nonzero, possibly trivial | meaningful |
| H5a | γ0 10, h 8, s_f 1 (frozen) | u + d share +0.920 [+0.902, +0.938] | held | u + d share +1.013 [+0.970, +1.057] | held | same reading | meaningful | meaningful |
| H5a | γ0 10, h 8, s_f 0.9 (drift) | u + d share +0.929 [+0.904, +0.954] | held | u + d share +0.983 [+0.931, +1.035] | held | same reading | meaningful | meaningful |
| H6 | γ0 0.1, h 4, s_f 1 (frozen) | ρ = +1.173 [+1.136, +1.209] | held | ρ = +1.262 [+1.181, +1.342] | held | same reading | meaningful | meaningful |
| H6 | γ0 0.1, h 4, s_f 0.9 (drift) | ρ = +1.164 [+1.126, +1.202] | held | ρ = +1.275 [+1.210, +1.339] | held | same reading | meaningful | meaningful |
| H6 | γ0 0.1, h 8, s_f 1 (frozen) | ρ = +1.125 [+1.095, +1.154] | held | ρ = +1.099 [+1.025, +1.173] | held | same reading | meaningful | meaningful |
| H6 | γ0 0.1, h 8, s_f 0.9 (drift) | ρ = +1.079 [+1.045, +1.113] | held | ρ = +1.136 [+1.068, +1.204] | held | same reading | meaningful | meaningful |
| H6 | γ0 1, h 4, s_f 1 (frozen) | ρ = +1.017 [+0.991, +1.044] | held | ρ = +1.003 [+0.996, +1.011] | held | same reading | nonzero, possibly trivial | nonzero, possibly trivial |
| H6 | γ0 1, h 4, s_f 0.9 (drift) | ρ = +1.006 [+0.977, +1.034] | held | ρ = +0.969 [+0.961, +0.977] | held | same reading | nonzero, possibly trivial | equivalent |
| H6 | γ0 1, h 8, s_f 1 (frozen) | ρ = +1.016 [+0.994, +1.038] | held | ρ = +0.990 [+0.974, +1.007] | held | same reading | nonzero, possibly trivial | nonzero, possibly trivial |
| H6 | γ0 1, h 8, s_f 0.9 (drift) | ρ = +0.969 [+0.944, +0.993] | held | ρ = +0.931 [+0.916, +0.946] | held | same reading | equivalent | equivalent |
| H6 | γ0 10, h 4, s_f 1 (frozen) | ρ = +0.978 [+0.924, +1.033] | held | ρ = +0.970 [+0.959, +0.981] | held | same reading | nonzero, possibly trivial | equivalent |
| H6 | γ0 10, h 4, s_f 0.9 (drift) | ρ = +0.978 [+0.930, +1.027] | held | ρ = +0.973 [+0.964, +0.983] | held | same reading | nonzero, possibly trivial | equivalent |
| H6 | γ0 10, h 8, s_f 1 (frozen) | ρ = +0.975 [+0.927, +1.023] | held | ρ = +0.921 [+0.904, +0.938] | held | same reading | nonzero, possibly trivial | equivalent |
| H6 | γ0 10, h 8, s_f 0.9 (drift) | ρ = +0.959 [+0.916, +1.001] | held | ρ = +0.933 [+0.917, +0.949] | held | same reading | equivalent | equivalent |

**Floors** (SD of log α_A over K = 8 estimator seeds, seed 8, representation after A).

| γ0 | criterion (docs/28 §9.2) | fixed budget |
|---|---|---|
| 0.1 | 0.0248 | 0.0252 |
| 1 | 0.0354 | 0.0360 |
| 10 | 0.0428 | 0.0473 |

**Final training loss per phase** under the fixed budget (criterion target 0.05; median [min, max] over seeds). Exposure is matched; fit is not.

| configuration | γ0 | A1 | B | A2 | fresh B |
|---|---|---|---|---|---|
| h 0, s_f 1 (frozen) | 0.1 | 3.70e-02 [3.0e-02, 4.9e-02] | 1.20e-02 [1.0e-02, 1.4e-02] | 6.06e-03 [5.3e-03, 7.2e-03] | 3.45e-02 [2.8e-02, 4.5e-02] |
| h 0, s_f 1 (frozen) | 1 | 2.24e-04 [1.9e-04, 2.6e-04] | 9.25e-05 [8.4e-05, 9.8e-05] | 5.97e-05 [5.4e-05, 6.5e-05] | 2.09e-04 [1.8e-04, 2.4e-04] |
| h 0, s_f 1 (frozen) | 10 | 6.49e-06 [5.5e-06, 7.2e-06] | 3.83e-06 [3.5e-06, 4.0e-06] | 3.11e-06 [2.8e-06, 3.3e-06] | 6.26e-06 [5.3e-06, 6.9e-06] |
| h 0, s_f 0.9 (drift) | 0.1 | 3.70e-02 [3.0e-02, 4.9e-02] | 1.52e-02 [1.2e-02, 1.6e-02] | 6.02e-03 [5.3e-03, 7.7e-03] | 3.94e-02 [2.8e-02, 4.7e-02] |
| h 0, s_f 0.9 (drift) | 1 | 2.24e-04 [1.9e-04, 2.6e-04] | 1.09e-04 [1.0e-04, 1.4e-04] | 6.47e-05 [6.2e-05, 7.2e-05] | 2.04e-04 [1.7e-04, 2.4e-04] |
| h 0, s_f 0.9 (drift) | 10 | 6.49e-06 [5.5e-06, 7.2e-06] | 5.33e-06 [4.9e-06, 5.9e-06] | 3.94e-06 [3.6e-06, 4.3e-06] | 6.09e-06 [5.4e-06, 6.6e-06] |
| h 0, s_f 0.1 (jump) | 0.1 | 3.70e-02 [3.0e-02, 4.9e-02] | 2.20e-02 [1.8e-02, 2.9e-02] | 1.14e-02 [8.1e-03, 1.4e-02] | 3.02e-02 [2.5e-02, 3.8e-02] |
| h 0, s_f 0.1 (jump) | 1 | 2.24e-04 [1.9e-04, 2.6e-04] | 1.79e-04 [1.5e-04, 2.4e-04] | 1.22e-04 [1.0e-04, 1.8e-04] | 1.93e-04 [1.7e-04, 2.3e-04] |
| h 0, s_f 0.1 (jump) | 10 | 6.49e-06 [5.5e-06, 7.2e-06] | 8.76e-06 [7.9e-06, 1.0e-05] | 7.87e-06 [7.3e-06, 8.7e-06] | 5.91e-06 [5.5e-06, 7.5e-06] |
| h 4, s_f 1 (frozen) | 0.1 | 3.70e-02 [3.0e-02, 4.9e-02] | 2.34e-02 [1.7e-02, 2.7e-02] | 1.96e-02 [1.6e-02, 2.4e-02] | 3.16e-02 [2.5e-02, 4.3e-02] |
| h 4, s_f 1 (frozen) | 1 | 2.24e-04 [1.9e-04, 2.6e-04] | 1.35e-04 [1.2e-04, 1.6e-04] | 1.33e-04 [1.0e-04, 1.6e-04] | 1.95e-04 [1.6e-04, 2.4e-04] |
| h 4, s_f 1 (frozen) | 10 | 6.49e-06 [5.5e-06, 7.2e-06] | 5.42e-06 [5.1e-06, 6.5e-06] | 5.58e-06 [5.4e-06, 7.2e-06] | 6.25e-06 [4.9e-06, 7.3e-06] |
| h 4, s_f 0.9 (drift) | 0.1 | 3.70e-02 [3.0e-02, 4.9e-02] | 2.10e-02 [1.9e-02, 2.8e-02] | 1.53e-02 [1.3e-02, 1.6e-02] | 3.01e-02 [2.6e-02, 4.9e-02] |
| h 4, s_f 0.9 (drift) | 1 | 2.24e-04 [1.9e-04, 2.6e-04] | 1.49e-04 [1.1e-04, 1.7e-04] | 1.13e-04 [9.2e-05, 1.2e-04] | 2.11e-04 [1.7e-04, 2.3e-04] |
| h 4, s_f 0.9 (drift) | 10 | 6.49e-06 [5.5e-06, 7.2e-06] | 6.33e-06 [5.7e-06, 7.0e-06] | 5.90e-06 [5.2e-06, 6.8e-06] | 6.43e-06 [5.3e-06, 7.5e-06] |
| h 4, s_f 0.1 (jump) | 0.1 | 3.70e-02 [3.0e-02, 4.9e-02] | 2.89e-02 [1.8e-02, 3.5e-02] | 1.32e-02 [9.8e-03, 1.7e-02] | 3.15e-02 [2.3e-02, 4.3e-02] |
| h 4, s_f 0.1 (jump) | 1 | 2.24e-04 [1.9e-04, 2.6e-04] | 1.85e-04 [1.6e-04, 2.4e-04] | 1.37e-04 [1.2e-04, 1.5e-04] | 2.09e-04 [1.6e-04, 2.5e-04] |
| h 4, s_f 0.1 (jump) | 10 | 6.49e-06 [5.5e-06, 7.2e-06] | 8.40e-06 [7.5e-06, 9.2e-06] | 8.58e-06 [7.4e-06, 9.1e-06] | 6.29e-06 [5.5e-06, 7.2e-06] |
| h 8, s_f 1 (frozen) | 0.1 | 3.70e-02 [3.0e-02, 4.9e-02] | 4.04e-02 [3.2e-02, 5.0e-02] | 3.70e-02 [3.3e-02, 4.7e-02] | 3.62e-02 [3.0e-02, 4.7e-02] |
| h 8, s_f 1 (frozen) | 1 | 2.24e-04 [1.9e-04, 2.6e-04] | 1.92e-04 [1.7e-04, 2.2e-04] | 1.86e-04 [1.5e-04, 2.3e-04] | 2.00e-04 [1.8e-04, 2.4e-04] |
| h 8, s_f 1 (frozen) | 10 | 6.49e-06 [5.5e-06, 7.2e-06] | 6.06e-06 [5.2e-06, 6.4e-06] | 6.91e-06 [6.4e-06, 8.6e-06] | 6.05e-06 [5.6e-06, 6.8e-06] |
| h 8, s_f 0.9 (drift) | 0.1 | 3.70e-02 [3.0e-02, 4.9e-02] | 3.78e-02 [3.1e-02, 4.5e-02] | 2.54e-02 [1.9e-02, 3.3e-02] | 3.78e-02 [2.8e-02, 4.6e-02] |
| h 8, s_f 0.9 (drift) | 1 | 2.24e-04 [1.9e-04, 2.6e-04] | 1.85e-04 [1.6e-04, 2.1e-04] | 1.40e-04 [1.2e-04, 1.5e-04] | 2.05e-04 [1.8e-04, 2.4e-04] |
| h 8, s_f 0.9 (drift) | 10 | 6.49e-06 [5.5e-06, 7.2e-06] | 6.37e-06 [5.8e-06, 6.9e-06] | 7.44e-06 [6.4e-06, 8.6e-06] | 6.09e-06 [5.7e-06, 7.1e-06] |
| h 8, s_f 0.1 (jump) | 0.1 | 3.70e-02 [3.0e-02, 4.9e-02] | 2.41e-02 [1.9e-02, 3.0e-02] | 1.40e-02 [1.2e-02, 1.7e-02] | 3.09e-02 [2.5e-02, 3.6e-02] |
| h 8, s_f 0.1 (jump) | 1 | 2.24e-04 [1.9e-04, 2.6e-04] | 1.82e-04 [1.6e-04, 2.2e-04] | 1.28e-04 [1.1e-04, 1.5e-04] | 1.94e-04 [1.7e-04, 2.4e-04] |
| h 8, s_f 0.1 (jump) | 10 | 6.49e-06 [5.5e-06, 7.2e-06] | 8.19e-06 [7.2e-06, 8.8e-06] | 8.19e-06 [7.7e-06, 9.2e-06] | 5.78e-06 [5.5e-06, 6.6e-06] |
<!-- END GENERATED: ablation results -->

## 8. Interpretation (2026-11-09, after the results; per stratum only)

**Pre-committed reading (§4).**
- No stratum shows a sign flip or a held ↔ not-held flip.
- **H1 at h 4 drift:** same sign. d = +0.184 [+0.058, +0.310], down from +0.333.
- **H1 at h 4 frozen:** unresolved. d = +0.100 [−0.076, +0.277], down from +0.332. By §4 this is neither
  agreement nor a flip.
- **H2 d₁ and d₂:** same sign. d₁ grows from +0.861 to +1.084; d₂ is +0.930 against +0.944.
- **H5a:** same reading in each of the four strata. The u + d share rises to 0.97–1.01.
- **H6:** same reading in each of the twelve strata.

**The expected direction (§4) was not met for H1.** The d = f(10) − f(0.1) at h 4 frozen and h 4 drift was
expected to keep its sign *or grow*. It shrank in both strata. Both terms moved:

| h 4 stratum | f(0.1), criterion → fixed | f(10), criterion → fixed |
|---|---|---|
| frozen | 0.387 → 0.480 | 0.719 → 0.580 |
| drift | 0.264 → 0.350 | 0.597 → 0.534 |

- **γ0 0.1:** B removed more of the gain. B ran 2400 updates instead of a median of about 1500.
- **γ0 10:** B removed **less** of a larger gain:
  - after A, the gain is 2.294 [2.229, 2.360] against 2.059 at criterion;
  - B ran 2400 updates instead of about 21.

So under matched-loss stopping, exposure did not work against the γ0 effect on fraction removed, as §4
assumed. In these two strata, part of the criterion-run H1 difference goes with the stopping rule. Exposure
and over-training of A both change under a fixed budget, and this design does not separate the two.

**Categories (98.75%, descriptive here).**
- **H1, h 4 frozen:** meaningful → inconclusive.
- **H1, h 4 drift:** meaningful → nonzero, possibly trivial (lower bound under +0.10).
- **H2:** meaningful in both runs.
- **H5a:** meaningful in each stratum under the fixed budget. Under criterion, h 4 frozen and h 4 drift are
  "nonzero, possibly trivial".
- **H6 at γ0 1 and 10:** mostly "equivalent" around 0.9 ± 0.10. Centring a threshold hypothesis at its
  threshold is this report's choice, not a decision, so the H6 and H5a categories are descriptive only.

**Fit (final-loss table).**
- **γ0 0.1:** each phase ends between 0.006 and 0.040, at or below the 0.05 criterion.
- **γ0 1:** about 1e-4 to 2e-4.
- **γ0 10:** about 3e-6 to 9e-6, four orders of magnitude below the criterion.

Exposure is matched; fit is not, as accepted in §2.

**Floors.** The fixed-budget floors (0.0252, 0.0360, 0.0473) are within 1.1× of the criterion floors.

**For K9 (docs/29).** By the pre-committed rule there is no flip, so matched-loss stopping can stand. Two
things should be weighed beside the rule:
- H1 at h 4 frozen is unresolved under a fixed budget;
- the §4 expectation that exposure works against the γ0 effect was wrong.
